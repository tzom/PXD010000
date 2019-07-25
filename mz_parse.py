from scipy.sparse import csr_matrix
import numpy as np
import tensorflow as tf

# TODO remove those dependencies:
import matplotlib.pyplot as plt
import time

MZ_MAX = 2000
SPECTRUM_RESOLUTION = 2

def preprocess_spectrum(dummy,mz,intensity):
    #global MZ_MAX, SPECTRUM_RESOLUTION
    #ID,mz,intensity = x

    def _parse_indices(element):
        resolution = SPECTRUM_RESOLUTION
        element = np.around(element,resolution)
        element = (element * (10**resolution))
        return [element]

    def _rescale_spectrum(indices,values):
        # get unique indices, and positions in the array
        y,idx = np.unique(indices,return_index=True)

        # Use the positions of the unique values as the segment ids to sum segments up:
        values = np.add.reduceat(np.append(values,0), idx)

        ## Truncate
        mask = np.less(y,MZ_MAX * (10**SPECTRUM_RESOLUTION))
        indices = y[mask]
        values = values[mask]

        #make nested list [1, 2, 3] -> [[1],[2],[3]], as later requiered by SparseTensor:
        #indices = tf.reshape(indices, [tf.size(indices),1])

        return indices,values

    def _to_sparse(indices,values):
        from scipy.sparse import csr_matrix
        zeros = np.zeros(len(indices),dtype=np.int32)
        intensities_array = csr_matrix((values,(indices,zeros)),shape=(MZ_MAX * (10**SPECTRUM_RESOLUTION),1), dtype=np.float32).toarray().flatten()
        return intensities_array

    #### PREPROCESSING BEGIN #######
    # round indices according to spectrum resolution, SPECTRUM_RESOLUTION:
    mz = list(map(_parse_indices,mz))
                        # aggregate intensities accordingly to the new indices:
    mz,intensity = _rescale_spectrum(mz,intensity)

    # mz,intensity -> dense matrix of fixed m/z-range populated with intensities:
    spectrum_dense = _to_sparse(mz,intensity)
    # normalize by maximum intensity
    max_int = np.max(intensity)
    spectrum_dense = np.log(spectrum_dense+1.) - np.log(max_int)
    spectrum_dense = spectrum_dense.astype(np.float32)
    #print(spectrum)
    #### PREPROCESSING END #########
    return dummy,spectrum_dense

def tf_preprocess_spectrum(dummy,mz,intensity):
    #global MZ_MAX, SPECTRUM_RESOLUTION
   
    n_spectrum = MZ_MAX * 10**SPECTRUM_RESOLUTION
    mz = mz*10**SPECTRUM_RESOLUTION
    
    # TODO: check this:
    indices = tf.math.floor(mz)
    indices = tf.cast(indices,tf.int64)


    uniq_indices, i = tf.unique(indices)
    # TODO: check what exactly to use here, sum, max, mean, ...
    uniq_values = tf.math.segment_sum(intensity,i)

    # create as mask to truncate between 0<mz<max
    # eliminate zeros:
    notzero_mask = tf.math.greater(uniq_indices,tf.zeros_like(uniq_indices))    
    # truncate :
    trunc_mask = tf.math.less_equal(uniq_indices,tf.zeros_like(uniq_indices)+n_spectrum)
    # put into joint mask:
    mask = tf.logical_and(notzero_mask,trunc_mask)
    # apply mask:
    uniq_indices = tf.boolean_mask(uniq_indices,mask)
    uniq_values = tf.boolean_mask(uniq_values,mask)
    

    #### workaroud, cause tf.SparseTensor only works with tuple indices, so with stack zeros
    zeros = tf.zeros_like(uniq_indices)
    uniq_indices_tuples = tf.stack([uniq_indices, zeros],axis = 1)
    sparse = tf.SparseTensor(indices = uniq_indices_tuples, values = uniq_values,dense_shape = [n_spectrum,1])
    dense = tf.sparse.to_dense(sparse)

    #dense = tf.expand_dims(dense,axis=0)
    return dummy,dense

def normalize(intensities):
    max_int = tf.reduce_max(intensities)
    normalized = tf.log(intensities+1.)-tf.log(max_int+1)
    return normalized

def tf_maxpool(dense):
    shape = dense.shape
    dense = tf.reshape(dense,[1,-1,1,1])
    k = 100
    n_spectrum = int(shape[0])
    x, i = tf.nn.max_pool_with_argmax(dense,[1,k,1,1],[1,k,1,1],padding='SAME')
    i0 = tf.constant(np.arange(0,n_spectrum,k))
    i0 = tf.reshape(i0,[1,int(n_spectrum/k),1,1]) 
    i = i-i0
    return x,i

def tf_maxpool_with_argmax(dense,k=100):
    dense = tf.reshape(dense,[-1,k])
    x = tf.reduce_max(dense,axis=-1)
    i = tf.arg_max(dense,dimension=-1)
    return x,i

def parse(dummy,mz,intensity):
    dummy, dense = tf_preprocess_spectrum(dummy,mz, intensity)
    x,i = tf_maxpool_with_argmax(dense)
    x = normalize(x)
    return dummy, x, i 

if __name__ == "__main__":
    n_peaks = 400

    mz = np.sort(np.random.uniform(0,2000,size=n_peaks))
    #intensity = np.random.standard_exponential(size=n_peaks)
    intensity = np.random.gamma(1.0,size=n_peaks)
    print(intensity)

    _,dense = preprocess_spectrum(None,mz,intensity)
    print(dense.shape)

    #_,tf_dense = tf_preprocess_spectrum(None,mz, intensity)
    #x,i = tf_maxpool(tf_dense)

    _,x,i = parse(None,mz,intensity)
  
    mz_p = tf.placeholder(tf.int64,[n_peaks])
    intensities_p = tf.placeholder(tf.float32,[n_peaks])

    with tf.Session() as sess:
        tf_dense = sess.run((tf_dense),feed_dict={mz_p:mz,intensities_p:intensity})
        x,i = sess.run((x,i),feed_dict={mz_p:mz,intensities_p:intensity})

    print(np.squeeze(x).shape)
    plt.figure()
    plt.title('max')
    plt.plot(np.squeeze(x),color='orange',alpha=.8)
    plt.xlabel('m/z')
    plt.figure()
    plt.title('argmax')
    plt.plot(np.squeeze(i),color='black',alpha=.8)
    plt.xlabel('m/z')
    #plt.show()    
    plt.figure()
    plt.title('original')
    plt.plot(tf_dense)
    plt.xlabel('(e-2) m/z')
    plt.show()
