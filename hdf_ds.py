
#%%
#from pyteomics import mgf
import pandas as pd
import tensorflow as tf
import numpy as np
import time
#import matplotlib.pyplot as plt

from one_hot_encode_seq import do_all_the_shit, rev_one_hot, to_aa, index_to_aa
AUTOTUNE = tf.data.experimental.AUTOTUNE


max_len=24

MZ_MAX=1900
SPECTRUM_RESOLUTION=2
k = 50

def set_k(new_k):
    global k
    k = new_k
    return k

def tf_preprocess_spectrum(mz,intensity):
    '''
    converts a peaks list (mz,intensity) into a dense spectrum.
    '''
    #global MZ_MAX, SPECTRUM_RESOLUTION
   
    n_spectrum = MZ_MAX * 10**SPECTRUM_RESOLUTION
    mz = mz*10**SPECTRUM_RESOLUTION
    
    # TODO: check this:
    indices = tf.math.floor(mz)
    indices = tf.cast(indices,tf.int64)


    uniq_indices, i = tf.unique(indices)
    # TODO: check what exactly to use here, sum, max, mean, ...
    uniq_values = tf.math.segment_max(intensity,i)

    # create as mask to truncate between min<mz<max
    # eliminate zeros:
    lower_bound = 100 * 10**SPECTRUM_RESOLUTION
    notzero_mask = tf.math.greater(uniq_indices,tf.zeros_like(uniq_indices)+lower_bound)    
    # truncate :
    trunc_mask = tf.math.less(uniq_indices,tf.zeros_like(uniq_indices)+n_spectrum)
    # put into joint mask:
    mask = tf.logical_and(notzero_mask,trunc_mask)
    # apply mask:
    uniq_indices = tf.boolean_mask(uniq_indices,mask)
    uniq_indices = uniq_indices - lower_bound
    uniq_values = tf.boolean_mask(uniq_values,mask)
    

    #### workaroud, cause tf.SparseTensor only works with tuple indices, so with stack zeros
    zeros = tf.zeros_like(uniq_indices)
    uniq_indices_tuples = tf.stack([uniq_indices, zeros],axis = 1)
    sparse_spectrum = tf.SparseTensor(indices = uniq_indices_tuples, values = uniq_values,dense_shape = [n_spectrum-lower_bound,1])
    dense_spectrum = tf.sparse.to_dense(sparse_spectrum)

    return dense_spectrum

# def normalize(intensities):
#     max_int = tf.reduce_max(intensities)
#     normalized = tf.log(intensities+1.)-tf.log(max_int+1)
#     return normalized

def tf_maxpool(dense):
    shape = dense.shape
    dense = tf.reshape(dense,[1,-1,1,1])
    #k = 100
    n_spectrum = int(shape[0])
    x, i = tf.nn.max_pool_with_argmax(dense,[1,k,1,1],[1,k,1,1],padding='SAME')
    i0 = tf.constant(np.arange(0,n_spectrum,k))
    i0 = tf.reshape(i0,[1,int(n_spectrum/k),1,1]) 
    i = i-i0
    return x,i

def tf_maxpool_with_argmax(dense,k):
    dense = tf.reshape(dense,[-1,k])
    x = tf.reduce_max(dense,axis=-1)
    i = tf.math.argmax(dense,axis=-1)
    return x,i

def ion_current_normalize(intensities):
    total_sum = tf.reduce_sum(intensities**2)
    normalized = intensities/total_sum
    return normalized

def parse(mz,intensity):
    '''
    converts a peaks list (mz,intensity) into a two-vector spectrum
    '''
    intensity = ion_current_normalize(intensity)
    
    spectrum_dense = tf_preprocess_spectrum(mz, intensity)
    
    x,i = tf_maxpool_with_argmax(spectrum_dense,k=k)
    x = tf.cast(x,tf.float32)
    i = tf.cast(i,tf.float32)
    i = i/tf.cast(k,tf.float32)
    spectrum_two_vec = tf.stack([x,i],axis=1)
    return  spectrum_two_vec

def fire_up_generator(file_path="./ptm21.hdf5",n=1,preshuffle=True):
    with pd.HDFStore(file_path) as hdf:
        keys = np.array(hdf.keys())
        df = pd.DataFrame()
        for key in keys:
            print(key)
            df = df.append(hdf.select(key=key))
    n = len(df)
    print('n datapoints:',len(df))
    if preshuffle:
      df = df.sample(frac=1)
      print('preshuffling done.')
    global index        
    index = 0
    def generator():        
        #r_entry = df.sample(n)
        global index
        r_entry = df
        if index > (n-1):
          index=0
        seq = str(r_entry.iloc[index]['seq'])
        seq = np.array(do_all_the_shit(seq,max_len=max_len,k=None))        
        mz,i = np.array(r_entry.iloc[index]['mz']), np.array(r_entry.iloc[index]['intensities'])
        index+=1

        yield mz,i,seq
    return generator

def get_dataset(generator,batch_size=16,training=True):
    ds = tf.data.Dataset.from_generator(generator,output_types=(tf.float32,tf.float32,tf.float32),output_shapes=(None,None,(None,max_len,22)))
    if training:
        ds = ds.shuffle(500000)
    ds = ds.repeat(batch_size)
    ds = ds.map(lambda mz,intensities,seq: (parse(mz,intensities),seq),num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.repeat()
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

if __name__ == "__main__":   

    # peaks-list:
    seq,mz,i=next(iter(fire_up_generator("./files/merged.hdf5")()))
    print(len(seq),len(mz),len(i))
    print(seq,mz,i)

    # two_vector representation
    spectra,seqs=next(iter(get_dataset(fire_up_generator("./files/merged.hdf5"))))
    print(spectra)
    print(seqs)
    # print(x[0][0])
    

