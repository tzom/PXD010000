
#%%
#from pyteomics import mgf
import pandas as pd
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

from one_hot_encode_seq import do_all_the_shit, rev_one_hot, to_aa, index_to_aa
AUTOTUNE = tf.data.experimental.AUTOTUNE

MZ_MAX=2000
SPECTRUM_RESOLUTION=2
max_len=24

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
    #trunc_mask = tf.math.less_equal(uniq_indices,tf.zeros_like(uniq_indices)+n_spectrum)
    trunc_mask = tf.math.less(uniq_indices,tf.zeros_like(uniq_indices)+n_spectrum)
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
    x = tf.cast(x,tf.float32)
    i = tf.cast(i,tf.float32)
    output = tf.stack([x,i],axis=1)
    return output, dummy 

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

        yield seq,mz,i 
    return generator

def get_dataset(generator,batch_size=16,training=True):
    ds = tf.data.Dataset.from_generator(generator,output_types=(tf.float32,tf.float32,tf.float32),output_shapes=((None,max_len,22),None,None))
    if training:
        ds = ds.shuffle(500000)
    ds = ds.repeat(batch_size)
    ds = ds.map(lambda seq,mz,intensities: tuple(parse(seq,mz,intensities)),num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.repeat()
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

if __name__ == "__main__":   
    generator = fire_up_generator("./files/merged.hdf5")
    it = get_dataset(generator,batch_size=16).make_one_shot_iterator()
    next_ = it.get_next()
    sess = tf.Session()
    for i in range(100):
      x,s = sess.run(next_)
      print(x.shape)


#%%