import tensorflow as tf
import tensorflow.data as data
from one_hot_encode_seq import to_index, one_hot
tf.compat.v1.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE

import pandas as pd
import time

from scipy.sparse import csr_matrix
import numpy as np

MZ_MAX = 2000
SPECTRUM_RESOLUTION = 0

def preprocess_spectrum(dummy,mz,intensity):
    global MZ_MAX, SPECTRUM_RESOLUTION
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

df = pd.read_hdf('merged.hdf5','df')#,dtype= {'mz': str, 'intensities': str})
dataset_size = len(df)
seq, mz, intensities = df.seq.values, df.mz.values, df.intensities.values

### BEGIN PREPROCESSING

seq = list(map(to_index,seq))
seq = np.array(list(map(one_hot,seq)))

### END PREPROCESSING

BATCH_SIZE = 32
steps_per_epoch = int(dataset_size/BATCH_SIZE)

seq = tf.ragged.constant(seq)
mz = tf.ragged.constant(mz)
intensities = tf.ragged.constant(intensities)

seq = seq.to_tensor()
mz = mz.to_tensor()
intensities = intensities.to_tensor()

ds = data.Dataset.from_tensor_slices((seq,mz,intensities))
ds = data.Dataset.shuffle(ds,buffer_size = 10*BATCH_SIZE)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
ds = ds.map(lambda dummy, mz, intensities: tuple(tf.numpy_function(preprocess_spectrum, [dummy,mz,intensities], [dummy.dtype,tf.float32])),num_parallel_calls=AUTOTUNE)
ds = ds.prefetch(buffer_size=AUTOTUNE)

def timeit(ds, steps_per_epoch=steps_per_epoch):
  overall_start = time.time()
  # Fetch a single batch to prime the pipeline (fill the shuffle buffer),
  # before starting the timer
  it = iter(ds.take(steps_per_epoch))
  next(it)

  start = time.time()
  for i,(seq,x) in enumerate(it):
    if i%10 == 0:
      print('.',end='')
  end = time.time()

  duration = end-start
  print("{} steps_per_epoch: {} s".format(steps_per_epoch, duration))
  print("{:0.5f} elements/s".format(BATCH_SIZE*steps_per_epoch/duration))
  print("Total time: {}s".format(end-overall_start))

timeit(ds)

### Test, to get a single mini-batches, just loop over tf.dataset:
for i,b in enumerate(ds):
  print(i,b)
  break