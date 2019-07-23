import tensorflow as tf
import tensorflow.data as data
from one_hot_encode_seq import do_all_the_shit
tf.compat.v1.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE

import pandas as pd
import time

#from scipy.sparse import csr_matrix
import numpy as np
from mz_parse import tf_preprocess_spectrum

MZ_MAX = 2000
SPECTRUM_RESOLUTION = 2


df = pd.read_hdf('./files/merged.hdf5','df')#,dtype= {'mz': str, 'intensities': str})
dataset_size = len(df)
seq, mz, intensities = df.seq.values, df.mz.values, df.intensities.values

### BEGIN PREPROCESSING

seq = np.array(list(map(do_all_the_shit,seq)))

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

print(data.get_output_shapes(ds))
#ds = ds.map(lambda dummy, mz, intensities: tuple(tf.numpy_function(preprocess_spectrum, [dummy,mz,intensities], [dummy.dtype,tf.float32])),num_parallel_calls=AUTOTUNE)
ds = ds.map(lambda dummy, mz, intensities: tuple(tf_preprocess_spectrum(dummy,mz,intensities)),num_parallel_calls=AUTOTUNE)
ds = ds.batch(BATCH_SIZE)
print(data.get_output_shapes(ds))
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