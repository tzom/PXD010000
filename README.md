# PXD010000

## Dependencies:
    wget, numpy, pandas, pyteomics, tensorflow>2.0


## step 1: download 

Download mzid and mzml files with wget from PRIDE:

```
python download_from_pride.py
```

## step 2: assign search results


Dumps all (peptide-spectrum)-pairs in a hdf-file:

```
python merge_mzid_mzml.py 
```

## step 3: Fire up a tensorflow.dataset

Performs loading,preprocessing,shuffle and batch-creation in a tensorflow.dataset

```
python hdf_ds.py
```


