from pyteomics import mzid, mzml
import pandas as pd
import numpy as np
import glob,os

os.chdir('./files')

mzid_files=glob.glob('*.mzid')

for mzid_file in mzid_files:
    print('reading...', mzid_file)
    indexed_mzid = mzid.chain.from_iterable([mzid_file],use_index=True)

    def _parse_mzid_entry(entry):
        spectrum_id = str(entry['spectrumID'])
        seq = str(entry['SpectrumIdentificationItem'][0]['PeptideSequence'])
        try: 
            mods = entry['SpectrumIdentificationItem'][0]['Modification'] 
        except:
            mods = None
        rank = int(entry['SpectrumIdentificationItem'][0]['rank'])
        file_location = str(entry['name'])
        return file_location,spectrum_id,seq,mods,rank

    #all_mzid = []
    #for i,entry in enumerate(indexed_mzid.map(_parse_mzid_entry)):
    #    all_mzid.append(entry)

    all_mzid = list(indexed_mzid.map(_parse_mzid_entry))

    file_location,spectrum_ids,seq,mods,rank = zip(*all_mzid)

    mzid_df = pd.DataFrame({'file':file_location,'id':spectrum_ids,'seq':seq})

    def _parse_mzml_entry(entry):
        ID = str(entry['id'])
        mz = np.array(entry['m/z array'])
        intensities = np.array(entry['intensity array'])
        return ID, mz, intensities

    all_spectra = []

    for file in np.unique(file_location):
        
        indexed = mzml.MzML(file)
        for i,entry in enumerate(indexed.map(_parse_mzml_entry)):
            tupl = (file,)+entry
            all_spectra.append(tupl)

    mzml_location, ids, mz, intensities = zip(*all_spectra)

    spectra_df = pd.DataFrame({'file':mzml_location,'id':ids,'mz':mz,'intensities':intensities})

    #### MERGE: mzid + mzml

    merged_df = pd.merge(mzid_df,spectra_df,how='left',on=['file','id'])

    merged_df = merged_df[['id','seq','mz','intensities']]
    print('writing...', mzid_file)
    out_file = 'merged.hdf5'
    if not(os.path.exists(out_file)):
        merged_df.to_hdf(out_file,key=mzid_file,mode='w')
    else:
        merged_df.to_hdf(out_file,key=mzid_file)