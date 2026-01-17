import numpy as np
import shutil
import pandas as pd
from ms_entropy import read_one_spectrum, standardize_spectrum, clean_spectrum
from dynamic_entropy_search.dynamic_entropy_search import DynamicEntropySearch

Standardize_Info = {
    "spectrum_id": (["spectrum_id", "id", "spectrumid", "scan_number", "index"], "", str),
    "precursor_mz": (["precursor_mz", "precursormz"], 0.0, float),
    "peaks": (["peaks", "intensities", "mzs_intensities"], [], list),
    "precursor_type": (["precursor_type", "precursortype", "adduct"], "", str),
    "smiles": (["smiles", "Smiles", "SMILES"], "", str),
    "inchikey": (["inchikey", "InChIKey", "INCHIKEY"], "", str),
    "retention_time": (["retentiontime", "retention_time","rt"], None, float),
    "name": (["name", "label"], "", str),  
}

def get_spectra_list(spectra_database):
    spec_list = []
    for spec in read_one_spectrum(spectra_database):
        spec = standardize_spectrum(spec, Standardize_Info)
        
        if len(spec['peaks']) ==0:
            continue
        
        spec['peaks'] = clean_spectrum(
            peaks=spec['peaks'], 
            noise_threshold=0.01,
            max_mz=spec['precursor_mz'] - 1.6, 
            min_ms2_difference_in_da=0.321,
            normalize_intensity=True)
        
        peaks = spec['peaks']
        peaks = np.array(peaks, dtype = np.float32)
        spectrum_id = spec.get('spectrum_id', "")
        name = spec.get('name', '')
        retention_time = spec.get('retention_time', '')
        precursor_mz = spec.get('precursor_mz', 0.0)
        precursor_type = spec.get('precursor_type', '')
        smiles = spec.get('smiles',  '')
        inchikey = spec.get('inchikey',  '')

        spectra = {
            'spectrum_id': spectrum_id,
            'name': name,
            'retention_time': retention_time,
            'precursor_mz': precursor_mz,
            'precursor_type': precursor_type,
            'smiles': smiles,
            'inchikey': inchikey,
            'peaks': peaks    
        }
        
        spec_list.append(spectra)
        
    return spec_list

def build_spectra_library(spec_list, library_path):
    entropy_search = DynamicEntropySearch(path_data=library_path,
                                          max_ms2_tolerance_in_da=0.16,
                                          )
    entropy_search.add_new_spectra(spectra_list=spec_list)
    entropy_search.build_index()
    entropy_search.write()
    
    
def library_search(search_data, library_path):
    entropy_search=DynamicEntropySearch(path_data=library_path)
    query_spec_list = get_spectra_list(search_data)
    match_list = []
    for spec in query_spec_list:
        result=entropy_search.search_topn_matches(
                precursor_mz=spec['precursor_mz'],
                peaks=spec['peaks'],
                ms1_tolerance_in_da=0.14, # You can change ms1_tolerance_in_da as needed.
                ms2_tolerance_in_da=0.16, # You can change ms2_tolerance_in_da as needed.
                method='identity', # or 'neutral_loss' or 'hybrid' or 'identity'.
                clean=False, # If you don't want to use the internal clean process in this function, set it to False.
                topn=1, # You can change topn as needed.
                need_metadata=True, # Set it to True if need metadata.
        )
        if result:
            match_result = {
                'spectrum_id': spec['spectrum_id'],
                'mz': spec['precursor_mz'],
                'retention_time': spec['retention_time'],
                'precursor_type': spec['precursor_type'],
                'top1_name': result[0]['name'],
                'adduct': result[0]['precursor_type'],
                'identity_score': result[0]['identity_search_entropy_similarity'],
                'smiles': result[0]['smiles'],
                'inchikey': result[0]['inchikey'],
            }
            match_list.append(match_result)
        
    return match_list
        
        
if __name__ == "__main__":

    spectra_database = '/y/zchen/Documents/spectral_database/all_clean_noMonaExp_neg.msp'
    library_path ='/y/zchen/DynamicEntropySearch_library/all_clean_noMonaExp_neg_ms2_016'
    search_data = '/y/zchen/Documents/ZJUH1/plasma_neg_mzML/masscube/project_files/features.msp'
    spec_list = get_spectra_list(spectra_database)
    
    shutil.rmtree(library_path, ignore_errors=True)
    build_spectra_library(spec_list, library_path)
    match_list = library_search(search_data, library_path)
    
    output_df = pd.DataFrame(match_list)
    output_df.to_excel('/y/zchen/Documents/ZJUH1/plasma_neg_mzML/masscube/project_files/identity_014_016_top1.xlsx', index=False, engine='openpyxl')

    