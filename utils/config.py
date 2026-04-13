'''
-- config.py 
-- author: lopsal
    Configuration of the full pipeline deployment
    Centralices configurable settings in the code 
    This file is included in all system scripts
'''

import os 
from pathlib import Path 

# Base paths to find audios on dataset
# Configure this path to the relevant directories
BASE_DIR = r"C:\Users\andre\Documents\giaa\MASMOVIL\data-asvspoof"
FLAC_DIRS = [
    os.path.join(BASE_DIR, "ASVspoof2021_DF_eval_part00", "ASVspoof2021_DF_eval", "flac"),
    os.path.join(BASE_DIR, "ASVspoof2021_DF_eval_part01", "ASVspoof2021_DF_eval", "flac"),
    os.path.join(BASE_DIR, "ASVspoof2021_DF_eval_part02", "ASVspoof2021_DF_eval", "flac"),
    os.path.join(BASE_DIR, "ASVspoof2021_DF_eval_part03", "ASVspoof2021_DF_eval", "flac"),
]

# Metadata -- included in util/metadata
VCC_METADATA    = "../utils/metadata/metadata-original/ASVspoof2021_DF_VCC_MetaInfo.tsv"
VCTK_METADATA   = "../utils/metadata/metadata-original/ASVspoof2021_DF_VCTK_MetaInfo.tsv"

# Directories for outputs
OUTPUT_DIR      = '.'
MODELS_DIR      = 'models'
RESULTS_DIR     = 'results'
PLOTS_DIR       = os.path.join(RESULTS_DIR, 'plots')
LOG_DIR         = 'logs'

# Parameters for the dataset 
N_TRAIN_PER_CLASS   = 5000    
N_TEST_PER_CLASS    = 300      
N_PRUEBA_PER_CLASS  = 200

# Random state for models
RANDOM_STATE    = 13 

# Audio parameters
SR              = 16000     # sample rate
N_MFCC          = 40        # number of coefficients to extract
N_FFT           = 2048      # size of FFT
HOP_LENGTH      = 512       # hop-length for STFT

# Silence detection threshold in decibels 
# A threshold between 30-35 is recommended
TOP_DB          = 30 

# Number of parallel processes
# None = Uses all cores available 
# Specific number = use that number of cores
N_JOBS          = None 

'''
This is where the parameters for all models would go. 
It is not necessary right now; they can be entered directly into the model training file
and the decision panel creation file
'''

# Number of components for PCA reduction 
N_PCA_COMPONENTS    = 100

# Durations we evaluate for model performance at thresholds
# We start at 0.3 because we need a minimum amount of time to calculate the coefficients
DURATIONS = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

# Extraction strategies
#   Best:   region with the highest energy 
#   Start:  first region with voice
#   End:    last region with voice
POSITIONS   = ['best', 'start', 'end']

# Output filenames
# Labels
LABELS_TRAIN    = "labels_train.csv"
LABELS_TEST     = "labels_test.csv"
LABELS_PRUEBA   = "labels_threshold_test.csv"

# Features
FEATURES_TRAIN   = "features_train.csv"
FEATURES_TEST    = "features_test.csv"

# Models
MODEL_HGB       = "modelo_histgradientboost.pkl"
MODEL_SVM       = "modelo_svm_pca.pkl"
IMPUTER_FILE    = "imputer.pkl"
SCALER_FILE     = "scaler.pkl"
PCA_FILE        = "pca.pkl"

# Metadata 
METADATA_FILE           = "metadata.json"
FEATURE_NAMES_FILE      = "feature_names.txt"
FEATURE_IMPORTANCE_FILE = "feature_importance_pca.csv"

# Results
RESULTS_COMPLETE_FILE   = "complete_results.csv"
RESULTS_SUMMARY_FILE    = "summary_metrics.csv"

# Constants
# Total number of features to be extracted
N_FEATURES = 751

# Classification Classes
CLASSES = ['HUMAN', 'TTS', 'VC']

'''
Auxiliary Methods
'''
# Obtain number of jobs 
def get_n_jobs(): 
    # returns the number of parallel processes to be used
    if N_JOBS is not None: 
        return N_JOBS
    try: 
        from multiprocessing import cpu_count
        return max(1, cpu_count()-1)
    except: 
        return 1

# Verify directories
def ensure_directories(): 
    # creates directories if they do not exist
    dirs = [MODELS_DIR, RESULTS_DIR, PLOTS_DIR, LOG_DIR]
    for d in dirs: 
        os.makedirs(d, exist_ok=True)
        
# Validate configuration 
def validate_config(): 
    errors = []
    warnings = []
    
    # Verify audio directories
    existing_dirs = []
    for i, flac_dir in enumerate(FLAC_DIRS): 
        if os.path.exists(flac_dir): 
            existing_dirs.append(flac_dir)
        else: 
            warnings.append(f"FLAC_DIRS[{i}] does not exist: {flac_dir}")
    
    if not existing_dirs: 
        errors.append("No valid audio directories were found in FLAC_DIRS")
    
    # Verify metadata files
    if not os.path.exists(VCC_METADATA):
        errors.append(f"VCC_METADATA does not exist {VCC_METADATA}")
    
    if not os.path.exists(VCTK_METADATA):
        errors.append(f"VCTK_METADATA does not exist: {VCTK_METADATA}")
        
    # Verify parameters
    if TOP_DB < 15 or TOP_DB > 50:
        warnings.append(f"TOP_DB={TOP_DB} outside the typical range (15-50)")
    
    if N_PCA_COMPONENTS > N_FEATURES:
        errors.append(f"N_PCA_COMPONENTS ({N_PCA_COMPONENTS}) > N_FEATURES ({N_FEATURES})")
    
    if N_TRAIN_PER_CLASS < 100:
        warnings.append(f"N_TRAIN_PER_CLASS too low ({N_TRAIN_PER_CLASS}), it can affect performance")
    
    return errors, warnings, existing_dirs

# Run on import
if __name__ == '__main__':
    print("=" * 80)
    print('PIPELINE CONFIGURATION (TSV-ONLY + MULTI-DIR)')
    print("=" * 80)
    
    print("\nDirectories of configured audio files")
    for i, flac_dir in enumerate(FLAC_DIRS):
        exists = "✓" if os.path.exists(flac_dir) else "✗"
        print(f"  {exists} part0{i}: {flac_dir}")

    print("\nTSV Files:")
    print(f"  VCC_METADATA: {VCC_METADATA}")
    print(f"  VCTK_METADATA: {VCTK_METADATA}")
    
    print("\nDataset:")
    print(f"  Train: {N_TRAIN_PER_CLASS} × 3 = {N_TRAIN_PER_CLASS * 3} audios")
    print(f"  Test: {N_TEST_PER_CLASS} × 3 = {N_TEST_PER_CLASS * 3} audios")
    print(f"  Threshold Test: {N_PRUEBA_PER_CLASS} × 3 = {N_PRUEBA_PER_CLASS * 3} audios")
    
    print("\nClassification (from TSV):")
    print(f"  HUMAN: VCC_ID/VCTK_ID != '-'")
    print(f"  TTS: TTS_text != '-'")
    print(f"  VC: VC_source_VCC_ID/VC_source_VCTK_ID != '-'")
    
    print("\nParameters:")
    print(f"  TOP_DB: {TOP_DB} dB")
    print(f"  N_PCA_COMPONENTS: {N_PCA_COMPONENTS}")
    print(f"  N_JOBS: {get_n_jobs()} cores")
    
    print("\nValidating configuration...")
    errors, warnings, existing_dirs = validate_config()
    
    if errors:
        print("\nFOUND ERRORS:")
        for error in errors:
            print(f"  • {error}")