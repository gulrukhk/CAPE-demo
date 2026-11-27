# CAPE Pretraining on MIMIC ECG Dataset

This repository contains a small working example demonstrating **CAPE** pretraining on the **MIMIC ECG** dataset and subsequent supervised training using the **PTB-XL** dataset.

---

## 📁 Folder Structure

```bash
.
├── data
│   ├── ECGdata
│   │   ├── ptb
│   │   |    ├── ecg_data.h5
│   │   |    └── ecg_labels.csv
│   │   └── mimic
│   │       ├── ecg_data.h5
│   │       └── ecg_labels.csv
│   └── DataSplits
│       └── ptb
│           └── ptb_*_split.csv
├── feats
│   └── ptb
│       └── CAPE_feats_ptb_bcsv.h5
├── indexes
│   └── mimic
│       └── indexes_mimic
│           └── indexes_x.h5
├── models
│   └── CAPE_mimic
├── utils
├── results
├── CAPE_training.ipynb
├── MLP_training_PTB_XL.ipynb
└── CAPE_get_feats.ipynb

📦 Directory and File Descriptions

    data/
    Contains the ECG data for pretraining:
        ECGdata/            
        The folder for each dataset contains:
            ECG signals as .h5 files.
            Labels as .csv files.
        DataSplits/
        The folder for each dataset contains:
            csv files with filenames for Train/validation/test splits 

    indexes/
    Contains index files for each pretraining cohort.
        The folder for each dataset contains:
            Indexes for two ECGs from the same patient for each epoch.
            CSV files for train and val.
            Shape: (samples, epochs, 2)

    models/
    Contains pretrained CAPE models.

    utils/
    Utility scripts and helper functions used across notebooks.

    results/
    Output folder for storing results of supervised training.

    CAPE_training.ipynb
    Notebook for CAPE model pretraining using the MIMIC ECG dataset.

    MLP_training_PTB_XL.ipynb
    Notebook for supervised training using PTB-XL labels and features extracted from the pretrained CAPE model.
    
    CAPE_get_feats.ipynb
    Notebook for generating features for a dataset

📝 Notes

    Make sure all required data and index files are placed correctly in the folder structure shown above.

    Pretrained models can be reused for downstream tasks or fine-tuned on other datasets.

🔧 Dependencies

To run the notebooks, install the required Python packages:

pip install -r requirements.txt

    It's recommended to use a virtual environment to avoid dependency conflicts.
