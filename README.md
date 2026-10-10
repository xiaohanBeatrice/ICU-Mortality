# ICU Mortality Prediction: Data Processing & GRU-D

In-hospital mortality prediction for ICU patients. This repository contains the **data processing pipeline** and the **GRU-D model** from our group project.

## Structure

- `data_processing/` – raw `.txt` loading, train/val/test split, mask/decay computation, unified preprocessing for all models
- `run_preprocessing.py` – CLI entry point for preprocessing (`--data-dir`, `--output-dir`)
- `example_data_usage.py` – example of loading the processed data
- `GRU_D/` – GRU-D model (`gru_d.py`), its data loader and preprocessor
- `GRU_D/Results/{1h,3h,6h,12h}/` – result plots and summaries per observation window

## Usage

```bash
python run_preprocessing.py --data-dir Data_group_7 --output-dir preprocessed_data
python GRU_D/gru_d.py --data-dir Data_group_7
```

Raw patient data (`Data_group_7/`, containing `Features_group_7/*.txt` and `Outcomes-7.txt`) and trained model weights (`.pt`) are **not** included in this repo. Place the data folder in the repo root as `Data_group_7/` before running preprocessing, or point to it with `--data-dir`.

## Credits

Group project collaboration. Original repository: Gokul2799/ICU-Mortality (the other models and files there are not part of this repo).
