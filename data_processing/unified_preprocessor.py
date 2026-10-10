# -*- coding: utf-8 -*-
"""
Unified Preprocessor for ICU Mortality Models

Generates preprocessed data in multiple formats:
1. Raw format (for GRU-D) - preserves NaN and masks
2. 2D format (for Logistic Regression/SVM/XGBoost) - flattened and forward-filled
3. 3D window format (for GRU) - time-windowed aggregation

Usage:
    from unified_preprocessor import UnifiedPreprocessor
    
    prep = UnifiedPreprocessor(
        data_dir="Data_group_7"  # relative to the repo root
    )
    prep.prepare_all_formats(output_dir="preprocessed_data")
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
import sys
import json

from .data_loader import DataLoader, ICUArrays


# Repo root (this file lives in <repo>/data_processing/)
REPO_ROOT = Path(__file__).resolve().parent.parent


class UnifiedPreprocessor:
    """
    Unified preprocessing pipeline for all 5 models.
    
    Handles:
    - Data loading and splitting
    - Forward-fill imputation (using training set statistics)
    - Multi-format data generation
    - Comprehensive metadata logging
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize preprocessor.
        
        Args:
            data_dir: Path to Data_group_7 directory
        """
        self.data_dir = Path(data_dir)
        self.loader = DataLoader(data_dir=str(self.data_dir))
        self.metadata = {}
        
    def prepare_all_formats(self, output_dir: str = str(REPO_ROOT / "preprocessed_data")) -> None:
        """
        Generate all data formats in one go.
        
        Pipeline:
        1. Load raw data
        2. Train/val/test split
        3. Compute training statistics
        4. Generate all formats
        5. Save metadata
        
        Args:
            output_dir: Base output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("UNIFIED PREPROCESSING PIPELINE")
        print("="*70)
        
        # ===== Step 1: Load raw data =====
        print("\n[1/7] Loading raw data...")
        arrays = self.loader.load()
        n_total = len(arrays.patient_ids)
        n_valid = sum(1 for m in arrays.Mortality if not np.isnan(m))
        print(f"  ✓ Loaded {n_total} patients ({n_valid} with valid mortality)")
        self.metadata['total_patients'] = n_total
        self.metadata['valid_patients'] = n_valid
        
        # ===== Step 2: Split data =====
        print("\n[2/7] Splitting data (train:val:test = 8:1:1)...")
        train, val, test = self.loader.split_data(arrays)
        print(f"  ✓ Train: {len(train.patient_ids)}, Val: {len(val.patient_ids)}, Test: {len(test.patient_ids)}")
        
        # Store split sizes
        self.metadata['split'] = {
            'train_size': len(train.patient_ids),
            'val_size': len(val.patient_ids),
            'test_size': len(test.patient_ids)
        }
        
        # ===== Step 3: Compute training statistics =====
        print("\n[3/7] Computing group means from training data...")
        ab_tr, g_tr = self.loader.makebin(train.X_static)
        ab_va, g_va = self.loader.makebin(val.X_static)
        ab_te, g_te = self.loader.makebin(test.X_static)
        
        group_mean, overall_mean = self.loader.group_mean_calculator(
            train.X_dynamic, train.M_dynamic, ab_tr, g_tr
        )
        print(f"  ✓ Computed {group_mean.shape[0]} age-gender groups")
        print(f"    Overall mean: {overall_mean[:5]}... (first 5 features)")
        
        # ===== Step 4: Forward fill =====
        print("\n[4/7] Forward-filling missing values...")
        Xtr_imp = self.loader.forwardfill(
            train.X_dynamic, train.M_dynamic, ab_tr, g_tr, group_mean, overall_mean
        )
        Xva_imp = self.loader.forwardfill(
            val.X_dynamic, val.M_dynamic, ab_va, g_va, group_mean, overall_mean
        )
        Xte_imp = self.loader.forwardfill(
            test.X_dynamic, test.M_dynamic, ab_te, g_te, group_mean, overall_mean
        )
        print(f"  ✓ Imputation complete")
        print(f"    Remaining NaN: train={np.isnan(Xtr_imp).sum()}, "
              f"val={np.isnan(Xva_imp).sum()}, test={np.isnan(Xte_imp).sum()}")
        
        # ===== Step 5: Save raw format (for GRU-D) =====
        print("\n[5/7] Saving raw format (for GRU-D)...")
        self._save_raw_format(
            train, val, test,
            output_dir / "raw_format"
        )
        
        # ===== Step 6: Save 2D format (for LR/SVM/XGBoost) =====
        print("\n[6/7] Saving 2D format (for Logistic/SVM/XGBoost)...")
        self._save_2d_format(
            Xtr_imp, train.M_dynamic, train.Mortality, train.X_static,
            Xva_imp, val.M_dynamic, val.Mortality, val.X_static,
            Xte_imp, test.M_dynamic, test.Mortality, test.X_static,
            output_dir / "2d_format"
        )
        
        # ===== Step 7: Save 3D window format (for GRU) =====
        print("\n[7/7] Saving 3D window aggregations (for GRU)...")
        for window_h in [3, 6, 12, 24]:
            self._save_3d_window_format(
                Xtr_imp, train.M_dynamic, train.Mortality,
                Xva_imp, val.M_dynamic, val.Mortality,
                Xte_imp, test.M_dynamic, test.Mortality,
                window_hours=window_h,
                output_dir=output_dir / "3d_window_format"
            )
        
        # Save metadata
        self._save_metadata(output_dir)
        
        print("\n" + "="*70)
        self._print_summary(output_dir)
        print("="*70)
    
    def _save_raw_format(self, train: ICUArrays, val: ICUArrays, test: ICUArrays, output_dir: Path) -> None:
        """
        Save original data with NaN (for GRU-D).
        
        GRU-D requires:
        - X: values with NaN
        - M: observation mask
        - D: time-to-event (decay)
        - y: mortality labels
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
            np.savez_compressed(
                output_dir / f"grud_raw_{split_name}.npz",
                X=split_data.X_dynamic,  # (N, 48, 36) with NaN
                M=split_data.M_dynamic,  # (N, 48, 36) observation mask
                D=split_data.D_dynamic,  # (N, 48, 36) time-to-event
                X_static=split_data.X_static,  # (N, 5)
                patient_ids=np.array(split_data.patient_ids, dtype=object),
                y=np.array(split_data.Mortality, dtype=np.float32)  # (N,)
            )
        
        print(f"  ✓ Raw format saved to {output_dir}")
        print(f"    Format: X:(N,48,36), M:(N,48,36), D:(N,48,36), y:(N,)")
    
    def _save_2d_format(
        self,
        Xtr, Mtr, ytr, X_static_tr,
        Xva, Mva, yva, X_static_va,
        Xte, Mte, yte, X_static_te,
        output_dir: Path
    ) -> None:
        """
        Flatten to 2D (for Logistic Regression/SVM/XGBoost).
        
        Converts (N, 48, 36) → (N, 48×36=1728)
        Each time step × feature pair becomes a single column.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Flatten spatial + temporal dimensions
        X_train_2d = Xtr.reshape(len(Xtr), -1)  # (N_tr, 1728)
        X_val_2d = Xva.reshape(len(Xva), -1)    # (N_va, 1728)
        X_test_2d = Xte.reshape(len(Xte), -1)   # (N_te, 1728)
        
        # Also flatten masks for optional feature engineering
        M_train_2d = Mtr.reshape(len(Mtr), -1)
        M_val_2d = Mva.reshape(len(Mva), -1)
        M_test_2d = Mte.reshape(len(Mte), -1)
        
        splits = [
            ("train", X_train_2d, M_train_2d, ytr, X_static_tr),
            ("val", X_val_2d, M_val_2d, yva, X_static_va),
            ("test", X_test_2d, M_test_2d, yte, X_static_te)
        ]
        
        for split_name, X, M, y, X_static in splits:
            np.savez_compressed(
                output_dir / f"2d_{split_name}.npz",
                X=X,  # (N, 1728) - flattened time×features
                M=M,  # (N, 1728) - optional observation mask
                X_static=X_static,  # (N, 5) - static features
                y=np.array(y, dtype=np.float32)  # (N,)
            )
        
        print(f"  ✓ 2D format saved to {output_dir}")
        print(f"    Shape: train {X_train_2d.shape}, val {X_val_2d.shape}, test {X_test_2d.shape}")
    
    def _save_3d_window_format(
        self,
        Xtr, Mtr, ytr,
        Xva, Mva, yva,
        Xte, Mte, yte,
        window_hours: int,
        output_dir: Path
    ) -> None:
        """
        Window aggregation to 3D (for GRU).
        
        Divides 48 hours into windows (e.g., 6-hour windows → 8 time steps).
        For each window, computes 5 statistics per feature: mean, max, min, std, last value.
        Output shape: (N, n_windows, n_features × 5)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        n_windows = 48 // window_hours
        n_vars = Xtr.shape[2]  # 36 features
        n_stats = 5  # mean, max, min, std, last
        
        def aggregate(X_filled: np.ndarray, M: np.ndarray) -> np.ndarray:
            """
            Aggregate X_filled over windows.
            
            Args:
                X_filled: (N, 48, 36) - forward-filled values
                M: (N, 48, 36) - observation mask
                
            Returns:
                (N, n_windows, 36×5=180) - aggregated features
            """
            N = X_filled.shape[0]
            all_windows = []
            
            for w in range(n_windows):
                start = w * window_hours
                end = (w + 1) * window_hours
                
                X_win = X_filled[:, start:end, :]  # (N, window_hours, 36)
                M_win = M[:, start:end, :]         # (N, window_hours, 36)
                
                # For each variable, compute statistics
                window_features = []
                for var_idx in range(n_vars):
                    X_var = X_win[:, :, var_idx]  # (N, window_hours)
                    M_var = M_win[:, :, var_idx]  # (N, window_hours)
                    
                    # Mask unobserved values
                    X_obs = np.where(M_var == 1, X_var, np.nan)
                    
                    # Compute 5 statistics
                    feat_mean = np.nanmean(X_obs, axis=1)
                    feat_max = np.nanmax(X_obs, axis=1)
                    feat_min = np.nanmin(X_obs, axis=1)
                    feat_std = np.nanstd(X_obs, axis=1)
                    feat_last = X_var[:, -1]  # Last value in window
                    
                    # Replace NaN with 0 (no observations)
                    feat_mean = np.nan_to_num(feat_mean, 0)
                    feat_max = np.nan_to_num(feat_max, 0)
                    feat_min = np.nan_to_num(feat_min, 0)
                    feat_std = np.nan_to_num(feat_std, 0)
                    
                    # Stack: (N, 5)
                    var_feat = np.stack([
                        feat_mean, feat_max, feat_min, feat_std, feat_last
                    ], axis=1)
                    
                    window_features.append(var_feat)
                
                # Concatenate all variables: (N, 36×5=180)
                window_all = np.concatenate(window_features, axis=1)
                all_windows.append(window_all)
            
            # Stack over windows: (N, n_windows, 180)
            return np.stack(all_windows, axis=1)
        
        # Aggregate all splits
        X_train_3d = aggregate(Xtr, Mtr)
        X_val_3d = aggregate(Xva, Mva)
        X_test_3d = aggregate(Xte, Mte)
        
        # Save
        prefix = f"gru_{window_hours}h"
        splits = [
            ("train", X_train_3d, ytr),
            ("val", X_val_3d, yva),
            ("test", X_test_3d, yte)
        ]
        
        for split_name, X, y in splits:
            np.savez_compressed(
                output_dir / f"{prefix}_{split_name}.npz",
                X=X,  # (N, n_windows, 180)
                y=np.array(y, dtype=np.float32)
            )
        
        print(f"  ✓ {window_hours}h window: shape {X_train_3d.shape}")
    
    def _save_metadata(self, output_dir: Path) -> None:
        """Save preprocessing metadata for reference."""
        metadata = {
            'data_dir': str(self.data_dir),
            'output_dir': str(output_dir),
            'loader_config': {
                'T': self.loader.T,
                'F': self.loader.F,
                'S': self.loader.S,
                'agg_method': self.loader.agg
            },
            'features': {
                'ts_features': self.loader.TS_FEATURES,
                'static_features': self.loader.STATIC_FEATURES
            },
            'preprocessing': self.metadata,
            'formats': {
                'raw': {
                    'description': 'Original data with NaN and masks (for GRU-D)',
                    'shape': '(N, 48, 36) with M and D',
                    'files': ['grud_raw_train.npz', 'grud_raw_val.npz', 'grud_raw_test.npz']
                },
                '2d': {
                    'description': 'Flattened forward-filled data (for LR/SVM/XGBoost)',
                    'shape': '(N, 1728)',
                    'files': ['2d_train.npz', '2d_val.npz', '2d_test.npz']
                },
                '3d_windows': {
                    'description': 'Time-windowed aggregations (for GRU)',
                    'windows': [3, 6, 12, 24],
                    'shape_per_window': '(N, n_windows, 180)',
                    'example': '6h window: (N, 8, 180)'
                }
            }
        }
        
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"\n  ✓ Metadata saved to {metadata_path}")
    
    def _print_summary(self, output_dir: Path) -> None:
        """Print comprehensive summary of generated files."""
        print("\n📊 PREPROCESSING COMPLETE\n")
        
        print("📁 Generated Files Structure:")
        print(f"   {output_dir}/")
        print(f"   ├── raw_format/           (for GRU-D)")
        print(f"   │   ├── grud_raw_train.npz")
        print(f"   │   ├── grud_raw_val.npz")
        print(f"   │   └── grud_raw_test.npz")
        print(f"   │")
        print(f"   ├── 2d_format/            (for Logistic/SVM/XGBoost)")
        print(f"   │   ├── 2d_train.npz      → X:(N,1728), y:(N,)")
        print(f"   │   ├── 2d_val.npz")
        print(f"   │   └── 2d_test.npz")
        print(f"   │")
        print(f"   ├── 3d_window_format/     (for GRU)")
        print(f"   │   ├── gru_3h_*.npz      → X:(N,16,180)")
        print(f"   │   ├── gru_6h_*.npz      → X:(N,8,180)")
        print(f"   │   ├── gru_12h_*.npz     → X:(N,4,180)")
        print(f"   │   └── gru_24h_*.npz     → X:(N,2,180)")
        print(f"   │")
        print(f"   └── metadata.json         (preprocessing info)")
        
        print("\n\n📋 Quick Reference:")
        print("   Logistic Regression: Load from 2d_format/")
        print("   SVM:                 Load from 2d_format/")
        print("   XGBoost:             Load from 2d_format/")
        print("   GRU-D:               Load from raw_format/")
        print("   GRU:                 Load from 3d_window_format/ (recommend 6h)")


def main():
    """
    Main execution function.
    
    Usage:
        python unified_preprocessor.py [data_dir] [output_dir]
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified ICU data preprocessing")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(REPO_ROOT / "Data_group_7"),
        help="Path to Data_group_7 directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "preprocessed_data"),
        help="Output directory for preprocessed data"
    )
    
    args = parser.parse_args()
    
    try:
        preprocessor = UnifiedPreprocessor(data_dir=args.data_dir)
        preprocessor.prepare_all_formats(output_dir=args.output_dir)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
