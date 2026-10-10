# -*- coding: utf-8 -*-
"""
Enhanced Data Loader for ICU Mortality Prediction
Improved version of data_process_v2.py with bug fixes and cross-platform support

"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ICUArrays:
    """Container for ICU patient data arrays"""
    X_dynamic: np.ndarray   # (N,48,F) - dynamic features
    M_dynamic: np.ndarray   # (N,48,F) - observation mask
    D_dynamic: np.ndarray   # (N,48,F) - time-to-event (decay)
    X_static: np.ndarray    # (N,S) - static features
    patient_ids: List[str]
    Mortality: List[float]  # Changed from List[int] to handle NaN


class DataLoader:
    """
    Unified data loader for ICU mortality dataset.
    
    Features:
    - Cross-platform path handling
    - Robust MechVent handling
    - NaN-aware mortality tracking
    - Multiple aggregation strategies (last/mean/median)
    """

    STATIC_FEATURES = ["Age", "Gender", "Height", "ICUType", "Weight"]

    TS_FEATURES = [
        "Albumin","ALP","ALT","AST","Bilirubin",
        "BUN","Cholesterol","Creatinine",
        "DiasABP","FiO2","GCS","Glucose","HCO3","HCT",
        "HR","K","Lactate","MAP","MechVent","Mg","Na",
        "NIDiasABP","NIMAP","NISysABP",
        "PaCO2","PaO2","pH","Platelets",
        "RespRate","SaO2","SysABP","Temp",
        "TroponinI","TroponinT","Urine","WBC"
    ]

    OUT_FEATURES = ['RecordID','SAPS-I','SOFA','Length_of_stay','Survival','In-hospital_death']

    def __init__(self, data_dir: str, agg: str = "last"):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Path to data directory containing Features_group_7/Features_group_10 and Outcomes-7.txt
            agg: Aggregation method for time series ("last", "mean", or "median")
        """
        self.data_dir = Path(data_dir)
        self.agg = agg.lower()
        if self.agg not in {"last", "mean", "median"}:
            raise ValueError("agg must be one of: last/mean/median")

        self.T = 48  # Hours
        self.F = len(self.TS_FEATURES)  # Number of time-series features
        self.S = len(self.STATIC_FEATURES)  # Number of static features
        self.O = len(self.OUT_FEATURES)

    @staticmethod
    def _time_to_hour(t: str) -> int:
        """
        Convert HH:MM format to hour index (0-47).
        """
        try:
            hh, mm = str(t).split(":")
            h = int(hh)
            return np.clip(h, 0, 47)
        except (ValueError, IndexError):
            return 0

    def _read_one(self, txt_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Read and process one patient's data file.
        
        Returns:
            (X, M, D, x_static) where:
            - X: (48, F) - imputed time series values
            - M: (48, F) - observation mask
            - D: (48, F) - time-to-event (decay)
            - x_static: (S,) - static features
        """
        df = pd.read_csv(txt_path)  # Time,Parameter,Value
        if not {"Time", "Parameter", "Value"}.issubset(df.columns):
            raise ValueError(f"{txt_path.name} missing required columns (Time,Parameter,Value)")

        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

        # ---- STATIC FEATURES ----
        s = df[df["Parameter"].isin(self.STATIC_FEATURES)].dropna(subset=["Value"]).copy()
        # if repeated, take last appearance
        s["_row"] = np.arange(len(s))
        s = s.sort_values(["Parameter", "_row"]).groupby("Parameter", as_index=False).tail(1)
        static_map: Dict[str, float] = dict(zip(s["Parameter"], s["Value"]))
        x_static = np.array([static_map.get(k, np.nan) for k in self.STATIC_FEATURES], dtype=np.float32)

        # ---- DYNAMIC FEATURES ----
        d = df[df["Parameter"].isin(self.TS_FEATURES)].dropna(subset=["Value"]).copy()
        d["Hour"] = d["Time"].map(self._time_to_hour)

        if self.agg == "last":
            d["_row"] = np.arange(len(d))
            d = d.sort_values(["Hour", "Parameter", "_row"])
            
            # FIX: Handle MechVent separately to avoid KeyError
            # Extract MechVent before pivot
            mv_data = d[d["Parameter"].eq("MechVent")].copy()
            if len(mv_data) > 0:
                mv = mv_data.groupby("Hour")["Value"].max()
            else:
                mv = pd.Series(dtype=np.float32)
            
            d = d.groupby(["Hour", "Parameter"], as_index=False).tail(1)
            pivot = d.pivot(index="Hour", columns="Parameter", values="Value")
            
            # FIX: Reindex pivot to ensure all hours exist
            pivot = pivot.reindex(index=range(self.T))
            
            # FIX: Properly merge MechVent with reindex
            if len(mv) > 0:
                pivot["MechVent"] = mv
            else:
                pivot["MechVent"] = np.nan
            
        elif self.agg == "mean":
            pivot = d.groupby(["Hour", "Parameter"])["Value"].mean().unstack("Parameter")
            pivot = pivot.reindex(index=range(self.T))
            
        else:  # median
            pivot = d.groupby(["Hour", "Parameter"])["Value"].median().unstack("Parameter")
            pivot = pivot.reindex(index=range(self.T))

        # FIX: Ensure all TS_FEATURES columns exist in pivot
        pivot = pivot.reindex(columns=self.TS_FEATURES)
        
        X = pivot.to_numpy(dtype=np.float32)                # (48, F)
        M = (~np.isnan(X)).astype(np.float32)              # (48, F)
        time = np.arange(self.T, dtype=np.int32)[:, None]   # (48, 1)
        
        # Compute decay (time-to-event): how long since last observation
        last = np.where(M == 1, time, -1)                   # (48, F)
        last = np.maximum.accumulate(last, axis=0)          # carry forward last observed hour
        D = np.where(M == 1, 0, time - last).astype(np.float32)  # (48, F)

        return X, M, D, x_static

    def load(self, limit: Optional[int] = None) -> ICUArrays:
        """
        Load all patient data from directory.
        
        Args:
            limit: Maximum number of patients to load (for testing)
            
        Returns:
            ICUArrays containing all loaded data
        """
        dim_data_path = self.data_dir / "Features_group_7" / "Features_group_10"
        out_data_path = self.data_dir / "Outcomes-7.txt"
        
        # Verify paths exist
        if not dim_data_path.exists():
            raise FileNotFoundError(f"Features directory not found: {dim_data_path}")
        if not out_data_path.exists():
            raise FileNotFoundError(f"Outcomes file not found: {out_data_path}")
        
        paths = sorted(dim_data_path.glob("*.txt"))
        if limit is not None:
            paths = paths[: int(limit)]

        N = len(paths)
        if N == 0:
            raise FileNotFoundError(f"No .txt files found in {dim_data_path}")

        X_dynamic = np.full((N, self.T, self.F), np.nan, dtype=np.float32)
        M_dynamic = np.zeros((N, self.T, self.F), dtype=np.float32)
        D_dynamic = np.zeros((N, self.T, self.F), dtype=np.float32)
        X_static  = np.full((N, self.S), np.nan, dtype=np.float32)
        patient_ids: List[str] = []
        Mortality: List[float] = []

        # Load outcomes
        df_outcomes = pd.read_csv(out_data_path)
        df_outcomes["RecordID"] = df_outcomes["RecordID"].astype(str)
        df_outcomes.drop(columns=['SAPS-I','SOFA','Length_of_stay','Survival'], inplace=True)
        df_outcomes["In-hospital_death"] = pd.to_numeric(df_outcomes["In-hospital_death"], errors="coerce")
        df_outcomes.set_index('RecordID', inplace=True)

        # Load each patient
        for i, p in enumerate(paths):
            X, M, D, xs = self._read_one(p)
            X_dynamic[i] = X
            M_dynamic[i] = M
            D_dynamic[i] = D
            X_static[i]  = xs
            patient_ids.append(p.stem)
            
            try:
                mortality_val = df_outcomes.loc[p.stem, "In-hospital_death"]
                Mortality.append(float(mortality_val))
            except KeyError:
                print(f"⚠️  Patient {p.stem} not found in outcomes")
                Mortality.append(np.nan)

        return ICUArrays(
            X_dynamic=X_dynamic,
            M_dynamic=M_dynamic,
            D_dynamic=D_dynamic,
            X_static=X_static,
            patient_ids=patient_ids,
            Mortality=Mortality
        )

    def save_npz(self, out_path: str, arrays: ICUArrays) -> None:
        """
        Save arrays to a single compressed .npz file.
        
        Args:
            out_path: Path to save file
            arrays: ICUArrays to save
        """
        np.savez_compressed(
            out_path,
            X_dynamic=arrays.X_dynamic,
            M_dynamic=arrays.M_dynamic,
            D_dynamic=arrays.D_dynamic,
            X_static=arrays.X_static,
            patient_ids=np.array(arrays.patient_ids, dtype=object),
            Mortality=np.array(arrays.Mortality, dtype=np.float32),
            ts_features=np.array(self.TS_FEATURES, dtype=object),
            static_features=np.array(self.STATIC_FEATURES, dtype=object),
        )
        print(f"✓ Saved to {out_path}")

    @staticmethod
    def load_npz(npz_path: str) -> ICUArrays:
        """
        Load preprocessed arrays from .npz file.
        
        Args:
            npz_path: Path to .npz file
            
        Returns:
            ICUArrays
        """
        data = np.load(npz_path, allow_pickle=True)
        return ICUArrays(
            X_dynamic=data["X_dynamic"],
            M_dynamic=data["M_dynamic"],
            D_dynamic=data["D_dynamic"],
            X_static=data["X_static"],
            patient_ids=list(data["patient_ids"].tolist()),
            Mortality=list(data["Mortality"].astype(float).tolist())
        )

    def split_data(
        self,
        arrays: ICUArrays,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        seed: int = 42
    ) -> Tuple[ICUArrays, ICUArrays, ICUArrays]:
        """
        Split data into train/val/test stratified by mortality.
        
        Args:
            arrays: Full dataset
            train_size, val_size, test_size: Split ratios (must sum to 1.0)
            seed: Random seed
            
        Returns:
            (train_arrays, val_arrays, test_arrays)
        """
        y_full = np.array(arrays.Mortality, dtype=float)
        idx_all = np.arange(len(y_full))
        
        # Only use samples with valid mortality labels
        keep = ~np.isnan(y_full)
        idx = idx_all[keep]
        y = y_full[keep].astype(int)
        
        # Validate split ratios
        total = train_size + val_size + test_size
        if abs(total - 1.0) > 1e-6:
            raise ValueError("train_size + val_size + test_size must sum to 1.0")
        
        # Split: train vs (val + test)
        temp_size = val_size + test_size
        train_idx, temp_idx = train_test_split(
            idx, test_size=temp_size, random_state=seed, stratify=y
        )
        
        # Split: val vs test
        y_temp = y_full[temp_idx].astype(int)
        test_frac_within_temp = test_size / temp_size
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=test_frac_within_temp, random_state=seed, stratify=y_temp
        )
        
        print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        print(f"Mortality rates: train={y_full[train_idx].mean():.3f}, "
              f"val={y_full[val_idx].mean():.3f}, test={y_full[test_idx].mean():.3f}")

        def pack(split_idx):
            return ICUArrays(
                X_dynamic=arrays.X_dynamic[split_idx],
                M_dynamic=arrays.M_dynamic[split_idx],
                D_dynamic=arrays.D_dynamic[split_idx],
                X_static=arrays.X_static[split_idx],
                patient_ids=np.array(arrays.patient_ids, dtype=object)[split_idx].tolist(),
                Mortality=y_full[split_idx].tolist(),
            )

        return pack(train_idx), pack(val_idx), pack(test_idx)
    
    def makebin(self, X_static: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create age bins and extract gender for group mean calculation.
        
        Args:
            X_static: (N, S) static features where column 0 is age, column 1 is gender
            
        Returns:
            (age_bin, gender) - indices for group stratification
        """
        age = X_static[:, 0].astype(float)
        gender = X_static[:, 1].astype(float)

        age = np.clip(age, 0, 120)
        age_bin = (age // 10).astype(int)
        age_bin = np.minimum(age_bin, 10)
        
        return age_bin, gender

    def group_mean_calculator(
        self,
        X_train: np.ndarray,
        M_train: np.ndarray,
        age_bin: np.ndarray,
        gd: np.ndarray,
        n_age_bins: int = 11
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate group-wise and overall mean values for imputation.
        
        Groups are stratified by age_bin × gender (11 age bins × 2 genders = 22 groups).
        
        Args:
            X_train: (N, T, F) - training time series
            M_train: (N, T, F) - observation mask
            age_bin: (N,) - age bin index
            gd: (N,) - gender
            n_age_bins: number of age bins (default 11)
            
        Returns:
            (group_mean, overall_mean)
            - group_mean: (22, F) - mean for each demographic group
            - overall_mean: (F,) - overall mean across all training data
        """
        N, T, F = X_train.shape
        G = n_age_bins * 2  # 22 groups total

        # Calculate overall mean using only observed values
        sum_all = np.nansum(X_train * M_train, axis=(0, 1))  # (F,)
        cnt_all = np.sum(M_train, axis=(0, 1))               # (F,)
        overall_mean = np.where(cnt_all > 0, sum_all / cnt_all, 0.0).astype(np.float32)

        # Calculate group means
        group_mean = np.zeros((G, F), dtype=np.float32)
        group_id = (age_bin * 2 + gd).astype(int)

        for g in range(G):
            members = (group_id == g)
            if not np.any(members):
                group_mean[g] = overall_mean
                continue

            Xm = X_train[members]  # (n_g, T, F)
            Mm = M_train[members]  # (n_g, T, F)

            sum_g = np.nansum(Xm * Mm, axis=(0, 1))  # (F,)
            cnt_g = np.sum(Mm, axis=(0, 1))          # (F,)
            mean_g = np.where(cnt_g > 0, sum_g / cnt_g, overall_mean)

            group_mean[g] = mean_g.astype(np.float32)

        return group_mean, overall_mean
    

    def forwardfill(
        self,
        X: np.ndarray,
        M: np.ndarray,
        age_bin: np.ndarray,
        gd: np.ndarray,
        group_mean: np.ndarray,
        overall_mean: np.ndarray
    ) -> np.ndarray:
        """
        Forward-fill imputation with group-wise means.
        
        For each sample:
        1. Use group mean (based on age/gender) as baseline
        2. Forward-fill observed values
        3. Special handling for GCS (clipped to 3-15) and MechVent (binary)
        
        Args:
            X: (N, T, F) - time series with NaN
            M: (N, T, F) - observation mask
            age_bin: (N,) - age bin index
            gd: (N,) - gender
            group_mean: (22, F) - group means from training
            overall_mean: (F,) - overall mean
            
        Returns:
            X_imp: (N, T, F) - imputed time series
        """
        N, T, F = X.shape
        G = group_mean.shape[0]

        group_id = (age_bin * 2 + gd).astype(int)
        group_id = np.clip(group_id, 0, G - 1)

        mu = group_mean[group_id]                         # (N, F)
        mu = np.where(np.isnan(mu), overall_mean, mu)     # safety check

        time = np.arange(T, dtype=np.int32)[None, :, None]     # (1, T, 1)

        # Find last seen hour index for each feature (or -1 if never seen)
        last_seen = np.where(M == 1, time, -1)
        last_seen = np.maximum.accumulate(last_seen, axis=1)   # (N, T, F)

        # Get values at last seen indices
        idx = np.clip(last_seen, 0, T - 1).astype(np.int32)
        ffill_vals = np.take_along_axis(X, idx, axis=1)        # (N, T, F)

        # Use forward-filled value if seen before, else use group mean
        has_prev = (last_seen >= 0)
        X_imp = np.where(has_prev, ffill_vals, mu[:, None, :]).astype(np.float32)

        # Special handling for GCS (integer 3-15)
        if "GCS" in self.TS_FEATURES:
            gcs_i = self.TS_FEATURES.index("GCS")
            X_imp[:, :, gcs_i] = np.clip(X_imp[:, :, gcs_i], 3, 15)
            X_imp[:, :, gcs_i] = np.rint(X_imp[:, :, gcs_i])

        # Special handling for MechVent (binary)
        if "MechVent" in self.TS_FEATURES:
            mv_i = self.TS_FEATURES.index("MechVent")
            never_seen = (last_seen[:, :, mv_i] < 0)
            X_imp[:, :, mv_i] = np.where(never_seen, 0.0, X_imp[:, :, mv_i])
            X_imp[:, :, mv_i] = (X_imp[:, :, mv_i] > 0).astype(np.float32)

        return X_imp
