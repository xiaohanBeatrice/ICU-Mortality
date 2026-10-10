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
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ICUArrays:
    """Container for ICU patient data arrays"""
    count: np.ndarray      # (N, 48, F)
    value_sum: np.ndarray  # (N, 48, F)
    value_min: np.ndarray  # (N, 48, F)
    value_max: np.ndarray  # (N, 48, F)
    value_first: np.ndarray # (N, 48, F)
    value_last: np.ndarray  # (N, 48, F)
    X_static: np.ndarray   # (N, S)
    patient_ids: List[str]
    Mortality: List[float]


class DataLoader:
    STATIC_FEATURES = ["Age", "Gender", "Height", "ICUType"]

    TS_FEATURES = [
        "Albumin","ALP","ALT","AST","Bilirubin",
        "BUN","Cholesterol","Creatinine",
        "BP_Dias","FiO2","GCS","Glucose","HCO3","HCT",
        "HR","K","Lactate","BP_MAP","MechVent","Mg","Na",
        "PaCO2","PaO2","pH","Platelets",
        "RespRate","SaO2","BP_Sys","Temp",
        "TroponinI","TroponinT","Urine","WBC","Weight"
    ]
    BOUNDS = {
    'Albumin': (0.6, 6.0),
    'ALP': (1.0, 4000.0),
    'ALT': (2.0, 10000.0),
    'AST': (6.0, 20000.0),
    'Bilirubin': (0.1, 60.0),
    'BUN': (0.1, 250.0),
    'Cholesterol': (32.0, 600.0),
    'Creatinine': (0.1, 18.0),
    'BP_Dias': (1.0, 375.0),
    'FiO2': (0.21, 1.0),
    'GCS': (3.0, 15.0),
    'Glucose': (30.0, 2000.0),
    'HCO3': (1.0, 60.0),
    'HCT': (5.0, 75.0),
    'HR': (1.0, 300.0),
    'K': (1.0, 12.0),
    'Lactate': (0.4, 30.0),
    'Mg': (0.5, 20.0),
    'BP_MAP': (14.0, 330.0),
    'Na': (100.0, 225.0),
    'PaCO2': (5.0, 200.0),
    'PaO2': (32.0, 700.0),
    'pH': (6.1, 7.9),
    'Platelets': (5.0, 2000.0),
    'RespRate': (0.1, 70.0),
    'SaO2': (0.1, 100.0),
    'BP_Sys': (1.0, 300.0),
    'Temp': (25.0, 45.0),
    'TroponinI':(0.01,50.0),
    'TroponinT':(0.01,20.0),
    'Urine': (0.0, 1000.0),
    'WBC': (0.1, 500.0),
    'Weight': (1.0, 300.0),
    'Height':(100.0,210.0)
    }
    BP_MERGE_MAP = {
    "DiasABP": "BP_Dias",
    "NIDiasABP": "BP_Dias",
    "MAP": "BP_MAP",
    "NIMAP": "BP_MAP",
    "SysABP": "BP_Sys",
    "NISysABP": "BP_Sys",
    }

    BP_SOURCE_PRIORITY = {
        "DiasABP": 0,     
        "MAP": 0,
        "SysABP": 0,
        "NIDiasABP": 1,  
        "NIMAP": 1,
        "NISysABP": 1,
    }
    
    OUT_FEATURES = ['RecordID','SAPS-I','SOFA','Length_of_stay','Survival','In-hospital_death']

    def __init__(self, data_dir: str):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Path to data directory containing Features_group_7/Features_group_10 and Outcomes-7.txt
        """
        self.data_dir = Path(data_dir)
        self.T = 48  # Hours
        self.F = len(self.TS_FEATURES)  # Number of time-series features
        self.S = len(self.STATIC_FEATURES)  # Number of static features


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
        
    @staticmethod
    def _time_to_minutes(t: str) -> int:
        """
        Convert HH:MM format to min index (minutes).
        """
        try:
            hh, mm = str(t).split(":")
            total = int(hh) * 60 + int(mm)
            return int(np.clip(total, 0, 47 * 60 + 59))
        except (ValueError, IndexError):
            return 0
        
    def _merge_bp_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        bp_raw = list(self.BP_MERGE_MAP.keys())
        bp = df[df["Parameter"].isin(bp_raw)].copy()
        if bp.empty:
            return df

        bp["MergedParam"] = bp["Parameter"].map(self.BP_MERGE_MAP)
        bp["Priority"] = bp["Parameter"].map(self.BP_SOURCE_PRIORITY)
        bp["Hour"] = bp["Time"].map(self._time_to_hour)
        bp["_row"] = np.arange(len(bp))

        best_priority = bp.groupby(["Hour", "MergedParam"])["Priority"].transform("min")
        bp = bp[bp["Priority"] == best_priority].copy()
        bp["Parameter"] = bp["MergedParam"]
        bp = bp[df.columns]
        other = df[~df["Parameter"].isin(bp_raw)].copy()
        out = pd.concat([other, bp], ignore_index=True)
        return out
        
    def apply_clinical_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace out-of-range physiological values with NaN.
        Only features listed in BOUNDS are checked.
        """
        df = df.copy()
        removed_counts = {}

        for feature, (low, high) in self.BOUNDS.items():
            mask = df["Parameter"].eq(feature)
            bad = mask & df["Value"].notna() & (
                (df["Value"] < low) | (df["Value"] > high)
            )
            n_bad = int(bad.sum())
            if n_bad > 0:
                removed_counts[feature] = n_bad
            df.loc[bad, "Value"] = np.nan

        return df

    def _read_one(self, txt_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Read and process one patient's data file.
        
        Returns:
            (X, M, D, x_static) where:
            - X: (48, F) - imputed time series values
            - M: (48, F) - observation mask
            - D: (48, F) - time-to-event (decay)
            - x_static: (S,) - static features
        """
        df = pd.read_csv(txt_path)  
        if not {"Time", "Parameter", "Value"}.issubset(df.columns):
            raise ValueError(f"{txt_path.name} missing required columns (Time,Parameter,Value)")

        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df = self.apply_clinical_bounds(df)
        df = self._merge_bp_features(df)

        # ---- STATIC FEATURES ----
        s = df[df["Parameter"].isin(self.STATIC_FEATURES)].dropna(subset=["Value"]).copy()
        s["_minutes"] = s["Time"].map(self._time_to_minutes)
        s["_row"] = np.arange(len(s))
        # choose lasts available static value
        s = s.sort_values(["Parameter", "_minutes", "_row"]).groupby("Parameter", as_index=False).head(1)
        static_map: Dict[str, float] = dict(zip(s["Parameter"], s["Value"]))
        x_static = np.array([static_map.get(k, np.nan) for k in self.STATIC_FEATURES], dtype=np.float32)


        # ---- DYNAMIC FEATURES ----
        d = df[df["Parameter"].isin(self.TS_FEATURES)].copy()
        d.loc[d["Parameter"] == "MechVent", "Value"] = d.loc[d["Parameter"] == "MechVent", "Value"].fillna(0)
        d = d.dropna(subset=["Value"])
        d["Hour"] = d["Time"].map(self._time_to_hour)
        d["Min"] = d["Time"].map(self._time_to_minutes)
        d["_row"] = np.arange(len(d))
        d = d.sort_values(["Parameter", "Hour", "Min", "_row"])
        shape = (self.T, self.F)
        count_arr = np.zeros(shape, dtype=np.float32)
        sum_arr   = np.zeros(shape, dtype=np.float32)
        min_arr   = np.full(shape, np.nan, dtype=np.float32)
        max_arr   = np.full(shape, np.nan, dtype=np.float32)
        first_arr = np.full(shape, np.nan, dtype=np.float32)
        last_arr  = np.full(shape, np.nan, dtype=np.float32)
        mv = np.zeros(self.T, dtype=np.float32) 
        feat_to_idx = {f: i for i, f in enumerate(self.TS_FEATURES)}

        grouped = d.groupby(["Hour", "Parameter"], sort=False)

        for (hour, param), g in grouped:
            j = feat_to_idx[param]
            vals = g["Value"].to_numpy(dtype=np.float32)

            count_arr[hour, j] = len(vals)
            sum_arr[hour, j] = np.sum(vals, dtype=np.float32)
            min_arr[hour, j] = np.min(vals)
            max_arr[hour, j] = np.max(vals)
            first_arr[hour, j] = vals[0]
            last_arr[hour, j] = vals[-1]

        return count_arr, sum_arr, min_arr, max_arr, first_arr, last_arr, x_static
        

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

        count_all = np.zeros((N, self.T, self.F), dtype=np.float32)
        sum_all   = np.zeros((N, self.T, self.F), dtype=np.float32)
        min_all   = np.full((N, self.T, self.F), np.nan, dtype=np.float32)
        max_all   = np.full((N, self.T, self.F), np.nan, dtype=np.float32)
        first_all = np.full((N, self.T, self.F), np.nan, dtype=np.float32)
        last_all  = np.full((N, self.T, self.F), np.nan, dtype=np.float32)
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
            cnt, sm, mn, mx, fst, lst, xs = self._read_one(p)

            count_all[i] = cnt
            sum_all[i]   = sm
            min_all[i]   = mn
            max_all[i]   = mx
            first_all[i] = fst
            last_all[i]  = lst
            X_static[i]  = xs
            patient_ids.append(p.stem)

            try:
                Mortality.append(float(df_outcomes.loc[p.stem, "In-hospital_death"]))
            except KeyError:
                Mortality.append(np.nan)

        return ICUArrays(
            count=count_all,
            value_sum=sum_all,
            value_min=min_all,
            value_max=max_all,
            value_first=first_all,
            value_last=last_all,
            X_static=X_static,
            patient_ids=patient_ids,
            Mortality=Mortality,
        )
