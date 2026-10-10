# -*- coding: utf-8 -*-
"""
Unified preprocessing for ICU mortality models.

Design:
- Uses DataLoader.load() hourly-base output only
- Hardcoded 3-hour window
- Reusable inside each CV fold
- Supports 3 branches:
    1) GRU-D
    2) Standard GRU
    3) ML (XGBoost / SVM / Logistic Regression)

Preprocessing policy:
- Holdout / fold splitting happens outside this file
- This file only fits on training fold and transforms train/val/test
- Selective log1p on chosen continuous features
- Z-score all continuous features
- Standardize D
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from data_loader_grud import DataLoader, ICUArrays


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

FEATURE_GROUPS = {
    "vitals_bp": [
        "BP_Dias", "BP_MAP", "BP_Sys",
        "HR", "RespRate", "Temp", "FiO2", "SaO2", "Weight",
    ],
    "urine": ["Urine"],
    "volatile_labs": ["Lactate", "Glucose", "K", "pH", "PaO2", "PaCO2", "Creatinine"],
    "slow_labs": ["Na", "Mg", "HCT", "Albumin", "BUN", "HCO3", "WBC", "Platelets"],
    "sparse_labs": ["TroponinI", "TroponinT", "ALP", "ALT", "AST", "Bilirubin", "Cholesterol"],
    "gcs": ["GCS"],
    "mechvent": ["MechVent"],
}

# Core selected features for GRU-D and Standard GRU
GRUD_X_CONFIG = {
    "vitals_bp": ["last", "range"],
    "urine": ["sum", "range"],
    "volatile_labs": ["last"],
    "slow_labs": ["last"],
    "sparse_labs": ["last"],
    "gcs": ["last"],
    "mechvent": ["last"],
}

# ML selected features
ML_X_CONFIG = {
    "vitals_bp": ["mean", "last", "range", "trend","mask"],
    "urine": ["sum", "range", "trend","mask"],
    "volatile_labs": ["mean", "last", "trend", "mask"],
    "slow_labs": ["last", "trend", "mask"],
    "sparse_labs": ["last", "mask"],
    "gcs": ["last", "trend", "mask"],
    "mechvent": ["last", "mask"],
}

# only these get forward-filled
FFILL_SUFFIXES = ("__last", "__mean", "__sum")

# selective log1p list from you
LOG1P_FEATURES = {
    "WBC", "TroponinT", "TroponinI", "ALT", "AST", "ALP",
    "Bilirubin", "Creatinine", "BUN", "Glucose", "Mg",
    "Lactate", "Urine", "PaO2", "PaCO2", "Weight",
}
LOG1P_STATS = {"sum", "mean", "last", "range"}

# do not z-score these temporal feature families
NON_CONTINUOUS_BASE = {"MechVent", "GCS"}
NON_CONTINUOUS_STATS = {"mask"}

# static transforms
STATIC_LOG1P = set()
STATIC_CONTINUOUS = {"Age", "Height"}


# ---------------------------------------------------------------------
# Returned batches
# ---------------------------------------------------------------------
@dataclass
class ProcessedBatch:
    X: np.ndarray
    X_static: np.ndarray
    y: np.ndarray
    patient_ids: List[str]
    feature_names: List[str]
    mask: Optional[np.ndarray] = None
    delta: Optional[np.ndarray] = None


# ---------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------
class UnifiedPreprocessor:
    def __init__(self, data_dir: str, window_hours: int = 3, use_forward_fill: bool = True):
        self.data_dir = Path(data_dir)
        self.loader = DataLoader(data_dir=str(self.data_dir))
        self.feature_to_idx = {f: i for i, f in enumerate(self.loader.TS_FEATURES)}
        self._validate_feature_groups()
        self._static_fill: Optional[np.ndarray] = None
        # fitted state
        self._branch: Optional[str] = None
        self._fitted: bool = False

        self.window_hours = int(window_hours)
        self.use_forward_fill = bool(use_forward_fill)
        if self.window_hours not in [1,3,6,12,24]:
            raise ValueError("Window_hours must be one of these, e.g. 1, 3, 6, 12, 24")

        self._selected_feature_names: List[str] = []
        self._fill_defaults: Optional[np.ndarray] = None

        self._x_mean: Optional[np.ndarray] = None
        self._x_std: Optional[np.ndarray] = None

        self._d_mean: Optional[np.ndarray] = None
        self._d_std: Optional[np.ndarray] = None

        self._static_mean: Optional[np.ndarray] = None
        self._static_std: Optional[np.ndarray] = None

    # -------------------------------------------------------------
    # public API
    # -------------------------------------------------------------
    def fit(self, train_arrays: ICUArrays, branch: str) -> "UnifiedPreprocessor":
        """
        Fit preprocessing parameters on a training split only.

        branch in {"grud", "gru", "ml"}
        """
        branch = branch.lower().strip()
        if branch not in {"grud", "gru", "ml"}:
            raise ValueError("branch must be one of {'grud','gru','ml'}")

        self._branch = branch

        base = self._aggregate_to_window(train_arrays)
        stats = self._build_stats_dict(base)

        # static transforms
        self._static_mean, self._static_std = self._fit_static_transform(train_arrays.X_static)

        if branch == "grud":
            X, names = self._select_branch_tensor(stats, GRUD_X_CONFIG)
            M, D = self._select_branch_mask_delta(stats, GRUD_X_CONFIG)

            X = self._apply_selective_log1p_3d(X, names)
            self._x_mean, self._x_std = self._fit_zscore_3d_selected(
                X, names, mask=M, observed_only=True
            )

            # delta standardized on full D values
            self._d_mean, self._d_std = None, None

            self._selected_feature_names = names

        elif branch == "gru":
            X, names = self._select_branch_tensor(stats, GRUD_X_CONFIG)
            M, D_aligned = self._select_branch_mask_delta(stats, GRUD_X_CONFIG)

            if self.use_forward_fill:
                self._fill_defaults = self._fit_fill_defaults(X, M, names)
                X = self._forward_fill_selected(X, M, names, self._fill_defaults)
                X = self._apply_selective_log1p_3d(X, names)

                self._x_mean, self._x_std = self._fit_zscore_3d_selected(
                    X, names, mask=None, observed_only=False
                )
            else:
                self._fill_defaults = None
                X = self._apply_selective_log1p_3d(X, names)

                self._x_mean, self._x_std = self._fit_zscore_3d_selected(
                    X, names, mask=M, observed_only=True
                )

            self._d_mean, self._d_std = self._fit_standardizer_3d(D_aligned)
            self._selected_feature_names = names

        else:  # ml
            # 1. Select the base features and stats for the ML branch
            X, names = self._select_branch_tensor(stats, ML_X_CONFIG)
            M_for_ffill, _ = self._select_branch_mask_delta(stats, ML_X_CONFIG)

            # 2. Identify unique base features to create a matching Delta set
            unique_base_feats = []
            seen = set()
            for n in names:
                base = n.split("__")[0]
                if base not in seen and base in self.feature_to_idx:
                    unique_base_feats.append(base)
                    seen.add(base)

            # 3. Pull only the deltas for the features actually present in names
            delta_indices = [self.feature_to_idx[f] for f in unique_base_feats]
            D_selected = stats["delta"][:, :, delta_indices]
            
            # 4. Fit transformations
            if self.use_forward_fill:
                self._fill_defaults = self._fit_fill_defaults(X, M_for_ffill, names)
                X = self._forward_fill_selected(X, M_for_ffill, names, self._fill_defaults)
            else:
                self._fill_defaults = None
            X = self._apply_selective_log1p_3d(X, names)

            self._x_mean, self._x_std = self._fit_zscore_3d_selected(
                X, names, mask=None, observed_only=False
            )

            # Standardize Delta based ONLY on the selected columns
            self._d_mean, self._d_std = self._fit_standardizer_3d(D_selected)

            self._selected_feature_names = names

        self._fitted = True
        return self

    def transform(self, arrays: ICUArrays) -> ProcessedBatch:
        if not self._fitted or self._branch is None:
            raise RuntimeError("Call fit(train_arrays, branch=...) first.")

        base = self._aggregate_to_window(arrays)
        stats = self._build_stats_dict(base)

        X_static = self._apply_static_transform(arrays.X_static, self._static_mean, self._static_std)
        y = np.array(arrays.Mortality, dtype=np.float32)

        if self._branch == "grud":
            X, names = self._select_branch_tensor(stats, GRUD_X_CONFIG)
            M, D = self._select_branch_mask_delta(stats, GRUD_X_CONFIG)

            X = self._apply_selective_log1p_3d(X, names)
            X = self._apply_zscore_3d_selected(
                X, self._x_mean, self._x_std, names, mask=M, observed_only=True
            )

            # keep missing X entries at 0 for GRU-D
            X = np.where(M == 1, X, 0.0).astype(np.float32)

            D = D.astype(np.float32)

            return ProcessedBatch(
                X=X.astype(np.float32),
                X_static=X_static.astype(np.float32),
                y=y,
                patient_ids=list(arrays.patient_ids),
                feature_names=names,
                mask=M.astype(np.float32),
                delta=D.astype(np.float32),
            )

        elif self._branch == "gru":
            X, names = self._select_branch_tensor(stats, GRUD_X_CONFIG)
            M, D_aligned = self._select_branch_mask_delta(stats, GRUD_X_CONFIG)

            if self.use_forward_fill:
                X = self._forward_fill_selected(X, M, names, self._fill_defaults)
                X = self._apply_selective_log1p_3d(X, names)
                X = self._apply_zscore_3d_selected(
                    X, self._x_mean, self._x_std, names, mask=None, observed_only=False
                )
            else:
                X = self._apply_selective_log1p_3d(X, names)
                X = self._apply_zscore_3d_selected(
                    X, self._x_mean, self._x_std, names, mask=M, observed_only=True
                )
                X = np.where(M == 1, X, 0.0).astype(np.float32)

            D_std = self._apply_standardizer_3d(D_aligned, self._d_mean, self._d_std)

            X_combined = np.concatenate([X, D_std], axis=2).astype(np.float32)
            delta_names = [f"{n}__delta" for n in names]
            out_names = names + delta_names

            return ProcessedBatch(
                X=X_combined,
                X_static=X_static.astype(np.float32),
                y=y,
                patient_ids=list(arrays.patient_ids),
                feature_names=out_names,
                mask=M.astype(np.float32),
                delta=D_std.astype(np.float32),
            )

        else:  # ml
            # 1. Selection
            X, names = self._select_branch_tensor(stats, ML_X_CONFIG)
            M_for_ffill, _ = self._select_branch_mask_delta(stats, ML_X_CONFIG)

            # 2. Identify unique base features to match the FIT step
            unique_base_feats = []
            seen = set()
            for n in names:
                base = n.split("__")[0]
                if base not in seen and base in self.feature_to_idx:
                    unique_base_feats.append(base)
                    seen.add(base)
            
            delta_indices = [self.feature_to_idx[f] for f in unique_base_feats]
            delta_names = [f"{f}__delta" for f in unique_base_feats]

            # 3. Apply Transformations
            if self.use_forward_fill:
                X = self._forward_fill_selected(X, M_for_ffill, names, self._fill_defaults)
            X = self._apply_selective_log1p_3d(X, names)
            X = self._apply_zscore_3d_selected(
                X, self._x_mean, self._x_std, names, mask=None, observed_only=False
            )

            # 4. Process and Align Delta
            D_full = stats["delta"][:, :, delta_indices]
            D_std = self._apply_standardizer_3d(D_full, self._d_mean, self._d_std)

            # 5. Concatenate everything: [Selected Stats + Masks (if in config)] + [Deltas]
            X_combined_3d = np.concatenate([X, D_std], axis=2).astype(np.float32)
            temporal_names = names + delta_names

            # 6. Flatten to (N, T * C)
            N, T, C = X_combined_3d.shape
            X_flat = X_combined_3d.reshape(N, -1)

            # Build final flat feature names (t00__feature, t01__feature...)
            out_names = []
            for t in range(T):
                out_names.extend([f"t{t:02d}__{name}" for name in temporal_names])

            return ProcessedBatch(
                X=X_flat,
                X_static=X_static.astype(np.float32),
                y=y,
                patient_ids=list(arrays.patient_ids),
                feature_names=out_names,
                mask=None,  # Now embedded in X_flat
                delta=None  # Now embedded in X_flat
            )

    # -------------------------------------------------------------
    # convenience
    # -------------------------------------------------------------
    def fit_transform(self, train_arrays: ICUArrays, branch: str) -> ProcessedBatch:
        self.fit(train_arrays, branch=branch)
        return self.transform(train_arrays)

    @staticmethod
    def subset_arrays(arrays: ICUArrays, idx: np.ndarray) -> ICUArrays:
        idx = np.asarray(idx)
        return ICUArrays(
            count=arrays.count[idx],
            value_sum=arrays.value_sum[idx],
            value_min=arrays.value_min[idx],
            value_max=arrays.value_max[idx],
            value_first=arrays.value_first[idx],
            value_last=arrays.value_last[idx],
            X_static=arrays.X_static[idx],
            patient_ids=np.array(arrays.patient_ids, dtype=object)[idx].tolist(),
            Mortality=np.array(arrays.Mortality, dtype=float)[idx].tolist(),
        )

    # -------------------------------------------------------------
    # validation
    # -------------------------------------------------------------
    def _validate_feature_groups(self) -> None:
        unknown = []
        for group, feats in FEATURE_GROUPS.items():
            for feat in feats:
                if feat not in self.feature_to_idx:
                    unknown.append((group, feat))
        if unknown:
            msg = ", ".join([f"{g}:{f}" for g, f in unknown])
            raise ValueError(f"Unknown features in FEATURE_GROUPS: {msg}")

    # -------------------------------------------------------------
    # aggregation
    # -------------------------------------------------------------
    def _aggregate_to_window(self, arrays: ICUArrays) -> Dict[str, np.ndarray]:
        count = arrays.count
        value_sum = arrays.value_sum
        value_min = arrays.value_min
        value_max = arrays.value_max
        value_first = arrays.value_first
        value_last = arrays.value_last

        N, T, F = count.shape
        W = self.window_hours
        n_windows = T // W

        count_r = count.reshape(N, n_windows, W, F)
        sum_r = value_sum.reshape(N, n_windows, W, F)
        min_r = value_min.reshape(N, n_windows, W, F)
        max_r = value_max.reshape(N, n_windows, W, F)
        first_r = value_first.reshape(N, n_windows, W, F)
        last_r = value_last.reshape(N, n_windows, W, F)

        count_w = np.sum(count_r, axis=2, dtype=np.float32)
        sum_w = np.sum(sum_r, axis=2, dtype=np.float32)

        with np.errstate(all="ignore"):
            min_w = np.nanmin(min_r, axis=2)
            max_w = np.nanmax(max_r, axis=2)

        no_obs = count_w == 0
        min_w = np.where(no_obs, np.nan, min_w).astype(np.float32)
        max_w = np.where(no_obs, np.nan, max_w).astype(np.float32)

        first_w = self._pick_first(first_r, count_r)
        last_w = self._pick_last(last_r, count_r)

        return {
            "count": count_w.astype(np.float32),
            "value_sum": sum_w.astype(np.float32),
            "value_min": min_w.astype(np.float32),
            "value_max": max_w.astype(np.float32),
            "value_first": first_w.astype(np.float32),
            "value_last": last_w.astype(np.float32),
        }

    @staticmethod
    def _pick_first(first_r: np.ndarray, count_r: np.ndarray) -> np.ndarray:
        has_obs = count_r > 0
        any_obs = np.any(has_obs, axis=2)
        first_idx = np.argmax(has_obs, axis=2)
        picked = np.take_along_axis(first_r, first_idx[:, :, None, :], axis=2).squeeze(axis=2)
        return np.where(any_obs, picked, np.nan).astype(np.float32)

    @staticmethod
    def _pick_last(last_r: np.ndarray, count_r: np.ndarray) -> np.ndarray:
        has_obs = count_r > 0
        any_obs = np.any(has_obs, axis=2)

        has_obs_rev = has_obs[:, :, ::-1, :]
        last_idx_from_end = np.argmax(has_obs_rev, axis=2)
        W = count_r.shape[2]
        last_idx = W - 1 - last_idx_from_end

        picked = np.take_along_axis(last_r, last_idx[:, :, None, :], axis=2).squeeze(axis=2)
        return np.where(any_obs, picked, np.nan).astype(np.float32)

    # -------------------------------------------------------------
    # stats dict
    # -------------------------------------------------------------
    def _build_stats_dict(self, base: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        count = base["count"]
        value_sum = base["value_sum"]
        value_min = base["value_min"]
        value_max = base["value_max"]
        value_first = base["value_first"]
        value_last = base["value_last"]

        mask = (count > 0).astype(np.float32)
        delta = self._build_delta(mask)

        mean = np.divide(
            value_sum,
            count,
            out=np.zeros_like(value_sum, dtype=np.float32),
            where=count > 0,
        ).astype(np.float32)

        value_sum = value_sum.astype(np.float32)
        last = np.nan_to_num(value_last, nan=0.0).astype(np.float32)

        range_feat = np.where(
            count > 0,
            np.nan_to_num(value_max - value_min, nan=0.0),
            0.0,
        ).astype(np.float32)

        trend = np.where(
            count > 0,
            np.nan_to_num(value_last - value_first, nan=0.0),
            0.0,
        ).astype(np.float32)

        return {
            "mean": mean,
            "sum": value_sum,
            "last": last,
            "range": range_feat,
            "trend": trend,
            "mask": mask,
            "delta": delta,
        }

    def _build_delta(self,mask: np.ndarray) -> np.ndarray:
        D = np.zeros_like(mask, dtype=np.float32)
        N, T, F = mask.shape

        for i in range(N):
            for f in range(F):
                last_seen = -1
                for t in range(T):
                    if mask[i, t, f] == 1:
                        D[i, t, f] = 0.0
                        last_seen = t
                    else:
                        if last_seen >= 0:
                            D[i, t, f] = (t - last_seen) * self.window_hours
                        else:
                            D[i, t, f] = (t + 1) * self.window_hours
        return D.astype(np.float32)

    # -------------------------------------------------------------
    # config-driven selection
    # -------------------------------------------------------------
    def _select_branch_tensor(
        self,
        stats: Dict[str, np.ndarray],
        config: Dict[str, List[str]],
    ) -> Tuple[np.ndarray, List[str]]:
        blocks = []
        names = []

        for group_name, stat_list in config.items():
            feats = FEATURE_GROUPS[group_name]
            idx = np.array([self.feature_to_idx[f] for f in feats], dtype=int)

            for stat_name in stat_list:
                block = stats[stat_name][:, :, idx].astype(np.float32)
                blocks.append(block)
                names.extend([f"{f}__{stat_name}" for f in feats])

        X = np.concatenate(blocks, axis=2).astype(np.float32)
        return X, names

    def _select_branch_mask_delta(
        self,
        stats: Dict[str, np.ndarray],
        config: Dict[str, List[str]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        m_blocks = []
        d_blocks = []

        for group_name, stat_list in config.items():
            feats = FEATURE_GROUPS[group_name]
            idx = np.array([self.feature_to_idx[f] for f in feats], dtype=int)

            for _ in stat_list:
                m_blocks.append(stats["mask"][:, :, idx].astype(np.float32))
                d_blocks.append(stats["delta"][:, :, idx].astype(np.float32))

        M = np.concatenate(m_blocks, axis=2).astype(np.float32)
        D = np.concatenate(d_blocks, axis=2).astype(np.float32)
        return M, D

    # -------------------------------------------------------------
    # transforms
    # -------------------------------------------------------------
    @staticmethod
    def _split_feature_name(name: str) -> Tuple[str, str]:
        if "__" not in name:
            return name, ""
        return name.split("__", 1)

    def _apply_selective_log1p_3d(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        X_out = X.copy().astype(np.float32)

        for j, name in enumerate(feature_names):
            feat, stat = self._split_feature_name(name)
            if feat in LOG1P_FEATURES and stat in LOG1P_STATS:
                X_out[:, :, j] = np.log1p(np.maximum(X_out[:, :, j], 0.0)).astype(np.float32)

        return X_out

    def _fit_zscore_3d_selected(
        self,
        X: np.ndarray,
        feature_names: List[str],
        mask: Optional[np.ndarray],
        observed_only: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.zeros((1, 1, X.shape[2]), dtype=np.float32)
        std = np.ones((1, 1, X.shape[2]), dtype=np.float32)

        for j, name in enumerate(feature_names):
            feat, stat = self._split_feature_name(name)

            if feat in NON_CONTINUOUS_BASE or stat in NON_CONTINUOUS_STATS:
                continue

            col = X[:, :, j]
            if observed_only and mask is not None:
                obs = col[mask[:, :, j] == 1]
            else:
                obs = col.reshape(-1)

            if obs.size == 0:
                m = 0.0
                s = 1.0
            else:
                m = float(obs.mean())
                s = float(obs.std())
                if s < 1e-6:
                    s = 1.0

            mean[0, 0, j] = np.float32(m)
            std[0, 0, j] = np.float32(s)

        return mean, std

    def _apply_zscore_3d_selected(
        self,
        X: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        feature_names: List[str],
        mask: Optional[np.ndarray],
        observed_only: bool,
    ) -> np.ndarray:
        X_out = X.copy().astype(np.float32)

        for j, name in enumerate(feature_names):
            feat, stat = self._split_feature_name(name)

            if feat in NON_CONTINUOUS_BASE or stat in NON_CONTINUOUS_STATS:
                continue

            X_out[:, :, j] = ((X_out[:, :, j] - mean[0, 0, j]) / std[0, 0, j]).astype(np.float32)

        if observed_only and mask is not None:
            X_out = np.where(mask == 1, X_out, 0.0).astype(np.float32)

        return X_out

    @staticmethod
    def _fit_standardizer_3d(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = X.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        std = X.std(axis=(0, 1), keepdims=True).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        return mean, std

    @staticmethod
    def _apply_standardizer_3d(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return ((X - mean) / std).astype(np.float32)

    @staticmethod
    def _fit_fill_defaults(X_train: np.ndarray, M_train: np.ndarray, feature_names: List[str]) -> np.ndarray:
        defaults = np.zeros(X_train.shape[2], dtype=np.float32)
        for j, name in enumerate(feature_names):
            if name.endswith(FFILL_SUFFIXES):
                obs = X_train[:, :, j][M_train[:, :, j] == 1]
                defaults[j] = float(obs.mean()) if obs.size > 0 else 0.0
        return defaults

    @staticmethod
    def _forward_fill_selected(
        X: np.ndarray,
        M: np.ndarray,
        feature_names: List[str],
        defaults: np.ndarray,
    ) -> np.ndarray:
        X_out = X.copy().astype(np.float32)
        N, T, C = X_out.shape

        for j, name in enumerate(feature_names):
            if not name.endswith(FFILL_SUFFIXES):
                continue

            if name == "Urine__sum":
                continue

            default_val = defaults[j]
            for i in range(N):
                last_val = None
                for t in range(T):
                    if M[i, t, j] == 1:
                        last_val = X_out[i, t, j]
                    else:
                        X_out[i, t, j] = last_val if last_val is not None else default_val

        return X_out.astype(np.float32)

    # -------------------------------------------------------------
    # static transforms
    # -------------------------------------------------------------
    def _fit_static_transform(self, X_static: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xs = X_static.copy().astype(np.float32)
        static_idx = {f: i for i, f in enumerate(self.loader.STATIC_FEATURES)}

        fill = np.zeros((1, Xs.shape[1]), dtype=np.float32)

        # categorical / discrete fills
        for feat in ["Gender", "ICUType"]:
            if feat in static_idx:
                j = static_idx[feat]
                col = Xs[:, j]
                obs = col[~np.isnan(col)]
                if obs.size > 0:
                    vals, counts = np.unique(obs, return_counts=True)
                    fill[0, j] = vals[np.argmax(counts)]
                else:
                    fill[0, j] = 0.0

        # continuous fills
        for feat in ["Age", "Height"]:
            if feat in static_idx:
                j = static_idx[feat]
                col = Xs[:, j]
                obs = col[~np.isnan(col)]
                fill[0, j] = np.median(obs) if obs.size > 0 else 0.0

        # apply imputation first
        Xs = np.where(np.isnan(Xs), fill, Xs).astype(np.float32)

        # static log1p if any remain in STATIC_LOG1P
        for feat in STATIC_LOG1P:
            if feat in static_idx:
                j = static_idx[feat]
                Xs[:, j] = np.log1p(np.maximum(Xs[:, j], 0.0)).astype(np.float32)

        mean = np.zeros((1, Xs.shape[1]), dtype=np.float32)
        std = np.ones((1, Xs.shape[1]), dtype=np.float32)

        for feat in STATIC_CONTINUOUS:
            if feat in static_idx:
                j = static_idx[feat]
                m = float(Xs[:, j].mean())
                s = float(Xs[:, j].std())
                if s < 1e-6:
                    s = 1.0
                mean[0, j] = np.float32(m)
                std[0, j] = np.float32(s)

        self._static_fill = fill
        return mean, std

    def _apply_static_transform(self, X_static: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        Xs = X_static.copy().astype(np.float32)
        static_idx = {f: i for i, f in enumerate(self.loader.STATIC_FEATURES)}

        # apply train-fold imputation values
        if self._static_fill is not None:
            Xs = np.where(np.isnan(Xs), self._static_fill, Xs).astype(np.float32)

        for feat in STATIC_LOG1P:
            if feat in static_idx:
                j = static_idx[feat]
                Xs[:, j] = np.log1p(np.maximum(Xs[:, j], 0.0)).astype(np.float32)

        for feat in STATIC_CONTINUOUS:
            if feat in static_idx:
                j = static_idx[feat]
                Xs[:, j] = ((Xs[:, j] - mean[0, j]) / std[0, j]).astype(np.float32)

        return Xs
