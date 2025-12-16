# src/utils_fusion.py
import numpy as np
import pandas as pd
import json
import os

def align_sources(dfs: dict):
    """
    dfs: dict name -> DataFrame indexed by date
    Returns: dict of aligned DataFrames restricted to the intersection (common days).
    """
    idx = None
    for df in dfs.values():
        if idx is None:
            idx = df.index
        else:
            idx = idx.intersection(df.index)
    if idx is None or len(idx) == 0:
        # fallback to union (we'll reindex and allow NaNs)
        out = {k: v.sort_index() for k, v in dfs.items()}
        return out
    aligned = {k: v.reindex(idx).sort_index() for k, v in dfs.items()}
    return aligned

def rmse_array(a, b):
    """
    Compute RMSE between two 1-D arrays, ignoring NaNs.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() == 0:
        return float("inf")
    return float(np.sqrt(np.mean((a[mask] - b[mask])**2)))

def compute_api_errors(aligned_dfs: dict, reference: str = "nasa", variables=None):
    """
    aligned_dfs: name -> df (same index)
    reference: which API to treat as reference (e.g., 'nasa')
    variables: list of variable names to compare
    Returns: dict api_name -> mean RMSE across variables (reference skipped)
    """
    if variables is None:
        variables = ['PRECTOT', 'T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'WS2M']
    errors = {}
    if reference not in aligned_dfs:
        # pick first as reference
        reference = list(aligned_dfs.keys())[0]
    ref_df = aligned_dfs[reference]
    for name, df in aligned_dfs.items():
        if name == reference:
            continue
        var_errs = []
        for v in variables:
            if v in df.columns and v in ref_df.columns:
                e = rmse_array(df[v].values, ref_df[v].values)
                var_errs.append(e)
        errors[name] = float(np.mean(var_errs)) if var_errs else float("inf")
    return errors

def compute_weights_from_errors(errors: dict, eps: float = 1e-6):
    """
    Convert errors -> normalized weights via inverse error.
    errors: dict api -> rmse
    Returns dict api -> weight (sum = 1)
    """
    inv = {}
    for k, v in errors.items():
        try:
            inv[k] = 1.0 / (float(v) + eps)
        except Exception:
            inv[k] = 0.0
    s = sum(inv.values())
    if s == 0:
        # fallback: equal weights
        n = max(1, len(inv))
        return {k: 1.0 / n for k in inv}
    return {k: (inv[k] / s) for k in inv}

def smooth_weights(old_weights: dict, new_weights: dict, alpha: float = 0.2):
    """
    Exponential smoothing: alpha * new + (1-alpha) * old
    old_weights may be None.
    """
    if not old_weights:
        return new_weights
    sm = {}
    for k in new_weights:
        sm[k] = alpha * new_weights[k] + (1 - alpha) * old_weights.get(k, new_weights[k])
    # renormalize
    s = sum(sm.values())
    if s == 0:
        n = len(sm)
        return {k: 1.0/n for k in sm}
    return {k: v/s for k, v in sm.items()}

def weighted_fusion(aligned_dfs: dict, weights: dict, variables):
    """
    Produce a fused dataframe by weighted sum across API dataframes.
    aligned_dfs: name -> df (reindexed to common index or union)
    weights: name -> weight (weights may not sum to 1, but will be treated as proportions)
    variables: list of variables to keep
    Returns fused_df indexed the union of indexes with variables columns.
    """
    # Determine union index
    all_indexes = sorted({d.index for d in aligned_dfs.values()}, key=lambda x: len(x))[0]  # not used; simpler approach below
    union_idx = pd.Index([]) 
    for df in aligned_dfs.values():
        union_idx = union_idx.union(df.index)
    union_idx = union_idx.sort_values()

    fused = pd.DataFrame(index=union_idx, columns=variables, dtype=float)
    fused = fused.fillna(0.0)
    total_weight = sum(weights.get(k, 0.0) for k in aligned_dfs.keys())

    for name, df in aligned_dfs.items():
        w = float(weights.get(name, 0.0))
        if w == 0:
            continue
        df_sel = df.reindex(union_idx)[variables].astype(float)
        # multiply by weight and add
        fused += df_sel.fillna(0.0) * w

    # if weights didn't sum to 1, normalize fused by total_weight (if >0)
    if total_weight > 0:
        fused = fused / total_weight
    return fused

def save_weights(weights: dict, path: str):
    with open(path, "w", encoding="utf8") as f:
        json.dump(weights, f, indent=2)
