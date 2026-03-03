"""
predict.py - 学習済みモデルで DataFrame に予測確率を追加する
"""

import json
import os
import numpy as np
import pandas as pd
import lightgbm as lgb


def predict(df: pd.DataFrame, model_dir: str = "./output/") -> pd.DataFrame:
    """
    学習済みモデルを読み込み、DataFrame に pred_prob / pred_label を追加して返す。

    Parameters
    ----------
    df        : 予測対象の DataFrame
    model_dir : save_model_package() で保存したディレクトリ

    Returns
    -------
    df_result : pred_prob, pred_label カラムを追加した DataFrame
    """

    # --- モデルとメタデータを読み込む ---
    gbm = lgb.Booster(model_file=os.path.join(model_dir, "lightgbm_model.txt"))
    with open(os.path.join(model_dir, "model_metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)

    feature_cols  = metadata["feature_names"]
    cat_cols      = metadata.get("cat_cols", [])
    threshold     = metadata.get("prob_threshold", 0.5)

    # --- 前走1着馬を除外 (学習時と同条件) ---
    if "前走確定着順" in df.columns:
        df_target = df[df["前走確定着順"] > 1].copy()
    else:
        df_target = df.copy()

    # --- カテゴリカル列を dtype 変換 ---
    for col in cat_cols:
        if col in df_target.columns:
            df_target[col] = df_target[col].astype("category")

    # --- 予測 ---
    raw   = gbm.predict(df_target[feature_cols])
    prob  = 1.0 / (1.0 + np.exp(-raw))

    df_target["pred_prob"]  = prob
    df_target["pred_label"] = (prob >= threshold).astype(int)

    return df_target
