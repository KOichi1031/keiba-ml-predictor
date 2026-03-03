"""
model_io.py - モデル保存ユーティリティ
"""

import json
import os
from datetime import datetime, timezone

import lightgbm as lgb


def save_model_package(
    gbm: lgb.Booster,
    feature_cols: list,
    cat_cols:     list  = None,
    best_params:  dict  = None,
    prob_threshold: float = 0.5,
    output_dir:   str   = "./output/",
):
    """
    学習済みモデルと推論に必要な情報をまとめて保存する。

    保存ファイル
    ------------
    lightgbm_model.txt  : LightGBM モデル本体
    model_metadata.json : 特徴量名・型・閾値・パラメータ等

    predict.py から読み込む際に使用。
    """
    cat_cols    = cat_cols    or []
    best_params = best_params or {}
    os.makedirs(output_dir, exist_ok=True)

    gbm.save_model(os.path.join(output_dir, "lightgbm_model.txt"))

    actual_names = gbm.feature_name()
    metadata = {
        "feature_names":   actual_names,
        "cat_cols":        cat_cols,
        "num_cols":        [c for c in actual_names if c not in cat_cols],
        "feature_dtype":   {c: ("category" if c in cat_cols else "numeric") for c in actual_names},
        "prob_threshold":  prob_threshold,
        "num_features":    len(actual_names),
        "num_trees":       gbm.num_trees(),
        "best_params":     best_params,
        "saved_at":        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(os.path.join(output_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"モデルを保存しました → {os.path.abspath(output_dir)}")
    print(f"  特徴量: {actual_names}")
    print(f"  カテゴリカル: {cat_cols or '(なし)'}")
    print(f"  ツリー数: {gbm.num_trees()}  閾値: {prob_threshold}")
