"""
corrected_code_v5.py - 学習コア
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import GroupKFold
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import roc_auc_score


def _apply_category_dtype(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """指定カラムを category dtype に変換して返す。"""
    df = df.copy()
    for col in (cat_cols or []):
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def softlabel_rank_gap(
    df: pd.DataFrame,
    alpha: float = 3.0,
    gamma: float = 0.3,
    eps: float = 1e-6,
) -> pd.Series:
    """
    前走結果から教師ラベルを生成する。
    必須カラム: '前走レースID(新/馬番無)', '前走着差タイム', '前走確定着順'
    """
    df_remain = df[df["前走確定着順"] > 1].copy()
    min_margin = df_remain.groupby("前走レースID(新/馬番無)")["前走着差タイム"].transform("min")
    margin_vs_2nd = (df_remain["前走着差タイム"] - min_margin).clip(lower=0.0)
    base = np.exp(-alpha * margin_vs_2nd.values)
    rank_decay = np.exp(-gamma * (df_remain["前走確定着順"].values - 2))
    score = base * rank_decay
    return pd.Series(np.clip(score, eps, 1.0 - eps), index=df_remain.index, name="y_soft")


def time_series_split(df: pd.DataFrame, date_col: str, train_ratio: float = 0.7):
    """時系列で学習/テストデータに分割する。"""
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    cut = int(len(df_sorted) * train_ratio)
    return df_sorted.iloc[:cut].reset_index(drop=True), df_sorted.iloc[cut:].reset_index(drop=True)


def make_objective_for_train(
    df_train_all: pd.DataFrame,
    feature_cols: list,
    cat_cols: list = None,
    n_splits: int = 5,
):
    """Optuna 用の目的関数を生成する (GroupKFold CV で AUC 最大化)。"""
    cat_cols = cat_cols or []
    df_train = _apply_category_dtype(
        df_train_all[df_train_all["前走確定着順"] > 1].copy(), cat_cols
    )
    groups  = df_train["前走レースID(新/馬番無)"].values
    y_bin   = df_train["next_top3"].values
    X_df    = df_train[feature_cols]
    idx_arr = np.arange(len(X_df))
    gkf     = GroupKFold(n_splits=n_splits)

    def objective(trial: optuna.trial.Trial):
        alpha = trial.suggest_float("alpha", 0.5, 20, log=True)
        gamma = trial.suggest_float("gamma", 0.0, 1.5)
        y_soft = softlabel_rank_gap(df_train_all, alpha=alpha, gamma=gamma).loc[df_train.index].values

        params_lgb = {
            "objective":        "regression",
            "metric":           "None",
            "learning_rate":    trial.suggest_float("learning_rate",    0.01, 0.2,  log=True),
            "num_leaves":       trial.suggest_int(  "num_leaves",       31,   255),
            "min_data_in_leaf": trial.suggest_int(  "min_data_in_leaf", 20,   200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6,  1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6,  1.0),
            "bagging_freq":     trial.suggest_int(  "bagging_freq",     0,    5),
            "lambda_l1":        trial.suggest_float("lambda_l1",        1e-4, 10.0, log=True),
            "lambda_l2":        trial.suggest_float("lambda_l2",        1e-4, 10.0, log=True),
            "max_depth":        trial.suggest_int(  "max_depth",        -1,   16),
            "max_bin":          trial.suggest_int(  "max_bin",          63,   255),
            "verbose":          -1,
        }

        aucs = []
        for tr_idx, va_idx in gkf.split(idx_arr, y_bin, groups):
            X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
            dtrain = lgb.Dataset(X_tr, label=y_soft[tr_idx],  free_raw_data=False)
            dvalid = lgb.Dataset(X_va, label=y_bin[va_idx],   free_raw_data=False, reference=dtrain)

            def auc_eval(preds, data):
                p = 1.0 / (1.0 + np.exp(-preds))
                return "roc_auc", roc_auc_score(data.get_label(), p), True

            gbm_cv = lgb.train(
                params_lgb, dtrain, num_boost_round=5000, valid_sets=[dvalid],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=200),
                    lgb.record_evaluation({}),
                    LightGBMPruningCallback(trial, "roc_auc", valid_name="valid_0"),
                ],
                feval=auc_eval,
            )
            va_probs = 1.0 / (1.0 + np.exp(-gbm_cv.predict(X_va)))
            aucs.append(roc_auc_score(y_bin[va_idx], va_probs))

        return float(np.mean(aucs))

    return objective


def final_train_and_eval_on_test(
    df_train_all: pd.DataFrame,
    df_test_all:  pd.DataFrame,
    feature_cols: list,
    best_params:  dict,
    alpha: float,
    gamma: float,
    cat_cols: list = None,
):
    """
    学習データで最終モデルを学習し、テストデータで AUC を評価する。
    戻り値: (gbm, X_train_df, auc_test)
    """
    cat_cols = cat_cols or []

    df_train   = _apply_category_dtype(df_train_all[df_train_all["前走確定着順"] > 1].copy(), cat_cols)
    y_soft     = softlabel_rank_gap(df_train_all, alpha=alpha, gamma=gamma).loc[df_train.index].values
    y_bin      = df_train["next_top3"].values
    X_train_df = df_train[feature_cols]

    params = {k: v for k, v in best_params.items() if k not in ("alpha", "gamma")}
    params.update({"objective": "regression", "metric": "None", "verbose": -1})

    def auc_eval(preds, data):
        p = 1.0 / (1.0 + np.exp(-preds))
        return "roc_auc", roc_auc_score(data.get_label(), p), True

    gbm = lgb.train(
        params,
        lgb.Dataset(X_train_df, label=y_soft, free_raw_data=False),
        num_boost_round=5000,
        valid_sets=[lgb.Dataset(X_train_df, label=y_bin, free_raw_data=False)],
        callbacks=[lgb.early_stopping(stopping_rounds=200)],
        feval=auc_eval,
    )

    df_test    = _apply_category_dtype(df_test_all[df_test_all["前走確定着順"] > 1].copy(), cat_cols)
    X_test     = df_test[feature_cols]
    p_test     = 1.0 / (1.0 + np.exp(-gbm.predict(X_test)))
    auc_test   = float(roc_auc_score(df_test["next_top3"].values, p_test))

    return gbm, X_train_df, auc_test
