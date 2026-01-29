"""
修正済みコード v4 - LightGBM 4.6.0対応 (LightGBMPruningCallback修正版)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import GroupKFold
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# ソフトラベル生成関数
def softlabel_rank_gap(
    df: pd.DataFrame,
    alpha: float = 3.0,
    gamma: float = 0.3,
    eps: float = 1e-6
) -> pd.Series:
    """
    目的:
     前走結果に対し、2着馬とのタイム差+着順をつかってを教師ラベル化
    数式:
     1着馬を除く集合 R=(前走確定着順 > 1)
     base = exp(-alpha * 2着とのタイム差) 僅差ほど高スコアになる
     decay = exp(-gamma * (前走確定着順-2)) 着順が下がるほどスコアが減衰
     score = clip(base * decay, eps, 1 - eps)
     入力:'前走レースID(新/馬番無)','前走着差タイム:float,'前走確定着順:int'が必須
     出力:y_softカラム
    """
    df_remain = df[df['前走確定着順'] > 1].copy()
    min_margin = df_remain.groupby('前走レースID(新/馬番無)')['前走着差タイム'].transform('min')
    margin_vs_2nd = (df_remain['前走着差タイム'] - min_margin).clip(lower=0.0)
    base = np.exp(-alpha * margin_vs_2nd.values)
    rank_decay = np.exp(-gamma * (df_remain['前走確定着順'].values - 2))
    score = base * rank_decay
    return pd.Series(np.clip(score, eps, 1.0 - eps), index=df_remain.index, name="y_soft")


# 時系列分割ユーティリティ関数
def time_series_split(df: pd.DataFrame, date_col: str, train_ratio: float = 0.7):
    """
    時系列データの学習/検証データ分割
    全レース(1着馬含む)を分割し、その後に学習対象(1着馬以外)フィルタを適用する
    入力:date_col: df, "前走日付",
    出力:df_train_all, df_test_all
    """
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n = len(df_sorted)
    cut = int(n * train_ratio)
    df_train_all = df_sorted.iloc[:cut].reset_index(drop=True)
    df_test_all = df_sorted.iloc[cut:].reset_index(drop=True)
    return df_train_all, df_test_all


def make_objective_for_train(
    df_train_all: pd.DataFrame,
    feature_cols,
    n_splits: int = 5,
):
    """
    学習セット(過去レース)内でGroupKFold(Race_id)を用いたCV(交差検証)
    Optunaでハイパラの同時選択でAUC最大化する

    1着馬は学習対象から除外
    CVは学習期間のみで実施
    'next_top3'('確定着順'<=3)、前走確定着順、前走レースID(新/馬番無)が必要
    """
    df_train = df_train_all[df_train_all['前走確定着順'] > 1].copy()
    groups = df_train['前走レースID(新/馬番無)'].values
    y_bin = df_train['next_top3'].values
    X = df_train[feature_cols].values
    gkf = GroupKFold(n_splits=n_splits)

    def objective(trial: optuna.trial.Trial):
        # ソフトラベル生成パラメータ
        alpha = trial.suggest_float('alpha', 0.5, 20, log=True)
        gamma = trial.suggest_float('gamma', 0.0, 1.5)
        # 学習セット全体からy_softを生成し、前走確定着順>1抽出indexで合わせる
        y_soft_sr = softlabel_rank_gap(df_train_all, alpha=alpha, gamma=gamma)
        y_soft = y_soft_sr.loc[df_train.index].values

        # LightGBMハイパラ
        params_lgb = {
            'objective': 'regression',
            'metric': 'None',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 5),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-4, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-4, 10.0, log=True),
            'max_depth': trial.suggest_int('max_depth', -1, 16),
            'max_bin': trial.suggest_int('max_bin', 63, 255),
            'verbose': -1,
        }
        
        aucs = []
        for tr_idx, va_idx in gkf.split(X, y_bin, groups):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr_soft, y_va_bin = y_soft[tr_idx], y_bin[va_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr_soft)
            dvalid = lgb.Dataset(X_va, label=y_va_bin)

            # カスタム評価関数
            def auc_eval(preds, data):
                y_true = data.get_label()
                p = 1.0 / (1.0 + np.exp(-preds))
                auc = roc_auc_score(y_true, p)
                return 'roc_auc', auc, True

            # 🔧 修正: valid_names を指定しない、またはLightGBMPruningCallbackを別の方法で使う
            gbm = lgb.train(
                params_lgb,
                dtrain,
                num_boost_round=5000,
                valid_sets=[dvalid],
                # valid_names を省略すると自動的に 'valid_0' になる
                callbacks=[
                    lgb.early_stopping(stopping_rounds=200),
                    lgb.record_evaluation({}),
                    # valid_0 を明示的に指定
                    LightGBMPruningCallback(trial, 'roc_auc', valid_name='valid_0')
                ],
                feval=auc_eval
            )
            
            # 検証データでのAUCを計算
            va_preds = gbm.predict(X_va)
            va_probs = 1.0 / (1.0 + np.exp(-va_preds))
            auc = roc_auc_score(y_va_bin, va_probs)
            aucs.append(auc)

        return float(np.mean(aucs))
    return objective


# 全データ学習(過去レースのみ)＆未来のレースで評価
def final_train_and_eval_on_test(
    df_train_all: pd.DataFrame,
    df_test_all: pd.DataFrame,
    feature_cols,
    best_params: dict,
    alpha: float,
    gamma: float,
):
    """
    過去レース(train_all)の1着馬を除く状態で再学習
    未来レース(test_all)の1着馬を除く状態で予測&AUC評価
    学習用X(DataFrame)とモデルを返し、SHAP可視化に利用

    出力:
     gbm: 学習済みLightGBMモデル
     X_train_df: 学習用特徴量DataFrame
     auc: 未来レースに対するAUCスコア
    """
    # 学習セットのソフトラベル生成と1着馬除外
    df_train = df_train_all[df_train_all['前走確定着順'] > 1].copy()
    y_soft_sr_all_train = softlabel_rank_gap(
        df_train_all, alpha=alpha, gamma=gamma)
    y_soft_train = y_soft_sr_all_train.loc[df_train.index].values
    y_bin_train = df_train['next_top3'].values
    X_train_df = df_train[feature_cols]

    params = dict(best_params)
    params.update({
        'objective': 'regression',
        'metric': 'None',
        'verbose': -1
    })
    
    # カスタム評価関数
    def auc_eval(preds, data):
        y_true = data.get_label()
        p = 1.0 / (1.0 + np.exp(-preds))
        auc = roc_auc_score(y_true, p)
        return 'roc_auc', auc, True
    
    gbm = lgb.train(
        params,
        lgb.Dataset(X_train_df.values, label=y_soft_train),
        num_boost_round=5000,
        valid_sets=[lgb.Dataset(X_train_df.values, label=y_bin_train)],
        callbacks=[lgb.early_stopping(stopping_rounds=200)],
        feval=auc_eval
    )

    # test
    df_test = df_test_all[df_test_all['前走確定着順'] > 1].copy()
    X_test = df_test[feature_cols].values
    y_test_bin = df_test['next_top3'].values
    
    # 予測
    raw_test = gbm.predict(X_test)
    p_test = 1.0 / (1.0 + np.exp(-raw_test))
    auc_test = float(roc_auc_score(y_test_bin, p_test))

    return gbm, X_train_df, auc_test


# SHAP(TreeSHAP)可視化関数
def compute_treeshap(gbm: lgb.Booster, X: pd.DataFrame):
    contrib = gbm.predict(X, pred_contrib=True)  # 最後はbias列
    shap_arr = contrib[:, :X.shape[1]]  # bias列を除外
    bias = contrib[:, X.shape[1]]
    shap_df = pd.DataFrame(shap_arr, columns=X.columns, index=X.index)
    return shap_df, bias


def plot_global_shap_bar(shap_df: pd.DataFrame, top_n: int = 20):
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    top = mean_abs.head(top_n)
    plt.figure(figsize=(8, max(4, 0.35 * top_n)))
    plt.barh(top.index, top.values, color="#1f77b4")
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Top {top_n} Feature Importances by Mean |SHAP value|")
    plt.tight_layout()
    plt.show()


def plot_dependence_scatter(shap_df: pd.DataFrame, X: pd.DataFrame, feature_name: str, color_feature: str = None):
    x = X[feature_name].values
    y = shap_df[feature_name].values
    plt.figure(figsize=(6, 4))
    if color_feature is not None and color_feature in X.columns:
        sc = plt.scatter(x, y, c=X[color_feature].values,
                         cmap="viridis", s=2, alpha=0.7)
        plt.colorbar(sc, label=f'{color_feature}')
    else:
        plt.scatter(x, y, s=12, alpha=0.7)
    plt.xlabel(f"Feature value: {feature_name}")
    plt.ylabel(f"SHAP value for {feature_name}")
    plt.title(f"Dependence Scatter Plot for {feature_name}")
    plt.tight_layout()
    plt.show()
