#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import GroupKFold
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# In[20]:

# ソフトラベル生成関数
def softlabel_rank_gap(
    df: pd.DataFrame,
    alpha: float = 3.0,
    gamma: float = 0.3,
    eps: float = 1e-6
) -> pd.Series:
    """
    目的：
     前走結果に対し、2着馬とのタイム差+着順をつかってを教師ラベル化
    数式：
     1着馬を除く集合 R=(前走確定着順 > 1)
     base = exp(-alpha * 2着とのタイム差) 僅差ほど高スコアになる
     decay = exp(-gamma * (前走確定着順-2)) 着順が下がるほどスコアが減衰
     scre = clip(base * decay, eps, 1 - eps)
     入力:'前走レースID(新/馬番無)','前走着差タイム:float,'前走確定着順:int'が必須
     出力：y_softカラム
    """
    df_remain = df[df['前走確定着順'] > 1].copy()
    min_margin = df_remain.groupby('前走レースID(新/馬番無)')['前走着差タイム'].transform(min)
    margin_vs_2nd = (df_remain['前走着差タイム'] - min_margin).clip(lower=0.0)
    base = np.exp(-alpha * margin_vs_2nd.values)
    rank_decay = np.exp(-gamma * (df_remain['前走確定着順'].values - 2))
    score = base * rank_decay
    return pd.Series(np.clip(score, eps, 1.0 - eps), index=df_remain, name="y_soft")


# 学習はソフトラベルのBVE最小化（fobj)。評価は2値ラベル（次走の3着以内）に対するAUC(febal)
def bec_softlabel_obj(preds, train_data):
    """
    LightGBM用のカスタム目的関数
      p = sigmoid(f)
      L = -[y*log(p) + (1-y)*log(1-p)], y[0,1]はSoftlabel
      grad = p - y, hess = p * (1 - p)
    """
    y = train_data.get_label()  # y_soft
    p = 1.0 / (1.0 + np.exp(-preds))  # シグモイド変換
    grad = p - y
    hess = p * (1.0 - p)
    return grad, hess


def acu_binary_metric(preds, train_data):
    """
    LightGBM用のカスタム評価関数
    feval: 次走3着以内（0/1）に対する予測確率pのROC-AUCを返す（higher_is_better=True）
    """
    y = train_data.get_label()  # 0/1ラベル
    p = 1.0 / (1.0 + np.exp(-preds))  # シグモイド変換
    return 'roc_auc', float(roc_auc_score(y, p)), True

# 　時系列分割ユーティリティ関数


def time_series_splilt(df: pd.DataFrame, data_col: str, train_ratio: float = 0.7):
    """
    時系列データの学習/検証データ分割
    全レース（1着馬含む）を分割し、その後に学習対象（1着馬以外）フィルタを適用する
    入力：data_col: df, "前走日付",
    出力：df_train_all, df_test_all
    """
    df_sorted = df.sort_values(data_col).reset_index(drop=True)
    n = len(df_sorted)
    cut = int(n * train_ratio)
    df_train_all = df_sorted.iloc[:cut].reset_index(drop=True)
    df_test_all = df_sorted.iloc[cut:].reset_index(drop=True)
    return df_train_all, df_test_all
