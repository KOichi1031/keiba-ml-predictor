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
