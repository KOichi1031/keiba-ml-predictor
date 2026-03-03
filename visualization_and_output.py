"""
visualization_and_output.py - 可視化とモデル出力
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report


def _apply_category_dtype(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in (cat_cols or []):
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


# ========================================
# 特徴量重要度
# ========================================

def plot_feature_importance(gbm, feature_cols=None, top_n=30, importance_type="gain"):
    """特徴量重要度を棒グラフで表示。feature_cols 省略時は gbm.feature_name() を使用。"""
    if feature_cols is None:
        feature_cols = gbm.feature_name()
    feature_imp = pd.DataFrame({
        "feature":    feature_cols,
        "importance": gbm.feature_importance(importance_type=importance_type),
    }).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, max(6, top_n * 0.3)))
    plt.barh(range(len(feature_imp)), feature_imp["importance"], color="steelblue")
    plt.yticks(range(len(feature_imp)), feature_imp["feature"])
    plt.xlabel(f"Importance ({importance_type})")
    plt.title(f"Top {top_n} Feature Importance ({importance_type})")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    return feature_imp


def plot_all_feature_importance(gbm, feature_cols=None, top_n=20):
    """gain / split 2種類の重要度を並べて表示。"""
    if feature_cols is None:
        feature_cols = gbm.feature_name()
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, top_n * 0.25)))
    for ax, imp_type in zip(axes, ["gain", "split"]):
        feature_imp = pd.DataFrame({
            "feature":    feature_cols,
            "importance": gbm.feature_importance(importance_type=imp_type),
        }).sort_values("importance", ascending=False).head(top_n)
        ax.barh(range(len(feature_imp)), feature_imp["importance"], color="steelblue")
        ax.set_yticks(range(len(feature_imp)))
        ax.set_yticklabels(feature_imp["feature"])
        ax.set_xlabel("Importance")
        ax.set_title(f"{imp_type.capitalize()} Importance")
        ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


# ========================================
# 予測結果の分析
# ========================================

def create_prediction_dataframe(df_test_all, feature_cols, gbm, alpha, gamma,
                                prob_threshold=0.5, cat_cols=None):
    """テストデータに予測確率・予測ラベルを付与した DataFrame を返す。"""
    df_test = _apply_category_dtype(
        df_test_all[df_test_all["前走確定着順"] > 1].copy(), cat_cols
    )
    raw_preds  = gbm.predict(df_test[feature_cols])
    pred_probs = 1.0 / (1.0 + np.exp(-raw_preds))

    df_pred = df_test.copy()
    df_pred["pred_prob"]  = pred_probs
    df_pred["pred_label"] = (pred_probs >= prob_threshold).astype(int)
    df_pred["true_label"] = df_test["next_top3"].values
    return df_pred


def plot_prediction_distribution(df_pred):
    """予測確率の分布をラベル別に可視化。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for label in [0, 1]:
        ax.hist(df_pred[df_pred["true_label"] == label]["pred_prob"],
                bins=50, alpha=0.6, label=f"True={label}")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Probability Distribution by True Label")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.boxplot([df_pred[df_pred["true_label"] == 0]["pred_prob"],
                df_pred[df_pred["true_label"] == 1]["pred_prob"]],
               labels=["True=0", "True=1"])
    ax.set_ylabel("Predicted Probability")
    ax.set_title("Prediction Probability Box Plot")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(df_pred):
    """ROC 曲線を描画して AUC を返す。"""
    fpr, tpr, _ = roc_curve(df_pred["true_label"], df_pred["pred_prob"])
    auc_score   = roc_auc_score(df_pred["true_label"], df_pred["pred_prob"])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return auc_score


def plot_confusion_matrix(df_pred, threshold=0.5):
    """混同行列を可視化。"""
    pred_labels = (df_pred["pred_prob"] >= threshold).astype(int)
    cm = confusion_matrix(df_pred["true_label"], pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred: 0", "Pred: 1"],
                yticklabels=["True: 0", "True: 1"])
    plt.title(f"Confusion Matrix (threshold={threshold})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()
    print("\n=== Classification Report ===")
    print(classification_report(df_pred["true_label"], pred_labels,
                                target_names=["Class 0", "Class 1"]))
    return cm


def analyze_by_probability_bins(df_pred, n_bins=10):
    """確率区間別の的中率を分析・可視化。"""
    df_tmp = df_pred.copy()
    df_tmp["prob_bin"] = pd.cut(df_tmp["pred_prob"], bins=n_bins)
    bin_analysis = df_tmp.groupby("prob_bin", observed=True).agg(
        count=("true_label", "count"),
        positive_count=("true_label", "sum"),
        accuracy=("true_label", "mean"),
    ).reset_index()
    bin_analysis["bin_center"] = bin_analysis["prob_bin"].apply(lambda x: x.mid)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(bin_analysis["bin_center"], bin_analysis["accuracy"], marker="o", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect Calibration")
    ax.set_xlabel("Predicted Probability (Bin Center)")
    ax.set_ylabel("Actual Positive Rate")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.bar(range(len(bin_analysis)), bin_analysis["count"], color="steelblue")
    ax.set_xlabel("Probability Bin")
    ax.set_ylabel("Count")
    ax.set_title("Sample Count per Bin")
    ax.set_xticks(range(len(bin_analysis)))
    ax.set_xticklabels([f"{x.left:.2f}-{x.right:.2f}"
                        for x in bin_analysis["prob_bin"]], rotation=45)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return bin_analysis


# ========================================
# SHAP 分析
# ========================================

def compute_treeshap(gbm: lgb.Booster, X: pd.DataFrame):
    """TreeSHAP を計算して (shap_df, bias) を返す。"""
    contrib  = gbm.predict(X, pred_contrib=True)
    shap_arr = contrib[:, :X.shape[1]]
    bias     = contrib[:, X.shape[1]]
    return pd.DataFrame(shap_arr, columns=X.columns, index=X.index), bias


def compute_and_plot_shap(gbm, X_train_df, top_n=20):
    """SHAP を計算してグローバル重要度を可視化。"""
    shap_df, bias = compute_treeshap(gbm, X_train_df)
    print(f"SHAP 計算完了: bias = {bias.mean():.4f}")
    plot_global_shap_bar(shap_df, top_n=top_n)
    return shap_df, bias


def plot_global_shap_bar(shap_df: pd.DataFrame, top_n: int = 20):
    """SHAP 値の平均絶対値を棒グラフで表示。"""
    top = shap_df.abs().mean().sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(10, max(6, top_n * 0.3)))
    plt.barh(range(len(top)), top.values, color="steelblue")
    plt.yticks(range(len(top)), top.index)
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Top {top_n} Feature Importances by SHAP")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_shap_dependence(shap_df, X_train_df, feature_name, color_feature=None, sample_size=5000):
    """特定の特徴量の SHAP Dependence Plot を描画。"""
    if len(shap_df) > sample_size:
        idx         = np.random.choice(len(shap_df), sample_size, replace=False)
        shap_sample = shap_df.iloc[idx]
        X_sample    = X_train_df.iloc[idx]
    else:
        shap_sample, X_sample = shap_df, X_train_df

    x = X_sample[feature_name].values
    if hasattr(x.dtype, "categories"):
        x = x.astype("category").cat.codes.values

    plt.figure(figsize=(8, 6))
    if color_feature and color_feature in X_sample.columns:
        c = X_sample[color_feature].values
        if hasattr(c.dtype, "categories") or pd.api.types.is_object_dtype(c):
            c = pd.Categorical(c).codes.astype(float)
        sc = plt.scatter(x, shap_sample[feature_name].values, c=c, cmap="viridis", s=10, alpha=0.6)
        plt.colorbar(sc, label=color_feature)
    else:
        plt.scatter(x, shap_sample[feature_name].values, s=10, alpha=0.6, color="steelblue")
    plt.xlabel(f"Feature value: {feature_name}")
    plt.ylabel(f"SHAP value for {feature_name}")
    plt.title(f"SHAP Dependence Plot: {feature_name}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ========================================
# 予測結果の保存
# ========================================

def save_predictions(df_pred, output_path="predictions.csv"):
    """予測結果を CSV に保存。"""
    output_cols = ["前走レースID(新/馬番無)", "前走確定着順", "前走着差タイム",
                   "true_label", "pred_prob", "pred_label"]
    existing    = [c for c in output_cols if c in df_pred.columns]
    df_pred[existing].to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"予測結果を保存しました: {output_path}  ({len(df_pred)} 件)")


# ========================================
# 統合実行関数
# ========================================

def full_analysis_and_output(gbm, X_train_df, df_test_all, feature_cols,
                             alpha, gamma, output_dir="./output/", cat_cols=None):
    """可視化・評価・結果保存を一括実行。"""
    os.makedirs(output_dir, exist_ok=True)

    print("[1/6] 特徴量重要度")
    plot_all_feature_importance(gbm, feature_cols, top_n=20)
    feature_imp = plot_feature_importance(gbm, feature_cols, top_n=30, importance_type="gain")
    feature_imp.to_csv(f"{output_dir}/feature_importance.csv", index=False, encoding="utf-8-sig")

    print("[2/6] 予測結果を作成")
    df_pred = create_prediction_dataframe(df_test_all, feature_cols, gbm,
                                          alpha, gamma, cat_cols=cat_cols)

    print("[3/6] 予測分布")
    plot_prediction_distribution(df_pred)

    print("[4/6] ROC 曲線")
    auc_score = plot_roc_curve(df_pred)
    print(f"Test AUC: {auc_score:.4f}")

    print("[5/6] 混同行列")
    plot_confusion_matrix(df_pred)

    print("[6/6] 確率区間別分析")
    bin_analysis = analyze_by_probability_bins(df_pred)
    bin_analysis.to_csv(f"{output_dir}/probability_bin_analysis.csv", index=False, encoding="utf-8-sig")

    print("[オプション] SHAP 分析")
    try:
        shap_df, _ = compute_and_plot_shap(gbm, X_train_df, top_n=20)
        for feat in feature_imp.head(3)["feature"].tolist():
            plot_shap_dependence(shap_df, X_train_df, feat)
    except Exception as e:
        print(f"SHAP 分析でエラー: {e}")

    save_predictions(df_pred, f"{output_dir}/predictions.csv")

    return df_pred, feature_imp
