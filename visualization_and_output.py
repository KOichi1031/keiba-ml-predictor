"""
特徴量可視化とモデル出力コード (LightGBM 4.x 対応版)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report


# ========================================
# 1. 特徴量重要度の可視化
# ========================================

def plot_feature_importance(gbm, feature_cols, top_n=30, importance_type='gain'):
    """
    LightGBMの特徴量重要度を可視化
    
    Parameters:
    -----------
    gbm : lgb.Booster
        学習済みモデル
    feature_cols : list
        特徴量名のリスト
    top_n : int
        表示する上位特徴量数
    importance_type : str
        'gain' または 'split' (LightGBM 4.x では 'weight' は非推奨)
    """
    importance = gbm.feature_importance(importance_type=importance_type)
    feature_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, max(6, top_n * 0.3)))
    plt.barh(range(len(feature_imp)), feature_imp['importance'], color='steelblue')
    plt.yticks(range(len(feature_imp)), feature_imp['feature'])
    plt.xlabel(f'Importance ({importance_type})')
    plt.title(f'Top {top_n} Feature Importance ({importance_type})')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return feature_imp


def plot_all_feature_importance(gbm, feature_cols, top_n=20):
    """
    2種類の重要度を並べて表示 (LightGBM 4.x対応)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, top_n * 0.25)))
    
    # gain と split のみ使用 (weight は廃止)
    for ax, imp_type in zip(axes, ['gain', 'split']):
        importance = gbm.feature_importance(importance_type=imp_type)
        feature_imp = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        ax.barh(range(len(feature_imp)), feature_imp['importance'], color='steelblue')
        ax.set_yticks(range(len(feature_imp)))
        ax.set_yticklabels(feature_imp['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'{imp_type.capitalize()} Importance')
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()


# ========================================
# 2. 予測結果の分析
# ========================================

def create_prediction_dataframe(
    df_test_all,
    feature_cols,
    gbm,
    alpha,
    gamma,
    prob_threshold=0.5
):
    """
    テストデータに対する予測結果を含むDataFrameを作成
    
    Returns:
    --------
    df_pred : pd.DataFrame
        予測確率、予測ラベル、実際のラベルを含むDataFrame
    """
    # 1着馬を除外
    df_test = df_test_all[df_test_all['前走確定着順'] > 1].copy()
    
    # 特徴量とラベル
    X_test = df_test[feature_cols].values
    y_test_bin = df_test['next_top3'].values
    
    # 予測
    raw_preds = gbm.predict(X_test)
    pred_probs = 1.0 / (1.0 + np.exp(-raw_preds))
    pred_labels = (pred_probs >= prob_threshold).astype(int)
    
    # 結果を元のDataFrameに追加
    df_pred = df_test.copy()
    df_pred['pred_prob'] = pred_probs
    df_pred['pred_label'] = pred_labels
    df_pred['true_label'] = y_test_bin
    
    return df_pred


def plot_prediction_distribution(df_pred):
    """
    予測確率の分布を可視化(正解/不正解別)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 正解ラベル別の予測確率分布
    ax = axes[0]
    for label in [0, 1]:
        probs = df_pred[df_pred['true_label'] == label]['pred_prob']
        ax.hist(probs, bins=50, alpha=0.6, label=f'True={label}')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Probability Distribution by True Label')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # BOX plot
    ax = axes[1]
    data_to_plot = [
        df_pred[df_pred['true_label'] == 0]['pred_prob'],
        df_pred[df_pred['true_label'] == 1]['pred_prob']
    ]
    ax.boxplot(data_to_plot, labels=['True=0', 'True=1'])
    ax.set_ylabel('Predicted Probability')
    ax.set_title('Prediction Probability Box Plot')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_roc_curve(df_pred):
    """
    ROC曲線を描画
    """
    fpr, tpr, thresholds = roc_curve(df_pred['true_label'], df_pred['pred_prob'])
    auc_score = roc_auc_score(df_pred['true_label'], df_pred['pred_prob'])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return auc_score


def plot_confusion_matrix(df_pred, threshold=0.5):
    """
    混同行列を可視化
    """
    pred_labels = (df_pred['pred_prob'] >= threshold).astype(int)
    cm = confusion_matrix(df_pred['true_label'], pred_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred: 0', 'Pred: 1'],
                yticklabels=['True: 0', 'True: 1'])
    plt.title(f'Confusion Matrix (threshold={threshold})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # 分類レポートを表示
    print("\n=== Classification Report ===")
    print(classification_report(df_pred['true_label'], pred_labels, 
                                target_names=['Class 0', 'Class 1']))
    
    return cm


def analyze_by_probability_bins(df_pred, n_bins=10):
    """
    予測確率を区間分割して、各区間での的中率を分析
    """
    df_pred['prob_bin'] = pd.cut(df_pred['pred_prob'], bins=n_bins)
    
    bin_analysis = df_pred.groupby('prob_bin', observed=True).agg({
        'true_label': ['count', 'sum', 'mean']
    }).reset_index()
    bin_analysis.columns = ['prob_bin', 'count', 'positive_count', 'accuracy']
    
    # 区間の中央値を計算
    bin_analysis['bin_center'] = bin_analysis['prob_bin'].apply(lambda x: x.mid)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 的中率
    ax = axes[0]
    ax.plot(bin_analysis['bin_center'], bin_analysis['accuracy'], 
            marker='o', linewidth=2, markersize=8)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect Calibration')
    ax.set_xlabel('Predicted Probability (Bin Center)')
    ax.set_ylabel('Actual Positive Rate')
    ax.set_title('Calibration Curve')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # サンプル数
    ax = axes[1]
    ax.bar(range(len(bin_analysis)), bin_analysis['count'], color='steelblue')
    ax.set_xlabel('Probability Bin')
    ax.set_ylabel('Count')
    ax.set_title('Sample Count per Bin')
    ax.set_xticks(range(len(bin_analysis)))
    ax.set_xticklabels([f'{x.left:.2f}-{x.right:.2f}' 
                        for x in bin_analysis['prob_bin']], rotation=45)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return bin_analysis


# ========================================
# 3. SHAP値による特徴量分析
# ========================================

def compute_and_plot_shap(gbm, X_train_df, top_n=20):
    """
    SHAP値を計算して可視化
    
    Parameters:
    -----------
    gbm : lgb.Booster
        学習済みモデル
    X_train_df : pd.DataFrame
        学習データの特徴量DataFrame
    top_n : int
        表示する上位特徴量数
    """
    # SHAP値を計算
    shap_df, bias = compute_treeshap(gbm, X_train_df)
    
    print(f"SHAP計算完了: bias = {bias.mean():.4f}")
    
    # グローバル重要度
    plot_global_shap_bar(shap_df, top_n=top_n)
    
    return shap_df, bias


def compute_treeshap(gbm, X):
    """
    TreeSHAPを計算
    """
    contrib = gbm.predict(X, pred_contrib=True)
    shap_arr = contrib[:, :X.shape[1]]
    bias = contrib[:, X.shape[1]]
    shap_df = pd.DataFrame(shap_arr, columns=X.columns, index=X.index)
    return shap_df, bias


def plot_global_shap_bar(shap_df, top_n=20):
    """
    SHAP値のグローバル重要度を棒グラフで表示
    """
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    top = mean_abs.head(top_n)
    
    plt.figure(figsize=(10, max(6, top_n * 0.3)))
    plt.barh(range(len(top)), top.values, color='steelblue')
    plt.yticks(range(len(top)), top.index)
    plt.xlabel('Mean |SHAP value|')
    plt.title(f'Top {top_n} Feature Importances by SHAP')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_shap_dependence(shap_df, X_train_df, feature_name, color_feature=None, sample_size=5000):
    """
    特定の特徴量のSHAP dependence plotを描画
    
    Parameters:
    -----------
    shap_df : pd.DataFrame
        SHAP値のDataFrame
    X_train_df : pd.DataFrame
        特徴量DataFrame
    feature_name : str
        分析する特徴量名
    color_feature : str
        色付けに使う特徴量名
    sample_size : int
        プロット点数(多すぎると重いのでサンプリング)
    """
    # サンプリング
    if len(shap_df) > sample_size:
        sample_idx = np.random.choice(len(shap_df), sample_size, replace=False)
        shap_sample = shap_df.iloc[sample_idx]
        X_sample = X_train_df.iloc[sample_idx]
    else:
        shap_sample = shap_df
        X_sample = X_train_df
    
    x = X_sample[feature_name].values
    y = shap_sample[feature_name].values
    
    plt.figure(figsize=(8, 6))
    if color_feature is not None and color_feature in X_sample.columns:
        sc = plt.scatter(x, y, c=X_sample[color_feature].values,
                         cmap="viridis", s=10, alpha=0.6)
        plt.colorbar(sc, label=f'{color_feature}')
    else:
        plt.scatter(x, y, s=10, alpha=0.6, color='steelblue')
    
    plt.xlabel(f'Feature value: {feature_name}')
    plt.ylabel(f'SHAP value for {feature_name}')
    plt.title(f'SHAP Dependence Plot: {feature_name}')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ========================================
# 4. モデル出力(予測結果の保存)
# ========================================

def save_predictions(df_pred, output_path='predictions.csv'):
    """
    予測結果をCSVファイルに保存
    
    Parameters:
    -----------
    df_pred : pd.DataFrame
        予測結果を含むDataFrame
    output_path : str
        保存先パス
    """
    # 必要なカラムを選択して保存
    output_cols = [
        '前走レースID(新/馬番無)', '前走確定着順', '前走着差タイム',
        'true_label', 'pred_prob', 'pred_label'
    ]
    
    # カラムが存在するもののみ選択
    existing_cols = [col for col in output_cols if col in df_pred.columns]
    df_output = df_pred[existing_cols].copy()
    
    df_output.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n予測結果を保存しました: {output_path}")
    print(f"保存レコード数: {len(df_output)}")
    
    return df_output


def save_model(gbm, model_path='lightgbm_model.txt'):
    """
    LightGBMモデルを保存
    
    Parameters:
    -----------
    gbm : lgb.Booster
        学習済みモデル
    model_path : str
        保存先パス
    """
    gbm.save_model(model_path)
    print(f"\nモデルを保存しました: {model_path}")


def load_model(model_path='lightgbm_model.txt'):
    """
    保存したモデルを読み込み
    
    Parameters:
    -----------
    model_path : str
        モデルファイルのパス
    
    Returns:
    --------
    gbm : lgb.Booster
        読み込んだモデル
    """
    gbm = lgb.Booster(model_file=model_path)
    print(f"\nモデルを読み込みました: {model_path}")
    return gbm


# ========================================
# 5. 統合実行関数
# ========================================

def full_analysis_and_output(
    gbm,
    X_train_df,
    df_test_all,
    feature_cols,
    alpha,
    gamma,
    output_dir='./output/'
):
    """
    全ての可視化とモデル出力を一括実行
    
    Parameters:
    -----------
    gbm : lgb.Booster
        学習済みモデル
    X_train_df : pd.DataFrame
        学習データの特徴量DataFrame
    df_test_all : pd.DataFrame
        テストデータ全体
    feature_cols : list
        特徴量名のリスト
    alpha : float
        ソフトラベル生成のalphaパラメータ
    gamma : float
        ソフトラベル生成のgammaパラメータ
    output_dir : str
        出力ディレクトリ
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("分析開始")
    print("=" * 60)
    
    # 1. 特徴量重要度
    print("\n[1/6] 特徴量重要度を計算中...")
    plot_all_feature_importance(gbm, feature_cols, top_n=20)
    feature_imp = plot_feature_importance(gbm, feature_cols, top_n=30, importance_type='gain')
    feature_imp.to_csv(f'{output_dir}/feature_importance.csv', index=False, encoding='utf-8-sig')
    
    # 2. 予測結果の作成
    print("\n[2/6] 予測結果を作成中...")
    df_pred = create_prediction_dataframe(df_test_all, feature_cols, gbm, alpha, gamma)
    
    # 3. 予測分布の可視化
    print("\n[3/6] 予測分布を可視化中...")
    plot_prediction_distribution(df_pred)
    
    # 4. ROC曲線
    print("\n[4/6] ROC曲線を描画中...")
    auc_score = plot_roc_curve(df_pred)
    print(f"Test AUC: {auc_score:.4f}")
    
    # 5. 混同行列
    print("\n[5/6] 混同行列を作成中...")
    cm = plot_confusion_matrix(df_pred, threshold=0.5)
    
    # 6. 確率区間別の分析
    print("\n[6/6] 確率区間別分析中...")
    bin_analysis = analyze_by_probability_bins(df_pred, n_bins=10)
    bin_analysis.to_csv(f'{output_dir}/probability_bin_analysis.csv', index=False, encoding='utf-8-sig')
    
    # 7. SHAP分析(時間がかかる場合はスキップ可)
    print("\n[オプション] SHAP分析中...")
    try:
        shap_df, bias = compute_and_plot_shap(gbm, X_train_df, top_n=20)
        
        # 上位3特徴量のdependence plot
        top_features = feature_imp.head(3)['feature'].tolist()
        for feat in top_features:
            print(f"  - {feat} のSHAP dependence plot")
            plot_shap_dependence(shap_df, X_train_df, feat)
    except Exception as e:
        print(f"SHAP分析でエラー: {e}")
    
    # 8. 結果の保存
    print("\n結果を保存中...")
    save_predictions(df_pred, f'{output_dir}/predictions.csv')
    save_model(gbm, f'{output_dir}/lightgbm_model.txt')
    
    print("\n" + "=" * 60)
    print("分析完了!")
    print("=" * 60)
    print(f"\n出力ディレクトリ: {output_dir}")
    print("  - feature_importance.csv")
    print("  - predictions.csv")
    print("  - probability_bin_analysis.csv")
    print("  - lightgbm_model.txt")
    
    return df_pred, feature_imp
