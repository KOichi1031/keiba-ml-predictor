"""
実行例: 学習から可視化・出力までの完全なワークフロー (v5)
変更点:
  - cat_cols を設定に追加 (性別などカテゴリカル特徴量)
  - corrected_code_v5 の関数に cat_cols を渡すよう修正
  - ステップ4 終了後に print_model_info() でカラム情報を表示
  - ステップ5 で save_model_package() によりモデル + メタデータを一括保存
    → 別スクリプト predict.py から読み込んで予測可能
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import optuna
from corrected_code import *
from visualization_and_output import *
from model_io import save_model_package    # モデル一括保存

# ========================================
# 設定
# ========================================

# データファイルパス
DATA_PATH = './sample.csv'

# 数値特徴量リスト
num_feature_cols = [
    '馬体重', '馬体重増減', '斤量', '斤量体重比',
    'キャリア', '間隔', '休み明け～戦目',
]

# ▼▼▼ カテゴリカル特徴量リスト (実際のカラム名に置き換えてください) ▼▼▼
# 例: 性別、毛色、騎手名、調教師名、馬場状態 など文字列 or コードで入っているもの
cat_feature_cols = ['性別']

# 全特徴量リスト (数値 + カテゴリカル の結合)
feature_cols = num_feature_cols + cat_feature_cols

# ハイパーパラメータ最適化の設定
N_TRIALS    = 100  # Optuna の試行回数
N_SPLITS    = 5    # GroupKFold の分割数
TRAIN_RATIO = 0.7  # 学習データの割合

# 出力ディレクトリ
OUTPUT_DIR = './output/'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ========================================
# 1. データ読み込み
# ========================================

print("=" * 60)
print("1. データ読み込み")
print("=" * 60)

df = pd.read_csv(DATA_PATH, encoding='cp932')
print(f"データ形状: {df.shape}")
print(f"\nカラム一覧(最初の10個):")
print(df.columns.tolist()[:10])

# 必須カラムの確認
required_cols = [
    '前走確定着順', '前走レースID(新/馬番無)',
    '前走着差タイム', 'next_top3', '前走日付',
]
print(f"\n必要なカラムの確認:")
for col in required_cols:
    exists = "✓" if col in df.columns else "✗"
    print(f"  {exists} {col}")

# 特徴量カラムの確認
print(f"\n特徴量カラムの確認:")
for col in feature_cols:
    exists = "✓" if col in df.columns else "✗"
    dtype  = "category" if col in cat_feature_cols else "numeric"
    print(f"  {exists} [{dtype}] {col}")


# ========================================
# 2. 時系列分割
# ========================================

print("\n" + "=" * 60)
print("2. 時系列分割")
print("=" * 60)

df_train_all, df_test_all = time_series_split(
    df, '前走日付', train_ratio=TRAIN_RATIO)
print(f"学習データ: {len(df_train_all)} レコード")
print(f"テストデータ: {len(df_test_all)} レコード")

n_train_filtered = len(df_train_all[df_train_all['前走確定着順'] > 1])
n_test_filtered  = len(df_test_all[df_test_all['前走確定着順']  > 1])
print(f"\n1着馬除外後:")
print(f"  学習データ: {n_train_filtered} レコード")
print(f"  テストデータ: {n_test_filtered} レコード")


# ========================================
# 3. ハイパーパラメータ最適化
# ========================================

print("\n" + "=" * 60)
print("3. ハイパーパラメータ最適化(Optuna)")
print("=" * 60)
print(f"試行回数: {N_TRIALS}")
print(f"CV分割数: {N_SPLITS}")
print(f"数値特徴量    : {num_feature_cols}")
print(f"カテゴリカル  : {cat_feature_cols}")
print("\n最適化を開始します...(時間がかかる場合があります)")

# ▼ cat_cols を渡して目的関数を生成
objective = make_objective_for_train(
    df_train_all,
    feature_cols,
    cat_cols=cat_feature_cols,   # ← 追加
    n_splits=N_SPLITS,
)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("\n最適化完了!")
print(f"Best AUC: {study.best_value:.4f}")
print(f"\nBest Parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")


# ========================================
# 4. 最終学習と評価
# ========================================

print("\n" + "=" * 60)
print("4. 最終学習と評価")
print("=" * 60)

best_params = study.best_params

# ▼ cat_cols を渡して最終学習
gbm, X_train_df, auc_test = final_train_and_eval_on_test(
    df_train_all,
    df_test_all,
    feature_cols,
    best_params,
    alpha=best_params['alpha'],
    gamma=best_params['gamma'],
    cat_cols=cat_feature_cols,   # ← 追加
)

print(f"\nTest AUC: {auc_test:.4f}")
print(f"学習データ形状: {X_train_df.shape}")


# ========================================
# 5. 可視化とモデル出力
# ========================================

print("\n" + "=" * 60)
print("5. 可視化とモデル出力")
print("=" * 60)

df_pred, feature_imp = full_analysis_and_output(
    gbm=gbm,
    X_train_df=X_train_df,
    df_test_all=df_test_all,
    feature_cols=feature_cols,
    alpha=best_params['alpha'],
    gamma=best_params['gamma'],
    output_dir=OUTPUT_DIR,
    cat_cols=cat_feature_cols,   # ← 追加: カテゴリカル特徴量を渡す
)

# ▼ モデル本体 + メタデータを一括保存 (predict.py で読み込み可能)
save_model_package(
    gbm,
    feature_cols,
    cat_cols=cat_feature_cols,
    best_params=best_params,
    prob_threshold=0.5,
    output_dir=OUTPUT_DIR,
)
print(f"  次のコマンドで予測を実行できます:")
print(f"  python predict.py --input <新しいCSV> --model_dir {OUTPUT_DIR}")


# ========================================
# 6. 追加の分析例
# ========================================

print("\n" + "=" * 60)
print("6. 追加分析(オプション)")
print("=" * 60)

print("\n予測確率が高い上位10件:")
top_predictions = df_pred.nlargest(10, 'pred_prob')[
    ['前走レースID(新/馬番無)', '前走確定着順', 'pred_prob', 'true_label']
]
print(top_predictions)

print("\n誤判定の統計:")
false_positives = df_pred[(df_pred['pred_label'] == 1) & (df_pred['true_label'] == 0)]
false_negatives = df_pred[(df_pred['pred_label'] == 0) & (df_pred['true_label'] == 1)]
print(f"  False Positives: {len(false_positives)} 件")
print(f"  False Negatives: {len(false_negatives)} 件")

print("\n確率区間別の的中率:")
for threshold in [0.3, 0.5, 0.7]:
    preds    = (df_pred['pred_prob'] >= threshold).astype(int)
    accuracy = (preds == df_pred['true_label']).mean()
    print(f"  threshold={threshold}: {accuracy:.4f}")


# ========================================
# 7. Optuna 最適化履歴の可視化
# ========================================

print("\n" + "=" * 60)
print("7. Optuna最適化履歴")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
trial_numbers = [t.number for t in study.trials]
trial_values  = [t.value if t.value is not None else 0 for t in study.trials]
ax.plot(trial_numbers, trial_values, marker='o', alpha=0.6)
ax.axhline(y=study.best_value, color='r', linestyle='--',
           label=f'Best: {study.best_value:.4f}')
ax.set_xlabel('Trial Number')
ax.set_ylabel('AUC Score')
ax.set_title('Optimization History')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
try:
    from optuna.importance import get_param_importances
    importances = get_param_importances(study)
    params_imp  = list(importances.keys())
    values_imp  = list(importances.values())
    ax.barh(params_imp, values_imp, color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title('Hyperparameter Importance')
    ax.invert_yaxis()
except Exception:
    ax.text(0.5, 0.5, 'Importance calculation not available',
            ha='center', va='center', transform=ax.transAxes)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/optuna_history.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n最適化履歴を保存しました: {OUTPUT_DIR}/optuna_history.png")


# ========================================
# 完了
# ========================================

print("\n" + "=" * 60)
print("全ての処理が完了しました!")
print("=" * 60)
print(f"\n出力ファイル一覧({OUTPUT_DIR}):")
if os.path.exists(OUTPUT_DIR):
    for file in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, file)
        size  = os.path.getsize(fpath)
        print(f"  - {file}  ({size:,} bytes)")
