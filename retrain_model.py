# ============================================================
# コーデ色バランス診断 - モデル再学習スクリプト（Supabase版）
# ============================================================
#
# アプリの「学習データ収集」ページで貯めたデータを使って、
# モデルを再学習します。データが増えるたびに実行してください。
#
# 使い方：
#   python retrain_model.py
#
# 必要なライブラリ：pandas, scikit-learn, supabase
#   pip install pandas scikit-learn supabase
#
# 実行前に、環境変数か .env で SUPABASE_URL / SUPABASE_KEY を
# 設定しておくか、下の SUPABASE_URL / SUPABASE_KEY に直接書いてください。
# （Streamlitのsecrets.tomlと同じ値を使えばOKです）

import os
import pickle
import pandas as pd
from supabase import create_client
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score

# ------------------------------------------------------------
# ① Supabase接続情報
#    st.secrets の中身と同じものをここに入れてください
# ------------------------------------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "ここにURLを入れる")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "ここにKEYを入れる")

MIN_SAMPLES_FOR_SPLIT = 15  # これ未満なら検証は行わず全データで学習だけする

# ------------------------------------------------------------
# ② Supabaseから学習データを取得
# ------------------------------------------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
res = supabase.table("ml_training_data").select("*").execute()
records = res.data

if not records:
    print("学習データがまだありません。アプリの「学習データ収集」ページでデータを追加してください。")
    exit()

df = pd.DataFrame(records)
print(f"=== 取得した学習データ: {len(df)}件 ===")
print(df[["filename", "p1", "p2", "p3", "my_score"]])

feature_cols = ['p1', 'p2', 'p3', 'hue1', 'hue2', 'hue3', 'sat1', 'sat2', 'sat3', 'diff_from_ideal']
X = df[feature_cols]
y = df['my_score']

# ------------------------------------------------------------
# ③ 学習（データ数に応じて検証の有無を切り替える）
# ------------------------------------------------------------
if len(df) >= MIN_SAMPLES_FOR_SPLIT:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    test_score = model.score(X_test, y_test)
    print(f"\n=== テストデータでの精度 (R^2): {test_score:.2f} ===")
    print("（1.0に近いほど良い。0.5未満ならまだデータ不足か特徴量の見直しが必要）")

    cv_scores = cross_val_score(model, X, y, cv=min(5, len(df)))
    print(f"交差検証スコア: {cv_scores.mean():.2f} (±{cv_scores.std():.2f})")

    # 最終的には全データで再学習してから保存する
    model.fit(X, y)
else:
    print(f"\n⚠ データが{MIN_SAMPLES_FOR_SPLIT}件未満のため、精度検証はスキップします。")
    print("  （今は動作確認フェーズです。データが増えたら自動で検証が始まります）")
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

# ------------------------------------------------------------
# ④ 特徴量の重要度を表示
# ------------------------------------------------------------
importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n=== 特徴量の重要度 ===")
print(importance)

# ------------------------------------------------------------
# ⑤ モデルを保存 → これを app.py と同じフォルダに置く
# ------------------------------------------------------------
with open("outfit_score_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nモデルを保存しました: outfit_score_model.pkl")
print("このファイルを app.py と同じフォルダに置いて、GitHubにpushしてください。")
