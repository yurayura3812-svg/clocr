# ============================================================
# コーデ色バランス診断 - 学習データ作成 & モデル訓練スクリプト
# Google Colabで実行してください
# ============================================================
#
# 使い方：
# 1. このファイルと一緒に、診断したい画像(4枚 or それ以上)を
#    Colabにアップロードする
# 2. IMAGE_SCORES の辞書に、ファイル名と自分の採点(0〜100)を書く
# 3. 上から順に実行する
#
# 必要なライブラリ（Colabなら大体プリインストール済み。
# 足りなければ !pip install で入れてください）
#   opencv-python, mediapipe, scikit-learn, pandas

import cv2
import numpy as np
import pandas as pd
import glob
import urllib.request
import os

# ------------------------------------------------------------
# ① 自分の採点をここに入力する
#    キー：ファイル名（アップロードした画像のファイル名と一致させる）
#    値　：0〜100点のあなたの主観スコア
# ------------------------------------------------------------
IMAGE_SCORES = {
    "outfit1.png": 80,   # ①ベージュ×赤バッグ
    "outfit2.png": 55,   # ②赤×青ジーンズ×緑スカーフ
    "outfit3.png": 65,   # ③黒一色
    "outfit4.png": 85,   # ④ネイビー×オレンジ
    # 画像を増やしたら、ここにどんどん追加していく
}

IMAGE_DIR = "."  # 画像を置いたフォルダ（Colabならそのままでも可）

# ------------------------------------------------------------
# ② 特徴量抽出（今のアプリのanalyze()と同じロジック）
# ------------------------------------------------------------

def download_model():
    model_path = "selfie_segmenter.tflite"
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
        urllib.request.urlretrieve(url, model_path)
    return model_path


def analyze(img_rgb, model_path):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    height, width = img_rgb.shape[:2]
    max_size = 600
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        img_rgb = cv2.resize(img_rgb, (int(width * scale), int(height * scale)))
        height, width = img_rgb.shape[:2]

    options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        output_category_mask=False,
        output_confidence_masks=True
    )
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = segmenter.segment(mp_image)
        confidence_mask = np.squeeze(result.confidence_masks[0].numpy_view())
        person_mask = (confidence_mask > 0.5).astype(np.uint8)

    img_hsv_init = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 45, 80], dtype=np.uint8)
    upper_skin = np.array([22, 140, 240], dtype=np.uint8)
    skin_mask = cv2.inRange(img_hsv_init, lower_skin, upper_skin)
    pure_clothing_mask = cv2.bitwise_and(person_mask, cv2.bitwise_not(skin_mask // 255))
    img_pure_clothing = img_rgb * pure_clothing_mask[:, :, np.newaxis]

    person_area = int(np.sum(person_mask))
    if person_area == 0:
        person_area = 1

    img_blurred = cv2.GaussianBlur(img_pure_clothing, (15, 15), 0)
    img_hsv = cv2.cvtColor(img_blurred, cv2.COLOR_RGB2HSV)
    pixels = img_hsv.reshape((-1, 3)).astype(np.float32)

    K = 6
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.flatten()
    counts = np.bincount(labels)
    sorted_indices = np.argsort(counts)[::-1]

    extracted_colors = []
    for idx in sorted_indices:
        hsv_color = np.uint8([[centers[idx]]])
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
        if rgb_color[0] < 20 and rgb_color[1] < 20 and rgb_color[2] < 20:
            continue
        percentage = (counts[idx] / person_area) * 100
        extracted_colors.append({'rgb': rgb_color, 'percentage': percentage})

    total = sum(c['percentage'] for c in extracted_colors)
    if total > 0:
        for c in extracted_colors:
            c['percentage'] = (c['percentage'] / total) * 100

    while len(extracted_colors) < 3:
        extracted_colors.append({'rgb': np.array([0, 0, 0]), 'percentage': 0.0})

    # 色相・彩度(特徴量として使う)
    hues, sats = [], []
    for c in extracted_colors[:3]:
        hsv = cv2.cvtColor(np.uint8([[[int(c['rgb'][0]), int(c['rgb'][1]), int(c['rgb'][2])]]]),
                            cv2.COLOR_RGB2HSV)[0][0]
        hues.append(int(hsv[0]) * 2)
        sats.append(int(hsv[1]))

    return {
        'p1': extracted_colors[0]['percentage'],
        'p2': extracted_colors[1]['percentage'],
        'p3': extracted_colors[2]['percentage'],
        'hue1': hues[0], 'hue2': hues[1], 'hue3': hues[2],
        'sat1': sats[0], 'sat2': sats[1], 'sat3': sats[2],
        'diff_from_ideal': abs(extracted_colors[0]['percentage'] - 70)
                           + abs(extracted_colors[1]['percentage'] - 25)
                           + abs(extracted_colors[2]['percentage'] - 5),
    }


# ------------------------------------------------------------
# ③ 画像を1枚ずつ処理して特徴量テーブルを作る
# ------------------------------------------------------------
model_path = download_model()

rows = []
for filename, score in IMAGE_SCORES.items():
    path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(path):
        print(f"⚠ {filename} が見つかりません。スキップします。")
        continue
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    feat = analyze(img_rgb, model_path)
    feat['filename'] = filename
    feat['my_score'] = score
    rows.append(feat)

df = pd.DataFrame(rows)
print("=== 特徴量テーブル ===")
print(df)

df.to_csv("outfit_training_data.csv", index=False, encoding="utf-8-sig")
print("\nCSVを保存しました: outfit_training_data.csv")

# ------------------------------------------------------------
# ④ 機械学習モデルを訓練する
#    注意：4枚だけだと学習というより「動作確認」レベルです。
#    データが20〜30枚を超えてきたら、本当に意味のある予測になります。
# ------------------------------------------------------------
from sklearn.ensemble import RandomForestRegressor

feature_cols = ['p1', 'p2', 'p3', 'hue1', 'hue2', 'hue3', 'sat1', 'sat2', 'sat3', 'diff_from_ideal']
X = df[feature_cols]
y = df['my_score']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

print("\n=== 学習後、同じデータで予測させてみる（参考値） ===")
pred = model.predict(X)
for fname, actual, p in zip(df['filename'], y, pred):
    print(f"{fname}: あなたの採点={actual}, モデルの予測={p:.1f}")

print("\n=== 特徴量の重要度（モデルが何を重視しているか） ===")
importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importance)

# ------------------------------------------------------------
# ⑤ モデルを保存する（あとでStreamlitアプリから読み込むため）
# ------------------------------------------------------------
import pickle
with open("outfit_score_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nモデルを保存しました: outfit_score_model.pkl")
