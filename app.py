import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import urllib.request
import os
import uuid

def download_model():
    model_path = "selfie_segmenter.tflite"
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
        urllib.request.urlretrieve(url, model_path)
    return model_path

def rgb_to_color_name(rgb):
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if s < 15:
        if v < 80: return "ブラック"
        elif v < 200: return "グレー"
        else: return "ホワイト"
    if s < 80:
        if h < 30 or h >= 160: return "ベージュ"
        elif h < 85: return "カーキ"
        else: return "グレー"
    if h < 15 or h >= 165: return "レッド"
    elif h < 25: return "オレンジ"
    elif h < 35: return "イエロー"
    elif h < 85: return "グリーン"
    elif h < 130: return "ブルー"
    elif h < 160: return "パープル"
    else: return "レッド"

def get_supabase():
    try:
        from supabase import create_client
        return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except Exception:
        return None

def analyze(img_rgb, model_path):
    height, width = img_rgb.shape[:2]
    max_size = 600
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        img_rgb = cv2.resize(img_rgb, (int(width * scale), int(height * scale)))
        height, width = img_rgb.shape[:2]

    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

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

    y_indices, x_indices = np.where(person_mask > 0)
    if len(y_indices) > 0:
        ymin, ymax = int(np.min(y_indices)), int(np.max(y_indices))
        xmin, xmax = int(np.min(x_indices)), int(np.max(x_indices))
        person_area = int(np.sum(person_mask))
    else:
        ymin, ymax, xmin, xmax = 0, height, 0, width
        person_area = width * height

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
    for c in extracted_colors:
        c['percentage'] = (c['percentage'] / total) * 100

    while len(extracted_colors) < 3:
        extracted_colors.append({'rgb': np.array([0, 0, 0]), 'percentage': 0.0})

    p1 = extracted_colors[0]['percentage']
    p2 = extracted_colors[1]['percentage']
    p3 = extracted_colors[2]['percentage']
    diff = abs(p1 - 70) + abs(p2 - 25) + abs(p3 - 5)

    unique_hues = set()
    for c in extracted_colors[:3]:
        if c['percentage'] > 5:
            h = cv2.cvtColor(np.uint8([[[int(c['rgb'][0]), int(c['rgb'][1]), int(c['rgb'][2])]]]),
                             cv2.COLOR_RGB2HSV)[0][0][0]
            unique_hues.add(h // 30)
    harmony_bonus = 5 if len(unique_hues) <= 2 else 0
    color_count_penalty = max(0, (len(extracted_colors) - 3) * 3)
    score = max(0, min(100, int(100 - diff * 0.7) + harmony_bonus - color_count_penalty))

    return {
        'img_rgb': img_rgb,
        'img_pure_clothing': img_pure_clothing,
        'extracted_colors': extracted_colors,
        'person_area': person_area,
        'score': score,
        'bbox': (xmin, ymin, xmax, ymax)
    }

def page_diagnosis():
    st.title("👗 コーデ色バランス診断")
    st.markdown("服装の写真をアップロードして、70:25:5の黄金比カラーバランスを診断します。")

    model_path = download_model()

    uploaded = st.file_uploader("全身コーデの写真をアップロード", type=["jpg", "jpeg", "png"])

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with st.spinner("AI解析中..."):
            result = analyze(img_rgb, model_path)

        score = result['score']
        colors = result['extracted_colors']

        st.subheader(f"スコア: {score} / 100")
        if score >= 80:
            st.success("バランス抜群！理想的なコーデです。")
        elif score >= 60:
            st.info("まずまずのバランスです。")
        else:
            st.warning("カラーバランスを調整するとより良くなります。")

        st.image(result['img_rgb'], caption="アップロード画像", use_container_width=True)

        st.subheader("カラー分析結果")
        valid_colors = [c for c in colors if c['percentage'] > 0.5]
        for i, c in enumerate(valid_colors[:5]):
            rgb = c['rgb']
            hex_color = f'#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}'
            color_name = rgb_to_color_name(rgb)
            role = ["ベース (目標70%)","アソート (目標25%)","アクセント (目標5%)"][i] if i < 3 else f"その他{i-2}"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0">'
                f'<div style="width:40px;height:40px;background:{hex_color};border-radius:6px;border:1px solid #ccc"></div>'
                f'<span><b>{color_name}</b> ({hex_color}) — {c["percentage"]:.1f}% — {role}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        fig, ax = plt.subplots(figsize=(5, 5))
        sizes = [c['percentage'] for c in valid_colors]
        hex_colors = [f'#{int(c["rgb"][0]):02x}{int(c["rgb"][1]):02x}{int(c["rgb"][2]):02x}' for c in valid_colors]
        labels_pie = [f"{c['percentage']:.1f}%" for c in valid_colors]
        ax.pie(sizes, labels=labels_pie, colors=hex_colors, startangle=90, counterclock=False,
               wedgeprops={'width': 0.4, 'edgecolor': 'white'})
        ax.set_title(f"Color Balance (Score: {score})")
        st.pyplot(fig)
        plt.close()

        st.divider()
        st.subheader("今日のコーデを記録する")

        col_a, col_b = st.columns(2)
        with col_a:
            category = st.selectbox("カテゴリ", ["コーデ全体", "トップス", "ボトムス", "アウター", "ワンピース"])
        with col_b:
            season = st.multiselect("季節", ["春", "夏", "秋", "冬"], default=["春", "秋"])

        memo = st.text_input("メモ（任意）", placeholder="今日のコーデメモ...")

        if st.button("保存する", type="primary"):
            supabase = get_supabase()
            if supabase is None:
                st.error("Supabase未設定です。")
            else:
                try:
                    img_bytes = cv2.imencode('.jpg', cv2.cvtColor(result['img_rgb'], cv2.COLOR_RGB2BGR))[1].tobytes()
                    file_name = f"{uuid.uuid4()}.jpg"
                    supabase.storage.from_("wardrobe-images").upload(
                        file_name, img_bytes, {"content-type": "image/jpeg"}
                    )
                    image_url = supabase.storage.from_("wardrobe-images").get_public_url(file_name)

                    top3 = valid_colors[:3]
                    color_data = [
                        {"rgb": [int(c['rgb'][0]), int(c['rgb'][1]), int(c['rgb'][2])],
                         "name": rgb_to_color_name(c['rgb']),
                         "percentage": round(c['percentage'], 1)}
                        for c in top3
                    ]

                    supabase.table("outfit_log").insert({
                        "category": category,
                        "season": season,
                        "score": score,
                        "colors": color_data,
                        "image_url": image_url,
                        "memo": memo
                    }).execute()

                    st.success("保存しました！")
                except Exception as e:
                    st.error(f"保存エラー: {e}")

def page_history():
    st.title("📋 コーデ履歴")
    supabase = get_supabase()
    if supabase is None:
        st.error("Supabase未設定です。")
        return
    try:
        res = supabase.table("outfit_log").select("*").order("registered_at", desc=True).execute()
        records = res.data
    except Exception as e:
        st.error(f"取得エラー: {e}")
        return

    if not records:
        st.info("まだ記録がありません。")
        return

    for rec in records:
        with st.container():
            col1, col2 = st.columns([1, 2])
            with col1:
                if rec.get("image_url"):
                    st.image(rec["image_url"], use_container_width=True)
            with col2:
                st.markdown(f"**スコア: {rec.get('score', '-')} / 100**")
                st.markdown(f"カテゴリ: {rec.get('category', '-')}")
                seasons = rec.get('season') or []
                st.markdown(f"季節: {' / '.join(seasons) if seasons else '-'}")
                if rec.get('memo'):
                    st.markdown(f"メモ: {rec['memo']}")
                colors = rec.get('colors') or []
                for c in colors:
                    hex_color = f'#{c["rgb"][0]:02x}{c["rgb"][1]:02x}{c["rgb"][2]:02x}'
                    st.markdown(
                        f'<div style="display:inline-flex;align-items:center;gap:6px;margin-right:8px">'
                        f'<div style="width:20px;height:20px;background:{hex_color};border-radius:4px;border:1px solid #ccc"></div>'
                        f'<span>{c["name"]} {c["percentage"]}%</span></div>',
                        unsafe_allow_html=True
                    )
                from datetime import datetime, timezone
                dt_str = rec.get("registered_at", "")
                if dt_str:
                    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00")).astimezone()
                    st.caption(dt.strftime("%Y/%m/%d %H:%M"))
            st.divider()

def main():
    st.set_page_config(page_title="コーデ色バランス診断", page_icon="👗")
    page = st.sidebar.radio("メニュー", ["診断する", "コーデ履歴"])
    if page == "診断する":
        page_diagnosis()
    else:
        page_history()

if __name__ == "__main__":
    main()
