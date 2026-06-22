import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import urllib.request
import os

def download_model():
    model_path = "selfie_segmenter.tflite"
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
            model_path
        )
    return model_path

def rgb_to_color_name(rgb):
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if s < 15:
        if v < 80:    return "ブラック"
        elif v < 200: return "グレー"
        else:         return "ホワイト"
    if s < 80:
        if h < 30 or h >= 160: return "ベージュ"
        elif h < 85:           return "カーキ"
        else:                  return "グレー"
    if h < 10 or h >= 170: return "レッド"
    elif h < 20:  return "オレンジ"
    elif h < 40:  return "イエロー"
    elif h < 85:  return "グリーン"
    elif h < 130: return "ブルー"
    elif h < 150: return "パープル"
    else:         return "ブラウン" if v < 140 else "ピンク"

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
        ymin, ymax = np.min(y_indices), np.max(y_indices)
        xmin, xmax = np.min(x_indices), np.max(x_indices)
        person_area = int(np.sum(person_mask))
    else:
        person_area = width * height
        ymin, ymax, xmin, xmax = 0, height, 0, width

    img_blurred = cv2.GaussianBlur(img_pure_clothing, (15, 15), 0)
    img_hsv = cv2.cvtColor(img_blurred, cv2.COLOR_RGB2HSV)
    pixels = np.float32(img_hsv.reshape((-1, 3)))
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
        extracted_colors.append({'rgb': rgb_color, 'percentage': percentage, 'name': rgb_to_color_name(rgb_color)})

    total = sum([c['percentage'] for c in extracted_colors])
    for c in extracted_colors:
        c['percentage'] = (c['percentage'] / total) * 100

    while len(extracted_colors) < 3:
        extracted_colors.append({'rgb': np.array([0,0,0]), 'percentage': 0.0, 'name': '―'})

    p1, p2, p3 = extracted_colors[0]['percentage'], extracted_colors[1]['percentage'], extracted_colors[2]['percentage']
    diff = abs(p1 - 70) + abs(p2 - 25) + abs(p3 - 5)
    score = max(0, int(100 - (diff * 0.7)))

    return img_rgb, img_pure_clothing, extracted_colors, score, (xmin, ymin, xmax, ymax), person_area

def main():
    st.set_page_config(page_title="コーデ診断", page_icon="👗")
    st.title("👗 コーデカラー診断")
    st.write("服装の写真をアップロードすると、色バランスをスコアリングします。")

    uploaded_file = st.file_uploader("写真をアップロード", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with st.spinner("解析中..."):
            model_path = download_model()
            img_rgb, img_pure_clothing, extracted_colors, score, bbox, person_area = analyze(img_rgb, model_path)

        if score >= 80:
            score_comment = "優秀！バランスの取れたコーデです"
        elif score >= 60:
            score_comment = "良好！もう少し調整するとさらに良くなります"
        elif score >= 40:
            score_comment = "もう一工夫でバランスが上がります"
        else:
            score_comment = "色数を絞るとまとまりが出ます"

        st.subheader(f"スコア: {score}点 / 100点")
        st.write(score_comment)

        roles = ["メインカラー", "サブカラー", "アクセント"]
        for i, c in enumerate(extracted_colors[:3]):
            hex_color = f'#{c["rgb"][0]:02x}{c["rgb"][1]:02x}{c["rgb"][2]:02x}'
            st.markdown(f"**【{roles[i]}】** {c['name']}系 ({c['percentage']:.1f}%) "
                       f"<span style='background:{hex_color};padding:0 20px;border-radius:3px;'>&nbsp;</span>",
                       unsafe_allow_html=True)

        p1, p2, p3 = extracted_colors[0]['percentage'], extracted_colors[1]['percentage'], extracted_colors[2]['percentage']
        st.subheader("💡 アドバイス")
        if p1 < 60:
            st.write(f"・{extracted_colors[0]['name']}をもっと増やしてメインカラーを強調しましょう（目安70%）")
        elif p1 > 80:
            st.write(f"・{extracted_colors[0]['name']}が多すぎます。サブカラーを足してみましょう")
        if p2 < 15:
            st.write(f"・{extracted_colors[1]['name']}などのサブカラーをもう少し取り入れると奥行きが出ます（目安25%）")
        if extracted_colors[2]['percentage'] < 3:
            st.write("・アクセントカラーが少なすぎます。小物や差し色で引き締めましょう（目安5%）")
        if p1 >= 60 and p1 <= 80 and p2 >= 15 and extracted_colors[2]['percentage'] >= 3:
            st.write("・このままのバランスを維持しましょう！")

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_rgb, caption="元画像", use_container_width=True)
        with col2:
            fig, ax = plt.subplots(figsize=(5, 5))
            colors_hex = [f'#{c["rgb"][0]:02x}{c["rgb"][1]:02x}{c["rgb"][2]:02x}' for c in extracted_colors if c['percentage'] > 0]
            sizes = [c['percentage'] for c in extracted_colors if c['percentage'] > 0]
            labels_pie = [f"{c['percentage']:.1f}%" for c in extracted_colors if c['percentage'] > 0]
            ax.pie(sizes, labels=labels_pie, colors=colors_hex, startangle=90, counterclock=False,
                   wedgeprops={'width': 0.4, 'edgecolor': 'white'})
            ax.set_title(f"Color Balance (Score: {score})")
            st.pyplot(fig)

        colors_hex = [f'#{c["rgb"][0]:02x}{c["rgb"][1]:02x}{c["rgb"][2]:02x}' for c in extracted_colors if c['percentage'] > 0]
        sizes = [c['percentage'] for c in extracted_colors if c['percentage'] > 0]
        labels_pie = [f"{c['percentage']:.1f}%" for c in extracted_colors if c['percentage'] > 0]
        ax.pie(sizes, labels=labels_pie, colors=colors_hex, startangle=90, counterclock=False,
               wedgeprops={'width': 0.4, 'edgecolor': 'white'})
        ax.set_title(f"Color Balance (Score: {score})")

if __name__ == "__main__":
    main()
