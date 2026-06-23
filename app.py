import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import urllib.request
import os
import uuid
import io

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

def get_color_harmony(hues):
    if len(hues) == 0:
        return "モノトーン", 8
    if len(hues) == 1:
        return "ワントーン", 5

    def hue_diff(a, b):
        d = abs(a - b)
        return min(d, 360 - d)

    diffs = []
    for i in range(len(hues)):
        for j in range(i + 1, len(hues)):
            diffs.append(hue_diff(hues[i], hues[j]))

    max_diff = max(diffs)

    if max_diff <= 30:
        return "類似色配色（まとまり感◎）", 8
    elif max_diff <= 90:
        return "中差色配色（バランス良い）", 5
    elif 150 <= max_diff <= 210:
        return "補色配色（インパクト大・上級者向け）", 2
    elif 110 <= max_diff <= 130:
        return "トライアド配色（鮮やか・カジュアル）", 4
    else:
        return "対比色配色（メリハリあり）", 3

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

    hues = []
    for c in extracted_colors[:3]:
        if c['percentage'] > 5:
            h = int(cv2.cvtColor(np.uint8([[[int(c['rgb'][0]), int(c['rgb'][1]), int(c['rgb'][2])]]]),
                             cv2.COLOR_RGB2HSV)[0][0][0])
            s = int(cv2.cvtColor(np.uint8([[[int(c['rgb'][0]), int(c['rgb'][1]), int(c['rgb'][2])]]]),
                             cv2.COLOR_RGB2HSV)[0][0][1])
            if s >= 40:
                hues.append(h * 2)

    harmony_label, harmony_bonus = get_color_harmony(hues)
    color_count_penalty = max(0, (len(extracted_colors) - 3) * 3)
    score = max(0, min(100, int(100 - diff * 0.7) + harmony_bonus - color_count_penalty))

    return {
        'img_rgb': img_rgb,
        'img_pure_clothing': img_pure_clothing,
        'extracted_colors': extracted_colors,
        'person_area': person_area,
        'score': score,
        'harmony_label': harmony_label,
        'bbox': (xmin, ymin, xmax, ymax)
    }

def render_pie_chart(valid_colors, score):
    fig, ax = plt.subplots(figsize=(5, 5))
    sizes = [c['percentage'] for c in valid_colors]
    hex_colors = [f'#{int(c["rgb"][0]):02x}{int(c["rgb"][1]):02x}{int(c["rgb"][2]):02x}' for c in valid_colors]
    labels_pie = [f"{c['percentage']:.1f}%" for c in valid_colors]
    ax.pie(sizes, labels=labels_pie, colors=hex_colors, startangle=90, counterclock=False,
           wedgeprops={'width': 0.4, 'edgecolor': 'white'})
    ax.set_title(f"Color Balance (Score: {score})")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.read()

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
        valid_colors = [c for c in colors if c['percentage'] > 0.5]

        harmony_label = result['harmony_label']
        st.subheader(f"スコア: {score} / 100")
        st.caption(f"配色タイプ: {harmony_label}")
        if score >= 80:
            st.success("バランス抜群！理想的なコーデです。")
        elif score >= 60:
            st.info("まずまずのバランスです。")
        else:
            st.warning("カラーバランスを調整するとより良くなります。")

        st.image(result['img_rgb'], caption="アップロード画像", use_container_width=True)

        st.subheader("カラー分析結果")
        for i, c in enumerate(valid_colors[:5]):
            rgb = c['rgb']
            hex_color = f'#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}'
            color_name = rgb_to_color_name(rgb)
            role = ["ベース (目標70%)", "アソート (目標25%)", "アクセント (目標5%)"][i] if i < 3 else f"その他{i-2}"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0">'
                f'<div style="width:40px;height:40px;background:{hex_color};border-radius:6px;border:1px solid #ccc"></div>'
                f'<span><b>{color_name}</b> ({hex_color}) — {c["percentage"]:.1f}% — {role}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        graph_bytes = render_pie_chart(valid_colors, score)
        st.image(graph_bytes)

        st.divider()
        st.subheader("💡 アドバイス")

        p1 = valid_colors[0]['percentage'] if len(valid_colors) > 0 else 0
        p2 = valid_colors[1]['percentage'] if len(valid_colors) > 1 else 0
        p3 = valid_colors[2]['percentage'] if len(valid_colors) > 2 else 0

        base_color_name = valid_colors[0]['rgb'] if len(valid_colors) > 0 else None
        base_name = rgb_to_color_name(base_color_name) if base_color_name is not None else "メインカラー"

        def is_neutral_rgb(rgb):
            hsv = cv2.cvtColor(np.uint8([[[int(rgb[0]), int(rgb[1]), int(rgb[2])]]]), cv2.COLOR_RGB2HSV)[0][0]
            return int(hsv[1]) < 60

        p1_is_neutral = is_neutral_rgb(valid_colors[0]['rgb']) if len(valid_colors) > 0 else True
        p2_is_neutral = is_neutral_rgb(valid_colors[1]['rgb']) if len(valid_colors) > 1 else True
        both_neutral = p1_is_neutral and p2_is_neutral

        if score >= 80:
            st.success("カラーバランスは理想的です。このコーデのまま着ていけばOKです！")
        else:
            if both_neutral and p1 < 60:
                st.warning(f"グレー・ベージュ系でまとまったコーデですが、色が分散しています（最大色{p1:.0f}%、目標70%）。同じ色のアイテムをもう1枚重ねるか、上下をより近い色で統一するとベース割合が上がります。")
            elif not p1_is_neutral and p1 < 60:
                st.warning(f"ベースカラー（{base_name}）が{p1:.0f}%と少なめです（目標70%）。トップスかボトムスをホワイト・グレー・ベージュ系の無彩色に変えると全体がまとまります。")
            elif p1 > 85:
                st.warning(f"ベースカラー（{base_name}）が{p1:.0f}%と多すぎます（目標70%）。ボトムスやアウターを別の色に変えるか、カラーの小物を足してメリハリをつけましょう。")

            if both_neutral and p2 > 35:
                st.warning(f"無彩色が2色で拮抗しています（{p1:.0f}%と{p2:.0f}%）。上下を同系色でまとめるか、どちらかにカラーアイテムを混ぜてメリハリをつけましょう。")
            elif not both_neutral:
                if p2 < 15:
                    st.warning(f"2色目が{p2:.0f}%と少なめです（目標25%）。ボトムスかアウターの色をもう少し主張させると、メリハリが出ます。")
                elif p2 > 35:
                    st.warning(f"2色目が{p2:.0f}%と多すぎます（目標25%）。どちらかをベース色寄りに抑えると落ち着きます。")

            if p3 < 2:
                if both_neutral:
                    st.warning("無彩色でまとまったコーデです。バッグ・シューズなど小物1点にカラーを入れると印象がぐっと締まります。")
                else:
                    st.warning("差し色がほぼゼロです（目標5%）。バッグ・シューズ・マフラーなど小物1点だけカラーを入れると印象が締まります。")
            elif p3 > 15:
                st.warning(f"差し色が{p3:.0f}%と多すぎます（目標5%）。アクセントは小物サイズに抑えるのがポイントです。")

        # 手持ちの服からの提案
        supabase = get_supabase()
        clothes_list = []
        if supabase:
            try:
                clothes_list = supabase.table("clothes").select("*").execute().data or []
            except Exception:
                pass

        if clothes_list:
            def item_role_hint(item_type):
                if item_type == "トップス": return "トップスとして着ると"
                if item_type == "ボトムス": return "ボトムスに合わせると"
                if item_type == "アウター": return "アウターに羽織ると"
                if item_type == "シューズ": return "足元に取り入れると"
                if item_type == "バッグ": return "バッグとして持つと"
                return "合わせると"

            base_candidate = None
            assort_candidate = None
            accent_candidate = None

            for item in clothes_list:
                if not item.get('color_hex') or not item.get('type'):
                    continue
                hex_val = item['color_hex'].lstrip('#')
                try:
                    ir, ig, ib = int(hex_val[0:2], 16), int(hex_val[2:4], 16), int(hex_val[4:6], 16)
                except Exception:
                    continue
                hsv = cv2.cvtColor(np.uint8([[[ir, ig, ib]]]), cv2.COLOR_RGB2HSV)[0][0]
                is_neutral = int(hsv[1]) < 60

                if p1 < 65 and is_neutral and base_candidate is None:
                    base_candidate = (item, f"**{item.get('color_name','')}の{item['type']}**を{item_role_hint(item['type'])}ベースカラーが増えてバランスが良くなります（目標70%、現在{p1:.0f}%）。")
                elif p3 < 3 and not is_neutral and accent_candidate is None:
                    accent_candidate = (item, f"**{item.get('color_name','')}の{item['type']}**を{item_role_hint(item['type'])}差し色になってコーデが引き締まります（目標5%、現在{p3:.0f}%）。")
                elif 60 <= p1 <= 80 and p2 < 20 and assort_candidate is None:
                    assort_candidate = (item, f"**{item.get('color_name','')}の{item['type']}**を{item_role_hint(item['type'])}アソートカラーとしてバランスが上がります（目標25%、現在{p2:.0f}%）。")

            suggestions = [c for c in [base_candidate, assort_candidate, accent_candidate] if c is not None]

            if suggestions:
                st.markdown("**👕 手持ちの服からの提案**")
                for item, msg in suggestions:
                    col_img, col_msg = st.columns([1, 3])
                    with col_img:
                        if item.get('image_url'):
                            st.image(item['image_url'], use_container_width=True)
                    with col_msg:
                        st.markdown(msg)
                        if item.get('brand'):
                            st.caption(item['brand'])
        else:
            if p1 < 60:
                st.info(f"ベースカラー（{base_name}）が{p1:.0f}%と少なめです。トップスかボトムスをホワイト・グレー・ベージュ系に変えると、ベース割合が70%に近づきます。服を登録すると具体的な提案ができます。")
            elif p3 < 3:
                st.info("差し色（3色目）がほぼゼロです。バッグ・スニーカー・マフラーなど小物1点にカラーを入れるだけでスコアが上がります。服を登録すると手持ちから具体的に提案します。")
            elif p2 < 15:
                st.info(f"2色目が{p2:.0f}%と少なめです。ボトムスやアウターを{base_name}以外の色にして、25%前後になるとバランスが良くなります。服を登録すると手持ちから提案します。")

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
                    img_name = f"{uuid.uuid4()}.jpg"
                    supabase.storage.from_("wardrobe-images").upload(img_name, img_bytes, {"content-type": "image/jpeg"})
                    image_url = supabase.storage.from_("wardrobe-images").get_public_url(img_name)

                    graph_name = f"graph_{uuid.uuid4()}.png"
                    supabase.storage.from_("wardrobe-images").upload(graph_name, graph_bytes, {"content-type": "image/png"})
                    graph_url = supabase.storage.from_("wardrobe-images").get_public_url(graph_name)

                    color_data = [
                        {"rgb": [int(c['rgb'][0]), int(c['rgb'][1]), int(c['rgb'][2])],
                         "name": rgb_to_color_name(c['rgb']),
                         "percentage": round(c['percentage'], 1)}
                        for c in valid_colors[:3]
                    ]

                    supabase.table("outfit_log").insert({
                        "category": category,
                        "season": season,
                        "score": score,
                        "colors": color_data,
                        "image_url": image_url,
                        "graph_url": graph_url,
                        "memo": memo
                    }).execute()

                    st.success("保存しました！")
                except Exception as e:
                    st.error(f"保存エラー: {e}")

def page_clothes_register():
    st.title("👕 服を登録")
    st.markdown("服単体の写真をアップロードして登録します。")

    model_path = download_model()
    uploaded = st.file_uploader("服の写真をアップロード", type=["jpg", "jpeg", "png"])

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with st.spinner("色を解析中..."):
            result = analyze(img_rgb, model_path)

        valid_colors = [c for c in result['extracted_colors'] if c['percentage'] > 0.5]
        main_color = valid_colors[0] if valid_colors else None

        st.image(result['img_rgb'], caption="アップロード画像", use_container_width=True)

        if main_color:
            hex_color = f'#{int(main_color["rgb"][0]):02x}{int(main_color["rgb"][1]):02x}{int(main_color["rgb"][2]):02x}'
            color_name = rgb_to_color_name(main_color['rgb'])
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;margin:8px 0">'
                f'<div style="width:40px;height:40px;background:{hex_color};border-radius:6px;border:1px solid #ccc"></div>'
                f'<span>メインカラー: <b>{color_name}</b> ({hex_color})</span></div>',
                unsafe_allow_html=True
            )

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            clothes_type = st.selectbox("種類", ["トップス", "ボトムス", "アウター", "シューズ", "バッグ", "その他"])
        with col_b:
            brand = st.text_input("ブランド（任意）", placeholder="例: UNIQLO")

        season = st.multiselect("季節", ["春", "夏", "秋", "冬"], default=["春", "秋"])
        memo = st.text_input("メモ（任意）", placeholder="お気に入りの一枚など...")

        if st.button("登録する", type="primary"):
            supabase = get_supabase()
            if supabase is None:
                st.error("Supabase未設定です。")
            else:
                try:
                    img_bytes = cv2.imencode('.jpg', cv2.cvtColor(result['img_rgb'], cv2.COLOR_RGB2BGR))[1].tobytes()
                    file_name = f"{uuid.uuid4()}.jpg"
                    supabase.storage.from_("wardrobe-images").upload(file_name, img_bytes, {"content-type": "image/jpeg"})
                    image_url = supabase.storage.from_("wardrobe-images").get_public_url(file_name)

                    color_data = [
                        {"rgb": [int(c['rgb'][0]), int(c['rgb'][1]), int(c['rgb'][2])],
                         "name": rgb_to_color_name(c['rgb']),
                         "percentage": round(c['percentage'], 1)}
                        for c in valid_colors[:3]
                    ]

                    supabase.table("clothes").insert({
                        "type": clothes_type,
                        "brand": brand or None,
                        "color_name": rgb_to_color_name(main_color['rgb']) if main_color else None,
                        "color_hex": hex_color if main_color else None,
                        "colors": color_data,
                        "season": season,
                        "image_url": image_url,
                        "memo": memo or None
                    }).execute()

                    st.success("登録しました！")
                except Exception as e:
                    st.error(f"登録エラー: {e}")

def page_clothes_list():
    st.title("🗂️ 服一覧")
    supabase = get_supabase()
    if supabase is None:
        st.error("Supabase未設定です。")
        return

    try:
        all_res = supabase.table("clothes").select("type,color_name,season").execute()
        all_records = all_res.data
        if all_records:
            st.subheader("💡 ワードローブアドバイス")
            types_owned = [r['type'] for r in all_records]
            type_counts = {t: types_owned.count(t) for t in set(types_owned)}
            all_types = ["トップス", "ボトムス", "アウター", "シューズ", "バッグ"]
            missing = [t for t in all_types if t not in type_counts]
            if missing:
                st.warning(f"まだ登録されていない種類: **{'、'.join(missing)}**")

            colors_owned = [r['color_name'] for r in all_records if r.get('color_name')]
            achromatic = ["ブラック", "ホワイト", "グレー", "ベージュ"]
            chromatic = [c for c in colors_owned if c not in achromatic]
            achromatic_owned = [c for c in colors_owned if c in achromatic]

            if not achromatic_owned:
                st.info("ベースカラー（ブラック・ホワイト・グレー・ベージュ）が少ないです。コーデの軸になる無彩色を追加するとバランスが上がります。")
            if len(set(chromatic)) >= 4:
                st.warning(f"カラフルな服が多め（{len(set(chromatic))}色）。アクセントカラーは1〜2色に絞るとまとまりが出ます。")
            elif len(chromatic) == 0:
                st.info("カラーアイテムがありません。差し色を1〜2点加えるとコーデに表情が出ます。")
            else:
                st.success(f"カラーバランスは良好です。（無彩色 {len(achromatic_owned)}点 / 有彩色 {len(chromatic)}点）")
            st.divider()
    except Exception:
        pass

    type_filter = st.selectbox("種類で絞り込み", ["すべて", "トップス", "ボトムス", "アウター", "シューズ", "バッグ", "その他"])

    try:
        query = supabase.table("clothes").select("*").order("registered_at", desc=True)
        if type_filter != "すべて":
            query = query.eq("type", type_filter)
        res = query.execute()
        records = res.data
    except Exception as e:
        st.error(f"取得エラー: {e}")
        return

    if not records:
        st.info("まだ服が登録されていません。")
        return

    cols = st.columns(3)
    for i, rec in enumerate(records):
        with cols[i % 3]:
            if rec.get("image_url"):
                st.image(rec["image_url"], use_container_width=True)
            hex_color = rec.get("color_hex", "#cccccc")
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:6px;margin:4px 0">'
                f'<div style="width:16px;height:16px;background:{hex_color};border-radius:3px;border:1px solid #ccc"></div>'
                f'<span><b>{rec.get("color_name","")}</b></span></div>',
                unsafe_allow_html=True
            )
            st.markdown(f"**{rec.get('type','')}**" + (f" / {rec['brand']}" if rec.get('brand') else ""))
            seasons = rec.get('season') or []
            if seasons:
                st.caption(" ".join(seasons))
            if rec.get('memo'):
                st.caption(rec['memo'])

            with st.expander("編集"):
                new_type = st.selectbox("種類", ["トップス", "ボトムス", "アウター", "シューズ", "バッグ", "その他"],
                                        index=["トップス", "ボトムス", "アウター", "シューズ", "バッグ", "その他"].index(rec.get('type', 'トップス')) if rec.get('type') in ["トップス", "ボトムス", "アウター", "シューズ", "バッグ", "その他"] else 0,
                                        key=f"type_{rec['id']}")
                new_brand = st.text_input("ブランド", value=rec.get('brand') or "", key=f"brand_{rec['id']}")
                season_opts = ["春", "夏", "秋", "冬"]
                new_season = st.multiselect("季節", season_opts, default=[s for s in (rec.get('season') or []) if s in season_opts], key=f"season_{rec['id']}")
                new_memo = st.text_input("メモ", value=rec.get('memo') or "", key=f"memo_{rec['id']}")

                col_save, col_del = st.columns(2)
                with col_save:
                    if st.button("保存", key=f"save_{rec['id']}"):
                        try:
                            supabase.table("clothes").update({
                                "type": new_type,
                                "brand": new_brand or None,
                                "season": new_season,
                                "memo": new_memo or None
                            }).eq("id", rec['id']).execute()
                            st.success("更新しました")
                            st.rerun()
                        except Exception as e:
                            st.error(f"更新エラー: {e}")
                with col_del:
                    if st.button("削除", key=f"del_{rec['id']}", type="secondary"):
                        try:
                            supabase.table("clothes").delete().eq("id", rec['id']).execute()
                            st.success("削除しました")
                            st.rerun()
                        except Exception as e:
                            st.error(f"削除エラー: {e}")

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
                if rec.get("graph_url"):
                    st.image(rec["graph_url"], use_container_width=True)
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
                from datetime import datetime
                dt_str = rec.get("registered_at", "")
                if dt_str:
                    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00")).astimezone()
                    st.caption(dt.strftime("%Y/%m/%d %H:%M"))

                with st.expander("編集・削除"):
                    cat_opts = ["コーデ全体", "トップス", "ボトムス", "アウター", "ワンピース"]
                    new_cat = st.selectbox("カテゴリ", cat_opts,
                                           index=cat_opts.index(rec.get('category', 'コーデ全体')) if rec.get('category') in cat_opts else 0,
                                           key=f"cat_{rec['id']}")
                    season_opts = ["春", "夏", "秋", "冬"]
                    new_season = st.multiselect("季節", season_opts,
                                                default=[s for s in (rec.get('season') or []) if s in season_opts],
                                                key=f"season_{rec['id']}")
                    new_memo = st.text_input("メモ", value=rec.get('memo') or "", key=f"memo_{rec['id']}")

                    col_save, col_del = st.columns(2)
                    with col_save:
                        if st.button("保存", key=f"save_{rec['id']}"):
                            try:
                                supabase.table("outfit_log").update({
                                    "category": new_cat,
                                    "season": new_season,
                                    "memo": new_memo or None
                                }).eq("id", rec['id']).execute()
                                st.success("更新しました")
                                st.rerun()
                            except Exception as e:
                                st.error(f"更新エラー: {e}")
                    with col_del:
                        if st.button("削除", key=f"del_{rec['id']}", type="secondary"):
                            try:
                                supabase.table("outfit_log").delete().eq("id", rec['id']).execute()
                                st.success("削除しました")
                                st.rerun()
                            except Exception as e:
                                st.error(f"削除エラー: {e}")
            st.divider()

def main():
    st.set_page_config(page_title="コーデ色バランス診断", page_icon="👗")
    page = st.sidebar.radio("メニュー", ["診断する", "コーデ履歴", "服を登録", "服一覧"])
    if page == "診断する":
        page_diagnosis()
    elif page == "コーデ履歴":
        page_history()
    elif page == "服を登録":
        page_clothes_register()
    else:
        page_clothes_list()

if __name__ == "__main__":
    main()
