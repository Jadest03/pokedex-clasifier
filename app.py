import os
import io
import sys
import glob
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image
from torchvision import transforms, models

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from korean_names import get_korean_name

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────────────────────────
# 커스텀 CSS
# ──────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Nunito:wght@400;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        min-height: 100vh;
    }
    .poke-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .poke-title {
        font-family: 'Press Start 2P', monospace;
        font-size: 1.8rem;
        color: #FFD700;
        text-shadow: 3px 3px 0px #CC0000, 6px 6px 0px rgba(0,0,0,0.3);
        letter-spacing: 2px;
        line-height: 1.6;
    }
    .poke-subtitle {
        font-family: 'Nunito', sans-serif;
        color: #a8c8f8;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 600;
    }
    .spin {
        display: inline-block;
        animation: spin 3s linear infinite;
    }
    @keyframes spin {
        from { transform: rotate(0deg); }
        to   { transform: rotate(360deg); }
    }
    .section-label {
        font-family: 'Press Start 2P', monospace;
        font-size: 0.55rem;
        color: #FFD700;
        letter-spacing: 2px;
        margin-bottom: 1rem;
        text-transform: uppercase;
    }
    .poke-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,215,0,0.2);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    .result-first {
        background: linear-gradient(135deg, #CC0000 0%, #FF6B6B 100%);
        border-radius: 20px;
        padding: 1.8rem 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(204,0,0,0.5);
        position: relative;
        overflow: hidden;
        margin-bottom: 1.2rem;
    }
    .result-first::after {
        content: '';
        position: absolute;
        top: -30px; right: -30px;
        width: 120px; height: 120px;
        background: rgba(255,255,255,0.06);
        border-radius: 50%;
    }
    .result-emoji  { font-size: 2.5rem; margin-bottom: 0.4rem; }
    .result-kor    {
        font-family: 'Nunito', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        color: #FFD700;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.4);
    }
    .result-eng    {
        font-family: 'Nunito', sans-serif;
        font-size: 0.9rem;
        color: rgba(255,255,255,0.75);
        margin-top: 0.1rem;
    }
    .result-pct    {
        font-family: 'Press Start 2P', monospace;
        font-size: 1.3rem;
        color: white;
        margin-top: 0.8rem;
    }
    .rank-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 0.75rem;
    }
    .rank-no {
        font-family: 'Press Start 2P', monospace;
        font-size: 0.45rem;
        color: #a8c8f8;
        width: 22px;
        flex-shrink: 0;
    }
    .rank-name {
        font-family: 'Nunito', sans-serif;
        font-weight: 700;
        color: white;
        font-size: 0.9rem;
        min-width: 90px;
        flex-shrink: 0;
    }
    .rank-bar-bg {
        flex: 1;
        background: rgba(255,255,255,0.08);
        border-radius: 99px;
        height: 10px;
        overflow: hidden;
    }
    .rank-bar-fill {
        height: 100%;
        border-radius: 99px;
        background: linear-gradient(90deg, #3a7bd5, #6dd5fa);
    }
    .rank-pct {
        font-family: 'Nunito', sans-serif;
        font-size: 0.82rem;
        color: #a8c8f8;
        font-weight: 700;
        min-width: 48px;
        text-align: right;
        flex-shrink: 0;
    }
    .exp-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Nunito', sans-serif;
    }
    .exp-table th {
        font-family: 'Press Start 2P', monospace;
        font-size: 0.42rem;
        color: #FFD700;
        padding: 0.6rem 0.5rem;
        border-bottom: 1px solid rgba(255,215,0,0.25);
        letter-spacing: 1px;
        text-align: center;
    }
    .exp-table td {
        text-align: center;
        padding: 0.55rem 0.5rem;
        color: rgba(255,255,255,0.85);
        font-size: 0.88rem;
        font-weight: 600;
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .exp-table tr:hover td { background: rgba(255,215,0,0.04); }
    .best-row td { color: #FFD700 !important; }
    .badge {
        display: inline-block;
        background: linear-gradient(135deg, #CC0000, #FF4444);
        border-radius: 99px;
        padding: 2px 10px;
        font-weight: 800;
    }
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        border: 2px dashed rgba(255,215,0,0.15);
        border-radius: 16px;
        color: rgba(255,255,255,0.25);
        font-family: 'Nunito', sans-serif;
        font-size: 0.95rem;
        line-height: 2;
    }
    [data-testid="stSidebar"] {
        background: rgba(8,16,36,0.95) !important;
        border-right: 1px solid rgba(255,215,0,0.15) !important;
    }
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(255,215,0,0.25) !important;
        border-radius: 12px !important;
        background: rgba(255,255,255,0.02) !important;
    }
    .stTextInput input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,215,0,0.25) !important;
        color: white !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #CC0000, #FF4444) !important;
        color: #FFD700 !important;
        font-family: 'Press Start 2P', monospace !important;
        font-size: 0.55rem !important;
        border: none !important;
        border-radius: 8px !important;
        letter-spacing: 1px !important;
        box-shadow: 0 4px 15px rgba(204,0,0,0.35) !important;
    }
    [data-testid="stMetricValue"] {
        color: #FFD700 !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] { color: #a8c8f8 !important; }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border-radius: 10px;
        padding: 0.4rem;
        border: 1px solid rgba(255,215,0,0.1);
    }
    [data-testid="stImage"] img {
        border-radius: 12px;
        border: 2px solid rgba(255,215,0,0.25);
    }
    p, li, label { color: rgba(255,255,255,0.82) !important; }
    hr { border-color: rgba(255,215,0,0.15) !important; }
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #1a1a2e; }
    ::-webkit-scrollbar-thumb { background: #CC0000; border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# 모델
# ──────────────────────────────────────────────────────────────────
def rebuild_model(model_name: str, num_classes: int):
    if model_name in ('resnet50_lp', 'resnet50_ft'):
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif model_name == 'vit':
        m = models.vit_b_16(weights=None)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    elif model_name == 'convnext':
        m = models.convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif model_name == 'vae':
        from results.vae.vae import VAEClassifier
        m = VAEClassifier(num_classes)
    else:
        raise ValueError(f"알 수 없는 모델: {model_name}")
    return m.to(DEVICE)


@st.cache_resource
def load_model(pth_path: str):
    ckpt = torch.load(pth_path, map_location=DEVICE, weights_only=False)
    model_name = ckpt['model_name']
    class_names = ckpt['class_names']
    metrics = ckpt.get('metrics', {})
    model = rebuild_model(model_name, len(class_names))
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, class_names, model_name, metrics


def predict(model, class_names, image: Image.Image, top_k=5):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
    top_probs, top_idx = torch.topk(probs, k=min(top_k, len(class_names)))
    return [(get_korean_name(class_names[i]), class_names[i], float(p)*100)
            for p, i in zip(top_probs.cpu().numpy(), top_idx.cpu().numpy())]


def load_image_from_url(url: str):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content))
    except Exception as e:
        st.error(f"❌ 이미지 로드 실패: {e}")
        return None


def scan_checkpoints(root_dir: str) -> dict:
    result = {}
    for p in glob.glob(os.path.join(root_dir, '**', '*.pth'), recursive=True):
        folder = os.path.basename(os.path.dirname(p)).lower()  # ← .lower() 추가
        result[folder] = p
    return result


# ──────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="PokeClassifier",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    # 헤더
    st.markdown("""
    <div class="poke-header">
        <div><span class="spin">⚡</span></div>
        <div class="poke-title">POKÉCLASSIFIER</div>
        <div class="poke-subtitle">
        </div>
    </div>
    <br>
    """, unsafe_allow_html=True)

    # 사이드바⚙
    with st.sidebar:
        st.markdown(
        '<div class="section-label" style="font-size: 16px;">⚙ 모델 설정</div>',
        unsafe_allow_html=True)

        checkpoints = scan_checkpoints(RESULTS_DIR)
        if not checkpoints:
            st.error(f"학습된 모델 없음\n`{RESULTS_DIR}`")
            st.stop()

        MODEL_LABELS = {
            'resnet50_lp': '① ResNet50  |  Linear Probing',
            'resnet50_ft': '② ResNet50  |  Full Fine-tuning',
            'vit':         '③ ViT-B/16  |  Linear Probing',
            'vae':         '④ SD-VAE Encoder',
            'convnext':    '⑤ ConvNeXt-Tiny  |  Fine-tuning',
        }
        ORDER = ['resnet50_lp', 'resnet50_ft', 'vit', 'vae', 'convnext']
        options = [k for k in ORDER if k in checkpoints] + \
                  [k for k in checkpoints if k not in ORDER]
        labels  = [MODEL_LABELS.get(k, k) for k in options]

        sel_idx  = st.selectbox("모델 선택", range(len(options)),
                                format_func=lambda i: labels[i],
                                index=options.index('convnext')
                                      if 'convnext' in options else 0)
        model_key = options[sel_idx]
        top_k     = st.slider("Top-K 예측 수", 1, 10, 5)

        st.markdown("---")

        with st.spinner("모델 로딩 중..."):
            model, class_names, model_name, metrics = \
                load_model(checkpoints[model_key])

        st.markdown(
        '<div class="section-label" style="font-size: 16px;">모델 성능</div>',
        unsafe_allow_html=True
        )   
        
        if metrics:
            c1, c2 = st.columns(2)
            c1.metric("Accuracy",  f"{metrics.get('accuracy',0)*100:.1f}%")
            c2.metric("Precision", f"{metrics.get('precision',0):.3f}")
            c1.metric("Recall",    f"{metrics.get('recall',0):.3f}")
            c2.metric("Classes",   str(len(class_names)))

        st.markdown("---")
        st.markdown(f"""
        <div style="font-family:'Nunito',sans-serif;font-size:0.78rem;
                    color:#a8c8f8;line-height:2">
        📦 7,000 Labeled Pokemon<br>
        🏷 150종 (1세대 포켓몬)<br>
        🖥 Device: {str(DEVICE).upper()}
        </div>
        """, unsafe_allow_html=True)

    # 메인 컬럼
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown('<div class="section-label" style="font-size: 16px;">🖼 이미지 입력</div>',
                    unsafe_allow_html=True)
        method = st.radio("방식", ["📁 파일 업로드", "🔗 이미지 URL"],
                          horizontal=True, label_visibility="collapsed")
        image = None

        if method == "📁 파일 업로드":
            up = st.file_uploader("포켓몬 이미지 업로드",
                                  type=["jpg","jpeg","png","webp"],
                                  label_visibility="collapsed")
            if up:
                image = Image.open(up)
        else:
            url = st.text_input("URL", placeholder="https://...",
                                label_visibility="collapsed")
            if url:
                with st.spinner("불러오는 중..."):
                    image = load_image_from_url(url)

        if image:
            st.image(image, use_container_width=True)

    with col_out:
        st.markdown('<div class="section-label" style="font-size: 16px;">🔍 예측 결과</div>',
                    unsafe_allow_html=True)

        if image is None:
            st.markdown("""
            <div class="empty-state">
                ⬅ 왼쪽에서 이미지를 입력하면<br>포켓몬 이름을 예측합니다
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("포켓몬 분석 중..."):
                results = predict(model, class_names, image, top_k=top_k)

            if results:
                kor, eng, prob = results[0]
                emoji = "⚡" if prob > 90 else "🔥" if prob > 70 else "💫"
                st.markdown(f"""
                <div class="result-first">
                    <div class="result-emoji">{emoji}</div>
                    <div class="result-kor">{kor}</div>
                    <div class="result-eng">{eng}</div>
                    <div class="result-pct">{prob:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

                if len(results) > 1:
                    st.markdown(
                        '<div class="section-label" style="margin-top:1.2rem" style="font-size: 16px;">후보 목록</div>',
                        unsafe_allow_html=True
                    )
                    max_p = results[0][2]
                    for rank, (k, e, p) in enumerate(results[1:], 2):
                        bar_w = int(p / max_p * 100)
                        st.markdown(f"""
                        <div class="rank-row">
                            <div class="rank-no">#{rank}</div>
                            <div class="rank-name">{k}</div>
                            <div class="rank-bar-bg">
                                <div class="rank-bar-fill" style="width:{bar_w}%"></div>
                            </div>
                           <div class="rank-pct">{p:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

    # 푸터
    st.markdown("""
    <br><br>
    <div style="text-align:center;font-family:'Press Start 2P',monospace;
                font-size:0.38rem;color:rgba(255,215,0,0.25);letter-spacing:2px">
        POKÉCLASSIFIER · TRANSFER LEARNING EXPERIMENT · CV 2026
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()