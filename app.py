import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import os
import tensorflow as tf
from keras.applications.efficientnet import preprocess_input

# =========================================================
#                MOBILE FRIENDLY PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Klasifikasi Batik Nusantara",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
#                     RESPONSIVE CSS
# =========================================================
st.markdown("""
<style>
/* Warna utama */
.stApp {
    background-color: #0E1117;
    color: #E6EEF8;
    font-family: "Segoe UI", sans-serif;
}

/* Mobile layout fix */
@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 1rem !important;
    }

    h1 { font-size: 1.7rem !important; }
    h2 { font-size: 1.4rem !important; }
    h3 { font-size: 1.2rem !important; }

    .prediction, .confidence {
        font-size: 1rem !important;
        padding: 10px !important;
    }
}

/* Komponen kotak hasil */
.prediction {
    background:#124734;
    padding:15px;
    border-radius:12px;
    color:#DFFFE5;
    font-size:1.15rem;
    font-weight:bold;
}

.confidence {
    background:#1E3A8A;
    padding:15px;
    border-radius:12px;
    color:#DDEBFF;
    font-size:1.15rem;
}

/* tombol */
.stButton button {
    padding: 0.8rem 1.2rem;
    border-radius: 10px;
    font-size: 1rem;
}
</style>
""", unsafe_allow_html=True)


st.title("🟦 Klasifikasi Motif Batik Nusantara — EfficientNetB0 (TFLite)")

# =========================================================
#                   LOAD TFLITE MODEL
# =========================================================
MODEL_PATH = "model/best_model_EfficientNetB0.tflite"

@st.cache_resource
def load_effnet_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

model = load_effnet_model()
if model is None:
    st.error("Model tidak ditemukan. Pastikan berada dalam folder /model/")
    st.stop()


# =========================================================
#                       LABELS
# =========================================================
labels = [
    "barong","celup","cendrawasih","ceplok","dayak",
    "insang","kawung","lontara","mataketeran",
    "megamendung","ondel-ondel","parang","pring",
    "rumah-minang"
]


# =========================================================
#                   PREPARE IMAGE
# =========================================================
def prepare_image(pil_img, target=(224,224)):
    img = pil_img.convert("RGB").resize(target)
    arr = np.array(img).astype(np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


# =========================================================
#                   PREDICT IMAGE
# =========================================================
def predict_image(interpreter, pil_img):
    arr = prepare_image(pil_img)
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()
    interpreter.set_tensor(inp[0]["index"], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(out[0]["index"])[0]
    idx = int(np.argmax(preds))
    return idx, float(np.max(preds)), preds


# =========================================================
#                    MAIN LAYOUT
# =========================================================
left, right = st.columns([1.1, 1])

with left:
    st.subheader("📤 Upload Gambar Batik")
    uploaded = st.file_uploader("Unggah file (jpg, jpeg, png)", type=["jpg","jpeg","png"])

    if uploaded:
        pil = Image.open(uploaded)
        st.image(pil, caption="Gambar diupload", use_container_width=True)

        if st.button("🔍 Prediksi Sekarang", use_container_width=True):
            idx, conf, probs = predict_image(model, pil)
            label = labels[idx]

            st.markdown(f"<div class='prediction'>Motif Terdeteksi: <b>{label}</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='confidence'>Confidence: {conf*100:.2f}%</div>", unsafe_allow_html=True)

            df = pd.DataFrame({"label": labels, "prob": probs}).sort_values("prob", ascending=False)
            st.bar_chart(df.set_index("label"))

with right:
    st.subheader("ℹ Tentang Model")
    st.info("""
    **Model EfficientNetB0 (TFLite)**  
    Ukuran input: 224x224  
    Jumlah kelas: 14  
    Dataset: Batik Nusantara  
    """)

    st.success("Aplikasi ini sudah **mobile friendly** dan siap digunakan di HP!")

