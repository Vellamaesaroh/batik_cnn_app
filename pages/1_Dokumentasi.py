import streamlit as st
import os
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input


# ==============================
# KONFIGURASI
# ==============================
DATASET_FOLDER = "dataset_images"
RIWAYAT_CSV = "riwayat_prediksi.csv"

os.makedirs(DATASET_FOLDER, exist_ok=True)


# ==============================
# LOAD LABEL
# ==============================
@st.cache_resource
def load_labels():
    path = os.path.join("model", "labels.txt")
    if not os.path.exists(path):
        st.error("❌ labels.txt tidak ditemukan")
        st.stop()

    with open(path, "r", encoding="utf-8") as f:
        return [x.strip() for x in f.readlines()]


# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_tflite_model():
    path = os.path.join("model", "best_model_EfficientNetB0.tflite")
    if not os.path.exists(path):
        st.error("❌ Model TFLite tidak ditemukan")
        st.stop()

    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter


# ==============================
# RIWAYAT
# ==============================
def load_riwayat():
    columns = ["filename", "prediksi", "confidence", "timestamp"]

    if os.path.exists(RIWAYAT_CSV):
        df = pd.read_csv(RIWAYAT_CSV)
        for c in columns:
            if c not in df.columns:
                df[c] = None
        return df[columns]

    return pd.DataFrame(columns=columns)


def save_riwayat(df):
    df.to_csv(RIWAYAT_CSV, index=False)


# ==============================
# PREDIKSI
# ==============================
def predict_tflite(image, labels):
    interpreter = load_tflite_model()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = image.resize((224, 224))
    img = np.array(img, dtype=np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = int(np.argmax(pred))

    return labels[idx], float(pred[idx])


# ==============================
# UI
# ==============================
st.set_page_config(page_title="Klasifikasi Motif Batik", layout="wide")
st.title("📸 Aplikasi Klasifikasi Motif Batik (CNN – EfficientNetB0)")

labels = load_labels()

mode = st.radio("Pilih metode input:", ["📷 Kamera", "📁 Upload File"])
img = None

if mode == "📷 Kamera":
    cam = st.camera_input("Ambil foto batik")
    if cam:
        img = Image.open(cam).convert("RGB")
else:
    up = st.file_uploader("Upload gambar batik", type=["jpg", "jpeg", "png"])
    if up:
        img = Image.open(up).convert("RGB")


# ==============================
# PREDIKSI
# ==============================
if img:
    st.image(img, caption="Gambar Input", use_container_width=True)

    if st.button("🔍 Prediksi Motif Batik"):
        label, conf = predict_tflite(img, labels)

        if conf < 0.6:
            st.warning("⚠️ Motif batik tidak dikenali dengan yakin")
        else:
            st.success(f"🧵 Hasil Prediksi: **{label}** ({conf:.3f})")

        fname = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        img.save(os.path.join(DATASET_FOLDER, fname))

        hist = load_riwayat()
        hist.loc[len(hist)] = {
            "filename": fname,
            "prediksi": label,
            "confidence": round(conf, 3),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_riwayat(hist)


# ==============================
# RIWAYAT + PREVIEW GAMBAR
# ==============================
st.markdown("---")
st.subheader("🕒 Riwayat Prediksi")

riwayat = load_riwayat().sort_values("timestamp", ascending=False)

if riwayat.empty:
    st.info("Belum ada riwayat prediksi.")
else:
    # HAPUS SEMUA
    if st.button("🗑️ Hapus Semua Riwayat"):
        save_riwayat(pd.DataFrame(columns=riwayat.columns))
        st.success("Semua riwayat dihapus")
        st.rerun()

    st.markdown("### 📋 Detail Riwayat")

    for i, row in riwayat.iterrows():
        img_path = os.path.join(DATASET_FOLDER, row["filename"])

        col1, col2, col3, col4, col5 = st.columns([1.2, 2, 2, 2, 0.8])

        # PREVIEW GAMBAR
        if os.path.exists(img_path):
            col1.image(img_path, use_container_width=True)
        else:
            col1.write("❌")

        col2.markdown(f"**Motif**  \n{row['prediksi']}")
        col3.markdown(f"**Confidence**  \n{row['confidence']}")
        col4.markdown(f"**Waktu**  \n{row['timestamp']}")

        # HAPUS PER BARIS
        if col5.button("❌", key=f"del_{i}"):
            df = load_riwayat()
            df = df[df["filename"] != row["filename"]]
            save_riwayat(df)
            st.rerun()

        st.markdown("---")
