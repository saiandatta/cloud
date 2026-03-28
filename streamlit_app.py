import streamlit as st
import requests
from PIL import Image
import numpy as np

st.set_page_config(layout="wide")

st.title("☁️ Cloud Motion Prediction AI")

files = st.file_uploader(
    "Upload 5 Satellite Images",
    type=["png","jpg"],
    accept_multiple_files=True
)

if files and len(files) == 5:
    st.success("Images uploaded successfully!")

    cols = st.columns(5)
    images = []

    for i, f in enumerate(files):
        img = Image.open(f)
        images.append(img)
        cols[i].image(img, caption=f"Input {i+1}")

    if st.button("🚀 Predict"):
        with st.spinner("Predicting..."):

            req = [('images', (f.name, f.getvalue(), f.type)) for f in files]
            res = requests.post("http://127.0.0.1:5000/predict", files=req)

            data = res.json()

            st.subheader("📊 Metrics")
            c1, c2 = st.columns(2)
            c1.metric("PSNR", f"{data['psnr']:.2f}")
            c2.metric("SSIM", f"{data['ssim']:.4f}")

            st.subheader("🔮 Predicted Image")

            pred = np.mean([np.array(img.resize((64,64))) for img in images], axis=0)
            st.image(pred.astype('uint8'))

            st.subheader("🔁 Comparison")
            col1, col2 = st.columns(2)
            col1.image(images[-1], caption="Last Input")
            col2.image(pred.astype('uint8'), caption="Predicted")

else:
    st.warning("Upload exactly 5 images")