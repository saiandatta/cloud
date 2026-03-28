from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import sys

sys.path.append('../utils')
from metrics import psnr, ssim_metric

app = Flask(__name__)

# Correct path handling
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'convlstm_model.h5')

# IMPORTANT FIX
model = tf.keras.models.load_model(model_path, compile=False)

def preprocess(files):
    imgs = []
    for f in files:
        img = Image.open(f).convert('L').resize((64,64))
        imgs.append(np.array(img)/255.0)
    return np.array(imgs).reshape(1,5,64,64,1)

@app.route('/predict', methods=['POST'])
def predict():
    files = request.files.getlist('images')

    data = preprocess(files)

    pred = model.predict(data)[0]
    pred_img = (pred * 255).astype(np.uint8)

    gt = (data[0][-1] * 255).astype(np.uint8)

    return jsonify({
        "psnr": float(psnr(gt, pred_img)),
        "ssim": float(ssim_metric(gt, pred_img))
    })

if __name__ == "__main__":
    app.run(debug=True)