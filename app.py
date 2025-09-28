import os
import json, traceback
import numpy as np
from PIL import Image
from io import BytesIO

# PyTorch
import torch
import torch.nn.functional as F
from torchvision import transforms

# TensorFlow
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from flask import Flask, request, jsonify, render_template
import requests

# ------------------ Flask Setup ------------------
app = Flask(__name__)

# ------------------ Google Drive Model Links ------------------
MODEL_URLS = {
    "AgriNet_ResNet": "https://drive.google.com/uc?export=download&id=1THaqGbAUNuatkkF65Eynbewh1aUrD53Y",
    "AgriNet_InceptionV3": "https://drive.google.com/uc?export=download&id=1WTGMOhF9vxfLJtTm96oxmrP9F9CROeXf",
    "AgriNet_VGG19": "https://drive.google.com/uc?export=download&id=1nJD2FICQHzZf8P259atkNeHe712yWBoG",
    "AgriNet_MobileNet": "https://drive.google.com/uc?export=download&id=1_9rs2_1vLrCuOSG7y2LLQ-XM75EAhFbz"
}

# ------------------ Ensure Models Exist ------------------
def download_file(name, url):
    os.makedirs("models", exist_ok=True)
    path = f"models/{name}"
    if not os.path.exists(path):
        print(f"⬇️ Downloading {name} ...")
        r = requests.get(url, stream=True)
        with open(path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print(f"✅ Downloaded: {name}")
    return path

# ------------------ Load Class Names ------------------
with open('models/class_names.json') as f:
    plant_class_names = json.load(f)

with open('models/agri_class_names.json') as f:
    agri_class_names = json.load(f)

# ------------------ Load Models ------------------
keras_models = {}
pytorch_models = {}

def load_models():
    for filename, url in MODEL_URLS.items():
        ext = ".hdf5" if "h5" not in filename.lower() else ".h5"
        path = download_file(filename + ext, url)

        # Load Keras models
        if path.endswith(('.h5', '.hdf5')):
            try:
                keras_models[filename] = load_model(path)
                print(f"✅ Loaded Keras model: {filename}")
            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")

        # Load PyTorch models
        elif path.endswith('.pt'):
            try:
                model = torch.load(path, map_location=torch.device('cpu'))
                model.eval()
                pytorch_models[filename] = model
                print(f"✅ Loaded PyTorch model: {filename}")
            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")

load_models()

# ------------------ Preprocessing ------------------
def preprocess_for_keras(img):
    img = img.convert("RGB").resize((224, 224))
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def preprocess_for_pytorch(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = img.convert("RGB")
    return transform(img).unsqueeze(0)

# ------------------ Label Mapper ------------------
def get_label(idx, model_name):
    if "plant" in model_name.lower():
        names = plant_class_names
    else:
        names = agri_class_names

    if isinstance(names, dict):
        return names.get(str(idx), f"unknown_class_{idx}")
    elif isinstance(names, list):
        return names[idx] if idx < len(names) else f"unknown_class_{idx}"
    return f"unknown_class_{idx}"

# ------------------ Prediction Handler ------------------
def get_predictions(img):
    predictions = {}
    best = {"class": "Unknown", "confidence": 0.0}

    # ---- Keras Models ----
    for name, model in keras_models.items():
        try:
            x = preprocess_for_keras(img)
            pred = model.predict(x)[0]
            idx = int(np.argmax(pred))
            confidence = float(np.max(pred))
            label = get_label(idx, name)
            predictions[name] = {"class": label, "confidence": round(confidence * 100, 2)}

            if confidence > best["confidence"]:
                best = {"class": label, "confidence": round(confidence * 100, 2)}
        except Exception as e:
            predictions[name] = {"class": "Error", "confidence": 0.0, "error": str(e)}

    # ---- PyTorch Models ----
    for name, model in pytorch_models.items():
        try:
            x = preprocess_for_pytorch(img)
            with torch.no_grad():
                pred = model(x)
                probs = F.softmax(pred, dim=1)[0]
                idx = int(torch.argmax(probs))
                confidence = float(probs[idx])
                label = get_label(idx, name)
                predictions[name] = {"class": label, "confidence": round(confidence * 100, 2)}

                if confidence > best["confidence"]:
                    best = {"class": label, "confidence": round(confidence * 100, 2)}
        except Exception as e:
            predictions[name] = {"class": "Error", "confidence": 0.0, "error": str(e)}

    return best, predictions

# ------------------ Routes ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if 'image' not in request.files:
                return render_template("index.html", error="No image uploaded.")

            file = request.files['image']
            if file.filename == "":
                return render_template("index.html", error="Empty file name.")

            img = Image.open(BytesIO(file.read()))
            best_pred, all_preds = get_predictions(img)

            return render_template("result.html",
                                   prediction=best_pred,
                                   all_preds=all_preds)
        except Exception as e:
            traceback.print_exc()
            return render_template("index.html", error=str(e))

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded."}), 400

        file = request.files['image']
        if file.filename == "":
            return jsonify({"error": "Empty file name."}), 400

        img = Image.open(BytesIO(file.read()))
        best_pred, all_preds = get_predictions(img)

        return jsonify({
            "most_confident_prediction": best_pred,
            "all_model_predictions": all_preds
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
