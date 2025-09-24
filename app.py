from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os, json, traceback
import numpy as np
from PIL import Image

# PyTorch
import torch
import torch.nn.functional as F
from torchvision import transforms

# TensorFlow
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ------------------ Flask Setup ------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ------------------ Load Class Names ------------------
with open('models/class_names.json') as f:
    plant_class_names = json.load(f)

with open('models/agri_class_names.json') as f:
    agri_class_names = json.load(f)

# ------------------ Load Models ------------------
keras_models = {}
pytorch_models = {}

def load_models():
    for file in os.listdir('models'):
        path = os.path.join('models', file)

        # Load Keras models
        if file.endswith(('.h5', '.hdf5')):
            try:
                keras_models[file] = load_model(path)
                print(f"✅ Loaded Keras model: {file}")
            except Exception as e:
                print(f"❌ Failed to load Keras model {file}: {e}")

        # Load PyTorch models
        elif file.endswith('.pt'):
            try:
                model = torch.load(path, map_location=torch.device('cpu'))
                model.eval()
                pytorch_models[file] = model
                print(f"✅ Loaded PyTorch model: {file}")
            except Exception as e:
                print(f"❌ Failed to load PyTorch model {file}: {e}")

load_models()

# ------------------ Preprocessing ------------------
def preprocess_for_keras(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def preprocess_for_pytorch(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)

# ------------------ Label Mapper ------------------
def get_label(idx, model_name):
    if model_name == 'plant_disease_model_latest.h5':
        names = plant_class_names
    else:
        names = agri_class_names

    if isinstance(names, dict):
        return names.get(str(idx), f"unknown_class_{idx}")
    elif isinstance(names, list):
        return names[idx] if idx < len(names) else f"unknown_class_{idx}"
    return f"unknown_class_{idx}"

# ------------------ Prediction Handler ------------------
def get_predictions(image_path):
    predictions = {}
    best = {"class": "Unknown", "confidence": 0.0}

    # ---- Keras Models ----
    for name, model in keras_models.items():
        try:
            x = preprocess_for_keras(image_path)
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
            x = preprocess_for_pytorch(image_path)
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

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            best_pred, all_preds = get_predictions(filepath)

            return render_template("result.html",
                                   prediction=best_pred,
                                   all_preds=all_preds,
                                   image_path=filepath)
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

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        best_pred, all_preds = get_predictions(filepath)

        return jsonify({
            "most_confident_prediction": best_pred,
            "all_model_predictions": all_preds
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
