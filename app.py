# app.py
import os, json, traceback
import numpy as np
from io import BytesIO
from PIL import Image

# Hugging Face
from huggingface_hub import HfApi, hf_hub_download

# PyTorch
import torch
import torch.nn.functional as F
from torchvision import transforms

# TensorFlow / Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from flask import Flask, request, jsonify, render_template

# ---------------- Config ----------------
HF_MODEL_REPO = "Rewant102/agri_sense_disease"  # <-- your repo
HF_TOKEN = os.environ.get("HF_TOKEN", None)     # set only if repo is private

os.makedirs("models", exist_ok=True)

# ---------------- Globals ----------------
app = Flask(__name__)
keras_models = {}
pytorch_models = {}
plant_class_names = None
agri_class_names = None

# ---------------- Utilities ----------------
def list_repo_files_safe(repo_id):
    api = HfApi()
    try:
        return api.list_repo_files(repo_id)
    except Exception as e:
        print("⚠️ Could not list HF repo files (maybe private or network issue):", e)
        return []

def hf_download(repo_id, filename):
    """Download via HF hub and return local path (uses cache)."""
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model", use_auth_token=HF_TOKEN)
    except Exception as e:
        print(f"❌ hf_hub_download failed for {filename}: {e}")
        raise

def load_json_local_or_hf(local_path, repo_files, possible_names):
    """
    Try local path first; otherwise find a matching filename in repo_files and download it.
    possible_names is a list like ['models/class_names.json','class_names.json']
    Returns loaded JSON or None.
    """
    # local first
    for p in possible_names:
        try:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print("Error reading local json", p, e)

    # try repo
    for candidate in possible_names:
        if candidate in repo_files:
            try:
                path = hf_download(HF_MODEL_REPO, candidate)
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print("Error downloading json from HF:", candidate, e)
    return None

def get_input_size_for_model(name):
    nc = name.lower()
    if "inception" in nc:
        return (299, 299)
    return (224, 224)

def preprocess_for_keras(img, size=(224,224)):
    img = img.convert("RGB").resize(size)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def preprocess_for_pytorch(img, size=(224,224)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    img = img.convert("RGB")
    return transform(img).unsqueeze(0)

def get_label(idx):
    """Try plant_class_names first, then agri_class_names, then fallback."""
    try:
        # dict keyed by strings
        if isinstance(plant_class_names, dict) and str(idx) in plant_class_names:
            return plant_class_names[str(idx)]
        if isinstance(plant_class_names, list) and idx < len(plant_class_names):
            return plant_class_names[idx]

        if isinstance(agri_class_names, dict) and str(idx) in agri_class_names:
            return agri_class_names[str(idx)]
        if isinstance(agri_class_names, list) and idx < len(agri_class_names):
            return agri_class_names[idx]
    except Exception:
        pass
    return f"unknown_class_{idx}"

# ---------------- Load models from HF (auto-discover) ----------------
def discover_and_load_models():
    global plant_class_names, agri_class_names

    repo_files = list_repo_files_safe(HF_MODEL_REPO)

    # Load class name JSONs (prefer local 'models/...' then repo)
    plant_class_names = load_json_local_or_hf(repo_files, repo_files, ["models/class_names.json", "class_names.json", "class_names/plant.json"])
    agri_class_names  = load_json_local_or_hf(repo_files, repo_files, ["models/agri_class_names.json", "agri_class_names.json", "class_names/agri.json"])

    if plant_class_names is None:
        print("⚠️ plant_class_names not found locally or in repo. Predictions will show numeric indices.")
    if agri_class_names is None:
        print("⚠️ agri_class_names not found locally or in repo. Predictions will show numeric indices.")

    # gather model files in repo
    model_candidates = [f for f in repo_files if f.lower().endswith(('.h5', '.hdf5', '.pt'))]

    # Also check local models/ folder as fallback (if user put local files)
    if not model_candidates:
        try:
            local_files = os.listdir("models")
            for f in local_files:
                if f.lower().endswith(('.h5', '.hdf5', '.pt')):
                    model_candidates.append(os.path.join("models", f))
        except Exception:
            pass

    if not model_candidates:
        print("❌ No model files found in HF repo or local models/ directory. Add .h5/.hdf5/.pt files and retry.")
        return

    print("Model files detected:", model_candidates)

    for mf in model_candidates:
        try:
            # mf may be "models/AgriNet_VGG19.h5" (repo path) or local path like "models/AgriNet_VGG19.h5"
            if os.path.exists(mf):  # local file
                local_path = mf
                model_name = os.path.splitext(os.path.basename(mf))[0]
            else:
                # download via HF
                local_path = hf_download(HF_MODEL_REPO, mf)
                model_name = os.path.splitext(os.path.basename(mf))[0]

            print(f"⬇️ Loading model file: {model_name} (from {local_path})")

            if local_path.lower().endswith(('.h5', '.hdf5')):
                try:
                    keras_models[model_name] = load_model(local_path, compile=False)
                    print(f"✅ Loaded Keras model: {model_name}")
                except Exception as e:
                    print(f"❌ Keras load failed for {model_name}: {e}")
                    # keep going
            elif local_path.lower().endswith('.pt'):
                try:
                    m = torch.load(local_path, map_location=torch.device("cpu"))
                    m.eval()
                    pytorch_models[model_name] = m
                    print(f"✅ Loaded PyTorch model: {model_name}")
                except Exception as e:
                    print(f"❌ PyTorch load failed for {model_name}: {e}")
        except Exception as e:
            print("Error while handling model file", mf, e)

# Run discovery at startup
discover_and_load_models()

# ---------------- Prediction logic ----------------
def get_predictions(img):
    predictions = {}
    best = {"class": "Unknown", "confidence": 0.0}

    # Keras
    for name, model in keras_models.items():
        try:
            size = get_input_size_for_model(name)
            x = preprocess_for_keras(img, size=size)
            pred = model.predict(x)[0]
            idx = int(np.argmax(pred))
            confidence = float(np.max(pred))
            label = get_label(idx)
            predictions[name] = {"class": label, "confidence": round(confidence * 100, 2)}
            if confidence > best["confidence"]:
                best = {"class": label, "confidence": round(confidence * 100, 2)}
        except Exception as e:
            predictions[name] = {"class": "Error", "confidence": 0.0, "error": str(e)}

    # PyTorch
    for name, model in pytorch_models.items():
        try:
            size = get_input_size_for_model(name)
            x = preprocess_for_pytorch(img, size=size)
            with torch.no_grad():
                out = model(x)
                # handle different shape outputs (batch, classes)
                if isinstance(out, tuple):
                    out = out[0]
                probs = F.softmax(out, dim=1)[0]
                idx = int(torch.argmax(probs))
                confidence = float(probs[idx])
                label = get_label(idx)
                predictions[name] = {"class": label, "confidence": round(confidence * 100, 2)}
                if confidence > best["confidence"]:
                    best = {"class": label, "confidence": round(confidence * 100, 2)}
        except Exception as e:
            predictions[name] = {"class": "Error", "confidence": 0.0, "error": str(e)}

    return best, predictions

# ---------------- Routes ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if 'image' not in request.files:
                return render_template("index.html", error="No image uploaded.")
            f = request.files['image']
            if f.filename == "":
                return render_template("index.html", error="Empty file name.")
            img = Image.open(BytesIO(f.read()))
            best_pred, all_preds = get_predictions(img)
            return render_template("result.html", prediction=best_pred, all_preds=all_preds)
        except Exception as e:
            traceback.print_exc()
            return render_template("index.html", error=str(e))
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded."}), 400
        f = request.files['image']
        if f.filename == "":
            return jsonify({"error": "Empty file name."}), 400
        img = Image.open(BytesIO(f.read()))
        best_pred, all_preds = get_predictions(img)
        return jsonify({"most_confident_prediction": best_pred, "all_model_predictions": all_preds})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ---------------- Run ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # HF Spaces default 7860; change if you want
    app.run(host="0.0.0.0", port=port, debug=False)
