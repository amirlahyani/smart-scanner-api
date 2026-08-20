from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
from PIL import Image
import io
import torch
from torchvision import transforms
import os
from model import DocumentDenoiser

app = Flask(__name__)
CORS(app)

# Charger le modèle
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "ep0020_psnr30.13_ssim0.9295.pth"

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = DocumentDenoiser(base_ch=32).to(DEVICE)
    state_dict = checkpoint.get("model_state", checkpoint)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("✅ Modèle chargé avec succès")
else:
    model = None
    print("❌ Modèle non trouvé")

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def denorm(t):
    return (t.float() * 0.5 + 0.5).clamp(0, 1)

@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "running", "model_loaded": model is not None})

@app.route("/full-process", methods=["POST"])
def full_process():
    try:
        file = request.files["file"]
        contents = file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        if model is not None:
            w_orig, h_orig = pil_img.size
            img_1024 = pil_img.resize((1024, 1024), Image.LANCZOS)
            t = to_tensor(img_1024).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = model(t)
            result = denorm(out.squeeze(0).cpu())
            pil_result = transforms.ToPILImage()(result)
            pil_result = pil_result.resize((w_orig, h_orig), Image.LANCZOS)
        else:
            pil_result = pil_img
        
        image_np = np.array(pil_result)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".png", image_bgr)
        b64 = base64.b64encode(buf).decode()
        
        return jsonify({
            "success": True,
            "enhanced_image": b64,
            "analysis": {
                "model_epoch": "20",
                "model_psnr": "30.13 dB",
                "model_ssim": "0.9295"
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)