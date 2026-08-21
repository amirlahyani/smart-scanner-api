"""
Smart Scanner API — Railway
Débruitage de documents U-Net | PSNR=30.13dB | SSIM=0.9295
"""

import os
import io
import base64
import traceback
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
import re

from model import DocumentDenoiser

# ─────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("===== Application Startup at {} =====".format(
    datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODEL_PTH = "ep0020_psnr30.13_ssim0.9295.pth"  # ✅ VOTRE NOM DE FICHIER
GOOGLE_DRIVE_ID = "1EDzgmyQuYzC7VbbSCHyb0zT0wP0Bl1Jz"  # ✅ VOTRE ID
BASE_CH = 32
device = torch.device("cpu")
logger.info("Device : {}".format(device))

# ─────────────────────────────────────────────────────────────
#  TÉLÉCHARGER LE MODÈLE DEPUIS GOOGLE DRIVE
# ─────────────────────────────────────────────────────────────
def download_model():
    """Télécharge le modèle depuis Google Drive"""
    if os.path.exists(MODEL_PTH):
        size = os.path.getsize(MODEL_PTH)
        if size > 80000000:  # 80 MB
            logger.info(f"✅ Modèle déjà présent ({size/1024/1024:.1f} MB)")
            return
        else:
            logger.warning(f"⚠️ Fichier corrompu ({size/1024/1024:.1f} MB), re-téléchargement...")
            os.remove(MODEL_PTH)

    logger.info("📥 Téléchargement du modèle depuis Google Drive...")

    url = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_ID}"

    try:
        response = requests.get(url, stream=True, timeout=300)

        # Gérer la confirmation Google Drive
        if "confirm" in response.text and "download_warning" in response.url:
            logger.info("   🔄 Confirmation Google Drive...")
            confirm_match = re.search(r'confirm=([^&]+)', response.text)
            if confirm_match:
                confirm_token = confirm_match.group(1)
                url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={GOOGLE_DRIVE_ID}"
                response = requests.get(url, stream=True, timeout=300)

        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        if total_size > 0:
            logger.info(f"   Taille totale: {total_size/1024/1024:.1f} MB")

        downloaded = 0
        with open(MODEL_PTH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if int(percent) % 10 == 0:
                            logger.info(f"   Téléchargement: {percent:.0f}%")

        size_mb = os.path.getsize(MODEL_PTH) / 1024 / 1024
        if size_mb > 80:
            logger.info(f"✅ Modèle téléchargé ({size_mb:.1f} MB)")
        else:
            logger.error(f"❌ Fichier trop petit ({size_mb:.1f} MB)")
            os.remove(MODEL_PTH)

    except Exception as e:
        logger.error(f"❌ Erreur téléchargement: {e}")

# Télécharger le modèle au démarrage
download_model()

# ─────────────────────────────────────────────────────────────
#  FASTAPI APP
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Scanner API",
    description="API de débruitage de documents avec U-Net",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
#  CHARGEMENT DU MODÈLE
# ─────────────────────────────────────────────────────────────
MODEL = None
PSNR_VAL = 30.13
EPOCH_VAL = 20
SSIM_VAL = 0.9295

def charger_modele():
    global MODEL, PSNR_VAL, EPOCH_VAL, SSIM_VAL

    if not Path(MODEL_PTH).exists():
        logger.error(f"❌ Modèle introuvable : {MODEL_PTH}")
        return False

    try:
        raw = torch.load(MODEL_PTH, map_location=device, weights_only=False)
        PSNR_VAL = raw.get("psnr", 30.13)
        EPOCH_VAL = raw.get("epoch", 20)
        SSIM_VAL = raw.get("ssim", 0.9295)
        sd = raw.get("model_state", raw)

        net = DocumentDenoiser(base_ch=BASE_CH).to(device)
        net_keys = set(net.state_dict().keys())
        sd_clean = {k: v for k, v in sd.items() if k in net_keys}
        manquants = net_keys - set(sd_clean.keys())

        if len(manquants) == 0:
            net.load_state_dict(sd_clean, strict=True)
            logger.info("Chargement STRICT complet")
        else:
            cur = net.state_dict()
            cur.update(sd_clean)
            net.load_state_dict(cur, strict=False)
            logger.warning(f"Chargement partiel ({len(manquants)} manquantes)")

        net.eval()
        MODEL = net
        logger.info(f"✅ Modèle chargé — Epoch={EPOCH_VAL} PSNR={PSNR_VAL} dB SSIM={SSIM_VAL}")
        return True

    except Exception as e:
        logger.error(f"❌ Erreur chargement: {e}")
        traceback.print_exc()
        return False

charger_modele()

# ─────────────────────────────────────────────────────────────
#  TRANSFORMATIONS
# ─────────────────────────────────────────────────────────────
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def denorm(t):
    return (t.float() * 0.5 + 0.5).clamp(0, 1)

def debruiter_pil(pil_img: Image.Image) -> Image.Image:
    w_orig, h_orig = pil_img.size
    img_1024 = pil_img.resize((1024, 1024), Image.LANCZOS)
    t = to_tensor(img_1024).unsqueeze(0).to(device)
    with torch.no_grad():
        out = MODEL(t)
    result = denorm(out.squeeze(0).cpu())
    pil_out = transforms.ToPILImage()(result)
    return pil_out.resize((w_orig, h_orig), Image.LANCZOS)

def pil_to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ─────────────────────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────────────────────
@app.get("/")
async def health_check():
    """Health check — Railway vérifie cet endpoint."""
    return {
        "status": "ok",
        "service": "Smart Scanner API",
        "model": "DocumentDenoiser U-Net",
        "epoch": EPOCH_VAL,
        "psnr": PSNR_VAL,
        "ssim": SSIM_VAL,
        "device": str(device),
        "model_loaded": MODEL is not None
    }

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.post("/full-process")
async def full_process(file: UploadFile = File(...)):
    """
    Endpoint principal — compatible Flutter/Dart.
    Reçoit une image, retourne l'image débruitée en base64.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé"
        )

    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        w, h = pil_img.size
        logger.info(f"Image reçue : {w}×{h} — {file.filename}")

        debruitee = debruiter_pil(pil_img)
        img_b64 = pil_to_base64(debruitee)

        return JSONResponse(content={
            "success": True,
            "enhanced_image": img_b64,
            "analysis": {
                "model_epoch": str(EPOCH_VAL),
                "model_psnr": f"{PSNR_VAL:.2f} dB",
                "model_ssim": f"{SSIM_VAL:.4f}",
                "summary": f"Débruitage terminé | Epoch {EPOCH_VAL} | PSNR={PSNR_VAL:.2f} dB"
            }
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

# ─────────────────────────────────────────────────────────────
#  LANCEMENT — PORT DYNAMIQUE RAILWAY
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Démarrage sur port {port}")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
