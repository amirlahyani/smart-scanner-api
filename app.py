"""
Smart Scanner API — Railway
Débruitage de documents U-Net | PSNR=30.13dB | SSIM=0.9295
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import base64
import io
import uvicorn

# ── Importer le modèle ──────────────────────────────────────
class ChannelAttention(torch.nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.avg = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(ch, ch // reduction, bias=False),
            torch.nn.ReLU(True),
            torch.nn.Linear(ch // reduction, ch, bias=False),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(
            self.avg(x).squeeze(-1).squeeze(-1)
        ).unsqueeze(-1).unsqueeze(-1)

class ResBlock(torch.nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(ch),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(ch)
        )
        self.ca = ChannelAttention(ch)
    def forward(self, x):
        return x + self.ca(self.body(x))

class EncoderBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n_res=2):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(True),
            *[ResBlock(out_ch) for _ in range(n_res)]
        )
        self.down = torch.nn.Conv2d(out_ch, out_ch, 2, stride=2, bias=False)
    def forward(self, x):
        s = self.conv(x)
        return self.down(s), s

class DecoderBlock(torch.nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, n_res=2):
        super().__init__()
        self.up = torch.nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2, bias=False)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(True),
            *[ResBlock(out_ch) for _ in range(n_res)]
        )
    def forward(self, x, s):
        x = self.up(x)
        if x.shape != s.shape:
            x = torch.nn.functional.interpolate(x, size=s.shape[2:],
                              mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, s], dim=1))

class DocumentDenoiser(torch.nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()
        b = base_ch
        self.enc1 = EncoderBlock(3, b, 2)
        self.enc2 = EncoderBlock(b, b * 2, 2)
        self.enc3 = EncoderBlock(b * 2, b * 4, 3)
        self.enc4 = EncoderBlock(b * 4, b * 8, 3)
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(b * 8, b * 16, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(b * 16),
            torch.nn.ReLU(True),
            ResBlock(b * 16),
            ResBlock(b * 16),
            ChannelAttention(b * 16)
        )
        self.dec4 = DecoderBlock(b * 16, b * 8, b * 8, 3)
        self.dec3 = DecoderBlock(b * 8, b * 4, b * 4, 3)
        self.dec2 = DecoderBlock(b * 4, b * 2, b * 2, 2)
        self.dec1 = DecoderBlock(b * 2, b, b, 2)
        self.head = torch.nn.Conv2d(b, 3, 1, bias=True)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)
        bn = self.bottleneck(x4)
        d4 = self.dec4(bn, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return torch.clamp(x - self.head(d1), -1.0, 1.0)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODEL_PTH = "ep0020_psnr30.13_ssim0.9295.pth"
IMG_SIZE = 1024
BASE_CH = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# CHARGER LE MODÈLE
# ─────────────────────────────────────────────────────────────
def charger_modele():
    p = Path(MODEL_PTH)
    if not p.exists():
        print(f"❌ Modèle introuvable : {MODEL_PTH}")
        return None, 0, 0, 0

    state = torch.load(str(p), map_location="cpu", weights_only=False)
    sd = state.get("model_state", state.get("net", state))
    if sd and next(iter(sd)).startswith("module."):
        sd = {k[7:]: v for k, v in sd.items()}

    model = DocumentDenoiser(base_ch=BASE_CH)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        model.load_state_dict(sd, strict=False)

    model.eval()

    ep = state.get("epoch", 20)
    psnr = state.get("psnr", 30.13)
    ssim = state.get("ssim", 0.9295)
    print(f"✅ Modèle chargé — Epoch {ep} | PSNR={psnr:.2f}dB | SSIM={ssim:.4f}")
    return model, ep, psnr, ssim

MODEL, EPOCH, PSNR_V, SSIM_V = charger_modele()

# ─────────────────────────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────────────────────────
TFM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# ─────────────────────────────────────────────────────────────
# FONCTION DE DÉBRUITAGE
# ─────────────────────────────────────────────────────────────
def denoise_image(image_bytes):
    """Débruite une image depuis des bytes"""
    try:
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except:
        return None, "❌ Erreur de lecture de l'image"

    w_orig, h_orig = img_pil.size

    if MODEL is None:
        return img_pil, "❌ Modèle non chargé"

    model = MODEL.to(DEVICE)
    x = TFM(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(x)

    out = (pred.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
    out_np = (out.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    result = Image.fromarray(out_np).resize((w_orig, h_orig), Image.LANCZOS)

    info = f"✅ Débruitage terminé | Epoch {EPOCH} | PSNR={PSNR_V:.2f} dB | SSIM={SSIM_V:.4f}"
    return result, info

# ─────────────────────────────────────────────────────────────
# API FASTAPI
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="Smart Scanner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "running", "model_loaded": MODEL is not None, "epoch": EPOCH}

@app.post("/full-process")
async def full_process(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if len(contents) < 100:
            return {"success": False, "error": "Fichier vide ou corrompu"}

        result, info = denoise_image(contents)
        if result is None:
            return {"success": False, "error": info}

        # Convertir en base64
        buffered = io.BytesIO()
        result.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {
            "success": True,
            "enhanced_image": img_str,
            "analysis": {
                "model_epoch": str(EPOCH),
                "model_psnr": f"{PSNR_V:.2f} dB",
                "model_ssim": f"{SSIM_V:.4f}",
                "summary": info
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ─────────────────────────────────────────────────────────────
# LANCEMENT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
