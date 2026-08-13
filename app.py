"""
Smart Scanner API — Débruitage de documents U-Net
HuggingFace Spaces + Gradio
PSNR=30.13 dB | SSIM=0.9295 | Epoch 20
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import gradio as gr
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "ep0020_psnr30.13_ssim0.9295.pth"
IMG_SIZE   = 1024
BASE_CH    = 32

# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE (identique à l'entraînement)
# ─────────────────────────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(ch, ch//r, bias=False), nn.ReLU(True),
            nn.Linear(ch//r, ch, bias=False), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(self.avg(x).flatten(1)).view(x.shape[0],-1,1,1)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch,ch,3,padding=1,bias=False), nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.Conv2d(ch,ch,3,padding=1,bias=False), nn.BatchNorm2d(ch))
        self.ca = ChannelAttention(ch)
    def forward(self, x): return x + self.ca(self.body(x))

class EncBlock(nn.Module):
    def __init__(self, ic, oc, nr=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ic,oc,3,padding=1,bias=False), nn.BatchNorm2d(oc), nn.ReLU(True),
            *[ResBlock(oc) for _ in range(nr)])
        self.down = nn.Conv2d(oc,oc,2,stride=2,bias=False)
    def forward(self, x):
        s = self.conv(x); return self.down(s), s

class DecBlock(nn.Module):
    def __init__(self, ic, sc, oc, nr=2):
        super().__init__()
        self.up   = nn.ConvTranspose2d(ic,ic,2,stride=2,bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(ic+sc,oc,3,padding=1,bias=False), nn.BatchNorm2d(oc), nn.ReLU(True),
            *[ResBlock(oc) for _ in range(nr)])
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))

class DocumentDenoiser(nn.Module):
    def __init__(self, b=BASE_CH):
        super().__init__()
        self.e1 = EncBlock(3,b,2);    self.e2 = EncBlock(b,b*2,2)
        self.e3 = EncBlock(b*2,b*4,3); self.e4 = EncBlock(b*4,b*8,3)
        self.bn = nn.Sequential(
            nn.Conv2d(b*8,b*16,3,padding=1,bias=False),
            nn.BatchNorm2d(b*16), nn.ReLU(True),
            ResBlock(b*16), ResBlock(b*16), ChannelAttention(b*16))
        self.d4 = DecBlock(b*16,b*8,b*8,3); self.d3 = DecBlock(b*8,b*4,b*4,3)
        self.d2 = DecBlock(b*4,b*2,b*2,2);  self.d1 = DecBlock(b*2,b,b,2)
        self.head = nn.Conv2d(b,3,1,bias=True)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x1,s1 = self.e1(x); x2,s2 = self.e2(x1)
        x3,s3 = self.e3(x2); x4,s4 = self.e4(x3)
        b  = self.bn(x4)
        d  = self.d1(self.d2(self.d3(self.d4(b,s4),s3),s2),s1)
        return torch.clamp(x - self.head(d), -1., 1.)

# ─────────────────────────────────────────────────────────────────────────────
# CHARGER LE MODÈLE
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL  = None
INFO   = {"epoch": 0, "psnr": 0.0, "ssim": 0.0}

def charger():
    global MODEL, INFO
    p = Path(MODEL_PATH)
    if not p.exists():
        print(f"⚠️  Modèle introuvable : {p}")
        for alt in [f"models/{MODEL_PATH}", f"weights/{MODEL_PATH}"]:
            if Path(alt).exists():
                p = Path(alt); break
        else:
            print("❌ Aucun checkpoint trouvé — vérifiez que le .pth est dans le Space")
            return

    state = torch.load(str(p), map_location=DEVICE, weights_only=False)
    sd    = state.get("model_state", state.get("net", state))
    if sd and next(iter(sd)).startswith("module."):
        sd = {k[7:]: v for k, v in sd.items()}

    MODEL = DocumentDenoiser().to(DEVICE)
    try:
        MODEL.load_state_dict(sd, strict=True)
    except RuntimeError:
        MODEL.load_state_dict(sd, strict=False)
    MODEL.eval()

    INFO["epoch"] = state.get("epoch", 20)
    INFO["psnr"]  = state.get("psnr",  30.13)
    INFO["ssim"]  = state.get("ssim",  0.9295)
    print(f"✅ Modèle — Epoch {INFO['epoch']} | PSNR={INFO['psnr']:.2f}dB | {DEVICE}")

charger()

# ─────────────────────────────────────────────────────────────────────────────
# FONCTION DE DÉBRUITAGE
# ─────────────────────────────────────────────────────────────────────────────
TFM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE),
                       interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def debruiter(image_input):
    if MODEL is None:
        return None, "❌ Modèle non chargé. Vérifiez que le .pth est dans le Space."

    if isinstance(image_input, np.ndarray):
        img_pil = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
    else:
        img_pil = image_input.convert("RGB")

    w_orig, h_orig = img_pil.size

    x = TFM(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = MODEL(x)

    out = (pred.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
    out_np = (out.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    result = Image.fromarray(out_np)
    result = result.resize((w_orig, h_orig), Image.LANCZOS)

    info = (f"✅ Débruitage terminé\n"
            f"Modèle : Epoch {INFO['epoch']} | "
            f"PSNR={INFO['psnr']:.2f} dB | SSIM={INFO['ssim']:.4f}\n"
            f"Taille : {w_orig}×{h_orig} px")

    return result, info

# ─────────────────────────────────────────────────────────────────────────────
# INTERFACE GRADIO
# ─────────────────────────────────────────────────────────────────────────────
description_md = f"""
## 📄 Smart Scanner API — Débruitage de Documents

Uploadez un document bruité (ombres, plis, taches, fissures) 
et obtenez la version nettoyée en couleur.

**Modèle :** U-Net Résiduel avec ChannelAttention  
**Performance :** PSNR = {INFO['psnr']:.2f} dB | SSIM = {INFO['ssim']:.4f}  
**Types de bruits traités :** ombres colorées, taches, fissures, plis
"""

with gr.Blocks(
    title="Smart Scanner API",
    theme=gr.themes.Soft(primary_hue="blue"),
    css=".gradio-container {max-width: 1200px; margin: auto;}"
) as demo:

    gr.Markdown(description_md)

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(
                label="📸 Image bruitée (entrée)",
                type="pil",
                height=500,
            )
            btn = gr.Button("🚀 Débruiter", variant="primary", size="lg")

        with gr.Column(scale=1):
            img_output = gr.Image(
                label="✨ Image débruitée (sortie)",
                type="pil",
                height=500,
            )
            info_box = gr.Textbox(
                label="ℹ️ Informations",
                interactive=False,
                lines=4,
            )

    btn.click(
        fn=debruiter,
        inputs=img_input,
        outputs=[img_output, info_box],
    )

# ─────────────────────────────────────────────────────────────────────────────
# LANCEMENT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )