"""
Smart Scanner API — HuggingFace Spaces
Débruitage de documents U-Net | PSNR=30.13dB | SSIM=0.9295
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import gradio as gr
from model import DocumentDenoiser

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODEL_PTH = "ep0020_psnr30.13_ssim0.9295.pth"
IMG_SIZE  = 1024
BASE_CH   = 32
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# CHARGER LE MODÈLE
# ─────────────────────────────────────────────────────────────
def charger_modele():
    from pathlib import Path

    p = Path(MODEL_PTH)
    if not p.exists():
        raise FileNotFoundError(f"❌ Fichier introuvable : {MODEL_PTH}")

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

    ep   = state.get("epoch", 20)
    psnr = state.get("psnr",  30.13)
    ssim = state.get("ssim",  0.9295)
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
# INFÉRENCE
# ─────────────────────────────────────────────────────────────
def debruiter(image_input):
    if image_input is None:
        return None, "⚠️ Aucune image fournie"

    if isinstance(image_input, np.ndarray):
        img_pil = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
    else:
        img_pil = image_input.convert("RGB")

    w_orig, h_orig = img_pil.size

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
# INTERFACE GRADIO
# ─────────────────────────────────────────────────────────────
with gr.Blocks(title="Smart Scanner API") as demo:
    gr.Markdown("# 📄 Smart Scanner — Débruitage de Documents")
    gr.Markdown(f"**Modèle :** U-Net Résiduel | Epoch {EPOCH} | PSNR **{PSNR_V:.2f} dB** | SSIM **{SSIM_V:.4f}**")

    with gr.Row():
        with gr.Column():
            img_in = gr.Image(label="📸 Document bruité", type="pil", height=480)
            btn = gr.Button("🚀 Débruiter", variant="primary", size="lg")
        with gr.Column():
            img_out = gr.Image(label="✨ Document nettoyé", type="pil", height=480)
            info = gr.Textbox(label="ℹ️ Informations", lines=3, interactive=False)

    # ✅ SUPPRIMER api_name pour éviter l'erreur
    btn.click(
        fn=debruiter,
        inputs=img_in,
        outputs=[img_out, info],
        # api_name="debruiter",  # ← SUPPRIMÉ
    )

# ─────────────────────────────────────────────────────────────
# LANCEMENT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        prevent_thread_lock=True,  # ✅ AJOUTÉ
        enable_api=False,          # ✅ AJOUTÉ
    )