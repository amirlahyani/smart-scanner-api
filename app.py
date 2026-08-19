# ============================================================
#  app.py — Smart Scanner API
#  ✅ Compatible HuggingFace Spaces + Gradio 4.44.1
#  ✅ Charge best_model.pth depuis le Space
#  ✅ Endpoint /full-process compatible Flutter
# ============================================================

import os
import io
import base64
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import gradio as gr

from model import DocumentDenoiser

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODEL_PATH = "ep0020_psnr30.13_ssim0.9295.pth"
BASE_CH    = 32
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)

# ─────────────────────────────────────────────────────────────
#  CHARGEMENT DU MODÈLE
# ─────────────────────────────────────────────────────────────
def charger_modele():
    if not Path(MODEL_PATH).exists():
        print("ERREUR : {} introuvable".format(MODEL_PATH))
        return None, None, None

    try:
        raw  = torch.load(MODEL_PATH, map_location=device,
                          weights_only=False)
        psnr = raw.get("psnr",  30.13)
        ep   = raw.get("epoch", 20)
        sd   = raw.get("model_state", raw)

        net      = DocumentDenoiser(base_ch=BASE_CH).to(device)
        net_keys = set(net.state_dict().keys())
        sd_clean = {k: v for k, v in sd.items() if k in net_keys}
        manquants = net_keys - set(sd_clean.keys())

        if len(manquants) == 0:
            net.load_state_dict(sd_clean, strict=True)
            print("Chargement STRICT OK")
        else:
            cur = net.state_dict()
            cur.update(sd_clean)
            net.load_state_dict(cur, strict=False)
            print("Chargement partiel ({} manquantes)".format(len(manquants)))

        net.eval()
        print("Modele pret — PSNR={} dB  Epoch={}".format(psnr, ep))
        return net, psnr, ep

    except Exception as e:
        print("ERREUR chargement modele :", e)
        traceback.print_exc()
        return None, None, None

MODEL, PSNR_VAL, EPOCH_VAL = charger_modele()

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
    """Débruite une image PIL et retourne l'image débruitée."""
    w_orig, h_orig = pil_img.size

    # Redimensionner en 1024×1024 pour le modèle
    img_1024 = pil_img.resize((1024, 1024), Image.LANCZOS)
    t        = to_tensor(img_1024).unsqueeze(0).to(device)

    with torch.no_grad():
        out = MODEL(t)

    result  = denorm(out.squeeze(0).cpu())
    pil_out = transforms.ToPILImage()(result)

    # Restaurer taille originale
    return pil_out.resize((w_orig, h_orig), Image.LANCZOS)

# ─────────────────────────────────────────────────────────────
#  FONCTION PRINCIPALE — GRADIO + FLUTTER COMPATIBLE
# ─────────────────────────────────────────────────────────────
def process_image(image: Image.Image):
    """
    Entrée  : image PIL (Gradio)
    Sortie  : (image_debruitee PIL, texte_info)
    """
    if image is None:
        return None, "❌ Aucune image fournie"

    if MODEL is None:
        return image, "❌ Modèle non chargé — vérifiez best_model.pth"

    try:
        img_rgb   = image.convert("RGB")
        debruitee = debruiter_pil(img_rgb)

        info = (
            "✅ Débruitage réussi\n"
            "Modèle : Epoch {} | PSNR {} dB | SSIM 0.9295\n"
            "Taille originale : {}×{} px"
        ).format(EPOCH_VAL, PSNR_VAL, image.width, image.height)

        return debruitee, info

    except Exception as e:
        traceback.print_exc()
        return image, "❌ Erreur : {}".format(str(e))

# ─────────────────────────────────────────────────────────────
#  INTERFACE GRADIO
# ─────────────────────────────────────────────────────────────
titre = "📄 Smart Scanner — Débruitage de Documents Couleur"
description = """
**U-Net résiduel avec Channel Attention** — entraîné sur 26 604 paires de documents couleur

| Métrique | Valeur |
|----------|--------|
| PSNR     | **{psnr} dB** |
| SSIM     | **0.9295** |
| Epoch    | **{ep}** |

**Bruits gérés** : ombres colorées · taches eau/huile · fissures · luminosité excessive
""".format(psnr=PSNR_VAL, ep=EPOCH_VAL)

with gr.Blocks(
    title=titre,
    theme=gr.themes.Soft(primary_hue="blue"),
    css=".gradio-container {max-width: 1200px}"
) as demo:

    gr.Markdown("# " + titre)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(
                type="pil",
                label="📁 Image bruitée",
                height=450
            )
            btn = gr.Button(
                value="🚀 Débruiter l'image",
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=1):
            out_img = gr.Image(
                type="pil",
                label="✅ Image débruitée",
                height=450
            )
            out_txt = gr.Textbox(
                label="Résultats",
                lines=4,
                interactive=False
            )

    btn.click(
        fn=process_image,
        inputs=[inp],
        outputs=[out_img, out_txt]
    )

    # Exemples si disponibles
    exemples_dir = Path("exemples")
    if exemples_dir.exists():
        imgs_ex = list(exemples_dir.glob("*.jpg")) + \
                  list(exemples_dir.glob("*.png"))
        if imgs_ex:
            gr.Examples(
                examples=[[str(p)] for p in imgs_ex[:4]],
                inputs=[inp],
                outputs=[out_img, out_txt],
                fn=process_image,
                cache_examples=False
            )

    gr.Markdown(
        "---\n"
        "**Architecture** : U-Net + ChannelAttention + ResBlocks  |  "
        "**Loss** : MSE(0.6) + L1(0.4)  |  "
        "**Données** : 26 604 paires documents couleur"
    )

# ─────────────────────────────────────────────────────────────
#  LANCEMENT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )