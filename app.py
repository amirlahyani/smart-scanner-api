"""
Smart Scanner API
Débruitage de documents avec U-Net résiduel

PSNR = 30.13 dB
SSIM = 0.9295
Epoch = 20
"""

import os
from pathlib import Path

import numpy as np
import torch

import gradio as gr

from PIL import Image
from torchvision import transforms

from model import DocumentDenoiser


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "ep0020_psnr30.13_ssim0.9295.pth"

IMG_SIZE = 1024
BASE_CH = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# INFORMATIONS MODÈLE
# ============================================================

MODEL = None

INFO = {
    "epoch": 20,
    "psnr": 30.13,
    "ssim": 0.9295,
}


# ============================================================
# TRANSFORMATIONS
# ============================================================

TFM = transforms.Compose([
    transforms.Resize(
        (IMG_SIZE, IMG_SIZE),
        interpolation=transforms.InterpolationMode.LANCZOS
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    ),
])


# ============================================================
# RECHERCHE DU CHECKPOINT
# ============================================================

def trouver_checkpoint():

    candidates = [
        Path(MODEL_PATH),

        Path("models") / MODEL_PATH,

        Path("weights") / MODEL_PATH,
    ]

    for path in candidates:

        if path.is_file():

            print(
                f"✅ Checkpoint trouvé : {path}"
            )

            return path

    print(
        "❌ Checkpoint introuvable."
    )

    print(
        "Fichiers recherchés :"
    )

    for path in candidates:
        print(f"   - {path}")

    return None


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

def charger_modele():

    global MODEL
    global INFO

    checkpoint_path = trouver_checkpoint()

    if checkpoint_path is None:

        MODEL = None

        return

    print(
        f"🔄 Chargement du modèle : "
        f"{checkpoint_path}"
    )

    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False
        )

    except Exception as e:

        print(
            f"❌ Erreur lors du chargement : {e}"
        )

        MODEL = None

        return


    # --------------------------------------------------------
    # Récupération du state_dict
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state" in checkpoint:

            state_dict = checkpoint["model_state"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        elif "net" in checkpoint:

            state_dict = checkpoint["net"]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint


    # --------------------------------------------------------
    # Suppression éventuelle du préfixe DataParallel
    # --------------------------------------------------------

    if isinstance(state_dict, dict):

        cleaned_state_dict = {}

        for key, value in state_dict.items():

            if key.startswith("module."):

                key = key[7:]

            cleaned_state_dict[key] = value

        state_dict = cleaned_state_dict


    # --------------------------------------------------------
    # Création du modèle
    # --------------------------------------------------------

    model = DocumentDenoiser(
        base_ch=BASE_CH
    )


    # --------------------------------------------------------
    # CHARGEMENT STRICT
    # --------------------------------------------------------

    try:

        result = model.load_state_dict(
            state_dict,
            strict=True
        )

    except RuntimeError as e:

        print(
            "❌ ERREUR : le checkpoint ne correspond "
            "pas exactement à l'architecture."
        )

        print(e)

        MODEL = None

        return


    # --------------------------------------------------------
    # Vérification
    # --------------------------------------------------------

    print(
        f"✅ Poids chargés correctement : "
        f"{len(state_dict)} tenseurs"
    )

    print(
        f"✅ Missing keys : {len(result.missing_keys)}"
    )

    print(
        f"✅ Unexpected keys : "
        f"{len(result.unexpected_keys)}"
    )


    # --------------------------------------------------------
    # Informations checkpoint
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        INFO["epoch"] = checkpoint.get(
            "epoch",
            20
        )

        INFO["psnr"] = checkpoint.get(
            "psnr",
            30.13
        )

        INFO["ssim"] = checkpoint.get(
            "ssim",
            0.9295
        )


    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    model = model.to(DEVICE)

    model.eval()

    MODEL = model


    print(
        f"✅ Modèle chargé"
    )

    print(
        f"   Epoch : {INFO['epoch']}"
    )

    print(
        f"   PSNR  : {INFO['psnr']:.2f} dB"
    )

    print(
        f"   SSIM  : {INFO['ssim']:.4f}"
    )

    print(
        f"   Device: {DEVICE}"
    )


# ============================================================
# CHARGEMENT AU DÉMARRAGE
# ============================================================

charger_modele()


# ============================================================
# DÉBRUITAGE
# ============================================================

def debruiter(image_input):

    if MODEL is None:

        return (
            None,
            "❌ Modèle non chargé.\n"
            "Vérifiez la présence du fichier .pth."
        )


    if image_input is None:

        return (
            None,
            "⚠️ Veuillez sélectionner une image."
        )


    # --------------------------------------------------------
    # Conversion en PIL
    # --------------------------------------------------------

    if isinstance(image_input, Image.Image):

        img_pil = image_input.convert("RGB")

    elif isinstance(image_input, np.ndarray):

        if image_input.dtype != np.uint8:

            image_input = np.clip(
                image_input,
                0,
                255
            ).astype(np.uint8)

        img_pil = Image.fromarray(
            image_input
        ).convert("RGB")

    else:

        return (
            None,
            "❌ Format d'image non supporté."
        )


    # --------------------------------------------------------
    # Dimensions originales
    # --------------------------------------------------------

    original_width, original_height = (
        img_pil.size
    )


    # --------------------------------------------------------
    # Préparation
    # --------------------------------------------------------

    x = TFM(
        img_pil
    ).unsqueeze(0).to(DEVICE)


    # --------------------------------------------------------
    # Inférence
    # --------------------------------------------------------

    try:

        with torch.inference_mode():

            prediction = MODEL(x)

    except Exception as e:

        print(
            f"❌ Erreur pendant l'inférence : {e}"
        )

        return (
            None,
            f"❌ Erreur pendant le débruitage :\n{e}"
        )


    # --------------------------------------------------------
    # Retour dans [0, 1]
    # --------------------------------------------------------

    output = (
        prediction.squeeze(0)
        .cpu()
        .clamp(-1.0, 1.0)
    )

    output = (
        output * 0.5
        + 0.5
    )

    output_np = (
        output
        .permute(1, 2, 0)
        .numpy()
    )

    output_np = (
        output_np * 255.0
    ).round().astype(np.uint8)


    # --------------------------------------------------------
    # PIL
    # --------------------------------------------------------

    result = Image.fromarray(
        output_np,
        mode="RGB"
    )


    # --------------------------------------------------------
    # Retour à la taille originale
    # --------------------------------------------------------

    result = result.resize(
        (
            original_width,
            original_height
        ),
        Image.Resampling.LANCZOS
    )


    # --------------------------------------------------------
    # Informations
    # --------------------------------------------------------

    info = (
        "✅ Débruitage terminé\n\n"
        f"Modèle : Epoch {INFO['epoch']}\n"
        f"PSNR : {INFO['psnr']:.2f} dB\n"
        f"SSIM : {INFO['ssim']:.4f}\n"
        f"Device : {DEVICE}\n"
        f"Taille : "
        f"{original_width} × "
        f"{original_height} px"
    )


    return result, info


# ============================================================
# INTERFACE
# ============================================================

description_md = f"""
# 📄 Smart Scanner API

### Débruitage intelligent de documents

Chargez une image de document contenant :

- ombres
- taches
- plis
- fissures
- bruit visuel

Le modèle produit automatiquement une version nettoyée.

---

### 🧠 Modèle

**U-Net résiduel + Channel Attention**

**Performance du checkpoint :**

- **PSNR : {INFO['psnr']:.2f} dB**
- **SSIM : {INFO['ssim']:.4f}**
- **Epoch : {INFO['epoch']}**

---
"""


# ============================================================
# GRADIO
# ============================================================

with gr.Blocks(
    title="Smart Scanner API",
    theme=gr.themes.Soft(
        primary_hue="blue"
    ),
    css="""
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    """
) as demo:

    gr.Markdown(
        description_md
    )


    with gr.Row():

        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------

        with gr.Column(
            scale=1
        ):

            img_input = gr.Image(
                label="📸 Document original",
                type="pil",
                height=500
            )

            btn = gr.Button(
                "🚀 Débruiter",
                variant="primary",
                size="lg"
            )


        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        with gr.Column(
            scale=1
        ):

            img_output = gr.Image(
                label="✨ Document débruité",
                type="pil",
                height=500
            )

            info_box = gr.Textbox(
                label="ℹ️ Informations",
                interactive=False,
                lines=7
            )


    # --------------------------------------------------------
    # EVENT
    # --------------------------------------------------------

    btn.click(
        fn=debruiter,
        inputs=img_input,
        outputs=[
            img_output,
            info_box
        ],
        api_name="debruiter"
    )


# ============================================================
# LANCEMENT HUGGING FACE SPACES
# ============================================================

if __name__ == "__main__":

    demo.launch(
        show_error=True,
        show_api=False
    )