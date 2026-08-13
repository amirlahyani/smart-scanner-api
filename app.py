# ============================================================
# app.py — Smart Scanner API pour Hugging Face Spaces (Gradio)
# ============================================================

import os
import cv2
import numpy as np
from PIL import Image, ImageOps
import gradio as gr
import torch
from torchvision import transforms

# Importer le modèle
from model import DocumentDenoiser

# ──────────────────────────────────────────────────────────────
#  CHARGEMENT DU MODÈLE
# ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {DEVICE}")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ep0020_psnr30.13_ssim0.9295.pth")

if not os.path.exists(MODEL_PATH):
    print(f"❌ Modèle non trouvé: {MODEL_PATH}")
    model = None
else:
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model = DocumentDenoiser(base_ch=32).to(DEVICE)
        state_dict = checkpoint.get("model_state", checkpoint)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        print("✅ Modèle chargé avec succès !")
        print(f"   PSNR     : {checkpoint.get('val_psnr', '?')} dB")
        print(f"   SSIM     : {checkpoint.get('val_ssim', '?')}")
    except Exception as e:
        print(f"❌ Erreur chargement modèle: {e}")
        model = None

# ──────────────────────────────────────────────────────────────
#  PRÉTRAITEMENT
# ──────────────────────────────────────────────────────────────
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

def denorm(t):
    return (t.float() * 0.5 + 0.5).clamp(0, 1)

def process_with_ai(pil_img):
    if model is None:
        return pil_img
    
    w_orig, h_orig = pil_img.size
    img_1024 = pil_img.resize((1024, 1024), Image.LANCZOS)
    t = to_tensor(img_1024).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        out = model(t)
    
    result = denorm(out.squeeze(0).cpu())
    pil_result = transforms.ToPILImage()(result)
    return pil_result.resize((w_orig, h_orig), Image.LANCZOS)

def correct_perspective_auto(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image_bgr
    
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    img_area = image_bgr.shape[0] * image_bgr.shape[1]
    
    if area < img_area * 0.4:
        return image_bgr
    
    rect = cv2.minAreaRect(largest)
    angle = rect[2]
    
    if angle < -45:
        angle += 90
    
    if abs(angle) < 0.5:
        return image_bgr
    
    h, w = image_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        image_bgr, M, (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE
    )

# ──────────────────────────────────────────────────────────────
#  FONCTION GRADIO
# ──────────────────────────────────────────────────────────────
def denoise_image(file):
    try:
        # Charger l'image
        pil_img = Image.open(file).convert("RGB")
        pil_img = ImageOps.exif_transpose(pil_img)
        
        # Traitement IA
        if model is not None:
            pil_denoised = process_with_ai(pil_img)
        else:
            pil_denoised = pil_img
        
        # Post-traitement OpenCV
        image_np = np.array(pil_denoised)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_bgr = correct_perspective_auto(image_bgr)
        
        # Convertir en RGB pour affichage
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        return image_rgb
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

# ──────────────────────────────────────────────────────────────
#  INTERFACE GRADIO
# ──────────────────────────────────────────────────────────────
with gr.Blocks(title="Smart Scanner IA", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📄 Smart Scanner - IA de Débruitage
    **U-Net Autoencoder - PSNR 30.13 dB, SSIM 0.9295**
    
    Uploader une image de document bruitée pour le débruitage automatique.
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="📸 Image bruitée", type="filepath")
            submit_btn = gr.Button("🧹 Débruiter", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="✨ Image débruitée")
    
    submit_btn.click(
        fn=denoise_image,
        inputs=input_image,
        outputs=output_image
    )
    
    gr.Markdown("""
    ---
    ### 🔬 Informations techniques
    - **Modèle** : U-Net Autoencoder
    - **Entraînement** : Epoch 20
    - **PSNR** : 30.13 dB
    - **SSIM** : 0.9295
    - **Framework** : PyTorch + OpenCV
    - **Hébergement** : Hugging Face Spaces (Gratuit)
    """)

# ──────────────────────────────────────────────────────────────
#  LANCEMENT
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)