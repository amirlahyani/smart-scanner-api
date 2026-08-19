---
title: Smart Scanner API
emoji: 📄
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
python_version: "3.10"
---

# Smart Scanner API
API de débruitage de documents basée sur un U-Net résiduel avec Channel Attention.

**Performance :**
- PSNR : 30.13 dB
- SSIM : 0.9295
- Entraînement : Epoch 20

Le modèle réduit :
- ombres noires, marron, violet, mauve, rose
- taches d'eau et d'huile
- fissures graves
- luminosité forte