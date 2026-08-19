
---

## 📄 **3. README.md (CORRIGÉ)**

```md
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

## Performance
- **PSNR** : 30.13 dB
- **SSIM** : 0.9295
- **Epoch** : 20

## Bruits traités
- Ombres noires, marron, violet, mauve, rose
- Taches d'eau et d'huile (tache œil)
- Fissures graves
- Luminosité très forte

## Utilisation API
```python
from gradio_client import Client
client = Client("amirlahyani/smart-scanner-api")
result = client.predict(image_input="photo.jpg", api_name="/debruiter")