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

API de débruitage de documents avec U-Net (PSNR 30.13 dB, SSIM 0.9295)

## Performance
- PSNR : 30.13 dB
- SSIM : 0.9295
- Epoch : 20

## Utilisation
```python
from gradio_client import Client
client = Client("mohamedamirlehyani/smart-scanner-api")
result = client.predict(image_input="photo.jpg", api_name="/debruiter")