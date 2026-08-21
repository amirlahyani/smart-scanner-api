---
title: Smart Scanner API
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
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

## Endpoints
- `GET /` — Health check
- `POST /full-process` — Débruitage (compatible Flutter)
- `POST /denoise` — Alias

## Utilisation Flutter
```dart
static const String baseUrl = 'https://smart-scanner-api.up.railway.app';
```
