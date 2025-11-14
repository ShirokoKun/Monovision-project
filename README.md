---
title: MonoVision Image Recognition
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.49.0"
app_file: app.py
pinned: false
python_version: "3.10"
---

# 🤖 MonoVision AI - Advanced Image Recognition Chatbot

Multi-modal AI system combining computer vision and natural language processing.

## Features

- 🖼️ **Image Captioning** - BLIP-base model
- 🔍 **Object Detection** - YOLOv8-nano
- 🧠 **Vision Embeddings** - CLIP ViT-B/32
- 💬 **Natural Language** - Flan-T5 + Context-aware prompts
- 📊 **Production Dashboard** - Real-time metrics

## Usage

1. Upload an image
2. Choose processing mode:
   - **Fast**: Quick responses (Flan-T5)
   - **Balanced**: Better quality (Phi-2) 
   - **Rich**: Best quality (API models)
3. Ask questions about the image!

## Performance

- **CPU (Free tier)**: 30-60 seconds per request
- **GPU ($7/month)**: 3-5 seconds per request

Upgrade to GPU tier for 10x faster responses!

## Architecture

- **Vision**: BLIP + CLIP + YOLOv8
- **Language**: Flan-T5-Small (Tier 1)
- **UI**: Gradio 5.49.0 (latest)
- **Cache**: SQLite + in-memory

Built by [KunShiroko](https://github.com/ShirokoKun)
