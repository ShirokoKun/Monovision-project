#!/usr/bin/env python3
"""
MonoVision for HuggingFace Spaces
Clean root-level app.py - Gradio 5.49.0 (latest stable)
Version: 3.0 - LOCKED VERSIONS
"""

import os
import sys

# Set cache directories for HuggingFace Spaces
os.environ["TRANSFORMERS_CACHE"] = os.environ.get("HF_HOME", "/tmp/cache/hf")
os.environ["TORCH_HOME"] = "/tmp/cache/torch"

print("🚀 MONOVISION - HUGGINGFACE SPACES v3.0 (Gradio 5.49.0)")
print("=" * 60)

# Check Gradio version
import gradio as gr
gradio_version = gr.__version__
print(f"📦 Gradio version: {gradio_version}")

# Expect exactly 4.44.1
if gradio_version == "4.44.1":
    print(f"✅ Gradio {gradio_version} - Locked stable version")
elif gradio_version.startswith("4.44"):
    print(f"⚠️  Gradio {gradio_version} - Close enough to 4.44.1, continuing")
else:
    print(f"⚠️  WARNING: Expected Gradio 4.44.1, got {gradio_version}")
    print("   If buttons/CSS don't work, do Factory reboot to reinstall")
print("=" * 60)

# Detect hardware
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU: {gpu_name}")
    print("⚡ Fast mode enabled")
else:
    print("⚠️ CPU mode (free tier)")
    print("⏱️ Expect 30-60s per request")

print("=" * 60)
print("📦 Loading models...")

# Import the interface creator
from gradio_v2_modern import create_modern_interface

print("✅ Models loaded!")
print("🌐 Starting server...")

# Create the Gradio app
demo = create_modern_interface()

# Launch - HuggingFace Spaces configuration
if __name__ == "__main__":
    # Gradio 4.44.1 stable configuration
    demo.launch(
        server_name="0.0.0.0",
        share=False,
        show_error=True
    )
