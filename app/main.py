import os
import io
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from ben2 import AutoModel
from PIL import Image
import torch

# ===== CRITICAL MEMORY OPTIMIZATIONS =====
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# ===== MODEL INITIALIZATION =====
device = torch.device("cpu")
model = AutoModel.from_pretrained("PramaLLC/BEN2").to(device).eval()
model.requires_grad_(False)  # Disable gradients

app = Flask(__name__)
CORS(app)

@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    if "image" not in request.files:
        return jsonify({"error": "Field 'image' missing"}), 400

    try:
        # Load and validate image
        img = Image.open(request.files["image"].stream)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Downscale to prevent OOM
        max_size = 1024
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size))
        
        # Process with memory safety
        with torch.inference_mode():
            result = model.inference(img)
        
        # Stream response
        buf = io.BytesIO()
        result.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))