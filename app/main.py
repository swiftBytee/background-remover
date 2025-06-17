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
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Additional memory saving

# ===== MODEL INITIALIZATION =====
device = torch.device("cpu")
try:
    model = AutoModel.from_pretrained("PramaLLC/BEN2").to(device).eval()
    # Freeze model and disable gradients
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

# ===== FLASK APP SETUP =====
app = Flask(__name__)
CORS(app, resources={
    r"/remove-bg": {
        "origins": ["*"],
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})

# ===== HELPER FUNCTIONS =====
def validate_image(file_stream):
    """Validate and optimize the input image"""
    try:
        img = Image.open(file_stream)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Downscale large images to prevent OOM
        max_dimension = 1024
        if max(img.size) > max_dimension:
            img.thumbnail((max_dimension, max_dimension))
        
        return img
    except Exception as e:
        raise ValueError(f"Invalid image: {str(e)}")

# ===== ROUTE HANDLER =====
@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    if "image" not in request.files:
        return jsonify({"error": "Field 'image' missing"}), 400

    try:
        # Process image with memory safety
        input_image = validate_image(request.files["image"].stream)
        
        # Perform background removal
        with torch.no_grad():  # Disable gradient calculation
            result = model.inference(input_image)
        
        # Prepare response
        output_buffer = io.BytesIO()
        result.save(output_buffer, format="PNG", optimize=True)
        output_buffer.seek(0)
        
        # Clear memory
        del input_image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return send_file(output_buffer, mimetype="image/png")
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except torch.cuda.OutOfMemoryError:
        return jsonify({"error": "Server overloaded, please try a smaller image"}), 413
    except Exception as e:
        app.logger.error(f"Processing error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# ===== HEALTH CHECK =====
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "device": str(device)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))