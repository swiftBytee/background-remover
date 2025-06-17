from flask import Flask, request, send_file
from flask_cors import CORS
from ben2 import AutoModel
from PIL import Image
import torch
import io
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# Force CPU usage for free tier
device = torch.device("cpu")
model = AutoModel.from_pretrained("PramaLLC/BEN2").to(device).eval()

app = Flask(__name__)
CORS(app)

@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    if "image" not in request.files:
        return {"error": "Field 'image' missing"}, 400

    try:
        file = request.files["image"]
        image = Image.open(file.stream).convert("RGB")
        result = model.inference(image)

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)