from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from ben2 import BEN_Base
from PIL import Image
import io, torch

app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BEN_Base.from_pretrained("PramaLLC/BEN2").to(device).eval()

# ✅ Home route
@app.get("/")
def home():
    return {"message": "You are connected"}

# 🎯 Background removal route
@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    fg = model.inference(img, refine_foreground=False)
    buf = io.BytesIO()
    fg.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

