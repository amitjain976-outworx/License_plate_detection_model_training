import torch
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from model import CRNN
from utils import transform, greedy_decode, load_charset, LabelEncoder

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load charset
charset = load_charset("charset.txt")
encoder = LabelEncoder(charset)

blank_idx = len(charset)
num_classes = len(charset) + 1

# Load model
model = CRNN(num_classes)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    img = transform(image)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():

        logits = model(img)

        decoded = greedy_decode(logits, blank_idx)

        text = encoder.decode(decoded[0])

    return {"prediction": text}