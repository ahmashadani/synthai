from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import os
import shutil
import json
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Try loading SAM model (optional)
try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SamPredictor, sam_model_registry = None, None
    SAM_AVAILABLE = False

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Directories
UPLOAD_DIR, LABEL_DIR, VIS_DIR, EXPORT_DIR = "uploads", "labels", "visuals", "exports"
for d in [UPLOAD_DIR, LABEL_DIR, VIS_DIR, EXPORT_DIR]: os.makedirs(d, exist_ok=True)

# Load YOLOv8
yolo_model = YOLO("yolov8n.pt")

# Load SAM if available
if SAM_AVAILABLE:
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
    sam.to("cuda" if torch.cuda.is_available() else "cpu")
    sam_predictor = SamPredictor(sam)
else:
    sam_predictor = None

@app.get("/")
def root(): return {"message": "SynthLabel API live."}

@app.post("/upload/")
async def upload_image(file: UploadFile):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f: shutil.copyfileobj(file.file, f)
    img = cv2.imread(file_path)
    results = yolo_model(img)
    detections = results[0].boxes.xyxy.cpu().numpy()
    overlay = img.copy()
    label_data = []

    for i, box in enumerate(detections):
        x1, y1, x2, y2 = map(int, box)
        color = np.random.randint(0, 255, (3,)).tolist()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        mask = None
        if sam_predictor:
            sam_predictor.set_image(img)
            transformed = sam_predictor.transform.apply_boxes_torch(torch.tensor([box]), img.shape[:2]).to(sam_predictor.device)
            masks, _, _ = sam_predictor.predict_torch(boxes=transformed, multimask_output=False)
            mask = masks[0].cpu().numpy().astype(np.uint8).tolist()
            overlay = np.where(np.expand_dims(masks[0].cpu().numpy(), -1), overlay * 0.5 + np.array(color) * 0.5, overlay).astype(np.uint8)
        label_data.append({"bbox": box.tolist(), "label": "object", "mask": mask or []})

    vis_path = os.path.join(VIS_DIR, f"{file.filename}_vis.jpg")
    label_path = os.path.join(LABEL_DIR, f"{file.filename}.json")
    cv2.imwrite(vis_path, overlay)
    with open(label_path, "w") as f: json.dump(label_data, f)

    return JSONResponse({
        "filename": file.filename,
        "num_detections": len(label_data),
        "label_path": label_path,
        "visual_path": f"/visual/{file.filename}_vis.jpg"
    })

@app.get("/visual/{filename}")
def get_visual(filename: str):
    path = os.path.join(VIS_DIR, filename)
    return FileResponse(path, media_type="image/jpeg") if os.path.exists(path) else JSONResponse(status_code=404, content={"error": "Not found"})

@app.get("/export/{filename}")
def export_yolo(filename: str):
    label_path = os.path.join(LABEL_DIR, f"{filename}.json")
    export_path = os.path.join(EXPORT_DIR, f"{filename}.txt")
    if not os.path.exists(label_path): return JSONResponse(status_code=404, content={"error": "No labels"})

    with open(label_path) as f: labels = json.load(f)
    lines = []
    for obj in labels:
        x1, y1, x2, y2 = obj["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    with open(export_path, "w") as f: f.writelines(lines)
    return FileResponse(export_path, media_type="text/plain")
