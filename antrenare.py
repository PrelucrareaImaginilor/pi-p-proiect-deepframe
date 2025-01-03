from ultralytics import YOLO


# Load a model
model = YOLO("best.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="VisDrone.yaml", epochs=80, imgsz=640, device="cuda", workers=0)