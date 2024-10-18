def main():
    from ultralytics import YOLO
    model = YOLO("models/yolov8m-obb.pt", task="obb")  # load a pretrained model (recommended for training)
    results = model.train(data="datasets/data.yaml", epochs=50, imgsz=640, batch=16, project="final-train", device="cuda:0")

if __name__ == '__main__':
    main()