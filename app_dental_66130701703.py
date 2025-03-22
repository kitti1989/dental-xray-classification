import ultralytics
from ultralytics import solutions
import gdown

url = "https://drive.google.com/uc?id=1bcge0W2AT9KWRlGXg-S4BS75cCA4TXvw"

model = 'yolo.pt'
gdown.download(url, model, quiet=False)

inf = solutions.Inference(
    model=model,  # you can use any model that Ultralytics support, i.e. YOLO11, or custom trained model
)

inf.inference()
