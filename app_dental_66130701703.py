import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

import gdown

url = "https://drive.google.com/uc?id=1bcge0W2AT9KWRlGXg-S4BS75cCA4TXvw"

model = 'yolo.pt'
gdown.download(url, model, quiet=False)


# โหลดโมเดล YOLO (ใช้ YOLOv8 ที่ pretrained)
model = YOLO(model)  # ใช้ yolov8n ที่เบากว่า

# ส่วนติดต่อผู้ใช้ (UI) บน Streamlit
st.title("🔍 Detecting Dental Health Conditions from Dental X-ray")
st.write("อัปโหลดภาพ แล้วให้ YOLO ตรวจจับวัตถุ")

# อัปโหลดรูปภาพ
uploaded_file = st.file_uploader("📤 เลือกรูปภาพ", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # แปลงเป็นภาพที่สามารถใช้ได้ใน OpenCV
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # ตรวจจับวัตถุ
    results = model(image_np)

    # ดึงภาพที่ตรวจจับแล้ว
    result_image = results[0].plot()  # แสดงผล bounding box บนรูป

    # แสดงภาพที่ตรวจจับแล้ว
    st.image(result_image, caption="ผลลัพธ์การตรวจจับ", use_column_width=True)

    # แสดงข้อมูลที่ตรวจจับได้
    st.subheader("🔹 รายการวัตถุที่พบ")
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            confidence = box.conf[0].item()
            label = model.names[class_id]
            st.write(f"🛑 {label}: {confidence:.2f}")
