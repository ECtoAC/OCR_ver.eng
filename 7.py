
# 必要なモジュールのインポート
import cv2
import os
import numpy as np
from PIL import Image
import pyocr
from docx import Document
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import streamlit as st

# プロセッサとモデルをロード
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
st.title("OCR App")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # OCRツールの設定
    tools = pyocr.get_available_tools()
    tool = tools[0]

    # 行ごとに画像を切り分け
    img_cv = np.array(image)
    word_boxes = tool.image_to_string(
        image,
        lang="eng",
        builder=pyocr.builders.WordBoxBuilder(tesseract_layout=11)
    )
    tolerance = 100
    reference_y = word_boxes[0].position[0][1]
    crop_imgs = []
    crop_dict = {"xmin": 0, "xmax": img_cv.shape[1], "ymin": None, "ymax": None}

    for i, box in enumerate(word_boxes):
        x1, y1 = box.position[0]
        x2, y2 = box.position[1]
        if not (reference_y - tolerance <= y1 <= reference_y + tolerance):
            reference_y = y1
            crop_imgs.append(crop_dict)
            crop_dict = {"xmin": 0, "xmax": img_cv.shape[1], "ymin": None, "ymax": None}

        if reference_y - tolerance <= y1 <= reference_y + tolerance:
            crop_dict["ymin"] = y1 if crop_dict["ymin"] is None else min(crop_dict["ymin"], y1)
            crop_dict["ymax"] = y2 if crop_dict["ymax"] is None else max(crop_dict["ymax"], y2)

    crop_imgs.append(crop_dict)

    # 切り抜いた画像をOCRで処理し、Wordファイルに出力
    doc = Document()
    for i, crop_dict in enumerate(crop_imgs):
        cropped_image = img_cv[crop_dict["ymin"]:crop_dict["ymax"], crop_dict["xmin"]:crop_dict["xmax"]]
        if cropped_image.size > 0:  # 切り取られた画像が空でない場合のみ処理
            cv2.imwrite(f"cropped_image_{i}.png", cropped_image)
            image = Image.open(f"cropped_image_{i}.png").convert("RGB")
            generated_ids = model.generate(processor(image, return_tensors="pt").pixel_values)
            S = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            doc.add_paragraph(S)  # テキストをWordファイルに追加

    # Wordファイルの保存
    doc.save("output.docx")
    st.success("The document has been successfully saved as output.docx")
