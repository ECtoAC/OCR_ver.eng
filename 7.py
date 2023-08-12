import cv2
import numpy as np
from PIL import Image
import pyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import streamlit as st
from io import BytesIO
from docx import Document

# モデルとプロセッサのロードは初回のみ行う
@st.cache(allow_output_mutation=True)
def load_model():
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    return processor, model

try:
    processor, model = load_model()

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
        tolerance = 110
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
        paragraphs = [] # パラグラフを一度に追加するためのリスト
        for i, crop_dict in enumerate(crop_imgs):
            cropped_image = img_cv[crop_dict["ymin"]:crop_dict["ymax"], crop_dict["xmin"]:crop_dict["xmax"]]
            if cropped_image.size > 0:  # 切り取られた画像が空でない場合のみ処理
                image = Image.fromarray(cropped_image).convert("RGB")
                generated_ids = model.generate(processor(image, return_tensors="pt").pixel_values)
                S = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                paragraphs.append(S)

        for para in paragraphs:  # ここで一度にパラグラフを追加
            doc.add_paragraph(para)

        # Wordファイルの保存
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)

        # Wordファイルのダウンロード
        st.download_button(
            label="Download output.docx",
            data=buffer,
            file_name='output.docx',
            mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )
except Exception as e:
    st.write(f"An error occurred: {str(e)}")
