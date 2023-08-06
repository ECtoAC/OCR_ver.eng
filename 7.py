import streamlit as st
from PIL import Image
import pytesseract
from pytesseract import Output
import cv2
import numpy as np
from docx import Document
from io import BytesIO
from transformers import pipeline
import torch

# Streamlit settings
st.set_page_config(page_title="OCR App", page_icon=None, layout='centered', initial_sidebar_state='auto')

model_name = "microsoft/trocr-base-handwritten"
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache(allow_output_mutation=True)
def load_model(model_name):
    return pipeline("text-generation-ocr", model=model_name, device=0 if device == "cuda" else -1)

try:
    st.title("OCR App")
    st.write("This application converts the text in images to a Word document.")
    image_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])

    if image_file is not None:
        image = Image.open(image_file)
        img_array = np.array(image)
        st.image(image, use_column_width=True)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        d = pytesseract.image_to_data(img_cv, output_type=Output.DICT)

        # 全てのテキストを包含する最小の矩形を求め、画像をその矩形に切り抜く
        word_boxes = []
        for i in range(len(d['text'])):
            if int(d['conf'][i]) > 60:  # confidenceが60以上のもののみを対象とする
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                word_boxes.append(pytesseract.Output.DICT(x=x, y=y, w=w, h=h))

        tolerance = 10
        crop_imgs = []
        crop_dict = {"xmin": 0, "xmax": img_cv.shape[1], "ymin": None, "ymax": None}

        if word_boxes:  # word_boxesが空でないことを確認
            reference_y = word_boxes[0].position[0][1]

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
        else:
            st.write("No text could be detected in the image.")

        # 切り抜いた画像をOCRで処理し、Wordファイルに出力
        doc = Document()
        paragraphs = []  # パラグラフを一度に追加するためのリスト
        model = load_model(model_name)
        for i, crop_dict in enumerate(crop_imgs):
            cropped_image = img_cv[crop_dict["ymin"]:crop_dict["ymax"], crop_dict["xmin"]:crop_dict["xmax"]]
            if cropped_image.size > 0:  # 切り取られた画像が空でない場合のみ処理
                image = Image.fromarray(cropped_image).convert("RGB")
                generated_ids = model.generate(processor(image, return_tensors="pt").pixel_values)

                # generated_idsが空でないことを確認
                if generated_ids:
                    S = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    paragraphs.append(S)
                else:
                    st.write("No text could be detected in the cropped image.")

        # crop_imgsが空でないことを確認
        if crop_imgs:
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
        else:
            st.write("No cropped images to process.")
except Exception as e:
    st.write(f"An error occurred: {str(e)}")
