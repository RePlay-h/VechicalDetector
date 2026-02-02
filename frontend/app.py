import streamlit as st
import requests
from dotenv import load_dotenv
import os

load_dotenv()
API = os.getenv("API_URL")

st.set_page_config(page_title="VehicleDetector", layout="wide")
st.title("VehicleDetector demo")

models = requests.get(f"{API}/models").json()["models"]
model_name = st.selectbox("Model", models, index=0)

score_thr = st.slider("score_thr", 0.0, 1.0, 0.25, 0.01)
iou_thr = st.slider("nms_iou", 0.1, 0.95, 0.6, 0.01)

file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if file is not None:
    with col1:
        st.image(file.getvalue(), caption="Input", use_container_width=True)

    # render endpoint (image with boxes)
    r = requests.post(
        f"{API}/detect/image_render",
        files={"file": (file.name, file.getvalue(), file.type)},
        data={"model": model_name, "score_thr": score_thr, "iou_thr": iou_thr},
        timeout=120,
    )
    if r.status_code != 200:
        st.error(r.text)
        st.stop()

    with col2:
        st.image(r.content, caption="Output", use_container_width=True)

    # json detections
    rj = requests.post(
        f"{API}/detect/image",
        files={"file": (file.name, file.getvalue(), file.type)},
        data={"model": model_name, "score_thr": score_thr, "iou_thr": iou_thr},
        timeout=120,
    ).json()

    st.subheader("Detections")
    st.write(rj["detections"])