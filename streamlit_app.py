import streamlit as st
from PIL import Image
import numpy as np
import cv2
from model_inference import KidneyStoneDetectionModel

# Streamlit app content
st.set_page_config(
    page_title="Kidney Stone Detection using YOLOv8",
    page_icon="download.jpeg",
    layout="wide"
)

# Header
st.header("Team SR2")
st.image("download.jpeg", width=40, caption="SR2 Logo")

# Main content
with st.container():
    st.subheader("Choose Detection Type:")
    detection_type = st.selectbox("", ["Kidney Stone Detection"])

    st.subheader("Upload Image:")
    uploaded_image = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

    if st.button("Analyze Image"):
        if uploaded_image:
            image = Image.open(uploaded_image)
            image_np = np.array(image)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Load and run model inference
            model = KidneyStoneDetectionModel(model_path="./ks_detection.pt")
            model.run_inference(image=image_np)
            annotated_image, severity = model.annotate_image_with_severity(image=image_np)

            # Display images
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            st.image(annotated_image, caption="Processed Image", use_column_width=True)

            # Display severity
            st.subheader("Risk Analysis")
            st.write(f"Severity Level: **{severity}**")

    # Result section
    result = st.empty()

# Footer
st.markdown("---")
