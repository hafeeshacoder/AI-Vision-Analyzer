import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import easyocr
from collections import Counter
import pandas as pd


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Vision Analyzer",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #080714 0%,
        #120d25 50%,
        #080714 100%
    );
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: white !important;
}

p, label {
    color: #cbd5e1 !important;
}

[data-testid="stSidebar"] {
    background: #0d091b;
    border-right: 1px solid #2d2149;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: white !important;
}

[data-testid="stFileUploader"] {
    background: #171027;
    border: 1px dashed #7c3aed;
    border-radius: 15px;
    padding: 10px;
}

.stButton > button {
    background: #7c3aed;
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: bold;
}

.stButton > button:hover {
    background: #9333ea;
    color: white;
}

div[data-testid="metric-container"] {
    background: #171027;
    border: 1px solid #30204f;
    padding: 20px;
    border-radius: 15px;
}

div[data-testid="metric-container"] label {
    color: #a78bfa !important;
}

div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: white !important;
}

hr {
    border-color: #30204f;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD YOLO
# ============================================================

@st.cache_resource
def load_yolo():

    model = YOLO("yolo11n.pt")

    return model


# ============================================================
# LOAD OCR
# ============================================================

@st.cache_resource
def load_ocr():

    reader = easyocr.Reader(
        ["en"],
        gpu=False
    )

    return reader


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("👁️ AI Vision Analyzer")

    st.caption("Computer Vision Intelligence")

    st.divider()

    st.subheader("⚙️ Detection Settings")

    confidence = st.slider(
        "Detection Confidence",
        min_value=0.10,
        max_value=0.95,
        value=0.35,
        step=0.05
    )

    st.caption(
        f"Objects below {confidence:.0%} confidence "
        "will be ignored."
    )

    st.divider()

    st.subheader("🧠 AI Modules")

    object_detection_enabled = st.checkbox(
        "🎯 Object Detection",
        value=True
    )

    ocr_enabled = st.checkbox(
        "🔤 Text Recognition",
        value=True
    )

    opencv_enabled = st.checkbox(
        "🖼️ OpenCV Processing",
        value=True
    )

    st.divider()

    st.subheader("📌 About")

    st.write(
        "AI Vision Analyzer is a computer vision application "
        "that analyzes images using YOLO, EasyOCR and OpenCV."
    )

    st.write("**Features:**")

    st.write("• Object detection")
    st.write("• Object counting")
    st.write("• Confidence scores")
    st.write("• Text recognition")
    st.write("• OpenCV processing")

    st.divider()

    st.subheader("🛠️ Technology")

    st.write("Python")
    st.write("Streamlit")
    st.write("YOLO")
    st.write("EasyOCR")
    st.write("OpenCV")
    st.write("NumPy")


# ============================================================
# HEADER
# ============================================================

st.title("👁️ AI Vision Analyzer")

st.subheader(
    "Computer Vision & AI Application"
)

st.write(
    "Upload an image and let AI detect objects, count them, "
    "recognize text and analyze the image."
)

st.success("🟢 AI SYSTEM ONLINE")

st.divider()


# ============================================================
# FEATURES
# ============================================================

st.header("🚀 Vision Intelligence")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.info(
        "🎯 **Object Detection**\n\n"
        "Identify objects automatically using YOLO."
    )

with col2:

    st.info(
        "🔢 **Smart Counting**\n\n"
        "Count detected objects automatically."
    )

with col3:

    st.info(
        "🔤 **Text Recognition**\n\n"
        "Extract visible text using EasyOCR."
    )

with col4:

    st.info(
        "📊 **AI Analytics**\n\n"
        "View confidence scores and analysis."
    )


st.divider()


# ============================================================
# UPLOAD
# ============================================================

st.header("📤 Upload Image")

st.write(
    "Choose a JPG, JPEG, PNG, WEBP or BMP image."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
        "bmp"
    ]
)


# ============================================================
# NO IMAGE
# ============================================================

if uploaded_file is None:

    st.info(
        "🖼️ Upload an image above to start your AI analysis."
    )

    st.divider()

    st.caption(
        "Built with Python • Streamlit • YOLO • EasyOCR • OpenCV"
    )

    st.stop()


# ============================================================
# IMAGE LOAD
# ============================================================

try:

    image = Image.open(uploaded_file).convert("RGB")

except Exception as error:

    st.error(
        "Unable to read this image."
    )

    st.exception(error)

    st.stop()


image_array = np.array(image)

height, width = image_array.shape[:2]


# ============================================================
# IMAGE INFORMATION
# ============================================================

st.header("📋 Image Information")

info1, info2, info3, info4 = st.columns(4)

with info1:

    st.metric(
        "Width",
        f"{width}px"
    )

with info2:

    st.metric(
        "Height",
        f"{height}px"
    )

with info3:

    st.metric(
        "File Size",
        f"{uploaded_file.size / 1024:.1f} KB"
    )

with info4:

    st.metric(
        "Format",
        uploaded_file.type.split("/")[-1].upper()
    )


st.divider()


# ============================================================
# INITIAL VARIABLES
# ============================================================

detections = []

annotated_image = image_array.copy()

ocr_results = []

gray_image = None


# ============================================================
# YOLO OBJECT DETECTION
# ============================================================

if object_detection_enabled:

    st.header("🎯 Object Detection")

    with st.spinner(
        "Loading YOLO model and detecting objects..."
    ):

        try:

            model = load_yolo()

            results = model.predict(
                source=image_array,
                conf=confidence,
                verbose=False
            )

            result = results[0]

            if result.boxes is not None:

                names = result.names

                for box in result.boxes:

                    class_id = int(
                        box.cls[0].item()
                    )

                    score = float(
                        box.conf[0].item()
                    )

                    coordinates = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                        .astype(int)
                    )

                    detections.append(
                        {
                            "object": names[class_id],
                            "confidence": score,
                            "x1": int(coordinates[0]),
                            "y1": int(coordinates[1]),
                            "x2": int(coordinates[2]),
                            "y2": int(coordinates[3])
                        }
                    )

            try:

                plotted = result.plot()

                annotated_image = cv2.cvtColor(
                    plotted,
                    cv2.COLOR_BGR2RGB
                )

            except Exception:

                annotated_image = image_array.copy()

        except Exception as error:

            st.error(
                "YOLO detection failed."
            )

            st.exception(error)


# ============================================================
# OCR
# ============================================================

if ocr_enabled:

    st.header("🔤 Text Recognition")

    with st.spinner(
        "Reading text from image..."
    ):

        try:

            reader = load_ocr()

            ocr_data = reader.readtext(
                image_array
            )

            for item in ocr_data:

                if len(item) >= 3:

                    ocr_results.append(
                        {
                            "text": str(item[1]),
                            "confidence": float(item[2])
                        }
                    )

        except Exception as error:

            st.error(
                "OCR processing failed."
            )

            st.exception(error)


# ============================================================
# OPENCV
# ============================================================

if opencv_enabled:

    st.header("🖼️ OpenCV Processing")

    try:

        gray_image = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

    except Exception as error:

        st.error(
            "OpenCV processing failed."
        )

        st.exception(error)


# ============================================================
# STATISTICS
# ============================================================

object_counter = Counter(
    item["object"]
    for item in detections
)

total_objects = len(detections)

unique_objects = len(object_counter)

if detections:

    average_confidence = (
        sum(
            item["confidence"]
            for item in detections
        )
        / len(detections)
    )

else:

    average_confidence = 0


total_text = len(ocr_results)


# ============================================================
# MAIN RESULTS
# ============================================================

st.divider()

st.header("📊 AI Analysis Results")

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:

    st.metric(
        "Objects Detected",
        total_objects
    )

with metric2:

    st.metric(
        "Object Classes",
        unique_objects
    )

with metric3:

    st.metric(
        "Average Confidence",
        f"{average_confidence:.1%}"
    )

with metric4:

    st.metric(
        "Text Regions",
        total_text
    )


# ============================================================
# IMAGE COMPARISON
# ============================================================

st.divider()

st.header("🔍 Visual Analysis")

image_col1, image_col2 = st.columns(2)

with image_col1:

    st.subheader("Original Image")

    st.image(
        image,
        use_container_width=True
    )

with image_col2:

    st.subheader("AI Detection")

    if object_detection_enabled:

        st.image(
            annotated_image,
            use_container_width=True
        )

    else:

        st.info(
            "Object detection is disabled."
        )


# ============================================================
# OBJECT SUMMARY
# ============================================================

st.divider()

st.header("🔢 Object Summary")

if object_counter:

    summary_data = []

    for object_name, count in object_counter.most_common():

        related = [
            item
            for item in detections
            if item["object"] == object_name
        ]

        avg_conf = (
            sum(
                item["confidence"]
                for item in related
            )
            / len(related)
        )

        summary_data.append(
            {
                "Object": object_name,
                "Count": count,
                "Average Confidence": f"{avg_conf:.1%}"
            }
        )

    summary_df = pd.DataFrame(
        summary_data
    )

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning(
        "No objects were detected above the selected "
        "confidence threshold."
    )


# ============================================================
# DETECTION DETAILS
# ============================================================

if detections:

    st.divider()

    st.header("📌 Detection Details")

    detection_data = []

    for index, item in enumerate(
        detections,
        start=1
    ):

        detection_data.append(
            {
                "ID": index,
                "Object": item["object"],
                "Confidence": (
                    f"{item['confidence']:.1%}"
                ),
                "X1": item["x1"],
                "Y1": item["y1"],
                "X2": item["x2"],
                "Y2": item["y2"]
            }
        )

    detection_df = pd.DataFrame(
        detection_data
    )

    st.dataframe(
        detection_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# OCR RESULTS
# ============================================================

st.divider()

st.header("🔤 OCR Results")

if ocr_results:

    ocr_data = []

    for index, item in enumerate(
        ocr_results,
        start=1
    ):

        ocr_data.append(
            {
                "ID": index,
                "Detected Text": item["text"],
                "Confidence": (
                    f"{item['confidence']:.1%}"
                )
            }
        )

    ocr_df = pd.DataFrame(
        ocr_data
    )

    st.dataframe(
        ocr_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📝 Extracted Text")

    for item in ocr_results:

        st.write(
            f"**{item['text']}** "
            f"— {item['confidence']:.1%}"
        )

else:

    st.info(
        "No readable text was detected."
    )


# ============================================================
# OPENCV RESULT
# ============================================================

if opencv_enabled and gray_image is not None:

    st.divider()

    st.header("🖼️ OpenCV Result")

    cv_col1, cv_col2 = st.columns(2)

    with cv_col1:

        st.subheader("Grayscale Image")

        st.image(
            gray_image,
            use_container_width=True
        )

    with cv_col2:

        st.subheader("Processing Information")

        st.write(
            "Original image: RGB"
        )

        st.write(
            "Processed image: Grayscale"
        )

        st.write(
            f"Resolution: {width} × {height}"
        )

        st.write(
            "Processing library: OpenCV"
        )


# ============================================================
# DOWNLOAD
# ============================================================

st.divider()

st.header("⬇️ Export Result")

try:

    result_bgr = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_RGB2BGR
    )

    success, encoded = cv2.imencode(
        ".png",
        result_bgr
    )

    if success:

        st.download_button(
            label="⬇️ Download AI Analyzed Image",
            data=encoded.tobytes(),
            file_name="ai_vision_analyzed.png",
            mime="image/png"
        )

except Exception as error:

    st.warning(
        f"Download preparation failed: {error}"
    )


# ============================================================
# FINAL STATUS
# ============================================================

st.divider()

if total_objects > 0 or total_text > 0:

    st.success(
        "✅ Analysis completed successfully!"
    )

else:

    st.warning(
        "⚠️ Analysis completed, but no objects or "
        "text were detected."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Vision Analyzer • Computer Vision & AI Application"
)

st.caption(
    "Built with Python • Streamlit • YOLO • EasyOCR • OpenCV"
)
