
import streamlit as st
import tensorflow as tf
import numpy as np

from pathlib import Path
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="DigitVision | Handwritten Digit Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0B1120;
    color: #F8FAFC;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

h1, h2, h3 {
    color: #F8FAFC;
}

p, label {
    color: #CBD5E1;
}

.hero {
    background: linear-gradient(120deg, #172554, #1D4ED8);
    padding: 35px;
    border-radius: 18px;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 38px;
}

.hero p {
    color: #DBEAFE;
    font-size: 17px;
}

.metric-card {
    background-color: #172033;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #263449;
    text-align: center;
}

.metric-value {
    font-size: 30px;
    font-weight: bold;
    color: #60A5FA;
}

.metric-label {
    color: #CBD5E1;
    font-size: 14px;
}

.result-card {
    background-color: #172033;
    border: 1px solid #2563EB;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
}

.result-digit {
    font-size: 85px;
    font-weight: bold;
    color: #60A5FA;
}

.success-message {
    background-color: #123524;
    padding: 18px;
    border-radius: 10px;
    color: #86EFAC;
}

.warning-message {
    background-color: #422006;
    padding: 18px;
    border-radius: 10px;
    color: #FDE68A;
}

.stButton > button {
    width: 100%;
    border-radius: 9px;
    border: 1px solid #3B82F6;
    background-color: #1D4ED8;
    color: white;
    padding: 10px;
}

.stButton > button:hover {
    background-color: #2563EB;
    border-color: #60A5FA;
}

.footer {
    text-align: center;
    color: #94A3B8;
    padding: 25px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "mnist_ann_model.keras"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
    model_loaded = True
except Exception as error:
    model_loaded = False
    st.error(f"Could not load the model: {error}")


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🔢 DigitVision")

st.sidebar.caption("Handwritten Digit Recognition System")

st.sidebar.markdown("---")
st.sidebar.subheader("Navigation")

pages = [
    "Dashboard",
    "Digit Recognition",
    "Model Performance",
    "About"
]

for page in pages:
    if st.sidebar.button(
        page,
        key=f"navigation_{page}"
    ):
        st.session_state.page = page
        st.rerun()

st.sidebar.markdown("---")

if model_loaded:
    st.sidebar.success("ANN Model Loaded")
else:
    st.sidebar.error("Model Not Available")

st.sidebar.markdown("""
### Project Information

**Dataset:** MNIST

**Model:** Artificial Neural Network

**Framework:** TensorFlow / Keras

**Classes:** 0–9
""")


# --------------------------------------------------
# REUSABLE PAGE HEADER
# --------------------------------------------------

def page_header(title, description):
    st.title(title)
    st.write(description)
    st.markdown("---")


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_drawing(canvas_image):
    """
    Convert a canvas drawing into an MNIST-style
    28 x 28 grayscale image.
    """

    image = Image.fromarray(
        canvas_image.astype("uint8")
    ).convert("L")

    image_array = np.array(image)

    # Identify the white drawing pixels.
    threshold = 30
    binary_image = image_array > threshold

    if not np.any(binary_image):
        return None

    # Find the digit boundaries.
    rows = np.where(np.any(binary_image, axis=1))[0]
    cols = np.where(np.any(binary_image, axis=0))[0]

    top, bottom = rows[0], rows[-1] + 1
    left, right = cols[0], cols[-1] + 1

    cropped = image.crop((left, top, right, bottom))

    # Resize while preserving the original proportions.
    cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Create a 28 x 28 black background.
    final_image = Image.new("L", (28, 28), 0)

    # Place the digit in the centre.
    x_position = (28 - cropped.width) // 2
    y_position = (28 - cropped.height) // 2

    final_image.paste(
        cropped,
        (x_position, y_position)
    )

    # Normalize pixel values.
    processed = np.array(final_image).astype("float32") / 255.0

    # Add the batch dimension.
    processed = np.expand_dims(processed, axis=0)

    return processed


# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------

def dashboard_page():

    st.markdown("""
    <div class="hero">
        <h1>DigitVision</h1>
        <p>
        An intelligent handwritten digit recognition
        platform powered by Artificial Neural Networks.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Project Overview")

    st.write("""
    DigitVision uses a trained Artificial Neural Network
    to recognize handwritten digits from 0 to 9.

    Users can draw a digit, submit it for prediction,
    and view the model's confidence scores.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">97.65%</div>
            <div class="metric-label">Test Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">60,000</div>
            <div class="metric-label">Training Images</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">10</div>
            <div class="metric-label">Digit Classes</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Explore DigitVision")

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            "**Digit Recognition**\n\n"
            "Draw a handwritten digit and receive a prediction."
        )

    with col2:
        st.info(
            "**Model Performance**\n\n"
            "Explore the model's evaluation results."
        )

    st.markdown("---")

    st.write("**Model Status:**")

    if model_loaded:
        st.success("The trained ANN model is ready.")
    else:
        st.error("The model could not be loaded.")


# --------------------------------------------------
# DIGIT RECOGNITION PAGE
# --------------------------------------------------

def recognition_page():

    page_header(
        "Digit Recognition Studio",
        "Draw a digit between 0 and 9 in the canvas below."
    )

    st.info(
        "Use white strokes on the black canvas. "
        "Draw one digit clearly and click Recognize Digit."
    )

    col1, col2 = st.columns([1, 1])

    with col1:

        st.subheader("Drawing Canvas")

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=18,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=280,
            height=280,
            drawing_mode="freedraw",
            return_image_data=True,
            key=f"digit_canvas_{st.session_state.canvas_key}"
        )

        predict_button = st.button(
            "Recognize Digit",
            type="primary"
        )

        clear_button = st.button("Clear Canvas")

        if clear_button:
            st.session_state.canvas_key += 1
            st.rerun()

    with col2:

        st.subheader("Recognition Result")

        if predict_button:

            if canvas_result.image_data is None:

                st.warning("Please draw a digit first.")

            elif not model_loaded:

                st.error("The ANN model is unavailable.")

            else:

                processed_image = preprocess_drawing(
                    canvas_result.image_data
                )

                if processed_image is None:

                    st.warning("The canvas is empty. Draw a digit first.")

                else:

                    predictions = model.predict(
                        processed_image,
                        verbose=0
                    )[0]

                    predicted_digit = int(
                        np.argmax(predictions)
                    )

                    confidence = float(
                        np.max(predictions)
                    )

                    sorted_scores = np.sort(predictions)

                    second_highest = float(
                        sorted_scores[-2]
                    )

                    margin = confidence - second_highest

                    # Confidence-based rejection.
                    # These thresholds are starting values
                    # and should be calibrated through testing.
                    confidence_threshold = 0.70
                    margin_threshold = 0.15

                    st.image(
                        processed_image[0],
                        caption="Processed 28 x 28 image",
                        width=180,
                        clamp=True
                    )

                    if (
                        confidence < confidence_threshold
                        or margin < margin_threshold
                    ):

                        st.markdown("""
                        <div class="warning-message">
                            <h3>Not a recognizable digit</h3>
                            The model is not sufficiently confident
                            that this drawing represents a digit.
                            Please try drawing it again.
                        </div>
                        """, unsafe_allow_html=True)

                    else:

                        st.markdown(f"""
                        <div class="result-card">
                            <p>Predicted Digit</p>
                            <div class="result-digit">
                                {predicted_digit}
                            </div>
                            <h3>{confidence * 100:.2f}% Confidence</h3>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("---")

                    st.subheader("Class Probabilities")

                    for digit, probability in enumerate(predictions):

                        st.write(
                            f"Digit {digit}: "
                            f"{probability * 100:.2f}%"
                        )

                        st.progress(
                            float(probability)
                        )

        else:

            st.write(
                "Your prediction will appear here after "
                "you draw and submit a digit."
            )


# --------------------------------------------------
# MODEL PERFORMANCE PAGE
# --------------------------------------------------

def performance_page():

    page_header(
        "Model Performance",
        "Evaluation results from the trained MNIST ANN model."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Test Accuracy",
            "97.65%"
        )

    with col2:
        st.metric(
            "Test Loss",
            "0.0788"
        )

    with col3:
        st.metric(
            "Test Samples",
            "10,000"
        )

    st.markdown("---")

    st.subheader("Model Architecture")

    architecture = {
        "Layer": [
            "Flatten",
            "Dense",
            "Dropout",
            "Dense",
            "Dense"
        ],
        "Configuration": [
            "28 x 28 input",
            "128 neurons, ReLU",
            "30% dropout",
            "64 neurons, ReLU",
            "10 neurons, Softmax"
        ]
    }

    st.table(architecture)

    st.subheader("Training Configuration")

    st.write("""
    - Optimizer: Adam
    - Loss Function: Categorical Crossentropy
    - Evaluation Metric: Accuracy
    - Epochs: 10
    - Batch Size: 128
    - Validation Split: 10%
    - Dataset: MNIST
    """)

    st.markdown("---")

    st.subheader("Classification Results")

    report = {
        "Digit": list(range(10)),
        "Precision": [
            0.9739, 0.9834, 0.9833, 0.9762, 0.9766,
            0.9774, 0.9721, 0.9689, 0.9732, 0.9789
        ],
        "Recall": [
            0.9908, 0.9930, 0.9700, 0.9733, 0.9786,
            0.9686, 0.9823, 0.9698, 0.9692, 0.9673
        ],
        "F1 Score": [
            0.9823, 0.9882, 0.9766, 0.9747, 0.9776,
            0.9730, 0.9772, 0.9694, 0.9712, 0.9731
        ]
    }

    st.dataframe(
        report,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "These are the results from the original test-set evaluation."
    )


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

def about_page():

    page_header(
        "About DigitVision",
        "Understanding the technology behind the application."
    )

    st.subheader("What is DigitVision?")

    st.write("""
    DigitVision is a handwritten digit recognition
    application developed using an Artificial Neural Network.

    The model was trained using the MNIST dataset,
    which contains grayscale images of handwritten digits
    from 0 to 9.
    """)

    st.subheader("How It Works")

    steps = [
        "A user draws a handwritten digit.",
        "The drawing is converted into a grayscale image.",
        "The image is resized to 28 x 28 pixels.",
        "Pixel values are normalized between 0 and 1.",
        "The trained ANN processes the image.",
        "The application displays the prediction and confidence."
    ]

    for index, step in enumerate(steps, start=1):
        st.write(f"**{index}.** {step}")

    st.markdown("---")

    st.subheader("Technologies Used")

    st.write("""
    - Python
    - TensorFlow and Keras
    - NumPy
    - Streamlit
    - Streamlit Drawable Canvas
    - Pillow
    - MNIST Dataset
    """)

    st.subheader("Important Note")

    st.write("""
    This application is designed to classify handwritten
    digits, not general letters or symbols.

    Its rejection mechanism uses prediction confidence
    and the difference between the two highest class
    probabilities. It is a practical safeguard, not a
    guaranteed letter detector.
    """)


# --------------------------------------------------
# PAGE ROUTING
# --------------------------------------------------

if st.session_state.page == "Dashboard":

    dashboard_page()

elif st.session_state.page == "Digit Recognition":

    recognition_page()

elif st.session_state.page == "Model Performance":

    performance_page()

elif st.session_state.page == "About":

    about_page()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    DigitVision | Handwritten Digit Recognition
    <br>
    Artificial Neural Network Project
</div>
""", unsafe_allow_html=True)
