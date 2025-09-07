import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2 # For blending images

# --- Utility Functions ---

# Function to apply colormap to a grayscale image
def apply_color_map(image_pil, cmap_name='jet'):
    """
    Apply a matplotlib colormap to a grayscale PIL Image and return an RGB numpy array.
    image_pil: PIL Image (will be converted to grayscale if not)
    cmap_name: Colormap name (e.g., 'jet', 'gray', 'viridis')
    """
    image_gray = image_pil.convert('L') # Ensure grayscale
    img_np_gray = np.array(image_gray)

    # Normalize pixel values to [0, 1]
    img_norm = (img_np_gray - img_np_gray.min()) / (img_np_gray.max() - img_np_gray.min() + 1e-7) # Add epsilon to avoid division by zero

    colormap = cm.get_cmap(cmap_name)
    img_colored = colormap(img_norm)[:, :, :3] # Get RGB channels, ignore alpha

    img_colored_uint8 = (img_colored * 255).astype(np.uint8)
    return img_colored_uint8

# Function to blend two images
def blend_images(img1_np, img2_np, alpha):
    """
    Blends two numpy images (same size) using an alpha value.
    img1_np: Base image (e.g., original grayscale converted to RGB)
    img2_np: Overlay image (e.g., colormapped image)
    alpha: Blending factor (0.0 to 1.0), where 0.0 is all img1_np, 1.0 is all img2_np.
    """
    return cv2.addWeighted(img1_np, 1 - alpha, img2_np, alpha, 0)


# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded" # Changed to expanded for settings
)

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
        color: #e0e0e0;
    }
    .main {
        background-color: #1a1a2e; /* Darker blue/purple */
        color: #e0e0e0;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #0f3460; /* Darker blue for buttons */
        color: white;
        border-radius: 12px;
        padding: 10px 25px;
        font-weight: 600;
        font-size: 16px;
        transition: background-color 0.3s ease, transform 0.2s ease;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .stButton>button:hover {
        background-color: #1b5299; /* Lighter blue on hover */
        transform: translateY(-2px);
    }
    .stFileUploader {
        background-color: #2e2e4e; /* Slightly lighter dark */
        border-radius: 12px;
        padding: 15px;
        border: 1px dashed #4a4a6e;
    }
    .stFileUploader label {
        color: #ffffff;
        font-weight: 600;
    }
    h1, h2, h3 {
        color: #b0c4de; /* Lighter blue for headers */
        font-weight: 700;
        margin-bottom: 0.5em;
    }
    .stMetric {
        background-color: #2e2e4e;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .stMetric > div > div:first-child { /* Label */
        color: #a0a0a0;
        font-size: 0.5em;
        font-weight: 250;
    }
    .stMetric > div > div:nth-child(2) { /* Value */
        color: #ffffff;
        font-size: 1em;
        font-weight: 500;
    }
    .stExpander {
        border: 1px solid #4a4a6e;
        border-radius: 12px;
        padding: 10px;
        background-color: #2e2e4e;
    }
    .stExpander > div > div > p {
        color: #d0d0d0;
    }
    .css-1d391kg { /* Target for selectbox/slider labels */
        color: #e0e0e0;
        font-weight: 500;
    }
    /* Custom scrollbar for better aesthetics */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #2e2e4e;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: #0f3460;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #1b5299;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Model Settings ---
model_key = 'efficientnetv2'
IMG_SIZE = (224, 224)

# --- Load Model with Caching ---
@st.cache_resource
def load_model():
    model_path = f'{model_key}_brain_tumor_classifier.h5'
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at '{model_path}'. Please ensure the model file is in the same directory as the app.")
        st.stop()
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Please check the model file integrity.")
        st.stop()

model = load_model()

# --- Load Class Names ---

DATA_DIR = 'brain-tumor-dataset' # <-- Suggestion: Create a 'data' folder
CLASS_NAMES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
if not CLASS_NAMES:
     st.error(f"No class subdirectories found in '{DATA_DIR}'. Please ensure your dataset directory contains subfolders for each class.")
     st.stop()

# --- Main App Title and Description ---
st.markdown("""
# 🧠 Brain Tumor Classifier
Upload a brain MRI image to classify the tumor type using a deep learning model.
""", unsafe_allow_html=True)

with st.expander("ℹ️ How to use this app & Model Info", expanded=True):
    st.markdown("""
    1.  **Upload Image:** Use the file uploader below to select a brain MRI image (PNG, JPG).
        Alternatively, click "Try Sample Image" to see how it works!
    2.  **View Visualization:** Observe the original image alongside a colormap visualization.
        Adjust the colormap and blending using the controls in the sidebar.
    3.  **Get Prediction:** The model will analyze the image and display the predicted tumor type, confidence score, and top class probabilities.

    ---
    ### Model Details:
    *   **Architecture:** EfficientNetV2 (pretrained on ImageNet, fine-tuned on brain MRI dataset).
    *   **Input Image Size:** 224 x 224 pixels.
    *   **Classes:** Recognizes the following types (  44 Brain Tumor Types):
    """)
    # --- THIS IS THE MODIFIED SECTION FOR CLASS DISPLAY ---
    # Dynamically display all classes in multiple columns for professional readability
    num_cols = 3 # You can adjust this number (e.g., 2, 3, 4) based on screen size and desired density
    cols = st.columns(num_cols)
    for i, class_name in enumerate(CLASS_NAMES):
        with cols[i % num_cols]: # Distribute items across columns
            st.markdown(f"- **{class_name}**")
    st.markdown("") # Add a newline after the columns for better spacing
    # --- END OF MODIFIED SECTION ---

# --- Sidebar for Visualization Settings ---
st.sidebar.header("🎨 Visualization Settings")

colormap_options = [
    'jet', 'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    'hot', 'bone', 'gray', 'gist_heat', 'spring', 'summer', 'autumn', 'winter'
]
selected_cmap = st.sidebar.selectbox("Choose Colormap", colormap_options, index=0)
blend_factor = st.sidebar.slider("Colormap Blending (Alpha)", 0.0, 1.0, 0.7, 0.05)


# --- Image Upload and Sample Image Logic ---
uploaded_file = st.file_uploader("📤 Choose an MRI image...", type=["jpg", "jpeg", "png"])

sample_img_path = "sample_mri.jpg" # Make sure you have a sample_mri.jpg in your project root!
# You could also use a base64 encoded string for a fully self-contained script:
# import base64
# with open("sample_mri.jpg", "rb") as f:
#     sample_img_b64 = base64.b64encode(f.read()).decode()
# sample_img_bytes = base64.b64decode(sample_img_b64)
    # --- Footer ---

if st.button("✨ Try Sample Image"):
    if os.path.exists(sample_img_path):
        with open(sample_img_path, "rb") as f:
            uploaded_file = f.read()
        st.session_state.uploaded_file_name = sample_img_path # Store filename for display
        st.success("Sample image loaded!")
    else:
        st.error(f"Sample image not found at '{sample_img_path}'. Please add a 'sample_mri.jpg' file to your app directory.")
        uploaded_file = None # Ensure no file is processed if sample is missing

if uploaded_file is not None:
    # If using bytes from sample_image:
    if isinstance(uploaded_file, bytes):
        image = Image.open(io.BytesIO(uploaded_file)).convert("RGB")
        file_name_display = st.session_state.get('uploaded_file_name', 'sample_mri.jpg')
    else: # Normal file uploader object
        image = Image.open(uploaded_file).convert("RGB")
        file_name_display = uploaded_file.name

    st.subheader(f"✨ Image: {file_name_display}")
    col_img1, col_img2 = st.columns([1, 1])

    with col_img1:
        st.image(image, caption="Original Image", use_container_width =True)

    # Prepare original image for blending (convert to grayscale then to RGB)
    original_gray_rgb = np.array(image.convert('L').convert('RGB'))

    # Apply colormap
    colormapped_img_np = apply_color_map(image, cmap_name=selected_cmap)

    # Blend original grayscale with colormapped image
    blended_img = blend_images(original_gray_rgb, colormapped_img_np, blend_factor)

    with col_img2:
        st.image(blended_img, caption=f"Colormap Visualization ({selected_cmap}, Blended)", use_container_width =True)

    # Prepare image for model prediction
    img_array = np.array(image.resize(IMG_SIZE))
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    if model_key == 'efficientnetv2':
        img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    # Add elif for other model_keys if you used different preprocessing

    with st.spinner("🔍 Analyzing image... Please wait."):
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0]) # Apply softmax to get probabilities
        predicted_class_index = np.argmax(score)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = 100 * np.max(score)

    # --- Display Prediction and Probabilities ---
    st.subheader("🩺 Prediction Result")

    col_pred, col_prob = st.columns([1, 2])

    with col_pred:
        st.metric(label="Predicted Class", value=predicted_class_name)
        st.metric(label="Confidence", value=f"{confidence:.2f} %")

    with col_prob:
        st.write("### 📊 Top Class Probabilities:")

        # Sort classes by probability in descending order
        sorted_indices = np.argsort(score.numpy())[::-1]
        top_indices = sorted_indices[:min(len(CLASS_NAMES), 5)] # Show top 5 or fewer if less classes

        top_class_names = [CLASS_NAMES[i] for i in top_indices]
        top_scores = score.numpy()[top_indices]

        # Create a horizontal bar chart
        fig, ax = plt.subplots(figsize=(7, max(3, len(top_class_names) * 0.8))) # Dynamic height
        bar_colors = plt.cm.Blues_r(top_scores / np.max(top_scores) if np.max(top_scores) > 0 else top_scores + 1e-9)
        bars = ax.barh(range(len(top_class_names)), top_scores, color=bar_colors)

        ax.set_yticks(range(len(top_class_names)))
        ax.set_yticklabels(top_class_names, fontsize=12)
        ax.set_xlabel("Probability", fontsize=12)
        ax.set_xlim([0, 1])
        ax.invert_yaxis() # Highest probability at the top

        # Add percentage labels to bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{width:.1%}", va='center', fontsize=10, color='white') # White text for contrast

        plt.title("Class Probabilities", color='#b0c4de', fontsize=14)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        fig.patch.set_facecolor('#2e2e4e') # Match plot background to metric background
        ax.set_facecolor('#2e2e4e')

        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")


    st.info("Disclaimer: This application is for educational and demonstrative purposes only and should not be used for medical diagnosis.")

else:
    st.info("Upload an image or click 'Try Sample Image' to get started!")
st.markdown(
        """
        <div class="footer">
            <p>
                Developed by <strong>Ahmad Hudhud</strong> | Created with 
                <a href="https://www.tensorflow.org/" target="_blank">TensorFlow</a> and 
                <a href="https://streamlit.io/" target="_blank">Streamlit</a>
                <br>
                <a href="https://github.com/AhmadHudhud83/BrainTumorNet-44" target="_blank">GitHub Repository</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )