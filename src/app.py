import streamlit as st
from PIL import Image
import numpy as np
import pickle
# from tensorflow.keras.models import load_model
import os
# Load all models

st.write("Current Working Directory:", os.getcwd())
st.write("Files in Directory:", os.listdir("models"))

@st.cache_data
def load_all_models():
    def load_models(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)
    

    models = {
        "Logistic Regression": load_models("models/logistic_regression_model.pkl"),
        "Random Forest": load_models("models/random_forest_model.pkl"),
        "SVM": load_models("models/svm_model.pkl"),
        # "CNN": load_model_tf("models/cnn_model.h5"),
    }
    return models

# Utility functions
def predict_with_model(model, X):
    if hasattr(model, "predict"):
        return model.predict(X)
    else:
        raise ValueError("The selected model does not support prediction.")

def preprocess_image(image):
    """
    Preprocess the image for the model.
    - Resize to 28x28 if required.
    - Convert to grayscale if required.
    - Normalize pixel values to [0, 1].
    """
    image = image.resize((28, 28)).convert("L")
    image_array = np.array(image) / 255.0
    return image_array.flatten()  # Flatten for traditional models

def load_example_images():

    examples_dir = "examples"
    example_images = []
    example_labels = []
    if os.path.exists(examples_dir):
        for filename in os.listdir(examples_dir):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                example_images.append(os.path.join(examples_dir, filename))
                example_labels.append(filename.split(".")[0])  # Assuming label is part of the filename
    return example_images, example_labels

def user_inputs():
    """
    Allow user to choose between uploading an image or selecting an example.
    """
    st.write("### Choose Input Method")
    input_method = st.radio("Select input method:", ["Upload an Image", "Use an Example Image"])

    if input_method == "Upload an Image":
        # File upload
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)
            return image

    elif input_method == "Use an Example Image":
        # Load and display example images
        example_images, example_labels = load_example_images()
        selected_example = st.selectbox("Select an example:", example_labels)
        if selected_example:
            selected_image_path = next((img for img, label in zip(example_images, example_labels) if label == selected_example), None)
            if selected_image_path:
                image = Image.open(selected_image_path)
                st.image(image, caption=f"Example: {selected_example}", width=300)
                return image
    return None
# Main app
st.title("ASL Gesture Classification")
st.write("##### Predict sign language using different models")

# Load models
models = load_all_models()

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_choice = st.sidebar.radio(
    "Choose a model for classification:",
    list(models.keys())
)
st.sidebar.write(f"### Selected Model: **{model_choice}**")

# Input section
input_data = user_inputs()

# Predict button
if st.button("Predict Sign Language"):
    if input_data:
        # Preprocess the image
        processed_input = preprocess_image(input_data).reshape(1, -1)  # Reshape for model input
        
        # Get the selected model
        selected_model = models[model_choice]
        
        # Predict
        prediction = predict_with_model(selected_model, processed_input)
        
        
        st.write(f"### Predicted Class: **{chr(int(prediction[0]) + 65)}**")
    else:
        st.write("Please upload an image!")
