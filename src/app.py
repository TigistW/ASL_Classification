import streamlit as st
from PIL import Image
import numpy as np
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.datasets import mnist

@st.cache_data
def load_all_models():
    def load_models(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)
    
    models = {
        "Logistic Regression": load_models("../models/logistic_regression_model.pkl"),
        "Random Forest": load_models("../models/random_forest_model.pkl"),
        "SVM": load_models("../models/svm_model.pkl"),
        "CNN": load_models("../models/cnn_model.pkl"),
        
        
    }
    return models

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
    
    return image_array.flatten()

def load_example_images():
    examples_dir = "examples"
    example_images = []
    example_labels = []
    if os.path.exists(examples_dir):
        for filename in os.listdir(examples_dir):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                example_images.append(os.path.join(examples_dir, filename))
                example_labels.append(filename.split(".")[0])
    return example_images, example_labels

def user_inputs():
    """
    Allow user to choose between uploading an image, selecting an example, or using the webcam.
    """
    st.write("### Choose Input Method")
    input_method = st.radio("Select input method:", ["Upload an Image", "Use an Example Image", "Capture from Webcam"])

    if input_method == "Upload an Image":
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)
            return image

    elif input_method == "Use an Example Image":
        example_images, example_labels = load_example_images()
        selected_example = st.selectbox("Select an example:", example_labels)
        if selected_example:
            selected_image_path = next((img for img, label in zip(example_images, example_labels) if label == selected_example), None)
            if selected_image_path:
                image = Image.open(selected_image_path)
                st.image(image, caption=f"Example: {selected_example}", width=300)
                return image

    elif input_method == "Capture from Webcam":
        captured_image = st.camera_input("Capture an image")
        if captured_image is not None:
            image = Image.open(captured_image)
            st.image(image, caption="Captured Image", width=300)
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
    # image_array = image_array.reshape(-1, 28, 28)
    selected_model = models[model_choice]
    
    processed_input = preprocess_image(input_data)
    
    if selected_model == models["CNN"]:
        input_data = processed_input.reshape(1, 28, 28, 1)
        st.write(input_data.shape)
        prediction = predict_with_model(selected_model, input_data)
        st.write(prediction)
        predicted_class_idx = np.argmax(prediction)
        st.write(f"### Predicted Class: **{chr(predicted_class_idx + 65)}**")
    else:
        input_data = np.array(processed_input).reshape(1, -1)
        st.write(input_data)


        prediction = predict_with_model(selected_model, input_data)
        st.write(prediction)
        
        st.write(f"### Predicted Class: **{chr(int(prediction[0]) + 65)}**")
else:
    st.write("Please provide an input image!")
