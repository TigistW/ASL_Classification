# **ASL Classification Project**

## **Project Overview**
This project focuses on building a machine learning application to classify American Sign Language (ASL) gestures into their corresponding letters using various machine learning models. The goal is to facilitate the recognition of static ASL gestures using pre-trained models and user-provided inputs, such as uploaded image files.

---

## **Features**
- **User Input Options:**
  - Upload an image of an ASL gesture for classification.
  - Select example images from a predefined set for quick predictions.
- **Model Selection:**
  - Choose from Logistic Regression, Random Forest, and SVM models for prediction.
- **Preprocessing:**
  - Automatic resizing, normalization, and grayscale conversion of images to prepare them for model input.
- **Prediction Results:**
  - Display the predicted ASL letter.

---

## **Project Structure**
```plaintext
.
├── app.py                  # Main Streamlit app script
├── models/
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   └── cnn_model.h5
├── examples/
│   ├── A.jpg
│   ├── B.jpg
│   ├── C.jpg
│   └── ...
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
```

---

## **How to Run the Project**

### **1. Prerequisites**
Ensure you have Python 3.8 or later installed on your system. Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

### **2. Directory Setup**
- Place the pre-trained models in the `models/` directory:
  - `logistic_regression_model.pkl`
  - `random_forest_model.pkl`
  - `svm_model.pkl`
- Place example images in the `examples/` directory with filenames representing their labels (e.g., `A.jpg`, `B.jpg`).

### **3. Run the Application**
Start the Streamlit app:

```bash
streamlit run app.py
```

---

## **Using the Application**

1. **Select a Model:**
   - Use the sidebar to choose a classification model (e.g., Logistic Regression, Random Forest, SVM, or CNN).

2. **Provide Input:**
   - Upload an image of an ASL gesture, or select an example image from the predefined set.

3. **Predict:**
   - Click the **Predict Sign Language** button to classify the gesture.

4. **View Results:**
   - The app displays the predicted ASL letter along with the uploaded or selected image.

---

## **Dependencies**
The project requires the following Python libraries:
- `streamlit`
- `pandas`
- `numpy`

All dependencies are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

---

## **Future Improvements**
- Add support for dynamic gestures (e.g., `J` and `Z`).
- Improve model accuracy with more training data.
- Enable real-time gesture recognition via webcam input.

---

## **Acknowledgments**
- The ASL gesture dataset used in this project is based on [Sign Language MNIST](https://www.kaggle.com/grassknoted/asl-alphabet).
- Inspiration for this application comes from accessibility needs for individuals with hearing impairments.
