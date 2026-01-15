# ♻️ Smart City Waste Classification
An end-to-end deep learning project that classifies waste images using a hierarchical deep learning approach.
The system first identifies whether waste is recyclable or non-recyclable, and then classifies the type of recyclable waste.
This project is designed to reflect real-world smart city waste management systems.

## 🚀 Live Demo
👉 Streamlit App: (http://localhost:8501/)
Upload a waste image and get:
  - Recyclable / Non-Recyclable prediction
  - Recyclable waste category (if applicable)
  - Prediction confidence

## Project Overview
Instead of using a single flat multi-class classifier, this project uses a two-stage hierarchical pipeline:
#### Input Image
   ↓
#### Binary Classifier
(Recyclable / Non-Recyclable)

   ↓
#### If Recyclable
   ↓
#### Multi-Class Classifier
(Cardboard / Paper / Plastic / Glass / Metal / E-waste)

Mimics real waste-sorting workflows.

## 📂 Repository Structure
#### Smart_City_Waste_classification/
#### ├── notebooks/     # Data preparation notebook, binary classifier training notebook, multiclassifier training notebook.
#### ├── app.py         # Streamlit application for prediction

## 📦 Trained Models (GitHub Releases)
Trained models are not provided in repository due to size issues thats why  distributed via GitHub Releases. go via releases to see how to load trained models to test deployed streamlit webapp.


## 🗂️ Dataset Description
The project uses a publicly available waste image dataset containing real-world images of various waste materials captured under different lighting conditions, backgrounds, and orientations.

Original Waste Categories :
- cardboard
- paper
- plastic
- glass
- metal
- e-waste
- organic
- textile

These categories include both recyclable and non-recyclable waste types.


## Data Preparation 
This project uses a hierarchical approach, so two different datasets were prepared: one for binary classification and one for multi-class classification to support our heirachichal classification pipeline.

## Train, Validation, and Test Split
First, the original dataset was split class-wise into:
- 70% Training
- 15% Validation
- 15% Test
This step was done before any further processing to make sure there is no data leakage and that model evaluation is fair.


## 1.Binary Classification Dataset
To train the binary classifier, waste categories were grouped as follows:
Recyclable :
cardboard, paper, plastic, glass, metal, e-waste
Non-Recyclable :
organic, textile

Images from the train, validation, and test splits were copied into a new folder structure :
#### binary_dataset/
#### ├── train/
#### ├── val/
#### └── test/

this dataset is what we used to train binary classiyer to distinguish btw recyclable and non-recyclable waste.

## Multi-Class Recyclable Dataset
For the second stage, only recyclable waste images were used to create a multi-class dataset with the following classes:
- cardboard
- paper
- plastic
- glass
- metal
- e-waste
- Non-recyclable categories were removed completely.

#### recyclable_dataset/
#### ├── train/
#### ├── val/
#### └── test/

This dataset is used to train multiclassifier to classify the type of recyclable waste.

## Image Preprocessing
For both datasets:
- Images resized to 224 × 224
- Pixel normalization applied
- Data augmentation (rotation, zoom, flip) used only on training data.

## Models Used
- Architecture: MobileNetV2 (Transfer Learning)
- Framework: TensorFlow / Keras
- Binary Model: Recyclable vs Non-Recyclable
- Multi-Class Model: Recyclable waste categories
- Early stopping was used to select the best generalizing model.

##  Prediction Pipeline
1. Image is passed to the binary classifier
2. If predicted as Non-Recyclable, prediction ends
3. If Recyclable, image is passed to the multi-class classifier
4. Final waste category and confidence are displayed

## Evaluation Summary
- Strong generalization on unseen test data
- Confusion mainly between visually similar classes (e.g., paper vs cardboard)
- E-waste shows highest accuracy due to distinct visual features.

## Key Highlights
- Hierarchical waste classification pipeline that mirrors real-world waste sorting systems
- Two-stage CNN architecture: Binary classification followed by recyclable material classification
- Transfer learning with MobileNetV2 for efficient training on limited data.
- Clean train–validation–test split performed before dataset restructuring to prevent data leakage
- Early stopping–based model selection to ensure strong generalization
- Class-level evaluation using confusion matrix 
- Streamlit-based interactive web application for real-time predictions.

## Future Work
- Integrate **YOLO-based object detection** to identify multiple waste items in a single image
- Add **Grad-CAM** visual explanations to highlight regions influencing model predictions
- Extend the system for real-time camera input in smart city environments.

