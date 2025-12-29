import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Load crop recommendation model and data
def load_crop_model():
    model_path = 'models/crop_model.pkl'
    data = pd.read_csv('datasets/Crop_recommendation.csv')
    X = data.drop('label', axis=1)
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        print("Crop model not found. Training a new one...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
    return model, X_test, y_test

# Load disease detection model
def load_disease_model():
    model_path = 'models/disease_model.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        print("Disease model not found.")
        return None

# For disease, we need to load test images, but for simplicity, assume we have labels
# This is a placeholder; in real scenario, load PlantVillage test data

def plot_confusion_matrix(cm, classes, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.close()
    print(f'Confusion matrix saved as {title.replace(" ", "_")}.png')

# Crop Recommendation Validation
print("Crop Recommendation Validation Matrix:")
crop_model, X_test_crop, y_test_crop = load_crop_model()
y_pred_crop = crop_model.predict(X_test_crop)
cm_crop = confusion_matrix(y_test_crop, y_pred_crop)
classes_crop = np.unique(y_test_crop)
plot_confusion_matrix(cm_crop, classes_crop, 'Confusion Matrix - Crop Recommendation')
print(classification_report(y_test_crop, y_pred_crop))

# Disease Detection Validation
print("\nDisease Detection Validation Matrix:")
disease_model = load_disease_model()
if disease_model:
    # Placeholder: Load test data
    # For demo, assume some test data
    # In real, load images and labels from PlantVillage
    print("Disease model loaded. For full validation, load test images and labels.")
    # Example placeholder
    # y_test_disease = ... load labels
    # y_pred_disease = disease_model.predict(test_images)
    # cm_disease = confusion_matrix(y_test_disease, y_pred_disease)
    # classes_disease = ['Healthy', 'Diseased', ...]  # Define classes
    # plot_confusion_matrix(cm_disease, classes_disease, 'Confusion Matrix - Disease Detection')
else:
    print("Disease model not available for validation.")