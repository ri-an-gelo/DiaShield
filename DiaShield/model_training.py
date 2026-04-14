import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create data and models directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

def load_and_preprocess_data(url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"):
    """Loads dataset, drops unnecessary columns, handles missing BMI values."""
    print("Loading dataset...")
    try:
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        df = pd.read_csv(url, names=columns)
        
        # 1. Drop unnecessary columns
        df = df.drop(columns=['Glucose', 'Insulin', 'SkinThickness', 'BloodPressure', 'Pregnancies'])

        # 2. Clean data: Replace missing/zero BMI values with median
        median_bmi = df['BMI'][df['BMI'] != 0].median()
        df['BMI'] = df['BMI'].replace(0, median_bmi)
            
        print("Dataset loaded and preprocessed successfully.")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def train_model(df):
    """Trains the Random Forest model on Age, BMI, and DiabetesPedigreeFunction."""
    if df is None:
        return
    
    print("Training the Random Forest model...")
    # Features (ONLY Age, BMI, Family history)
    X = df[['Age', 'BMI', 'DiabetesPedigreeFunction']]
    y = df['Outcome']
    
    # Train/test split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf_model.predict(X_test)
    
    # Evaluate accuracy and display confusion matrix
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Model Accuracy: {acc * 100:.2f}%")
    print(f"Confusion Matrix:\n{cm}")
    
    # Save the model
    with open('models/diabetes_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
        
    print("Model saved to models/diabetes_model.pkl")
    
    # Save the preprocessed dataset for visualization
    df.to_csv('data/preprocessed_dataset.csv', index=False)
    print("Preprocessed dataset saved to data/preprocessed_dataset.csv")

if __name__ == "__main__":
    df = load_and_preprocess_data()
    train_model(df)
