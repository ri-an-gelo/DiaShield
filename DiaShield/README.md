# 🛡️ DiaShield — AI-Based Diabetes Risk Predictor

A complete, production-quality machine learning web application that predicts the risk of Type 2 Diabetes using user lifestyle and basic health inputs, providing personalized suggestions via a rule-based AI assistant.

⚠️ **Disclaimer:** This application provides risk estimation only and is not a substitute for professional medical advice.

## 🌟 Key Features

* **Focused Machine Learning Model**: Built with a Random Forest Classifier trained strictly on Age, BMI, and Family History using the Pima Indians Diabetes Dataset.
* **Built-in BMI Calculator**: Easily converts height and weight inputs into BMI.
* **Modern Interface**: Clean medical theme (Blue/Green/Red risk indicators) created via Streamlit.
* **Smart Suggestions Engine**: Rule-based dynamic feedback providing lifestyle advice.
* **Data Visualization**: Rich visual insights into the dataset, including Age vs Risk, Feature Importance, and a BMI Category chart.
* **Integrated AI Chatbot**: Simulate friendly, professional AI interactions right inside the app.

## 🚀 How to Run Locally

### 1. Install Dependencies
Run the command below to install Streamlit, Scikit-Learn, Pandas, Plotly/Seaborn, etc. 
*(If you were facing build errors before, this should now safely pull pre-compiled packages for your system)*:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
You must pull the dataset and train the Machine Learning model before the web app can run. Simply run:
```bash
python model_training.py
```
This drops unnecessary columns, calculates BMI medians for missing data, and creates `diabetes_model.pkl`.

### 3. Start the Web App
Run your Streamlit server:
```bash
streamlit run app.py
```

## ☁️ Deployment on Streamlit Cloud

1. Create a GitHub repository and push your project files.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and link your GitHub account.
3. Choose your repository and specify `app.py` as the main route.
4. Click **Deploy**. The app is entirely self-sufficient (once `model_training.py` gets pre-emptively loaded or committed along with its `models/diabetes_model.pkl` file). 

## 📁 Files Included
* `model_training.py` - Script driving the Random Forest creation pipeline.
* `app.py` - Core web application.
* `requirements.txt` - Python module requisites.
* `README.md` - Technical setup guide.
