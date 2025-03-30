from flask import Flask, render_template, request
import pandas as pd
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract user input
        age = float(request.form['age'])
        credit_score = float(request.form['credit_score'])
        balance = float(request.form['balance'])
        
        # Use default values for missing features (adjust as needed)
        gender = 0  # Example default for gender
        education = 0  # Example default for education

        # Prepare input data (5 features in total)
        input_data = [[credit_score, age, balance, gender, education]]

        # Scale the input data
        scaled_data = scaler.transform(input_data)

        # Make prediction
        churn_prediction = model.predict(scaled_data)
        churn_probability = model.predict_proba(scaled_data)[0][1] * 100  # Probability of churn

        # Display result
        result = "Churn" if churn_prediction == 1 else "No Churn"
        
        return render_template('index.html', prediction=result, churn_percentage=round(churn_probability, 2))

if __name__ == "__main__":
    app.run(debug=True)
