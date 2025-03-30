import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the synthetic dataset (or you can load your actual CSV file here)
df = pd.read_csv('data.csv')

# Features and target variable
X = df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts']]  # Features
y = df['Exited']  # Target variable (1 for churn, 0 for no churn)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (important for some ML models, especially when they rely on distance metrics)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model accuracy
accuracy = model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the model to a file
with open('churn_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the scaler to a file
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully!")
