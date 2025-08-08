import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import google.generativeai as genai

# Replace 'YOUR_API_KEY' with your actual Gemini API key.
API_KEY = 'YOUR_API_KEY'
genai.configure(api_key=API_KEY)
model_name = 'gemini-1.5-flash-latest'  
gemini_model = genai.GenerativeModel(model_name)

# 1. Generate a synthetic dataset
def create_synthetic_dataset(n_samples=100):
    """
    Generates a synthetic dataset for predicting child malnutrition risk.
    """
    np.random.seed(42)
    data = {}
    data['age'] = np.random.randint(1, 6, n_samples)
    data['weight'] = np.random.normal(loc=15, scale=3, size=n_samples).round(2)
    data['height'] = np.random.normal(loc=95, scale=10, size=n_samples).round(2)
    data['muac'] = np.random.normal(loc=15, scale=2, size=n_samples).round(2)
    data['meals_per_day'] = np.random.randint(1, 5, n_samples)
    data['recent_illness'] = np.random.choice(['yes', 'no'], n_samples, p=[0.3, 0.7])
    data['breastfed'] = np.random.choice(['yes', 'no'], n_samples, p=[0.6, 0.4])

    df = pd.DataFrame(data)

    # Correcting for unrealistic values
    df.loc[df['weight'] < 5, 'weight'] = 5
    df.loc[df['height'] < 50, 'height'] = 50
    df.loc[df['muac'] < 10, 'muac'] = 10

    # Derive the 'risk' target variable
    def determine_risk(row):
        risk_score = 0
        if row['weight'] < 12:
            risk_score += 1
        if row['muac'] < 13.5:
            risk_score += 1
        if row['meals_per_day'] < 3:
            risk_score += 1
        if row['recent_illness'] == 'yes':
            risk_score += 1
        
        return 'high' if risk_score >= 2 else 'low'

    df['risk'] = df.apply(determine_risk, axis=1)

    return df

df = create_synthetic_dataset()
print("Generated Dataset:")
print(df.head())
print("\nRisk distribution:")
print(df['risk'].value_counts())

# Preprocessing for the model
df_encoded = pd.get_dummies(df, columns=['recent_illness', 'breastfed'], drop_first=True)
le = LabelEncoder()
df_encoded['risk'] = le.fit_transform(df_encoded['risk'])

X = df_encoded.drop('risk', axis=1)
y = df_encoded['risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train a classification model (Extra Trees Classifier)
model = ExtraTreesClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

# 4. Predict with data validation and Gemini-powered XAI
def predict_new_child_risk_with_gemini(model, label_encoder, normal_ranges):
    """
    Prompts the user for new child data, validates it, and predicts malnutrition risk
    with a Gemini-powered natural language explanation.
    """
    print("\nEnter new child data for prediction:")
    try:
        age = float(input("Enter age (1-5 years): "))
        weight = float(input("Enter weight (kg): "))
        height = float(input("Enter height (cm): "))
        muac = float(input("Enter MUAC (cm): "))
        meals = int(input("Enter meals per day (1-4): "))
        illness = input("Has the child had a recent illness? (yes/no): ").lower()
        breastfed = input("Is the child breastfed? (yes/no): ").lower()

        new_data = pd.DataFrame([{
            'age': age,
            'weight': weight,
            'height': height,
            'muac': muac,
            'meals_per_day': meals,
            'recent_illness_yes': 1 if illness == 'yes' else 0,
            'breastfed_yes': 1 if breastfed == 'yes' else 0
        }])
        
        new_data = new_data.reindex(columns=X.columns, fill_value=0)

        # Data validation checks
        anomalies = []
        if not (normal_ranges['age'][0] <= age <= normal_ranges['age'][1]):
            anomalies.append(f"Age of {age} years is outside the typical range (1-5 years).")
        if not (normal_ranges['weight'][0] <= weight <= normal_ranges['weight'][1]):
            if weight < normal_ranges['weight'][0]:
                anomalies.append(f"Weight of {weight} kg is unusually low for a child aged {age}.")
            else:
                anomalies.append(f"Weight of {weight} kg is unusually high for a child aged {age}.")
        if not (normal_ranges['height'][0] <= height <= normal_ranges['height'][1]):
            if height < normal_ranges['height'][0]:
                anomalies.append(f"Height of {height} cm is unusually low for a child aged {age}.")
            else:
                anomalies.append(f"Height of {height} cm is unusually high for a child aged {age}.")
        if not (normal_ranges['muac'][0] <= muac <= normal_ranges['muac'][1]):
            if muac < normal_ranges['muac'][0]:
                anomalies.append(f"MUAC of {muac} cm is unusually low and suggests a measurement error.")
            else:
                anomalies.append(f"MUAC of {muac} cm is unusually high and suggests a measurement error.")
        if not (normal_ranges['meals_per_day'][0] <= meals <= normal_ranges['meals_per_day'][1]):
            anomalies.append(f"Meals per day of {meals} is outside the typical range (1-4).")

        prediction = model.predict(new_data)
        predicted_risk = label_encoder.inverse_transform(prediction)[0]
        
        print(f"\nBased on the data, the predicted malnutrition risk is: {predicted_risk}")

        if anomalies:
            print("\n⚠  *Warning: The provided data contains some unusual values.*")
            for anomaly in anomalies:
                print(f"- {anomaly}")
            print("\nThis model's prediction is based on the input, but these values may indicate a data entry error or other health issues that the model is not trained to detect. Please consult with a healthcare professional.")

        # Gemini-powered natural language explanation
        print("\n--- Prediction Explanation (Gemini) ---")
        try:
            # Construct a prompt with the prediction and key input data
            prompt = f"A machine learning model predicted a child's malnutrition risk is '{predicted_risk}'. The child is {age} years old, weighs {weight} kg, is {height} cm tall, has a MUAC of {muac} cm, eats {meals} meals per day, has a recent illness: {illness}, and is breastfed: {breastfed}. Provide a concise, natural language explanation for this prediction based on the data. Focus on the most influential factors."
            
            response = gemini_model.generate_content(prompt)
            print(response.text)
        except Exception as e:
            print(f"An error occurred while generating the explanation: {e}")

    except ValueError:
        print("Invalid input. Please ensure all numerical fields are numbers.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Define some plausible normal ranges for data validation
normal_ranges = {
    'age': (1, 5),
    'weight': (5, 25),
    'height': (50, 120),
    'muac': (10, 25),
    'meals_per_day': (1, 4)
}

# Run the user prediction function
predict_new_child_risk_with_gemini(model, le, normal_ranges)