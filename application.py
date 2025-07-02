import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__, template_folder="templates")

# Load the trained model
MODEL_PATH = "artifacts/models/random_forest_model.pkl"
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

# List of feature names (must match training order)
FEATURE_NAMES = [
    'Pclass', 'Age', 'Fare', 'Sex', 'Embarked', 'Familysize', 'Isalone',
    'HasCabin', 'Title', 'Pclass_Fare', 'Age_Fare'
]

@app.route('/')
def home():
    return render_template('index.html', prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        form_data = request.form

        age = float(form_data['Age'])
        fare = float(form_data['Fare'])
        pclass = int(form_data['Pclass'])
        sex = int(form_data['Sex'])
        embarked = int(form_data['Embarked'])
        familysize = int(form_data['Familysize'])
        isalone = int(form_data['Isalone'])
        hascabin = int(form_data['HasCabin'])
        title = int(form_data['Title'])
        pclass_fare = float(form_data['Pclass_Fare'])
        age_fare = float(form_data['Age_Fare'])

        # Create a DataFrame for prediction (match training order)
        print("Model expects:", getattr(model, 'feature_names_in_', 'N/A'))
        print("You provide:", FEATURE_NAMES)
        print("DataFrame columns:", FEATURE_NAMES)
        print("Values:", [pclass, age, fare, sex, embarked, familysize, isalone, hascabin, title, pclass_fare, age_fare])
        features = pd.DataFrame([[
            pclass, age, fare, sex, embarked, familysize, isalone,
            hascabin, title, pclass_fare, age_fare
        ]], columns=FEATURE_NAMES)

        # For extra debugging, print the DataFrame
        print("DataFrame for prediction:\n", features)

        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        if prediction == 1:
            prediction_text = f"🎉 Congratulations! You would have survived! 🛟 (Survival Probability: {probability:.2%})"
        else:
            prediction_text = f"💔 Unfortunately, you would not have survived. (Survival Probability: {probability:.2%})"

        # Render same page with prediction
        return render_template('index.html', prediction_text=prediction_text)

    except Exception as e:
        return render_template('index.html', prediction_text=f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)
