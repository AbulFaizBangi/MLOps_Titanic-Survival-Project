import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from src.logger import get_logger
try:
    from alibi_detect.cd import KSDrift
except ImportError:
    # Fallback if TensorFlow dependencies are missing
    KSDrift = None
from src.feature_store import RedisFeatureStore
from sklearn.preprocessing import StandardScaler

from prometheus_client import start_http_server,Counter,Gauge

logger = get_logger(__name__)

app = Flask(__name__, template_folder="templates")


prediction_count = Counter('prediction_count' , " Number of prediction count" )
drift_count = Counter('drift_count' , "Number of times data drift is detected")

# Load the trained model
MODEL_PATH = "artifacts/models/random_forest_model.pkl"
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

# List of feature names (must match training order)
FEATURE_NAMES = [
    'Pclass', 'Age', 'Fare', 'Sex', 'Embarked', 'Familysize', 'Isalone',
    'HasCabin', 'Title', 'Pclass_Fare', 'Age_Fare'
]

feature_store = RedisFeatureStore()
scaler = StandardScaler()

def fit_scaler_on_ref_data():
    entity_ids = feature_store.get_all_entity_ids()
    all_features = feature_store.get_batch_features(entity_ids)

    all_features_df = pd.DataFrame.from_dict(all_features , orient='index')[FEATURE_NAMES]

    scaler.fit(all_features_df)
    return scaler.transform(all_features_df)


historical_data = fit_scaler_on_ref_data()
if KSDrift is not None:
    ksd = KSDrift(x_ref=historical_data , p_val=0.05)
else:
    ksd = None
    logger.warning("KSDrift not available - drift detection disabled")


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
        
        ##### Data Drift Detection
        if ksd is not None:
            features_scaled = scaler.transform(features)
            drift = ksd.predict(features_scaled)
            print("Drift Response : ",drift)
            
            drift_response = drift.get('data',{})
            is_drift = drift_response.get('is_drift' , None)
            
            if is_drift is not None and is_drift==1:
                print("Drift Detected....")
                logger.info("Drift Detected....")
                drift_count.inc()

        # For extra debugging, print the DataFrame
        print("DataFrame for prediction:\n", features)

        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        prediction_count.inc()

        if prediction == 1:
            prediction_text = f"🎉 Congratulations! You would have survived! 🛟 (Survival Probability: {probability:.2%})"
        else:
            prediction_text = f"💔 Unfortunately, you would not have survived. (Survival Probability: {probability:.2%})"

        # Render same page with prediction
        return render_template('index.html', prediction_text=prediction_text)

    except Exception as e:
        return render_template('index.html', prediction_text=f"⚠️ Error: {str(e)}")

@app.route('/metrics')
def metrics():
    from prometheus_client import generate_latest
    from flask import Response

    return Response(generate_latest() , content_type='text/plain')
    
if __name__ =="__main__":
    import socket
    
    def find_free_port(start_port=8000):
        for port in range(start_port, start_port + 100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    return port
        return None
    
    metrics_port = find_free_port(8000)
    if metrics_port:
        start_http_server(metrics_port)
        print(f"Metrics server started on port {metrics_port}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
