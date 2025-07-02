# Titanic Survival Prediction - MLOps Pipeline

A complete MLOps pipeline for predicting Titanic passenger survival using Apache Airflow, Redis Feature Store, Flask web application, and machine learning best practices.

## 🏗️ Architecture Overview

```
GCP Cloud Storage → Apache Airflow → PostgreSQL → Feature Engineering → Redis Feature Store → ML Model Training → Model Artifacts → Flask Web App
```

## 📁 Project Structure

```
Titanic_Survival/
├── src/                          # Core application modules
│   ├── data_ingestion.py         # Data extraction from PostgreSQL
│   ├── data_processing.py        # Feature engineering & SMOTE resampling
│   ├── feature_store.py          # Redis-based feature store
│   ├── model_training.py         # ML model training with hyperparameter tuning
│   ├── custom_exception.py       # Custom exception handling
│   └── logger.py                 # Centralized logging
├── dags/                         # Airflow DAGs
│   ├── extract_data_from_gcp.py  # GCS to PostgreSQL data pipeline
│   └── exampledag.py            # Example Airflow DAG
├── pipeline/                     # End-to-end training pipeline
│   └── training_pipeline.py      # Complete ML pipeline orchestration
├── config/                       # Configuration files
│   ├── database_config.py        # Database connection settings
│   └── paths_config.py          # File paths configuration
├── templates/                    # Flask web application templates
│   └── index.html               # Interactive prediction interface
├── artifacts/                    # Model artifacts and data
│   ├── models/                   # Trained models (RandomForest: 94.9% accuracy)
│   └── raw/                      # Raw datasets
├── logs/                         # Application logs
├── tests/                        # Test files
│   └── dags/                     # DAG tests
├── notebook/                     # Jupyter notebooks for exploration
├── application.py                # Flask web application
├── Dockerfile                    # Container configuration
├── requirements.txt              # Python dependencies
└── setup.py                      # Package setup
```

## 🚀 Features

### Data Pipeline
- **Data Ingestion**: Automated extraction from GCP Cloud Storage to PostgreSQL
- **Feature Engineering**: Advanced feature creation including family size, titles, and interaction features
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Oversampling Technique) implementation
- **Feature Store**: Redis-based feature storage for real-time serving

### ML Pipeline
- **Model**: Random Forest Classifier with hyperparameter tuning
- **Performance**: 94.9% accuracy on test set
- **Features**: 11 engineered features including Pclass, Age, Fare, Sex, Embarked, etc.
- **Validation**: Cross-validation with RandomizedSearchCV

### Web Application
- **Interactive Interface**: Flask-based web application for real-time predictions
- **User-Friendly UI**: Modern, responsive design with gradient backgrounds
- **Real-Time Predictions**: Instant survival probability calculations
- **Input Validation**: Comprehensive form validation and error handling

### MLOps Components
- **Orchestration**: Apache Airflow for workflow management
- **Monitoring**: Comprehensive logging system with Prometheus integration
- **Containerization**: Docker support with Astronomer runtime
- **Feature Store**: Redis for feature serving and storage
- **Web Deployment**: Flask application ready for production deployment

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Redis Server
- PostgreSQL
- GCP Account with Cloud Storage
- Flask (for web application)

### Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd Titanic_Survival

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Redis server
redis-server

# Start Airflow (using Astronomer)
astro dev start
```

### Configuration
1. **Database Configuration**: Update `config/database_config.py` with your PostgreSQL credentials
2. **GCP Setup**: Place your GCP service account key in `include/gcp-key.json`
3. **Airflow Connections**: Configure GCS and PostgreSQL connections in Airflow UI

## 🔄 Pipeline Execution

### Option 1: Complete Pipeline
```bash
python pipeline/training_pipeline.py
```

### Option 2: Individual Components
```bash
# Data ingestion
python src/data_ingestion.py

# Data processing & feature engineering
python src/data_processing.py

# Model training
python src/model_training.py
```

### Option 3: Airflow DAG
1. Access Airflow UI at `http://localhost:8080`
2. Trigger `extract_titanic_data` DAG
3. Monitor pipeline execution

### Option 4: Web Application
```bash
# Run Flask web application
python application.py

# Access web interface at http://localhost:5001
```

## 📊 Model Performance

- **Algorithm**: Random Forest Classifier
- **Accuracy**: 94.9%
- **Best Parameters**: 
  - n_estimators: 300
  - max_depth: 30
  - min_samples_split: 2
  - min_samples_leaf: 1

### Feature Engineering
- **Age**: Median imputation for missing values
- **Embarked**: Mode imputation
- **Sex**: Binary encoding (male: 0, female: 1)
- **Title**: Extracted from names (Mr, Miss, Mrs, Master, Rare)
- **Family Size**: SibSp + Parch + 1
- **Is Alone**: Binary indicator for solo travelers
- **Has Cabin**: Binary indicator for cabin information
- **Interaction Features**: Pclass_Fare, Age_Fare

## 🗄️ Feature Store

Redis-based feature store with:
- **Original Features**: Stored with PassengerId as key
- **SMOTE Features**: Synthetic samples with `smote_` prefix
- **Real-time Retrieval**: Fast feature serving for inference
- **Batch Operations**: Efficient bulk feature storage

### Usage Example
```python
from src.feature_store import RedisFeatureStore

feature_store = RedisFeatureStore()
features = feature_store.get_feature(entity_id=332)
print(features)
# Output: {'Pclass': 1, 'Age': 45.5, 'Fare': 28.5, ...}
```

## 📈 Monitoring & Logging

- **Centralized Logging**: All components log to `logs/log_YYYY-MM-DD.log`
- **Error Tracking**: Custom exception handling with detailed error messages
- **Pipeline Monitoring**: Airflow UI for DAG monitoring and debugging

### Recent Pipeline Execution (from logs):
```
2025-07-02 19:46:17,552 - INFO - Accuracy is 0.949438202247191
2025-07-02 19:46:17,579 - INFO - Model saved at artifacts/models/random_forest_model.pkl
2025-07-02 19:46:17,580 - INFO - End of Model Training pipeline...
```

## 🌐 Web Application

The Flask web application provides an interactive interface for making survival predictions:

### Features
- **Interactive Form**: User-friendly input form with 11 feature fields
- **Real-Time Predictions**: Instant survival probability calculations
- **Modern UI**: Responsive design with gradient backgrounds and animations
- **Error Handling**: Comprehensive validation and error messages

### Usage
1. Start the application: `python application.py`
2. Navigate to `http://localhost:5001`
3. Fill in passenger details:
   - Age, Fare, Passenger Class
   - Gender, Port of Embarkation
   - Family Size, Cabin Status
   - Title, Interaction Features
4. Click "Predict My Fate!" for instant results

### Input Features
- **Age**: Passenger age (numeric)
- **Fare**: Ticket fare paid (numeric)
- **Pclass**: Passenger class (1, 2, or 3)
- **Sex**: Gender (Male: 0, Female: 1)
- **Embarked**: Port (Cherbourg: 0, Queenstown: 1, Southampton: 2)
- **Familysize**: Total family members aboard
- **Isalone**: Solo traveler indicator (Yes: 1, No: 0)
- **HasCabin**: Cabin assignment indicator (Yes: 1, No: 0)
- **Title**: Passenger title (Mr: 1, Miss: 2, etc.)
- **Pclass_Fare**: Interaction feature (Pclass × Fare)
- **Age_Fare**: Interaction feature (Age × Fare)

## 🔧 Configuration Files

### Database Configuration
```python
# config/database_config.py
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'titanic_db',
    'user': 'your_username',
    'password': 'your_password'
}
```

### Paths Configuration
```python
# config/paths_config.py
RAW_DIR = "artifacts/raw/"
TRAIN_PATH = "artifacts/raw/titanic_train.csv"
TEST_PATH = "artifacts/raw/titanic_test.csv"
```

### Flask Application Configuration
```python
# application.py
MODEL_PATH = "artifacts/models/random_forest_model.pkl"
FEATURE_NAMES = [
    'Pclass', 'Age', 'Fare', 'Sex', 'Embarked', 'Familysize', 'Isalone',
    'HasCabin', 'Title', 'Pclass_Fare', 'Age_Fare'
]
```

## 🚀 Deployment

### Local Development
```bash
# Start Airflow
astro dev start

# Start Flask application
python application.py
```

### Production (Astronomer)
```bash
astro deploy
```

### Docker Deployment
```bash
# Build container
docker build -t titanic-survival .

# Run container
docker run -p 5001:5001 titanic-survival
```

### Container Configuration
- **Base Image**: Astronomer Runtime 12.6.0
- **Additional Providers**: Google Cloud providers for Airflow
- **Port**: 5001 (Flask application)
- **Dependencies**: All requirements from requirements.txt

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Technologies Used

- **ML/Data**: pandas, scikit-learn, imbalanced-learn, numpy
- **Web Framework**: Flask
- **Orchestration**: Apache Airflow, Astronomer
- **Storage**: Redis, PostgreSQL, GCP Cloud Storage
- **Containerization**: Docker
- **Monitoring**: Custom logging system, Prometheus, Alibi-detect
- **Frontend**: HTML5, CSS3, JavaScript (responsive design)
- **Database**: PostgreSQL with psycopg2-binary connector

---

**Note**: This project demonstrates a production-ready MLOps pipeline with proper separation of concerns, error handling, and monitoring capabilities.