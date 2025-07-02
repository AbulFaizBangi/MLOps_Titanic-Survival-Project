# Titanic Survival Prediction - MLOps Pipeline

A complete MLOps pipeline for predicting Titanic passenger survival using Apache Airflow, Redis Feature Store, and machine learning best practices.

## 🏗️ Architecture Overview

```
GCP Cloud Storage → Apache Airflow → PostgreSQL → Feature Engineering → Redis Feature Store → ML Model Training → Model Artifacts
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
│   └── extract_data_from_gcp.py  # GCS to PostgreSQL data pipeline
├── pipeline/                     # End-to-end training pipeline
│   └── training_pipeline.py      # Complete ML pipeline orchestration
├── config/                       # Configuration files
│   ├── database_config.py        # Database connection settings
│   └── paths_config.py          # File paths configuration
├── artifacts/                    # Model artifacts and data
│   ├── models/                   # Trained models (RandomForest: 94.9% accuracy)
│   └── raw/                      # Raw datasets
├── logs/                         # Application logs
└── notebook/                     # Jupyter notebooks for exploration
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

### MLOps Components
- **Orchestration**: Apache Airflow for workflow management
- **Monitoring**: Comprehensive logging system
- **Containerization**: Docker support with Astronomer
- **Feature Store**: Redis for feature serving and storage

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Redis Server
- PostgreSQL
- GCP Account with Cloud Storage

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

## 🚀 Deployment

### Local Development
```bash
astro dev start
```

### Production (Astronomer)
```bash
astro deploy
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Technologies Used

- **ML/Data**: pandas, scikit-learn, imbalanced-learn
- **Orchestration**: Apache Airflow, Astronomer
- **Storage**: Redis, PostgreSQL, GCP Cloud Storage
- **Containerization**: Docker
- **Monitoring**: Custom logging system

---

**Note**: This project demonstrates a production-ready MLOps pipeline with proper separation of concerns, error handling, and monitoring capabilities.