import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.feature_store import RedisFeatureStore
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *

logger = get_logger(__name__)

class DataProcessing:
      
      def __init__(self, train_data_path, test_data_path, feature_store: RedisFeatureStore):
            self.train_data_path = train_data_path
            self.test_data_path = test_data_path
            self.data= None
            self.test_data = None
            self.X_train = None
            self.y_train = None
            self.X_test = None
            self.y_test = None
            
            self.X_resampled = None
            self.y_resampled = None
            
            self.feature_store = feature_store
            
            logger.info("DataProcessing class initialized...")
            
      def load_data(self):
            try:
                  self.data = pd.read_csv(self.train_data_path)
                  self.test_data = pd.read_csv(self.test_data_path)
                  logger.info("Data loaded successfully...")
            except Exception as e:
                  logger.error(f"Error while loading data: {e}")
                  raise CustomException(str(e), sys)
            
      def preprocess_data(self):
            try:
                  self.data['Age'] = self.data['Age'].fillna(self.data['Age'].median())
                  self.data['Embarked'] = self.data['Embarked'].fillna(self.data['Embarked'].mode()[0])
                  self.data['Fare'] = self.data['Fare'].fillna(self.data['Fare'].median())
                  self.data['Sex'] = self.data['Sex'].map({'male': 0, 'female': 1})
                  self.data['Embarked'] = self.data['Embarked'].astype('category').cat.codes


                  self.data['Familysize'] = self.data['SibSp'] + self.data['Parch'] + 1
                  self.data['Isalone'] = (self.data['Familysize'] == 1).astype(int)
                  self.data['HasCabin'] = self.data['Cabin'].notnull().astype(int)
                  self.data['Title'] = self.data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False).map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}).fillna(4)
                  self.data['Pclass_Fare'] = self.data['Pclass'] * self.data['Fare']
                  self.data['Age_Fare'] = self.data['Age'] * self.data['Fare']
                  
                  logger.info("Data preprocessing completed...")
            except Exception as e:
                  logger.error(f"Error while preprocessing data: {e}")
                  raise CustomException(str(e), sys)
            
      def handle_imbalance(self):
            try:
                  # Store original data before SMOTE for comparison
                  self.original_data = self.data.copy()
                  
                  X = self.data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Familysize', 'Isalone', 'HasCabin', 'Title', 'Pclass_Fare', 'Age_Fare']]
                  y = self.data['Survived']
                  smote = SMOTE(random_state=42)
                  self.X_resampled, self.y_resampled = smote.fit_resample(X, y)
                  
                  logger.info("Imbalance handling completed using SMOTE...")
            except Exception as e:
                  logger.error(f"Error while handling imbalance: {e}")
                  raise CustomException(str(e), sys)
            
      def store_features_in_redis(self):
            try:
                  # Store original data first (without SMOTE)
                  batch_data = {}
                  for idx, row in self.original_data.iterrows():
                        entity_id = int(row['PassengerId'])  # Use actual PassengerId
                        features = {
                              "Pclass": int(row['Pclass']),
                              "Age": float(row['Age']),
                              "Fare": float(row['Fare']),
                              "Sex": int(row['Sex']),
                              "Embarked": int(row['Embarked']),
                              "Familysize": int(row['Familysize']),
                              "Isalone": int(row['Isalone']),
                              "HasCabin": int(row['HasCabin']),
                              "Title": float(row['Title']),
                              "Pclass_Fare": float(row['Pclass_Fare']),
                              "Age_Fare": float(row['Age_Fare']),
                              "Survived": int(row['Survived'])
                        }
                        batch_data[entity_id] = features
                  
                  self.feature_store.store_batch_features(batch_data)
                  logger.info("Original features stored in Redis successfully...")
                  
                  # Store SMOTE resampled data with synthetic IDs
                  smote_batch_data = {}
                  for idx, row in self.X_resampled.iterrows():
                        entity_id = f"smote_{int(idx)}"  # Prefix synthetic data
                        features = {
                              "Pclass": int(row['Pclass']),
                              "Age": float(row['Age']),
                              "Fare": float(row['Fare']),
                              "Sex": int(row['Sex']),
                              "Embarked": int(row['Embarked']),
                              "Familysize": int(row['Familysize']),
                              "Isalone": int(row['Isalone']),
                              "HasCabin": int(row['HasCabin']),
                              "Title": float(row['Title']),
                              "Pclass_Fare": float(row['Pclass_Fare']),
                              "Age_Fare": float(row['Age_Fare']),
                              "Survived": int(self.y_resampled.iloc[idx])
                        }
                        smote_batch_data[entity_id] = features
                  
                  self.feature_store.store_batch_features(smote_batch_data)
                  logger.info("SMOTE resampled features stored in Redis successfully...")
            except Exception as e:
                  logger.error(f"Error while storing features in Redis: {e}")
                  raise CustomException(str(e), sys)
            
      def retrive_feature_redis_store(self, entity_id):
            features = self.feature_store.get_feature(entity_id)
            if features:
                  logger.info(f"Features retrieved for entity {entity_id}: {features}")
            else:
                  logger.warning(f"No features found for entity {entity_id}")
            return features
      
      def run(self):
            try:
                  logger.info("Data Processing Pipeline Started...")
                  self.load_data()
                  self.preprocess_data()
                  self.handle_imbalance()
                  self.store_features_in_redis()
                  logger.info("Data Processing Pipeline Completed.")
            except Exception as e:
                  logger.error(f"Error in Data Processing Pipeline: {e}")
                  raise CustomException(str(e), sys)
            
if __name__ == "__main__":
      features_store = RedisFeatureStore()
      
      data_processor = DataProcessing(TRAIN_PATH, TEST_PATH, features_store)
      data_processor.run()
      
      print(data_processor.retrive_feature_redis_store(entity_id=332))  # Example entity_id to retrieve features
