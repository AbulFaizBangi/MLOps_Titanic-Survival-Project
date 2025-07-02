import json
import redis

class RedisFeatureStore:
      def __init__(self, host='localhost', port=6379, db=0):
            self.redis_client = redis.StrictRedis(host=host, port=port, db=db, decode_responses=True)

      # Store Row by row
      def store_features(self, entity_id, features):
            """Store features for a given entity in the Redis store."""
            key = f"entity:{entity_id}"  # Construct the key for the entity
            self.redis_client.set(key, json.dumps(features))  # Store features as JSON ,mapping=features)

      # Store features for an entity
      def get_feature(self, entity_id):
            key = f"entity:{entity_id}"  # Construct the key for the entity
            features = self.redis_client.get(key)  # Retrieve features as JSON
            if features:
                  return json.loads(features)
            return None  # Return None if no features are found for the entity

      # Store features for multiple entities
      def store_batch_features(self, batch_data):
            """Store a batch of features for multiple entities in the Redis store."""
            for entity_id, features in batch_data.items():
                  self.store_features(entity_id, features)
      
      # Retrieve features for multiple entities
      def get_batch_features(self, entity_ids):
            """Retrieve features for a batch of entities from the Redis store."""
            batch_features = {}
            for entity_id in entity_ids:
                  batch_features[entity_id] = self.get_feature(entity_id)
            return batch_features
      
      # Retrieve all entity IDs
      def get_all_entity_ids(self):
            """Retrieve all entity IDs stored in the Redis store."""
            keys = self.redis_client.keys("entity:*")  # Get all keys matching the pattern
            entity_ids = [key.split(":")[1] for key in keys]  # Extract entity IDs from keys
            return entity_ids  # Return a list of entity IDs
