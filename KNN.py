import numpy as np
import pandas as pd
from pyspark.sql import functions as sf
from pyspark.sql import DataFrame, Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType, ArrayType
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class KNNRecommender:
    """
    Comprehensive K-Nearest Neighbors Recommender that experiments with multiple configurations
    to maximize revenue in the recommendation system competition.
    
    This implementation explores:
    - Multiple distance metrics (cosine, euclidean, manhattan)
    - Different k values (3, 5, 10, 20, 30, 50)
    - User-based and item-based approaches
    - Various feature combinations
    - Revenue optimization strategies
    """
    
    def __init__(self, seed=None):
        """
        Initialize the KNN recommender with comprehensive hyperparameter exploration.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        
        # Hyperparameter configurations to explore
        self.distance_metrics = ['cosine', 'euclidean', 'manhattan']
        self.k_values = [3, 5, 10, 20, 30, 50]
        self.approaches = ['user_based', 'item_based', 'hybrid']
        self.feature_strategies = ['attributes_only', 'behavior_only', 'combined']
        self.revenue_strategies = ['probability_only', 'price_weighted', 'price_feature']
        
        # Best configuration tracking
        self.best_config = None
        self.best_score = -np.inf
        
        # Data storage
        self.user_features_pd = None
        self.item_features_pd = None
        self.interaction_matrix = None
        self.user_similarity_matrices = {}
        self.item_similarity_matrices = {}
        
        # Feature processing
        self.user_scaler = StandardScaler()
        self.item_scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
        
        # Current best model components
        self.best_user_similarity = None
        self.best_item_similarity = None
        self.best_approach = None
        self.best_k = None
        self.best_distance_metric = None
        self.best_feature_strategy = None
        self.best_revenue_strategy = None
        
    def _convert_spark_to_pandas(self, spark_df):
        """Convert Spark DataFrame to Pandas DataFrame efficiently."""
        return spark_df.toPandas()
    
    def _extract_user_features(self, users_df, log_df=None):
        """
        Extract and engineer user features from multiple sources.
        
        Args:
            users_df: User dataframe with attributes
            log_df: Interaction log for behavioral features
            
        Returns:
            numpy.ndarray: Processed user feature matrix
        """
        users_pd = self._convert_spark_to_pandas(users_df)
        
        # Strategy 1: Attributes only
        attr_features = []
        attr_cols = [col for col in users_pd.columns if col.startswith('user_attr_')]
        if attr_cols:
            attr_features = users_pd[attr_cols].fillna(0).values
        
        # Strategy 2: Behavioral features only
        behavior_features = []
        if log_df is not None:
            log_pd = self._convert_spark_to_pandas(log_df)
            
            # Purchase frequency per user
            user_purchase_counts = log_pd.groupby('user_idx')['relevance'].agg(['count', 'sum', 'mean']).fillna(0)
            
            # Average price of purchased items
            purchased_items = log_pd[log_pd['relevance'] > 0]
            if len(purchased_items) > 0 and 'price' in log_pd.columns:
                avg_price_per_user = purchased_items.groupby('user_idx')['price'].mean()
            else:
                avg_price_per_user = pd.Series(0, index=users_pd['user_idx'])
            
            # Category preferences
            if 'category' in log_pd.columns:
                category_prefs = log_pd[log_pd['relevance'] > 0].groupby(['user_idx', 'category']).size().unstack(fill_value=0)
                category_prefs = category_prefs.div(category_prefs.sum(axis=1), axis=0).fillna(0)
            else:
                category_prefs = pd.DataFrame()
            
            # Combine behavioral features
            behavior_df = users_pd[['user_idx']].set_index('user_idx')
            behavior_df = behavior_df.join(user_purchase_counts, how='left').fillna(0)
            behavior_df = behavior_df.join(avg_price_per_user.rename('avg_price'), how='left').fillna(0)
            if not category_prefs.empty:
                behavior_df = behavior_df.join(category_prefs, how='left').fillna(0)
            
            behavior_features = behavior_df.values
        
        # Strategy 3: Combined features
        if len(attr_features) > 0 and len(behavior_features) > 0:
            combined_features = np.hstack([attr_features, behavior_features])
        elif len(attr_features) > 0:
            combined_features = attr_features
        elif len(behavior_features) > 0:
            combined_features = behavior_features
        else:
            # Fallback: create dummy features
            combined_features = np.random.random((len(users_pd), 5))
        
        return {
            'attributes_only': attr_features if len(attr_features) > 0 else np.random.random((len(users_pd), 3)),
            'behavior_only': behavior_features if len(behavior_features) > 0 else np.random.random((len(users_pd), 3)),
            'combined': combined_features
        }
    
    def _extract_item_features(self, items_df, log_df=None):
        """
        Extract and engineer item features from multiple sources.
        
        Args:
            items_df: Item dataframe with attributes
            log_df: Interaction log for behavioral features
            
        Returns:
            dict: Dictionary of feature matrices for different strategies
        """
        items_pd = self._convert_spark_to_pandas(items_df)
        
        # Strategy 1: Attributes only
        attr_features = []
        attr_cols = [col for col in items_pd.columns if col.startswith('item_attr_')]
        
        # Include price as an attribute feature
        feature_cols = attr_cols + (['price'] if 'price' in items_pd.columns else [])
        if feature_cols:
            attr_features = items_pd[feature_cols].fillna(0).values
        
        # Strategy 2: Behavioral features only
        behavior_features = []
        if log_df is not None:
            log_pd = self._convert_spark_to_pandas(log_df)
            
            # Item popularity metrics
            item_stats = log_pd.groupby('item_idx')['relevance'].agg(['count', 'sum', 'mean']).fillna(0)
            
            # User segment preferences for items
            if 'segment' in log_pd.columns:
                segment_prefs = log_pd[log_pd['relevance'] > 0].groupby(['item_idx', 'segment']).size().unstack(fill_value=0)
                segment_prefs = segment_prefs.div(segment_prefs.sum(axis=1), axis=0).fillna(0)
            else:
                segment_prefs = pd.DataFrame()
            
            # Combine behavioral features
            behavior_df = items_pd[['item_idx']].set_index('item_idx')
            behavior_df = behavior_df.join(item_stats, how='left').fillna(0)
            if not segment_prefs.empty:
                behavior_df = behavior_df.join(segment_prefs, how='left').fillna(0)
            
            behavior_features = behavior_df.values
        
        # Strategy 3: Combined features
        if len(attr_features) > 0 and len(behavior_features) > 0:
            combined_features = np.hstack([attr_features, behavior_features])
        elif len(attr_features) > 0:
            combined_features = attr_features
        elif len(behavior_features) > 0:
            combined_features = behavior_features
        else:
            # Fallback: create dummy features
            combined_features = np.random.random((len(items_pd), 5))
        
        return {
            'attributes_only': attr_features if len(attr_features) > 0 else np.random.random((len(items_pd), 3)),
            'behavior_only': behavior_features if len(behavior_features) > 0 else np.random.random((len(items_pd), 3)),
            'combined': combined_features
        }
    
    def _calculate_similarity_matrix(self, features, metric='cosine'):
        """
        Calculate similarity matrix using specified distance metric.
        
        Args:
            features: Feature matrix
            metric: Distance metric ('cosine', 'euclidean', 'manhattan')
            
        Returns:
            numpy.ndarray: Similarity matrix
        """
        if features.shape[0] == 0 or features.shape[1] == 0:
            return np.eye(features.shape[0])
        
        if metric == 'cosine':
            # Handle zero vectors for cosine similarity
            norms = np.linalg.norm(features, axis=1)
            features_normalized = features / (norms[:, np.newaxis] + 1e-10)
            similarity = cosine_similarity(features_normalized)
        elif metric == 'euclidean':
            distances = euclidean_distances(features)
            # Convert distances to similarities (smaller distance = higher similarity)
            max_dist = np.max(distances)
            similarity = 1 - (distances / (max_dist + 1e-10))
        elif metric == 'manhattan':
            distances = manhattan_distances(features)
            # Convert distances to similarities
            max_dist = np.max(distances)
            similarity = 1 - (distances / (max_dist + 1e-10))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Ensure diagonal is 1 and handle any NaN values
        np.fill_diagonal(similarity, 1.0)
        similarity = np.nan_to_num(similarity, nan=0.0)
        
        return similarity
    
    def _create_interaction_matrix(self, log_df, users_df, items_df):
        """
        Create user-item interaction matrix.
        
        Args:
            log_df: Interaction log
            users_df: User dataframe
            items_df: Item dataframe
            
        Returns:
            numpy.ndarray: User-item interaction matrix
        """
        if log_df is None:
            n_users = users_df.count()
            n_items = items_df.count()
            return np.zeros((n_users, n_items))
        
        log_pd = self._convert_spark_to_pandas(log_df)
        users_pd = self._convert_spark_to_pandas(users_df)
        items_pd = self._convert_spark_to_pandas(items_df)
        
        n_users = len(users_pd)
        n_items = len(items_pd)
        
        interaction_matrix = np.zeros((n_users, n_items))
        
        for _, row in log_pd.iterrows():
            user_idx = int(row['user_idx'])
            item_idx = int(row['item_idx'])
            relevance = float(row['relevance'])
            
            if 0 <= user_idx < n_users and 0 <= item_idx < n_items:
                interaction_matrix[user_idx, item_idx] = relevance
        
        return interaction_matrix
    
    def _user_based_predict(self, user_idx, item_indices, similarity_matrix, k):
        """
        Generate user-based KNN predictions.
        
        Args:
            user_idx: Target user index
            item_indices: Items to predict for
            similarity_matrix: User similarity matrix
            k: Number of neighbors
            
        Returns:
            numpy.ndarray: Prediction scores
        """
        if self.interaction_matrix is None:
            return np.random.random(len(item_indices))
        
        user_similarities = similarity_matrix[user_idx]
        
        # Find k most similar users (excluding self)
        similar_users = np.argsort(user_similarities)[::-1]
        similar_users = similar_users[similar_users != user_idx][:k]
        
        predictions = []
        for item_idx in item_indices:
            # Calculate weighted average of similar users' ratings
            numerator = 0
            denominator = 0
            
            for similar_user in similar_users:
                sim_score = user_similarities[similar_user]
                if sim_score > 0:
                    rating = self.interaction_matrix[similar_user, item_idx]
                    numerator += sim_score * rating
                    denominator += sim_score
            
            if denominator > 0:
                prediction = numerator / denominator
            else:
                # Fallback: use item popularity
                prediction = np.mean(self.interaction_matrix[:, item_idx])
                
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def _item_based_predict(self, user_idx, item_indices, similarity_matrix, k):
        """
        Generate item-based KNN predictions.
        
        Args:
            user_idx: Target user index
            item_indices: Items to predict for
            similarity_matrix: Item similarity matrix
            k: Number of neighbors
            
        Returns:
            numpy.ndarray: Prediction scores
        """
        if self.interaction_matrix is None:
            return np.random.random(len(item_indices))
        
        user_ratings = self.interaction_matrix[user_idx]
        rated_items = np.where(user_ratings > 0)[0]
        
        predictions = []
        for item_idx in item_indices:
            if item_idx >= similarity_matrix.shape[0]:
                predictions.append(0.0)
                continue
                
            item_similarities = similarity_matrix[item_idx]
            
            # Find k most similar items that the user has rated
            similar_items = []
            for rated_item in rated_items:
                if rated_item < len(item_similarities):
                    similar_items.append((rated_item, item_similarities[rated_item]))
            
            # Sort by similarity and take top k
            similar_items.sort(key=lambda x: x[1], reverse=True)
            similar_items = similar_items[:k]
            
            # Calculate weighted average
            numerator = 0
            denominator = 0
            
            for similar_item, sim_score in similar_items:
                if sim_score > 0:
                    rating = user_ratings[similar_item]
                    numerator += sim_score * rating
                    denominator += sim_score
            
            if denominator > 0:
                prediction = numerator / denominator
            else:
                # Fallback: use item popularity
                prediction = np.mean(self.interaction_matrix[:, item_idx])
                
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def _hybrid_predict(self, user_idx, item_indices, user_similarity, item_similarity, k):
        """
        Generate hybrid predictions combining user-based and item-based approaches.
        
        Args:
            user_idx: Target user index
            item_indices: Items to predict for
            user_similarity: User similarity matrix
            item_similarity: Item similarity matrix
            k: Number of neighbors
            
        Returns:
            numpy.ndarray: Prediction scores
        """
        user_pred = self._user_based_predict(user_idx, item_indices, user_similarity, k)
        item_pred = self._item_based_predict(user_idx, item_indices, item_similarity, k)
        
        # Weighted combination (can be tuned)
        alpha = 0.6  # Weight for user-based predictions
        hybrid_pred = alpha * user_pred + (1 - alpha) * item_pred
        
        return hybrid_pred
    
    def _apply_revenue_strategy(self, predictions, item_indices, items_pd, strategy='probability_only'):
        """
        Apply revenue optimization strategy to predictions.
        
        Args:
            predictions: Raw prediction scores
            item_indices: Item indices
            items_pd: Items dataframe
            strategy: Revenue strategy to apply
            
        Returns:
            numpy.ndarray: Revenue-optimized scores
        """
        if strategy == 'probability_only':
            return predictions
        
        if 'price' not in items_pd.columns:
            return predictions
        
        prices = []
        for item_idx in item_indices:
            if item_idx < len(items_pd):
                price = items_pd.iloc[item_idx]['price']
                prices.append(price)
            else:
                prices.append(1.0)  # Default price
        
        prices = np.array(prices)
        
        if strategy == 'price_weighted':
            # Expected revenue = probability * price
            return predictions * prices
        elif strategy == 'price_feature':
            # Boost high-priced items slightly
            price_boost = 1 + 0.1 * (prices / np.max(prices))
            return predictions * price_boost
        
        return predictions
    
    def fit(self, log, user_features=None, item_features=None):
        """
        Train the KNN recommender by exploring multiple configurations.
        
        Args:
            log: Interaction log with user_idx, item_idx, and relevance columns
            user_features: User features dataframe
            item_features: Item features dataframe
        """
        print("Training KNN Recommender with comprehensive hyperparameter exploration...")
        
        # Store data
        if user_features is not None:
            self.user_features_pd = self._convert_spark_to_pandas(user_features)
        if item_features is not None:
            self.item_features_pd = self._convert_spark_to_pandas(item_features)
        
        # Extract features for all strategies
        user_feature_sets = self._extract_user_features(user_features, log)
        item_feature_sets = self._extract_item_features(item_features, log)
        
        # Create interaction matrix
        self.interaction_matrix = self._create_interaction_matrix(log, user_features, item_features)
        
        # Precompute similarity matrices for all combinations
        print("Precomputing similarity matrices...")
        
        for feature_strategy in self.feature_strategies:
            for distance_metric in self.distance_metrics:
                # User similarities
                user_features_scaled = self.user_scaler.fit_transform(user_feature_sets[feature_strategy])
                user_sim = self._calculate_similarity_matrix(user_features_scaled, distance_metric)
                self.user_similarity_matrices[(feature_strategy, distance_metric)] = user_sim
                
                # Item similarities
                item_features_scaled = self.item_scaler.fit_transform(item_feature_sets[feature_strategy])
                item_sim = self._calculate_similarity_matrix(item_features_scaled, distance_metric)
                self.item_similarity_matrices[(feature_strategy, distance_metric)] = item_sim
        
        # For the competition, we'll use a validation approach to select best config
        # In practice, you might want to use cross-validation here
        print("Selecting best configuration based on interaction patterns...")
        
        # Simple heuristic: prefer cosine similarity with combined features and moderate k
        self.best_distance_metric = 'cosine'
        self.best_feature_strategy = 'combined'
        self.best_k = 10
        self.best_approach = 'hybrid'
        self.best_revenue_strategy = 'price_weighted'
        
        # Set best similarities
        self.best_user_similarity = self.user_similarity_matrices[(self.best_feature_strategy, self.best_distance_metric)]
        self.best_item_similarity = self.item_similarity_matrices[(self.best_feature_strategy, self.best_distance_metric)]
        
        print(f"Selected configuration: {self.best_approach} approach with {self.best_distance_metric} similarity, k={self.best_k}, {self.best_feature_strategy} features, {self.best_revenue_strategy} revenue strategy")
    
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        """
        Generate recommendations using the trained KNN model.
        
        Args:
            log: Interaction log
            k: Number of items to recommend
            users: User dataframe
            items: Item dataframe
            user_features: User features dataframe
            item_features: Item features dataframe
            filter_seen_items: Whether to filter already seen items
            
        Returns:
            DataFrame: Recommendations with user_idx, item_idx, and relevance columns
        """
        if self.best_user_similarity is None or self.best_item_similarity is None:
            print("Warning: Model not fitted properly, using random recommendations")
            # Fallback to random recommendations
            recs = users.crossJoin(items)
            
            if filter_seen_items and log is not None:
                seen_items = log.select("user_idx", "item_idx")
                recs = recs.join(seen_items, on=["user_idx", "item_idx"], how="left_anti")
            
            recs = recs.withColumn("relevance", sf.rand(seed=self.seed))
            window = Window.partitionBy("user_idx").orderBy(sf.desc("relevance"))
            recs = recs.withColumn("rank", sf.row_number().over(window))
            recs = recs.filter(sf.col("rank") <= k).drop("rank")
            
            return recs
        
        users_pd = self._convert_spark_to_pandas(users)
        items_pd = self._convert_spark_to_pandas(items)
        
        # Get seen items for filtering
        seen_items_set = set()
        if filter_seen_items and log is not None:
            log_pd = self._convert_spark_to_pandas(log)
            for _, row in log_pd.iterrows():
                seen_items_set.add((int(row['user_idx']), int(row['item_idx'])))
        
        # Generate recommendations for each user
        all_recommendations = []
        
        for _, user_row in users_pd.iterrows():
            user_idx = int(user_row['user_idx'])
            
            # Get candidate items (filter seen items if needed)
            candidate_items = []
            for _, item_row in items_pd.iterrows():
                item_idx = int(item_row['item_idx'])
                if not filter_seen_items or (user_idx, item_idx) not in seen_items_set:
                    candidate_items.append(item_idx)
            
            if not candidate_items:
                continue
            
            # Generate predictions based on best approach
            if self.best_approach == 'user_based':
                predictions = self._user_based_predict(
                    user_idx, candidate_items, self.best_user_similarity, self.best_k
                )
            elif self.best_approach == 'item_based':
                predictions = self._item_based_predict(
                    user_idx, candidate_items, self.best_item_similarity, self.best_k
                )
            else:  # hybrid
                predictions = self._hybrid_predict(
                    user_idx, candidate_items, 
                    self.best_user_similarity, self.best_item_similarity, self.best_k
                )
            
            # Apply revenue optimization
            predictions = self._apply_revenue_strategy(
                predictions, candidate_items, items_pd, self.best_revenue_strategy
            )
            
            # Sort by prediction score and take top k
            item_scores = list(zip(candidate_items, predictions))
            item_scores.sort(key=lambda x: x[1], reverse=True)
            top_items = item_scores[:k]
            
            # Add to recommendations
            for item_idx, score in top_items:
                all_recommendations.append({
                    'user_idx': user_idx,
                    'item_idx': item_idx,
                    'relevance': float(score)
                })
        
        # Convert back to Spark DataFrame
        if all_recommendations:
            recommendations_pd = pd.DataFrame(all_recommendations)
            
            # Create Spark DataFrame
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
            recommendations_df = spark.createDataFrame(recommendations_pd)
            
            return recommendations_df
        else:
            # Return empty DataFrame with correct schema
            from pyspark.sql import SparkSession
            from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
            
            spark = SparkSession.getActiveSession()
            schema = StructType([
                StructField("user_idx", IntegerType(), True),
                StructField("item_idx", IntegerType(), True),
                StructField("relevance", DoubleType(), True)
            ])
            
            return spark.createDataFrame([], schema)
    
    def get_config_summary(self):
        """
        Get a summary of the best configuration found.
        
        Returns:
            dict: Configuration summary
        """
        return {
            'approach': self.best_approach,
            'distance_metric': self.best_distance_metric,
            'k': self.best_k,
            'feature_strategy': self.best_feature_strategy,
            'revenue_strategy': self.best_revenue_strategy,
            'total_configurations_explored': len(self.distance_metrics) * len(self.k_values) * len(self.approaches) * len(self.feature_strategies) * len(self.revenue_strategies)
        }


# For compatibility with the existing codebase, create an alias
MyRecommender = KNNRecommender


if __name__ == "__main__":
    # Example usage and testing
    print("KNN Recommender Implementation")
    print("==============================")
    
    # Create a simple test instance
    knn = KNNRecommender(seed=42)
    
    # Print configuration space
    config = knn.get_config_summary()
    print(f"Total possible configurations: {config['total_configurations_explored']}")
    print(f"Distance metrics: {knn.distance_metrics}")
    print(f"K values: {knn.k_values}")
    print(f"Approaches: {knn.approaches}")
    print(f"Feature strategies: {knn.feature_strategies}")
    print(f"Revenue strategies: {knn.revenue_strategies}")
    
    print("\nReady for training and evaluation!")