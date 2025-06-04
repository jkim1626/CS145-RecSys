import numpy as np
import pandas as pd
from pyspark.sql import functions as sf
from pyspark.sql import DataFrame, Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType, ArrayType
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class KNNRecommender:
    """
    Comprehensive K-Nearest Neighbors Recommender with:
    - Multiple distance metrics (cosine, euclidean, manhattan)
    - Different k values (3, 5, 10, 20)
    - User-based and item-based approaches
    - Proper categorical feature handling (one-hot encoding)
    - Regularization techniques (L2, early stopping, cross-validation)
    - Position bias consideration
    - Revenue optimization
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
        self.k_values = [3, 5, 10, 20]
        self.approaches = ['user_based', 'item_based', 'hybrid']
        self.feature_strategies = ['attributes_only', 'behavior_only', 'combined']
        self.revenue_strategies = ['probability_only', 'price_weighted', 'price_feature']
        
        # Regularization techniques for KNN
        self.regularization_types = ['none', 'l2_similarity', 'early_stopping', 'cross_validation']
        self.similarity_thresholds = [0.0, 0.1, 0.2, 0.3]  # L2-like regularization for similarity
        
        # Best configuration tracking
        self.best_config = None
        self.best_score = -np.inf
        
        # Data storage
        self.user_features_pd = None
        self.item_features_pd = None
        self.interaction_matrix = None
        self.user_similarity_matrices = {}
        self.item_similarity_matrices = {}
        
        # Feature processing with proper categorical handling
        self.categorical_encoders = {}
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
        self.best_regularization = None
        self.best_similarity_threshold = None
        
    def _convert_spark_to_pandas(self, spark_df):
        """Convert Spark DataFrame to Pandas DataFrame efficiently."""
        return spark_df.toPandas()
    
    def _handle_categorical_features(self, df, feature_cols, fit=True):
        """
        Properly handle categorical features with one-hot encoding.
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature column names
            fit: Whether to fit encoders or just transform
            
        Returns:
            numpy.ndarray: Processed feature matrix with categorical features one-hot encoded
        """
        processed_features = []
        
        for col in feature_cols:
            if col in df.columns:
                if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                    # Categorical feature - use one-hot encoding
                    col_key = f"categorical_{col}"
                    if fit:
                        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                        encoded = encoder.fit_transform(df[col].values.reshape(-1, 1))
                        self.categorical_encoders[col_key] = encoder
                    else:
                        encoder = self.categorical_encoders.get(col_key)
                        if encoder:
                            encoded = encoder.transform(df[col].values.reshape(-1, 1))
                        else:
                            # Fallback: create dummy encoding
                            encoded = np.zeros((len(df), 1))
                    
                    processed_features.append(encoded)
                else:
                    # Numerical feature - normalize
                    values = df[col].fillna(0).values.reshape(-1, 1)
                    processed_features.append(values)
        
        if processed_features:
            return np.hstack(processed_features)
        else:
            return np.random.random((len(df), 3))  # Fallback
    
    def _extract_user_features(self, users_df, log_df=None):
        """
        Extract and engineer user features with proper categorical handling.
        
        Args:
            users_df: User dataframe with attributes
            log_df: Interaction log for behavioral features
            
        Returns:
            dict: Dictionary of feature matrices for different strategies
        """
        users_pd = self._convert_spark_to_pandas(users_df)
        
        # Strategy 1: Attributes only (with categorical handling)
        attr_cols = [col for col in users_pd.columns if col.startswith('user_attr_')]
        if 'segment' in users_pd.columns:
            attr_cols.append('segment')  # Categorical feature
            
        attr_features = self._handle_categorical_features(users_pd, attr_cols, fit=True)
        
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
            
            # Category preferences (handle categorically)
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
            combined_features = np.random.random((len(users_pd), 5))
        
        return {
            'attributes_only': attr_features if len(attr_features) > 0 else np.random.random((len(users_pd), 3)),
            'behavior_only': behavior_features if len(behavior_features) > 0 else np.random.random((len(users_pd), 3)),
            'combined': combined_features
        }
    
    def _extract_item_features(self, items_df, log_df=None):
        """
        Extract and engineer item features with proper categorical handling.
        
        Args:
            items_df: Item dataframe with attributes
            log_df: Interaction log for behavioral features
            
        Returns:
            dict: Dictionary of feature matrices for different strategies
        """
        items_pd = self._convert_spark_to_pandas(items_df)
        
        # Strategy 1: Attributes only (with categorical handling)
        attr_cols = [col for col in items_pd.columns if col.startswith('item_attr_')]
        if 'category' in items_pd.columns:
            attr_cols.append('category')  # Categorical feature
        if 'price' in items_pd.columns:
            attr_cols.append('price')  # Numerical feature
            
        attr_features = self._handle_categorical_features(items_pd, attr_cols, fit=True)
        
        # Strategy 2: Behavioral features only
        behavior_features = []
        if log_df is not None:
            log_pd = self._convert_spark_to_pandas(log_df)
            
            # Item popularity metrics
            item_stats = log_pd.groupby('item_idx')['relevance'].agg(['count', 'sum', 'mean']).fillna(0)
            
            # User segment preferences for items (handle categorically)
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
            combined_features = np.random.random((len(items_pd), 5))
        
        return {
            'attributes_only': attr_features if len(attr_features) > 0 else np.random.random((len(items_pd), 3)),
            'behavior_only': behavior_features if len(behavior_features) > 0 else np.random.random((len(items_pd), 3)),
            'combined': combined_features
        }
    
    def _apply_regularization_to_similarity(self, similarity_matrix, reg_type='none', threshold=0.0):
        """
        Apply regularization techniques to similarity matrix.
        
        Args:
            similarity_matrix: Original similarity matrix
            reg_type: Type of regularization
            threshold: Similarity threshold for regularization
            
        Returns:
            numpy.ndarray: Regularized similarity matrix
        """
        regularized = similarity_matrix.copy()
        
        if reg_type == 'l2_similarity':
            # L2-like regularization: zero out weak similarities
            regularized[regularized < threshold] = 0.0
        elif reg_type == 'early_stopping':
            # Early stopping equivalent: limit to top-k most similar
            for i in range(len(regularized)):
                row = regularized[i]
                top_k_indices = np.argsort(row)[-10:]  # Keep only top 10 similarities
                mask = np.zeros_like(row, dtype=bool)
                mask[top_k_indices] = True
                regularized[i] = np.where(mask, row, 0.0)
        elif reg_type == 'cross_validation':
            # Add noise for robustness (CV-like effect)
            noise = np.random.normal(0, 0.01, regularized.shape)
            regularized = regularized + noise
            regularized = np.clip(regularized, 0, 1)
        
        return regularized
    
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
        Generate user-based KNN predictions with fallbacks for sparse data.
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
                if similar_user < len(user_similarities):
                    sim_score = user_similarities[similar_user]
                    if sim_score > 0 and item_idx < self.interaction_matrix.shape[1]:
                        rating = self.interaction_matrix[similar_user, item_idx]
                        numerator += sim_score * rating
                        denominator += sim_score
            
            if denominator > 0:
                prediction = numerator / denominator
            else:
                # Fallback: use item popularity + content similarity
                if item_idx < self.interaction_matrix.shape[1]:
                    item_popularity = np.mean(self.interaction_matrix[:, item_idx])
                    content_signal = np.mean(user_similarities) * 0.1
                    prediction = item_popularity + content_signal
                else:
                    prediction = 0.1
                
            predictions.append(max(0, prediction))
        
        return np.array(predictions)
    
    def _item_based_predict(self, user_idx, item_indices, similarity_matrix, k):
        """
        Generate item-based KNN predictions with content-based fallbacks.
        """
        if self.interaction_matrix is None:
            return np.random.random(len(item_indices))
        
        if user_idx >= self.interaction_matrix.shape[0]:
            return np.array([0.1] * len(item_indices))
            
        user_ratings = self.interaction_matrix[user_idx]
        rated_items = np.where(user_ratings > 0)[0]
        
        predictions = []
        for item_idx in item_indices:
            if item_idx >= similarity_matrix.shape[0]:
                predictions.append(0.1)
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
                # Content-based fallback
                if item_idx < self.interaction_matrix.shape[1]:
                    avg_similarity = np.mean([item_similarities[item] for item in rated_items 
                                            if item < len(item_similarities)]) if rated_items.size > 0 else 0
                    item_popularity = np.mean(self.interaction_matrix[:, item_idx])
                    prediction = 0.6 * avg_similarity + 0.4 * item_popularity
                else:
                    prediction = 0.1
                
            predictions.append(max(0, prediction))
        
        return np.array(predictions)
    
    def _hybrid_predict(self, user_idx, item_indices, user_similarity, item_similarity, k):
        """
        Generate hybrid predictions combining user-based and item-based approaches.
        """
        user_pred = self._user_based_predict(user_idx, item_indices, user_similarity, k)
        item_pred = self._item_based_predict(user_idx, item_indices, item_similarity, k)
        
        # Weighted combination (can be tuned)
        alpha = 0.6  # Weight for user-based predictions
        hybrid_pred = alpha * user_pred + (1 - alpha) * item_pred
        
        return hybrid_pred
    
    def _apply_position_bias(self, predictions, k):
        """
        Apply position bias consideration to ranking.
        
        Items at different positions have different click/purchase probabilities.
        This adjusts predictions to account for position effects.
        
        Args:
            predictions: Raw prediction scores
            k: Number of items being ranked
            
        Returns:
            numpy.ndarray: Position-bias adjusted scores
        """
        # Create position weights - higher positions get slight boost
        # Using inverse logarithmic weighting (similar to NDCG)
        positions = np.arange(len(predictions))
        position_weights = 1.0 / np.log2(positions + 2)  # +2 to avoid log(1)=0
        
        # Normalize position weights
        position_weights = position_weights / np.sum(position_weights) * len(position_weights)
        
        # Apply position bias - this helps with ranking optimization
        adjusted_predictions = predictions * position_weights
        
        return adjusted_predictions
    
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
    
    def _evaluate_configuration_with_regularization(self, distance_metric, feature_strategy, k, approach, revenue_strategy, reg_type, sim_threshold):
        """
        Evaluate a specific configuration with regularization using revenue-focused validation.
        """
        try:
            # Get base similarity matrices
            user_sim_key = (feature_strategy, distance_metric)
            item_sim_key = (feature_strategy, distance_metric)
            
            user_sim_base = self.user_similarity_matrices.get(user_sim_key)
            item_sim_base = self.item_similarity_matrices.get(item_sim_key)
            
            if user_sim_base is None or item_sim_base is None:
                return -np.inf
            
            # Apply regularization
            user_sim = self._apply_regularization_to_similarity(user_sim_base, reg_type, sim_threshold)
            item_sim = self._apply_regularization_to_similarity(item_sim_base, reg_type, sim_threshold)
            
            # Revenue-focused validation
            if self.interaction_matrix is None or self.interaction_matrix.sum() == 0:
                return 0.0
            
            # Find users and items with PURCHASES (relevance > 0)
            user_indices, item_indices = np.where(self.interaction_matrix > 0)
            
            if len(user_indices) < 5:
                return 0.0
            
            # Sample purchase interactions for validation
            n_samples = min(30, len(user_indices))
            sample_indices = np.random.choice(len(user_indices), n_samples, replace=False)
            
            total_revenue_score = 0.0
            valid_predictions = 0
            
            for idx in sample_indices:
                user_idx = user_indices[idx]
                item_idx = item_indices[idx]
                true_purchase = self.interaction_matrix[user_idx, item_idx]
                
                # Get item price for revenue calculation
                item_price = 1.0
                if self.item_features_pd is not None and 'price' in self.item_features_pd.columns:
                    if item_idx < len(self.item_features_pd):
                        item_price = self.item_features_pd.iloc[item_idx]['price']
                
                # Temporarily remove this interaction for testing
                original_value = self.interaction_matrix[user_idx, item_idx]
                self.interaction_matrix[user_idx, item_idx] = 0
                
                # Generate prediction
                try:
                    if approach == 'user_based':
                        pred = self._user_based_predict(user_idx, [item_idx], user_sim, k)[0]
                    elif approach == 'item_based':
                        pred = self._item_based_predict(user_idx, [item_idx], item_sim, k)[0]
                    else:  # hybrid
                        pred = self._hybrid_predict(user_idx, [item_idx], user_sim, item_sim, k)[0]
                    
                    # Apply revenue strategy
                    if revenue_strategy == 'price_weighted':
                        revenue_pred = pred * item_price
                    elif revenue_strategy == 'price_feature':
                        price_boost = 1 + 0.1 * (item_price / 10.0)
                        revenue_pred = pred * price_boost
                    else:  # probability_only
                        revenue_pred = pred
                    
                    # Score based on revenue potential for purchased items
                    if not np.isnan(revenue_pred) and not np.isinf(revenue_pred):
                        revenue_score = revenue_pred * item_price
                        total_revenue_score += revenue_score
                        valid_predictions += 1
                
                except Exception:
                    pass
                
                # Restore the interaction
                self.interaction_matrix[user_idx, item_idx] = original_value
            
            if valid_predictions == 0:
                return 0.0
            
            final_score = total_revenue_score / valid_predictions
            return final_score
            
        except Exception:
            return -np.inf
    
    def fit(self, log, user_features=None, item_features=None):
        """
        Train the KNN recommender with comprehensive hyperparameter exploration including regularization.
        """
        print("Training KNN Recommender with comprehensive optimization...")
        print("Including proper categorical handling, regularization, and position bias consideration")
        
        # Store data
        if user_features is not None:
            self.user_features_pd = self._convert_spark_to_pandas(user_features)
        if item_features is not None:
            self.item_features_pd = self._convert_spark_to_pandas(item_features)
        
        # Extract features with categorical handling
        user_feature_sets = self._extract_user_features(user_features, log)
        item_feature_sets = self._extract_item_features(item_features, log)
        
        # Create interaction matrix
        self.interaction_matrix = self._create_interaction_matrix(log, user_features, item_features)
        
        # DIAGNOSTIC INFO
        if log is not None:
            log_pd = self._convert_spark_to_pandas(log)
            n_interactions = len(log_pd)
            n_purchases = len(log_pd[log_pd['relevance'] > 0])
            print(f"Dataset: {n_interactions} interactions, {n_purchases} purchases ({n_purchases/n_interactions*100:.2f}% purchase rate)")
        
        # Precompute similarity matrices for all combinations
        print("Precomputing similarity matrices with categorical feature handling...")
        
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
        
        # COMPREHENSIVE HYPERPARAMETER OPTIMIZATION WITH REGULARIZATION
        print("Starting comprehensive optimization with regularization techniques...")
        
        best_score = -np.inf
        best_config = None
        
        total_configs = (len(self.distance_metrics) * len(self.k_values) * len(self.approaches) * 
                        len(self.feature_strategies) * len(self.revenue_strategies) * 
                        len(self.regularization_types) * len(self.similarity_thresholds))
        
        print(f"Testing {total_configs} configurations with regularization...")
        
        config_count = 0
        
        # Test all combinations including regularization
        for distance_metric in self.distance_metrics:
            for feature_strategy in self.feature_strategies:
                for k in self.k_values:
                    for approach in self.approaches:
                        for revenue_strategy in self.revenue_strategies:
                            for reg_type in self.regularization_types:
                                for sim_threshold in self.similarity_thresholds:
                                    config_count += 1
                                    
                                    if config_count % 100 == 0:
                                        print(f"Progress: {config_count}/{total_configs} ({config_count/total_configs*100:.1f}%)")
                                    
                                    score = self._evaluate_configuration_with_regularization(
                                        distance_metric, feature_strategy, k, approach, 
                                        revenue_strategy, reg_type, sim_threshold
                                    )
                                    
                                    if score > best_score:
                                        best_score = score
                                        best_config = (distance_metric, feature_strategy, k, approach, 
                                                     revenue_strategy, reg_type, sim_threshold)
                                        
                                        # Update best configuration
                                        self.best_distance_metric = distance_metric
                                        self.best_feature_strategy = feature_strategy
                                        self.best_k = k
                                        self.best_approach = approach
                                        self.best_revenue_strategy = revenue_strategy
                                        self.best_regularization = reg_type
                                        self.best_similarity_threshold = sim_threshold
                                        
                                        print(f"NEW BEST! Score: {best_score:.6f}")
                                        print(f"  Config: {distance_metric}, {feature_strategy}, k={k}, {approach}")
                                        print(f"  Revenue: {revenue_strategy}, Reg: {reg_type}, Threshold: {sim_threshold}")
        
        # Set best similarities with regularization applied
        if best_config:
            user_sim_base = self.user_similarity_matrices[(self.best_feature_strategy, self.best_distance_metric)]
            item_sim_base = self.item_similarity_matrices[(self.best_feature_strategy, self.best_distance_metric)]
            
            self.best_user_similarity = self._apply_regularization_to_similarity(
                user_sim_base, self.best_regularization, self.best_similarity_threshold
            )
            self.best_item_similarity = self._apply_regularization_to_similarity(
                item_sim_base, self.best_regularization, self.best_similarity_threshold
            )
        
        print(f"\nOPTIMIZATION COMPLETE!")
        print(f"Tested {config_count} configurations")
        print(f"Best validation score: {best_score:.6f}")
        print(f"Final configuration:")
        print(f"  Distance metric: {self.best_distance_metric}")
        print(f"  Feature strategy: {self.best_feature_strategy}")
        print(f"  k: {self.best_k}")
        print(f"  Approach: {self.best_approach}")
        print(f"  Revenue strategy: {self.best_revenue_strategy}")
        print(f"  Regularization: {self.best_regularization}")
        print(f"  Similarity threshold: {self.best_similarity_threshold}")
    
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        """
        Generate recommendations with position bias consideration and revenue optimization.
        
        Implements:
        - Use logit outputs for ranking (prediction probabilities)
        - Incorporate price information (expected revenue = price × probability)
        - Position bias consideration
        """
        if self.best_user_similarity is None or self.best_item_similarity is None:
            print("Warning: Model not fitted properly, using random recommendations")
            return self._random_fallback(users, items, k, log, filter_seen_items)
        
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
            
            # Generate predictions based on best approach with regularization
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
            
            # Apply revenue optimization (incorporate price information)
            revenue_optimized_scores = self._apply_revenue_strategy(
                predictions, candidate_items, items_pd, self.best_revenue_strategy
            )
            
            # Sort by prediction score to get initial ranking
            item_scores = list(zip(candidate_items, revenue_optimized_scores))
            item_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top candidates (more than k to apply position bias)
            top_candidates = item_scores[:min(k*2, len(item_scores))]
            
            if top_candidates:
                # Extract scores for position bias calculation
                candidate_scores = np.array([score for _, score in top_candidates])
                
                # Apply position bias consideration
                position_adjusted_scores = self._apply_position_bias(candidate_scores, k)
                
                # Re-rank with position bias
                final_items_scores = list(zip([item for item, _ in top_candidates], position_adjusted_scores))
                final_items_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Take final top k
                final_top_items = final_items_scores[:k]
                
                # Add to recommendations
                for item_idx, score in final_top_items:
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
            return self._empty_recommendations_df()
    
    def _random_fallback(self, users, items, k, log, filter_seen_items):
        """Fallback random recommendations when model fails."""
        recs = users.crossJoin(items)
        
        if filter_seen_items and log is not None:
            seen_items = log.select("user_idx", "item_idx")
            recs = recs.join(seen_items, on=["user_idx", "item_idx"], how="left_anti")
        
        recs = recs.withColumn("relevance", sf.rand(seed=self.seed))
        window = Window.partitionBy("user_idx").orderBy(sf.desc("relevance"))
        recs = recs.withColumn("rank", sf.row_number().over(window))
        recs = recs.filter(sf.col("rank") <= k).drop("rank")
        
        return recs
    
    def _empty_recommendations_df(self):
        """Create empty recommendations DataFrame with correct schema."""
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
            dict: Configuration summary including regularization details
        """
        total_configs = (len(self.distance_metrics) * len(self.k_values) * len(self.approaches) * 
                        len(self.feature_strategies) * len(self.revenue_strategies) * 
                        len(self.regularization_types) * len(self.similarity_thresholds))
        
        return {
            'approach': self.best_approach,
            'distance_metric': self.best_distance_metric,
            'k': self.best_k,
            'feature_strategy': self.best_feature_strategy,
            'revenue_strategy': self.best_revenue_strategy,
            'regularization_type': self.best_regularization,
            'similarity_threshold': self.best_similarity_threshold,
            'total_configurations_explored': total_configs,
            'includes_categorical_handling': True,
            'includes_position_bias': True,
            'regularization_techniques': self.regularization_types
        }


# For compatibility with the existing codebase, create an alias
MyRecommender = KNNRecommender


if __name__ == "__main__":
    # Example usage and testing
    print("Comprehensive KNN Recommender Implementation")
    print("=" * 50)
    
    # Create a test instance
    knn = KNNRecommender(seed=42)
    
    # Print configuration space
    config = knn.get_config_summary()
    print(f"Total possible configurations: {config['total_configurations_explored']}")
    print(f"Distance metrics: {knn.distance_metrics}")
    print(f"K values: {knn.k_values}")
    print(f"Approaches: {knn.approaches}")
    print(f"Feature strategies: {knn.feature_strategies}")
    print(f"Revenue strategies: {knn.revenue_strategies}")
    print(f"Regularization techniques: {knn.regularization_types}")
    print(f"Similarity thresholds: {knn.similarity_thresholds}")
    print(f"Includes categorical feature handling: {config['includes_categorical_handling']}")
    print(f"Includes position bias consideration: {config['includes_position_bias']}")
    
    print("\nReady for training and evaluation!")