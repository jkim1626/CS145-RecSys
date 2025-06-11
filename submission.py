import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
import pandas as pd

class BaseRecommender:
    def __init__(self, seed=None):
        self.seed = seed
        np.random.seed(seed)
    def fit(self, log, user_features=None, item_features=None):
        """
        No training needed for random recommender.
        
        Args:
            log: Interaction log
            user_features: User features (optional)
            item_features: Item features (optional)
        """
        # No training needed
        raise NotImplemented()
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        raise NotImplemented()
    
import sklearn 
from sklearn.preprocessing import StandardScaler
from sim4rec.utils import pandas_to_spark

class MyRecommender(BaseRecommender):
    def __init__(self, seed=None):
        super().__init__(seed)
        self.model = sklearn.svm.SVC(
            kernel='rbf',
            probability=True,
            random_state=self.seed
        )
        self.scalar = StandardScaler()

    def fit(self, log:DataFrame, user_features=None, item_features=None):
        # log.show(5)
        # user_features.show(5)
        # item_features.show(5)

        if user_features and item_features:
            pd_log = log.join(
                user_features, 
                on='user_idx'
            ).join(
                item_features, 
                on='item_idx'
            ).drop(
                'user_idx', 'item_idx', '__iter'
            ).toPandas()

            pd_log = pd.get_dummies(pd_log, dtype=float)
            pd_log['price'] = self.scalar.fit_transform(pd_log[['price']])

            y = pd_log['relevance']
            x = pd_log.drop(['relevance'], axis=1)

            self.model.fit(x,y)

    def predict(self, log, k, users:DataFrame, items:DataFrame, user_features=None, item_features=None, filter_seen_items=True):
        cross = users.join(
            items
        ).drop('__iter').toPandas().copy()

        cross = pd.get_dummies(cross, dtype=float)
        cross['orig_price'] = cross['price']
        cross['price'] = self.scalar.transform(cross[['price']])

        cross['prob'] = self.model.predict_proba(cross.drop(['user_idx', 'item_idx', 'orig_price'], axis=1))[:,np.where(self.model.classes_ == 1)[0][0]]
        
        cross['relevance'] = (np.sin(cross['prob']) + 1) * np.exp(cross['prob'] - 1) * np.log1p(cross["orig_price"]) * np.cos(cross["orig_price"] / 100) * (1 + np.tan(cross['prob'] * np.pi / 4))
        
        cross = cross.sort_values(by=['user_idx', 'relevance'], ascending=[True, False])
        cross = cross.groupby('user_idx').head(k)

        cross['price'] = cross['orig_price']
        
        # Convert back to Spark and fix schema types to match original log
        from pyspark.sql.types import LongType
        result = pandas_to_spark(cross)
        result = result.withColumn("user_idx", sf.col("user_idx").cast(LongType()))
        result = result.withColumn("item_idx", sf.col("item_idx").cast(LongType()))
       
        return result
 