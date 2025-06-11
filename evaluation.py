import os
import sys
import numpy as np
import shutil

from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from pyspark.sql import DataFrame, Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType, ArrayType

# Initialize Spark session
spark = SparkSession.builder \
    .appName("RecSysEvaluation") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

# Set log level to warnings only
spark.sparkContext.setLogLevel("WARN")

# Import competition modules
from data_generator import CompetitionDataGenerator
from simulator import CompetitionSimulator
from config import DEFAULT_CONFIG, EVALUATION_METRICS

# Import our custom recommender
from submission import MyRecommender


def run_evaluation():
    """
    Run evaluation of MyRecommender using train-test split.
    This function creates a synthetic dataset and evaluates the custom recommender.
    """
    # Create a smaller dataset for experimentation
    config = DEFAULT_CONFIG.copy()
    config['data_generation']['n_users'] = 1000  # Reduced from 10,000
    config['data_generation']['n_items'] = 200   # Reduced from 1,000
    config['data_generation']['seed'] = 42       # Fixed seed for reproducibility
    
    # Get train-test split parameters
    train_iterations = config['simulation']['train_iterations']
    test_iterations = config['simulation']['test_iterations']
    
    print(f"Running train-test simulation with {train_iterations} training iterations and {test_iterations} testing iterations")
    
    # Initialize data generator
    data_generator = CompetitionDataGenerator(
        spark_session=spark,
        **config['data_generation']
    )
    
    # Generate user data
    users_df = data_generator.generate_users()
    print(f"Generated {users_df.count()} users")
    
    # Generate item data
    items_df = data_generator.generate_items()
    print(f"Generated {items_df.count()} items")
    
    # Generate initial interaction history
    history_df = data_generator.generate_initial_history(
        config['data_generation']['initial_history_density']
    )
    print(f"Generated {history_df.count()} initial interactions")
    
    # Set up data generators for simulator
    user_generator, item_generator = data_generator.setup_data_generators()
    
    # Initialize MyRecommender
    recommender = MyRecommender(seed=42)
    
    # Initialize recommender with initial history
    recommender.fit(log=data_generator.history_df, 
                    user_features=users_df, 
                    item_features=items_df)
    
    print(f"\nEvaluating MyRecommender:")
    
    # Clean up any existing simulator data directory
    simulator_data_dir = "simulator_train_test_data_MyRecommender"
    if os.path.exists(simulator_data_dir):
        shutil.rmtree(simulator_data_dir)
        print(f"Removed existing simulator data directory: {simulator_data_dir}")
    
    # Initialize simulator
    simulator = CompetitionSimulator(
        user_generator=user_generator,
        item_generator=item_generator,
        data_dir=simulator_data_dir,
        log_df=data_generator.history_df,
        conversion_noise_mean=config['simulation']['conversion_noise_mean'],
        conversion_noise_std=config['simulation']['conversion_noise_std'],
        spark_session=spark,
        seed=config['data_generation']['seed']
    )
    
    # Run simulation with train-test split
    train_metrics, test_metrics, train_revenue, test_revenue = simulator.train_test_split(
        recommender=recommender,
        train_iterations=train_iterations,
        test_iterations=test_iterations,
        user_frac=config['simulation']['user_fraction'],
        k=config['simulation']['k'],
        filter_seen_items=config['simulation']['filter_seen_items'],
        retrain=config['simulation']['retrain']
    )
    
    # Calculate average metrics
    train_avg_metrics = {}
    for metric_name in train_metrics[0].keys():
        values = [metrics[metric_name] for metrics in train_metrics]
        train_avg_metrics[f"train_{metric_name}"] = np.mean(values)
    
    test_avg_metrics = {}
    for metric_name in test_metrics[0].keys():
        values = [metrics[metric_name] for metrics in test_metrics]
        test_avg_metrics[f"test_{metric_name}"] = np.mean(values)
    
    # Store results
    results = {
        "name": "MyRecommender",
        "train_total_revenue": sum(train_revenue),
        "test_total_revenue": sum(test_revenue),
        "train_avg_revenue": np.mean(train_revenue),
        "test_avg_revenue": np.mean(test_revenue),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_revenue": train_revenue,
        "test_revenue": test_revenue,
        **train_avg_metrics,
        **test_avg_metrics
    }
    
    # Print summary
    print(f"\n=== MyRecommender Evaluation Results ===")
    print(f"Training Phase - Total Revenue: {sum(train_revenue):.2f}")
    print(f"Testing Phase - Total Revenue: {sum(test_revenue):.2f}")
    print(f"Training Phase - Average Revenue per Iteration: {np.mean(train_revenue):.2f}")
    print(f"Testing Phase - Average Revenue per Iteration: {np.mean(test_revenue):.2f}")
    
    performance_change = ((sum(test_revenue) / len(test_revenue)) / (sum(train_revenue) / len(train_revenue)) - 1) * 100
    print(f"Performance Change (Train → Test): {performance_change:.2f}%")
    
    # Print detailed metrics
    print(f"\n=== Training Phase Metrics ===")
    for metric_name, value in train_avg_metrics.items():
        clean_name = metric_name.replace('train_', '').replace('_', ' ').title()
        print(f"{clean_name}: {value:.4f}")
    
    print(f"\n=== Testing Phase Metrics ===")
    for metric_name, value in test_avg_metrics.items():
        clean_name = metric_name.replace('test_', '').replace('_', ' ').title()
        print(f"{clean_name}: {value:.4f}")
    
    # Print revenue trajectory
    print(f"\n=== Revenue Trajectory ===")
    print("Training iterations:", [f"{rev:.2f}" for rev in train_revenue])
    print("Testing iterations:", [f"{rev:.2f}" for rev in test_revenue])
    
    return results


if __name__ == "__main__":
    results = run_evaluation()
    print(f"\nEvaluation completed successfully!") 