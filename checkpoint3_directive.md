# CS145 Recommendation System Competition - Checkpoint 3: Graph-Based Recommenders

## Overview

The third checkpoint focuses on **Graph-Based Recommenders** that leverage user-item interaction networks for link prediction. Your goal is to implement algorithms that utilize graph structure and collaborative filtering signals to maximize revenue by predicting missing user-item edges.

## Task

Implement at least three different graph-based recommendation approaches using link prediction techniques. Your implementation should:

1. Construct user-item bipartite graphs from interaction histories
2. Build models that predict user-item edge probabilities using graph neural networks
3. Generate ranked recommendations that maximize revenue. For example, one way of estimating revenue would be `expected_revenue = price × link_probability`. 
4. Compare performance of different graph architectures

## Dataset

You'll work with the same synthetic dataset, treating it as a dynamic graph:

- **Nodes**: Users and items with optional feature attributes
- **Edges**: Historical interactions (clicks, purchases) with weights
- **Graph evolution**: Incrementally update graph after each training iteration

## Implementation Requirements

### 1. Graph Construction

- Build bipartite user-item graphs from interaction logs
- Consider edge weights (frequency, recency, purchase amount)
- Handle graph updates as new interactions arrive
- Optionally add user-user or item-item connections

### 2. Model Implementation

Implement at least three approaches from:

a) **Graph Convolutional Networks (GCN)**:
   - Message passing on bipartite graphs
   - Tune embedding size (64, 128, 256) and layers (2-4)

b) **GraphSAGE**:
   - Inductive learning with neighborhood sampling
   - Experiment with sampling sizes and aggregator types

c) **Graph Attention Networks (GAT)**:
   - Learn attention weights for neighbor importance
   - Tune attention heads (2-8) and dropout rates

d) **LightGCN**:
   - Simplified GCN for collaborative filtering
   - Focus on embedding propagation without transformations

e) **Node2Vec/DeepWalk**:
   - Random walk-based graph embeddings
   - Tune walk parameters and embedding dimensions

### 3. Link Prediction

Implement effective prediction strategies:

- **Embedding fusion**: Combine user and item embeddings (dot product, cosine, MLP)
- **Negative sampling**: Sample non-interacted user-item pairs for training
- **Revenue ranking**: Sort by predicted probability × item price

### 4. Regularization Techniques

Experiment with at least two regularization schemes:

- **Embedding regularization**: L2 penalty on learned embeddings
- **Graph dropout**: Randomly remove nodes/edges during training
- **Early stopping**: Monitor validation performance
- **Batch normalization**: Stabilize deep GNN training

### 5. Evaluation and Comparison

Compare implemented models using:

- **Primary metric**: Discounted revenue
- **Secondary metrics**: Total revenue, Precision@K, NDCG@K, MRR, Hit Rate
- **Graph metrics**: Link prediction AUC, embedding visualization
- **Ablation studies**: Impact of graph construction choices

## Optimization Tips

1. **Graph sampling**: Use mini-batch training with neighbor sampling for scalability
2. **Cold start**: Initialize new user/item embeddings with content features
3. **Temporal decay**: Weight recent interactions more heavily
4. **Multi-hop reasoning**: Leverage higher-order graph connectivity

## Allowed Tools and Libraries

You may use:
- PyTorch Geometric (PyG), NetworkX
- pandas, numpy, scikit-learn, PyTorch

## Deliverables

Be prepared to submit:

1. **Code**: Your implementation in a well-organized Python module
2. **Report** containing:
   - Graph construction methodology
   - Implemented GNN architectures and design choices
   - Hyperparameter tuning results
   - Performance comparison between graph models
   - Comparison with content-based and sequence-based approaches

3. **Presentation slides** (5-8 slides) summarizing your approach and results

## Grading Criteria (Tentative) 

- **Implementation correctness**: 40%
- **Model performance**: 30%
- **Approach diversity**: 15%
- **Report quality and insights**: 15%

## Getting Started

1. Extend your existing pipeline to build user-item graphs
2. Start with Node2Vec or basic GCN as baseline
3. Implement more sophisticated GNN architectures
4. Experiment with different graph construction strategies
5. Analyze learned embeddings and graph statistics

## Integration with Previous Checkpoints

Consider combining insights from all approaches:
- **Hybrid models**: Merge content features, sequences, and graph structure
- **Ensemble methods**: Weighted combination of different paradigms
- **Feature fusion**: Use content/sequence features as node attributes

Remember: The goal is to maximize **discounted revenue** by leveraging graph structure to recommend items that users are likely to purchase! 