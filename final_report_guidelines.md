# CS145 Recommendation System Competition - Final Report Guidelines

## Submission Deadline

**Sunday, June 15th, 2025 at 23:59 PM**

## Overview

Your final submission should demonstrate mastery of all three recommendation paradigms covered in this competition: content-based, sequence-based, and graph-based recommenders. The report should synthesize your learnings, present comprehensive experimental results, and articulate your winning strategy.

## Submission Components

### 1. Final Report (PDF)
- **Main Text**: 6-8 pages
- **Appendix**: Unlimited pages for figures, additional results, and detailed specifications

### 2. Code Repository
- Complete codebase covering all three checkpoints
- Final `submission.py` uploaded to the leaderboard platform
- Clear documentation and reproducible experiments

### 3. Presentation Materials
- Slides for 10-minute presentation
- Recorded presentation video

## Report Structure and Requirements

### Main Text (6-8 pages)

#### 1. Introduction and Team Information (0.5 page)
- **Team name and members**
- **Final score on leaderboard**
- Brief overview of your approach and key contributions

#### 2. Methodology Overview (1.5 pages)
- **Content-Based Recommenders** (Checkpoint 1): Key models implemented
- **Sequence-Based Recommenders** (Checkpoint 2): Sequential modeling approaches
- **Graph-Based Recommenders** (Checkpoint 3): Graph neural network architectures
- **Integration strategy**: How you combined insights across paradigms

#### 3. Experimental Setup and Implementation (1.5 pages)
- **Data preprocessing and feature engineering**
- **Model architectures and design choices**
- **Training procedures and optimization strategies**
- **Evaluation framework and metrics**

#### 4. Results and Analysis (2.5 pages)
- **Performance comparison tables**: All implemented models across all metrics
- **Ablation studies**: Impact of key design choices
- **Error analysis**: What worked well and what didn't

#### 5. Final Strategy and Optimization Path (1 page)
- **Evolution of your approach**: How you progressed through checkpoints
- **Winning strategy**: Final model architecture and key insights
- **Hyperparameter optimization**: Search strategies and final selections

#### 6. Conclusion and Course Connection (0.5 page)
- **Connection to CS145 data mining concepts** 
- **Key learnings and takeaways**
- **Future work and potential improvements**

### Appendix (Unlimited Pages)

Contents you can include in the Appendix: 

#### A. Detailed Experimental Results
- **Complete performance tables** for all models and metrics
- **Learning curves and convergence plots**
- **Hyperparameter sensitivity analysis**
- **Additional ablation studies**

#### B. Model Specifications
- **Detailed hyperparameters** for each implemented model
- **Architecture diagrams** for neural network models
- **Training configurations** (batch size, learning rate schedules, etc.)

#### C. Visualizations and Additional Analysis
- **t-SNE/UMAP embeddings** (for graph-based models)
- **Attention visualizations** (for sequence/graph models)
- **Error distribution analysis**
- **User/item interaction patterns**

#### D. Code Documentation
- **Repository structure explanation**
- **Key functions and classes overview**
- **Reproduction instructions**

## Code Submission Requirements

### Example Repository Structure
```
cs145-recsys-team/
├── checkpoint1/          # Content-based recommenders
├── checkpoint2/          # Sequence-based recommenders  
├── checkpoint3/          # Graph-based recommenders
├── final/               # Final integrated approach
├── submission.py        # Final leaderboard submission
├── requirements.txt     # Dependencies
├── README.md           # Setup and usage instructions
└── experiments/        # Experimental scripts and configs
```

### Code Quality Standards
- **Clean, documented code** with meaningful variable names
- **Modular design** with reusable components
- **Reproducible experiments** with fixed random seeds
- **Clear README** with setup and execution instructions

## Presentation Requirements

### Slides (10 minutes)
- **Slide count**: 8-12 slides maximum
- **Content structure**:
  1. Team introduction and problem overview
  2. Approach summary (1-2 slides per checkpoint)
  3. Key experimental results and insights
  4. Final strategy and winning approach
  5. Lessons learned and course connections

### Video Presentation
- **Duration**: 10 minutes (±30 seconds)
- **Format**: MP4 or similar standard format
- **Quality**: Clear audio and readable slides
- **Content**: Present your slides as if giving a conference talk

## Evaluation Criteria

### Report Quality (40%)
- **Technical depth and accuracy**
- **Experimental rigor and statistical analysis**
- **Clear presentation of results and insights**
- **Writing quality and organization**

### Code Implementation (35%)
- **Correctness and completeness**
- **Code quality and documentation**
- **Reproducibility of results**
- **Integration across checkpoints**

### Leaderboard Performance (15%)
- **Final ranking on the competition leaderboard**
- **Improvement over baseline approaches**

### Presentation (10%)
- **Clarity and organization of slides**
- **Quality of video presentation**
- **Effective communication of key insights**

## Specific Requirements

### Tables and Figures
- **All experimental results must be presented in tables or charts**
- **Compare across all metrics**: discounted revenue, total revenue, precision@K, NDCG@K, MRR, hit rate

### Model Comparison
- **Quantitative comparison** of all implemented approaches
- **Analysis of strengths and weaknesses** for each paradigm
- **Discussion of metric trade-offs** (e.g., revenue vs. precision)
- **Computational efficiency analysis** (training time, inference speed)

### Reproducibility
- **All experiments must be reproducible** from submitted code
- **Include random seeds and version information**
- **Provide clear instructions** for replicating results
- **Document any external dependencies** or data preprocessing steps

## Submission Instructions

1. **Upload report PDF** to Gradescope (one submission per team)
2. **Submit code repository** as a ZIP file or GitHub link
3. **Upload presentation slides** (PDF format)
4. **Submit presentation video** (provide download link if file is large)
5. **Ensure final `submission.py` is uploaded** to the leaderboard platform

## Academic Integrity

- **All code must be original work** by team members
- **Properly cite any external libraries** or code snippets used
- **Collaboration between teams is not permitted**
- **AI assistance must be disclosed** if used for code generation or writing

## Support and Questions

- **Office hours**: TA Office Hour is offered on Wednesday 7-8 via Zoom. 
- **Leaderboard issues**: Contact TAs immediately if you encounter submission problems

Remember: This competition synthesizes core data mining concepts from CS145 including supervised learning, feature engineering, model evaluation, and performance optimization. Your final report should demonstrate deep understanding of these principles applied to real-world recommendation scenarios.

**Good luck, and may the best recommender win!** 