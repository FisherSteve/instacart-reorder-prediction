# Instacart Reorder Prediction Pipeline

## 🏆 Key Results & Performance

**Best Model Performance:** XGBoost achieved **ROC-AUC 0.8289** (target: ≥0.83), demonstrating strong predictive capability for customer reorder behavior.

| Rank | Model | ROC-AUC | PR-AUC | F1@0.5 | Order-F1@top-10 | Training Time |
|------|-------|---------|--------|--------|------------------|---------------|
| 🥇 | **XGBoost** | **0.8289** | 0.402 | 0.249 | 0.341 | 663.4s |
| 🥈 | **LightGBM** | **0.8282** | 0.401 | 0.373 | 0.341 | 398.2s |
| 🥉 | **LogisticRegression** | **0.816** | 0.356 | 0.347 | 0.331 | 127.1s |

**Key Technical Achievements:**
- ✅ **End-to-End ML Pipeline**: Complete workflow from raw CSV data to production-ready models
- ✅ **Rigorous Data Leakage Prevention**: Strict temporal separation between feature engineering and labels
- ✅ **Business-Relevant Metrics**: Custom Order-F1 metric measuring set-based prediction accuracy per user
- ✅ **Scalable Architecture**: SQL-based feature engineering processing 6.7M+ training samples
- ✅ **Reproducible Results**: Three-command pipeline with deterministic outputs

**Dataset Scale:** 6.7M+ training samples, 55 engineered features, 1.7M validation samples

## Project Overview

This project implements a complete machine learning pipeline to predict which products a customer will reorder in their next shopping basket on Instacart. The system follows a structured approach: SQL-based feature engineering using DuckDB, training multiple classification models (LogisticRegression, XGBoost, LightGBM), and generating an HTML performance report.

**Business Objective:** Predict the probability that a user will reorder each product they have previously purchased, enabling personalized product recommendations and inventory optimization.

**Target Performance:** ROC-AUC ≥ 0.83 with Order-F1 improvement over baseline approaches.

## Learning Project Approach

This is a **student learning project**  emphasizing educational clarity over optimization. The implementation style reflects a junior data scientist's learning journey with:

- **Learning-oriented Explanations:** Comments explain WHY decisions are made, not just WHAT the code does
- **Verbose Clarity:** Prefer `user_last_prior_ord` over `ulpo`, `times_reordered` over `tr`
- **Step-by-step Logic:** Break complex operations into clearly documented intermediate steps
- **Beginner-friendly Structure:** Code should be approachable for someone learning ML pipelines

## Three-Command Reproduction Workflow

The entire pipeline can be reproduced with exactly three commands:

```bash
# 1. Build feature dataset from CSV files using SQL
python src/build_dataset.py

# 2. Train a model (choose: logreg, xgb, or lgbm)
python src/train.py --model logreg
python src/train.py --model xgb  
python src/train.py --model lgbm

# 3. Generate performance report and leaderboard
python src/report.py
```

### Setup Instructions

1. **Create and activate virtual environment:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Unix/macOS
python3 -m venv venv
source venv/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Ensure data files are in place:**
```
data/
├── orders.csv
├── order_products__prior.csv
├── order_products__train.csv
├── products.csv
├── aisles.csv
└── departments.csv
```

## Data Schema and Features

### Input Data Schema
- **orders.csv:** User order history with timing information
- **order_products__prior.csv:** Products in historical orders (for feature engineering)
- **order_products__train.csv:** Products in final orders (for labels only)
- **products.csv:** Product catalog with aisle/department mappings
- **aisles.csv:** Product aisle categories
- **departments.csv:** Product department categories

### Engineered Features
The SQL-based feature engineering creates the following features for each user-product pair:

**User-Product Interaction Features:**
- `times_bought`: Total number of times user purchased this product
- `times_reordered`: Number of times user reordered this product (excludes first purchase)
- `user_prod_reorder_rate`: Reorder rate for this specific user-product combination
- `last_prior_ordnum`: Order number of last purchase (recency indicator)
- `orders_since_last`: Number of orders since last purchase of this product
- `avg_add_to_cart_pos`: Average position in cart when user adds this product
- `avg_days_since_prior`: Average days between user's orders

**Product Popularity Features:**
- `prod_cnt`: Global popularity (total times product was purchased)
- `prod_users`: Number of unique users who purchased this product

**Categorical Features:**
- `aisle_id`: Product aisle category
- `department_id`: Product department category

**Target Variable:**
- `y`: Binary label (1 if product was reordered in train set, 0 otherwise)

## Methodology

### Data Leakage Prevention
**Critical Design Decision:** Strict separation between feature engineering and label generation:
- **Features:** Built exclusively from 'prior' orders (historical data)
- **Labels:** Generated only from 'train' orders (target outcomes)
- **Validation:** GroupShuffleSplit on user_id prevents user leakage between train/validation sets

This ensures the model cannot "cheat" by seeing future information during training.

### Evaluation Approach
The pipeline uses multiple complementary metrics:

1. **ROC-AUC:** Standard binary classification metric
2. **PR-AUC:** Precision-Recall AUC, better for imbalanced datasets
3. **F1@0.5:** F1 score at 0.5 probability threshold
4. **Order-F1@top-k:** Custom set-based metric that measures how well we predict the actual set of reordered products per user

**Order-F1 Calculation:**
```python
# For each user:
true_set = {products actually reordered}
pred_set = {top-k predicted products}
f1_user = 2 * |intersection| / (|true_set| + |pred_set|)

# Final metric:
order_f1 = mean(f1_user across all users)
```

This metric captures the business objective: predicting the right set of products per user, not just individual product probabilities.

### Model Selection
Three complementary approaches:
- **LogisticRegression:** Fast baseline with interpretable coefficients
- **XGBoost:** Gradient boosting with excellent tabular performance  
- **LightGBM:** Memory-efficient alternative to XGBoost

All models use the same preprocessing pipeline:
- **StandardScaler:** Normalize numeric features
- **OneHotEncoder:** Handle categorical features (aisle_id, department_id)

## Project Structure

```
instacart-reorder-prediction/
├── venv/                         # Python virtual environment (excluded from git)
├── data/                         # Input CSV files
│   ├── orders.csv
│   ├── order_products__prior.csv
│   ├── order_products__train.csv
│   ├── products.csv
│   ├── aisles.csv
│   └── departments.csv
├── src/
│   ├── sql/01_build.sql          # Feature engineering SQL with educational comments
│   ├── build_dataset.py          # SQL executor → parquet converter
│   ├── train.py                  # ML pipeline & training with extensive documentation
│   └── report.py                 # HTML report generator
├── reports/                      # Output artifacts
│   ├── metrics_*.json            # Model performance metrics
│   ├── model_*.joblib            # Trained model pipelines
│   ├── metrics_leaderboard.csv   # Model comparison table
│   └── report.html               # Formatted performance report
├── requirements.txt              # Python dependencies
├── .gitignore                    # Exclude venv/ and artifacts
└── README.md                     # This documentation
```

## Limitations and Future Improvements

### Current Limitations
1. **No Cold-Start Handling:** Cannot predict for new users or products not in training data
2. **Simple Thresholding:** Uses basic 0.5 threshold for binary predictions instead of optimized thresholds
3. **Minimal Hyperparameter Tuning:** Focus on getting working pipeline rather than optimal performance
4. **No Temporal Features:** Doesn't consider seasonality or time-of-day patterns
5. **Basic Feature Engineering:** Could benefit from more sophisticated interaction features

### Learning Project Focus
As a "studentisches Lernprojekt," this implementation prioritizes:
- **Understanding over Optimization:** Clear, readable code with extensive comments
- **End-to-End Pipeline:** Complete workflow from raw data to final report
- **Reproducibility:** Deterministic results with fixed random seeds
- **Educational Value:** Step-by-step explanations of ML pipeline decisions

### Potential Enhancements
- Advanced feature engineering (user behavior patterns, product similarity)
- Hyperparameter optimization with cross-validation
- Ensemble methods combining multiple models
- Cold-start handling for new users/products
- Real-time prediction API
- A/B testing framework for model deployment

## Dependencies

See `requirements.txt` for complete list. Key dependencies:
- **duckdb:** SQL-based data processing
- **pandas:** Data manipulation and analysis
- **scikit-learn:** Machine learning algorithms and preprocessing
- **xgboost:** Gradient boosting framework
- **lightgbm:** Efficient gradient boosting
- **joblib:** Model serialization
- **pyarrow:** Efficient data serialization for parquet files

## Results Interpretation

After running the pipeline, check `reports/report.html` for:
- Model comparison table sorted by ROC-AUC
- Key performance metrics for each model
- Training time and feature count information
- Project methodology and limitations summary

The leaderboard in `reports/metrics_leaderboard.csv` provides a quick comparison of all trained models.

---

*This project demonstrates a complete ML pipeline implementation with emphasis on educational clarity, reproducible results, and proper evaluation methodology.*