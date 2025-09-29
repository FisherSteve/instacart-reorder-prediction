# Design Document

## Overview

The Instacart reorder prediction system implements an end-to-end machine learning pipeline using a three-stage approach: SQL-based feature engineering with DuckDB, multi-model training with scikit-learn/XGBoost/LightGBM, and automated HTML reporting. The design prioritizes reproducibility, data leakage prevention, and lean implementation following the PACE methodology (Plan-Analyze-Construct-Execute).

**Code Philosophy:** This project follows a "studentisches Lernprojekt" (student learning project) approach, emphasizing clear, readable, well-commented code over optimization. The implementation style reflects a junior data scientist's learning journey with:

- **Educational Comments:** "Ich mache hier bewusst X, weil...", "Für die erste Version halte ich Y simpel..."
- **Learning-oriented Explanations:** Comments explain WHY decisions are made, not just WHAT the code does
- **Verbose Clarity:** Prefer `user_last_prior_ord` over `ulpo`, `times_reordered` over `tr`
- **Step-by-step Logic:** Break complex operations into clearly documented intermediate steps
- **No Enterprise Jargon:** Avoid over-optimized patterns; focus on understandable implementations
- **Beginner-friendly Structure:** Code should be approachable for someone learning ML pipelines

## Architecture

### High-Level Flow
```
CSV Data → DuckDB SQL → features.parquet → ML Pipeline → Model + Metrics → HTML Report
```

### Directory Structure
```
instacart-e2e/
├── venv/                         # Python virtual environment (excluded from git)
├── data/                         # Input CSV files
│   ├── orders.csv
│   ├── order_products__prior.csv
│   ├── order_products__train.csv
│   ├── products.csv
│   ├── aisles.csv
│   └── departments.csv
├── src/
│   ├── sql/01_build.sql          # Feature engineering SQL
│   ├── build_dataset.py          # SQL executor → parquet
│   ├── train.py                  # ML pipeline & training
│   └── report.py                 # HTML report generator
├── reports/                      # Output artifacts
│   ├── metrics_*.json
│   ├── model_*.joblib
│   ├── metrics_leaderboard.csv
│   └── report.html
├── requirements.txt
├── .gitignore                    # Exclude venv/ and other artifacts
└── README.md
```

### Environment Setup
The project uses a Python virtual environment to ensure dependency isolation and reproducibility:

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Unix/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Components and Interfaces

### 1. Feature Engineering Component (build_dataset.py)

**Purpose:** Execute SQL-based feature engineering and output structured dataset

**Input:** 
- CSV files in /data/ directory
- SQL query file (src/sql/01_build.sql)

**Output:**
- features.parquet with schema: user_id, product_id, y, [feature_columns]

**Key Design Decisions:**
- Use DuckDB for direct CSV processing (no intermediate loading)
- Strict separation: 'prior' orders for features, 'train' orders for labels only
- Window functions for user-product interaction history
- Product popularity and categorical lookups (aisle/department)
- **Learning-focused code style:** Extensive comments, clear variable names, educational explanations over concise implementations

### 2. Training Component (train.py)

**Purpose:** Train and evaluate multiple ML models with proper validation

**Input:**
- features.parquet
- Command-line parameters: --model, --out, --topk, --seed

**Output:**
- model_{model_name}.joblib (serialized pipeline)
- metrics_{model_name}.json (performance metrics)

**Key Design Decisions:**
- GroupShuffleSplit on user_id to prevent user leakage
- Preprocessing pipeline: StandardScaler + OneHotEncoder
- Support for three model types: logreg, xgb, lgbm
- Custom Order-F1 metric implementation for set-based evaluation

### 3. Reporting Component (report.py)

**Purpose:** Generate comparative analysis and HTML visualization

**Input:**
- All metrics_*.json files in reports/ directory

**Output:**
- metrics_leaderboard.csv (sorted by ROC-AUC)
- report.html (formatted performance summary)

**Key Design Decisions:**
- Automatic discovery of completed model runs
- Simple HTML template with embedded CSS
- Focus on key metrics: ROC-AUC, PR-AUC, F1@0.5, Order-F1@top-k

## Data Models

### Feature Schema
```python
features.parquet columns:
- user_id: int64                    # User identifier
- product_id: int64                 # Product identifier  
- y: int32                          # Binary reorder label (0/1)
- times_bought: int64               # Historical purchase count
- times_reordered: int64            # Historical reorder count
- user_prod_reorder_rate: float64   # Reorder rate for this user-product
- last_prior_ordnum: int64          # Last order number containing product
- orders_since_last: int64          # Recency measure
- avg_add_to_cart_pos: float64      # Average cart position
- avg_days_since_prior: float64     # Average days between orders
- aisle_id: int64                   # Product aisle (categorical)
- department_id: int64              # Product department (categorical)
- prod_cnt: int64                   # Global product popularity
- prod_users: int64                 # Number of users who bought product
```

### Metrics Schema
```python
metrics_{model}.json structure:
{
  "model": str,                     # Model name (logreg/xgb/lgbm)
  "roc_auc": float,                 # ROC-AUC score
  "pr_auc": float,                  # Precision-Recall AUC
  "f1_at_0.5": float,               # F1 score at 0.5 threshold
  "order_f1_topk": float,           # Set-based F1 (top-k per user)
  "topk": int,                      # K value used for Order-F1
  "n_train": int,                   # Training set size
  "n_valid": int,                   # Validation set size
  "train_secs": float,              # Training time
  "num_features": int,              # Number of numeric features
  "cat_features": int               # Number of categorical features
}
```

## Error Handling

### Data Validation
- **Missing CSV files:** DuckDB will raise clear error if files don't exist in /data/
- **Schema mismatches:** SQL query validates expected column names
- **Empty results:** Check for non-zero rows in features.parquet

### Model Training
- **Insufficient data:** Validate minimum samples per class for training
- **Memory constraints:** Use chunked processing if dataset exceeds memory
- **Model convergence:** Set reasonable max_iter and early stopping where applicable

### Report Generation
- **No metrics files:** Display informative message if no models have been trained
- **Malformed JSON:** Skip corrupted metrics files with warning
- **HTML generation:** Fallback to plain text summary if HTML templating fails

## Testing Strategy

### Unit Testing Approach
1. **SQL Validation:** Test feature engineering logic with small synthetic datasets
2. **Pipeline Testing:** Validate preprocessing and model training with mock data
3. **Metrics Testing:** Verify Order-F1 calculation with known examples
4. **Integration Testing:** End-to-end pipeline test with sample Instacart data

### Data Quality Checks
- Verify no data leakage (train orders not used in features)
- Validate feature distributions and missing value handling
- Check user-product pair uniqueness in feature set
- Confirm label distribution matches expected reorder rates

### Performance Validation
- Baseline comparison: Random classifier and global popularity
- Cross-validation stability across different random seeds
- Training time benchmarks for each model type
- Memory usage profiling for large datasets

## Implementation Notes

### DuckDB SQL Strategy
- Direct CSV reading eliminates ETL complexity
- Window functions for efficient user-product aggregations
- CTEs (Common Table Expressions) for readable query structure
- Explicit eval_set filtering to prevent leakage

### Model Selection Rationale
- **LogisticRegression:** Fast baseline with interpretable coefficients
- **XGBoost:** Gradient boosting with excellent tabular performance
- **LightGBM:** Memory-efficient alternative to XGBoost

### Preprocessing Pipeline
- **StandardScaler:** Normalize numeric features for LogReg
- **OneHotEncoder:** Handle categorical features (aisle_id, department_id)
- **Pipeline:** Ensure consistent preprocessing across train/validation

### Order-F1 Metric Design
```python
# Set-based F1 per user, then averaged
for each user:
    true_set = {products actually reordered}
    pred_set = {top-k predicted products}
    f1_user = 2 * |intersection| / (|true_set| + |pred_set|)
order_f1 = mean(f1_user across all users)
```

This metric captures the business objective: predicting the right set of products per user, not just individual product probabilities.

## Code Style Guidelines

### Comment Style Examples
```python
# Gut: Educational explanation
# Ich verwende hier GroupShuffleSplit, weil normale train_test_split 
# Nutzer zwischen Train/Val aufteilen könnte → unrealistische Evaluation

# Schlecht: Nur beschreibend
# Split data into train and validation
```

### Variable Naming Philosophy
```python
# Gut: Selbsterklärend
user_last_prior_ord = df.groupby('user_id')['order_number'].max()
times_reordered_by_user_product = prior_df.groupby(['user_id', 'product_id'])['reordered'].sum()

# Schlecht: Kryptisch
ulpo = df.groupby('uid')['ord_num'].max()
tr_up = pdf.groupby(['uid', 'pid'])['reord'].sum()
```

### Documentation Approach
- Explain business logic: "Recency matters because recent purchases indicate current preferences"
- Justify technical choices: "DuckDB für direkte CSV-Verarbeitung ohne ETL-Overhead"
- Acknowledge limitations: "Für die erste Version halte ich das Thresholding simpel"
- Learning notes: "Als Lernprojekt bewusst ausführlich kommentiert"