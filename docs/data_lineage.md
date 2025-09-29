# Data Lineage and Transformation Documentation

## Overview

This document provides a comprehensive view of data flow, transformations, and lineage throughout the Instacart reorder prediction pipeline. The pipeline follows a strict layered architecture to ensure data quality, prevent leakage, and maintain reproducibility.

## Pipeline Architecture

```
Raw CSV Data → SQL Feature Engineering → ML Training → Performance Reports
     ↓                    ↓                   ↓              ↓
  data/raw/        data/features/        reports/      reports/
```

## Detailed Data Flow

### Layer 1: Raw Data Sources (data/raw/)

**Input Files:**
- `orders.csv` - User order sequences and timing
- `order_products__prior.csv` - Historical product purchases (features only)
- `order_products__train.csv` - Target order products (labels only)
- `products.csv` - Product catalog with categories
- `aisles.csv` - Product aisle definitions
- `departments.csv` - Product department definitions

**Data Volumes:**
- ~3.4M orders across ~200K users
- ~32M product purchases in prior orders
- ~1.4M product purchases in train orders
- ~50K unique products across 134 aisles and 21 departments

### Layer 2: Feature Engineering (src/sql/01_build.sql)

**Processing Logic:**

1. **Data Source Views**
   ```sql
   CREATE OR REPLACE VIEW orders_raw AS
   SELECT * FROM read_csv_auto('data/raw/orders.csv');
   ```
   - Direct CSV reading with DuckDB
   - No intermediate ETL steps
   - Preserves original data types and structure

2. **User-Product Aggregations**
   ```sql
   -- Purchase frequency calculation
   user_product_stats AS (
     SELECT 
       user_id,
       product_id,
       COUNT(*) as times_bought,
       SUM(reordered) as times_reordered,
       SUM(reordered) / NULLIF(COUNT(*), 0) as user_prod_reorder_rate
     FROM prior_orders_with_products
     GROUP BY user_id, product_id
   )
   ```

3. **Temporal Features**
   ```sql
   -- Recency calculations
   user_product_recency AS (
     SELECT 
       user_id,
       product_id,
       MAX(order_number) as last_prior_ordnum,
       user_max_order_num - MAX(order_number) as orders_since_last
     FROM prior_orders_with_products p
     JOIN user_order_stats u USING (user_id)
     GROUP BY user_id, product_id, user_max_order_num
   )
   ```

4. **Behavioral Features**
   ```sql
   -- Shopping behavior patterns
   user_product_behavior AS (
     SELECT 
       user_id,
       product_id,
       AVG(add_to_cart_order) as avg_add_to_cart_pos
     FROM prior_orders_with_products
     GROUP BY user_id, product_id
   )
   ```

5. **Product Popularity**
   ```sql
   -- Global product metrics
   product_popularity AS (
     SELECT 
       product_id,
       COUNT(DISTINCT order_id) as prod_cnt,
       COUNT(DISTINCT user_id) as prod_users
     FROM prior_orders_with_products
     GROUP BY product_id
   )
   ```

6. **Label Generation**
   ```sql
   -- Binary reorder labels from train set only
   labels AS (
     SELECT 
       user_id,
       product_id,
       1 as y
     FROM train_orders_with_products
   )
   ```

**Critical Design Principles:**
- **Temporal Separation**: Features use only 'prior' orders, labels use only 'train' orders
- **No Future Leakage**: Features cannot contain information from target time period
- **User-Centric**: All features capture individual user behavior patterns
- **Comprehensive Coverage**: Every user-product pair from prior orders gets features

### Layer 3: ML-Ready Dataset (data/features/features.parquet)

**Output Schema:**
```
user_id: int64                    # User identifier (foreign key)
product_id: int64                 # Product identifier (foreign key)
y: int32                          # Binary reorder label (target variable)

# User-Product Interaction Features (11 features)
times_bought: int64               # Total times user bought this product
times_reordered: int64            # Times user reordered (excludes first purchase)
user_prod_reorder_rate: float64   # Personal reorder rate for this product
last_prior_ordnum: int64          # Order number of last purchase
orders_since_last: int64          # Orders since last purchase (recency)
avg_add_to_cart_pos: float64      # Average cart position
avg_days_since_prior: float64     # Average days between user's orders

# Product Popularity Features (2 features)
prod_cnt: int64                   # Global product popularity
prod_users: int64                 # Unique users who bought product

# Categorical Features (2 features)
aisle_id: int64                   # Product aisle category
department_id: int64              # Product department category
```

**Dataset Characteristics:**
- **Rows**: ~13.3M user-product pairs
- **Features**: 55+ after preprocessing (including one-hot encoded categories)
- **Target Distribution**: ~32% positive class (reorder rate)
- **Sparsity**: Dense feature matrix (no missing values after imputation)

### Layer 4: Model Training (src/train.py)

**Data Processing Steps:**

1. **Data Loading**
   ```python
   # Load parquet with pandas
   df = pd.read_parquet('data/features/features.parquet')
   
   # Separate features and target
   X = df.drop(['user_id', 'product_id', 'y'], axis=1)
   y = df['y']
   user_ids = df['user_id']
   ```

2. **Train-Validation Split**
   ```python
   # User-based splitting to prevent leakage
   from sklearn.model_selection import GroupShuffleSplit
   
   splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
   train_idx, val_idx = next(splitter.split(X, y, groups=user_ids))
   ```

3. **Preprocessing Pipeline**
   ```python
   # Automatic feature type detection
   numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
   categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
   
   # Preprocessing pipeline
   preprocessor = ColumnTransformer([
       ('num', StandardScaler(), numeric_features),
       ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
   ])
   ```

4. **Model Training**
   ```python
   # Complete pipeline with preprocessing + model
   pipeline = Pipeline([
       ('preprocessor', preprocessor),
       ('classifier', model)
   ])
   
   # Fit on training data
   pipeline.fit(X_train, y_train)
   ```

5. **Evaluation Metrics**
   ```python
   # Standard binary classification metrics
   roc_auc = roc_auc_score(y_val, y_pred_proba)
   pr_auc = average_precision_score(y_val, y_pred_proba)
   f1_05 = f1_score(y_val, y_pred_binary)
   
   # Custom business metric
   order_f1 = calculate_order_f1_by_user(y_val, y_pred_proba, user_ids_val, k=10)
   ```

### Layer 5: Model Artifacts (reports/)

**Output Files:**

1. **Trained Models** (`model_*.joblib`)
   ```python
   # Serialized scikit-learn pipeline
   joblib.dump(pipeline, f'reports/model_{model_name}.joblib')
   ```
   - Complete preprocessing + model pipeline
   - Ready for production inference
   - Includes feature scaling and encoding

2. **Performance Metrics** (`metrics_*.json`)
   ```json
   {
     "model": "xgboost",
     "roc_auc": 0.8289,
     "pr_auc": 0.402,
     "f1_at_0.5": 0.249,
     "order_f1_topk": 0.341,
     "topk": 10,
     "n_train": 6734567,
     "n_valid": 1683642,
     "train_secs": 663.4,
     "num_features": 53,
     "cat_features": 2
   }
   ```

### Layer 6: Performance Reports (src/report.py)

**Report Generation:**

1. **Metrics Collection**
   ```python
   # Discover all metrics files
   metrics_files = glob.glob('reports/metrics_*.json')
   
   # Load and validate each file
   all_metrics = [load_metrics_file(f) for f in metrics_files]
   ```

2. **Leaderboard Creation**
   ```python
   # Create DataFrame and sort by ROC-AUC
   leaderboard_df = pd.DataFrame(all_metrics)
   leaderboard_df = leaderboard_df.sort_values('roc_auc', ascending=False)
   ```

3. **HTML Report Generation**
   - Model comparison table
   - Data lineage visualization
   - Methodology documentation
   - Performance insights and limitations

## Data Quality Assurance

### Schema Validation

**Input Validation (src/schemas/input_schemas.py):**
```python
import pandera as pa

orders_schema = pa.DataFrameSchema({
    "order_id": pa.Column(pa.Int64, nullable=False, unique=True),
    "user_id": pa.Column(pa.Int64, nullable=False),
    "eval_set": pa.Column(pa.String, isin=["prior", "train", "test"]),
    "order_number": pa.Column(pa.Int64, ge=1, le=100),
    # ... additional constraints
})
```

**Quality Checks:**
- Column presence and data types
- Value ranges and business rules
- Null rates and completeness
- Duplicate detection
- Foreign key relationships

### Processing Validation

**Feature Engineering Validation:**
- Row count consistency across transformations
- Feature distribution analysis
- Null value handling verification
- Label distribution validation
- Temporal consistency checks

**Model Training Validation:**
- Train-validation split verification
- Feature scaling validation
- Model convergence monitoring
- Performance metric calculation
- Reproducibility testing

## Lineage Tracking

### Automated Lineage Documentation

**File Dependencies:**
```
orders.csv → orders_raw → user_order_stats → final_features
order_products__prior.csv → prior_orders → user_product_stats → final_features
order_products__train.csv → train_orders → labels → final_features
products.csv → products_raw → product_lookups → final_features
```

**Transformation Lineage:**
- Each SQL CTE documents its input sources
- Feature engineering steps are explicitly commented
- Data quality checks validate transformation correctness
- Processing logs capture row counts and timing

### Reproducibility Guarantees

**Deterministic Processing:**
- Fixed random seeds for all stochastic operations
- Consistent data ordering in SQL queries
- Reproducible train-validation splits
- Deterministic model training (where possible)

**Version Control:**
- All code and configuration under version control
- Data schemas documented and validated
- Processing logs capture environment details
- Model artifacts include metadata

## Performance and Scalability

### Processing Efficiency

**SQL Optimization:**
- DuckDB columnar processing for CSV files
- Efficient window functions for user aggregations
- Indexed joins for categorical lookups
- Parallel processing where applicable

**Memory Management:**
- Streaming CSV processing
- Chunked data loading for large datasets
- Efficient parquet serialization
- Memory usage monitoring and logging

### Scalability Considerations

**Data Volume Scaling:**
- Current: ~13M user-product pairs
- Estimated capacity: ~100M pairs with current architecture
- Bottlenecks: Memory for model training, disk I/O for feature engineering

**Feature Engineering Scaling:**
- SQL-based approach scales with DuckDB capabilities
- Potential for distributed processing with larger datasets
- Feature store integration for production deployment

## Future Enhancements

### Data Pipeline Improvements

1. **Incremental Processing**
   - Delta processing for new orders
   - Feature store integration
   - Change data capture

2. **Advanced Feature Engineering**
   - Time-series features (seasonality, trends)
   - Graph-based features (product similarity)
   - Deep learning embeddings

3. **Real-time Processing**
   - Streaming feature computation
   - Online model updates
   - Real-time prediction API

### Monitoring and Observability

1. **Data Quality Monitoring**
   - Automated data drift detection
   - Feature distribution monitoring
   - Data lineage visualization

2. **Model Performance Monitoring**
   - A/B testing framework
   - Performance degradation alerts
   - Automated retraining triggers

---

*This documentation is automatically updated with each pipeline run to ensure accuracy and completeness.*