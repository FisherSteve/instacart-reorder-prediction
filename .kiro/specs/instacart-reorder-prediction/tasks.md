# Implementation Plan

- [x] 1. Set up project structure and virtual environment


  - Create Python virtual environment: python -m venv venv (or python3 -m venv venv on Unix)
  - Activate virtual environment: venv\Scripts\activate (Windows) or source venv/bin/activate (Unix)
  - Create directory structure: data/, src/, src/sql/, reports/
  - Create requirements.txt with all necessary packages: duckdb, pandas, numpy, pyarrow, scikit-learn, xgboost, lightgbm, joblib, matplotlib
  - Install dependencies: pip install -r requirements.txt
  - Create empty __init__.py files where needed for Python module structure
  - Add venv/ to .gitignore to exclude virtual environment from version control
  - _Requirements: 5.1, 5.2_

- [x] 2. Implement SQL-based feature engineering





- [x] 2.1 Create comprehensive SQL feature engineering script


  - Write src/sql/01_build.sql with complete feature engineering logic and extensive educational comments
  - Add detailed comments explaining each CTE step: "-- 1) Quellen einlesen", "-- 2) Prior/Train trennen", etc.
  - Implement CSV reading views for all data sources with clear naming and documentation
  - Create user-product aggregation features with explanatory comments about business logic
  - Add recency features with comments explaining why recency matters for reorder prediction
  - Include product popularity and categorical lookups with clear variable naming
  - Implement proper label generation from train data with comments about leakage prevention
  - Use verbose, self-documenting SQL style prioritizing readability over performance
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 5.5_



- [x] 2.2 Create dataset builder script





  - Write src/build_dataset.py with extensive docstring explaining purpose and learning approach
  - Add detailed comments: "Als Lernprojekt bewusst ausführlich kommentiert"
  - Implement command-line argument parsing with clear variable names and help text
  - Add DuckDB connection and SQL execution logic with step-by-step comments
  - Create COPY statement to export SQL results with explanation of DuckDB parquet export
  - Add error handling with educational comments about common failure modes
  - Include verbose logging showing each step of the process
  - Use clear, readable code structure over concise implementations
  - _Requirements: 1.4, 4.2, 5.5_


- [x] 3. Implement machine learning training pipeline



- [x] 3.1 Create core training infrastructure


  - Write src/train.py with argument parsing for --model, --out, --topk, --seed parameters
  - Implement data loading from features.parquet with proper column separation
  - Create GroupShuffleSplit logic for user-based train/validation splitting
  - Build preprocessing pipeline with ColumnTransformer (StandardScaler + OneHotEncoder)
  - Add automatic detection of numeric vs categorical columns
  - _Requirements: 2.1, 2.2, 4.3_

- [x] 3.2 Implement model training and evaluation


  - Create build_model() function supporting logreg, xgb, lgbm model types
  - Implement model training with preprocessing pipeline
  - Add comprehensive metrics calculation: ROC-AUC, PR-AUC, F1@0.5
  - Create custom order_f1_by_user() function for set-based evaluation
  - Implement model and metrics serialization to reports/ directory
  - Add training time measurement and feature count reporting
  - _Requirements: 2.3, 2.4, 2.5, 2.6_

- [x] 4. Create automated reporting system





- [x] 4.1 Implement metrics collection and leaderboard generation


  - Write src/report.py to discover and load all metrics_*.json files
  - Create pandas DataFrame from metrics and sort by ROC-AUC
  - Generate metrics_leaderboard.csv output
  - Add error handling for missing or malformed metrics files
  - _Requirements: 3.1, 3.5_

- [x] 4.2 Generate HTML performance report


  - Create HTML template with embedded CSS for clean formatting
  - Build model comparison table from leaderboard data
  - Add project description, methodology notes, and limitations section
  - Include all key metrics in formatted table: ROC-AUC, PR-AUC, F1@0.5, Order-F1@top-k
  - Write report.html to reports/ directory with proper encoding
  - _Requirements: 3.2, 3.3, 3.4_
-

- [x] 5. Create project documentation and validation









- [x] 5.1 Write comprehensive README documentation


  - Create README.md with project overview and business objective
  - Document the three-command reproduction workflow
  - Include data schema description and feature explanations
  - Add methodology notes about data leakage prevention and evaluation approach
  - Document limitations: no cold-start handling, simple thresholding, minimal tuning focus
  - Emphasize learning project approach: "studentisches Lernprojekt" with educational comments
  - _Requirements: 5.3, 5.5_



- [x] 5.2 Implement end-to-end pipeline validation


  - Create simple test script to validate the three-command workflow
  - Add data quality checks: verify no leakage, validate feature distributions
  - Test deterministic behavior with fixed random seeds
  - Validate output file generation and proper error handling
  - Create sample data validation to ensure pipeline works with expected inputs
  - _Requirements: 4.1, 4.5_

- [x] 6. Add model-specific optimizations and configurations




- [x] 6.1 Configure XGBoost and LightGBM parameters


  - Set appropriate hyperparameters for XGBoost: tree_method="hist", n_estimators=2000, learning_rate=0.05
  - Configure LightGBM parameters: n_estimators=4000, learning_rate=0.03, num_leaves=127
  - Add proper evaluation metrics and regularization settings
  - Implement basic parameter validation and error handling
  - _Requirements: 2.3, 6.4_

- [x] 6.2 Enhance error handling and robustness


  - Add comprehensive error handling for memory constraints and large datasets
  - Implement graceful degradation for missing dependencies
  - Add input validation for command-line parameters
  - Create informative error messages for common failure scenarios
  - Add memory usage monitoring and chunked processing fallbacks if needed
  - _Requirements: 6.5, 4.4_