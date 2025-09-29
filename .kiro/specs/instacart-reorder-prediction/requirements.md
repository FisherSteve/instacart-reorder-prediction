# Requirements Document

## Introduction

This project implements a complete machine learning pipeline to predict which products a customer will reorder in their next shopping basket on Instacart. The system follows a structured approach: SQL-based feature engineering using DuckDB, training multiple classification models (LogisticRegression, XGBoost, LightGBM), and generating an HTML performance report. The pipeline must be reproducible with exactly 3 commands and target performance benchmarks of ROC-AUC ≥ 0.83 and Order-F1 improvement over baseline.

## Requirements

### Requirement 1: SQL-Based Feature Engineering

**User Story:** As a data scientist, I want to build features using SQL on CSV files with DuckDB, so that I can create a clean feature dataset avoiding data leakage.

#### Acceptance Criteria

1. WHEN building features THEN the system SHALL use only 'prior' orders for feature engineering and 'train' orders only for labels
2. WHEN processing data THEN the system SHALL create src/sql/01_build.sql that reads CSV files directly from /data/ directory
3. WHEN feature engineering runs THEN the system SHALL generate features including: times_bought, user_prod_reorder_rate, orders_since_last, product popularity, aisle/department lookups
4. WHEN SQL execution completes THEN the system SHALL output features.parquet with columns: user_id, product_id, y (binary label), and all engineered features
5. IF data leakage is detected THEN the system SHALL prevent using train data for feature creation

### Requirement 2: Multi-Model Training Pipeline

**User Story:** As a data scientist, I want to train LogisticRegression, XGBoost, and LightGBM models with proper evaluation, so that I can compare performance and select the best approach.

#### Acceptance Criteria

1. WHEN splitting data THEN the system SHALL use GroupShuffleSplit on user_id to prevent user leakage between train/validation sets
2. WHEN preprocessing THEN the system SHALL apply StandardScaler to numeric features and OneHotEncoder to categorical features
3. WHEN training models THEN the system SHALL support --model parameter with choices: logreg, xgb, lgbm
4. WHEN evaluating performance THEN the system SHALL calculate and report ROC-AUC, PR-AUC, F1@0.5, and Order-F1@top-k metrics with target ROC-AUC ≥ 0.83
5. WHEN training completes THEN the system SHALL save model_*.joblib and metrics_*.json files in reports/ directory
6. WHEN measuring Order-F1 THEN the system SHALL implement set-based F1 using top-k products per user (default k=10)

### Requirement 3: Automated Report Generation

**User Story:** As a stakeholder, I want an automated HTML report with model comparison and metrics, so that I can quickly assess model performance and make decisions.

#### Acceptance Criteria

1. WHEN generating reports THEN the system SHALL collect all metrics_*.json files and create a leaderboard sorted by ROC-AUC
2. WHEN creating HTML report THEN the system SHALL include model comparison table, project description, and key limitations
3. WHEN report generation completes THEN the system SHALL output both metrics_leaderboard.csv and report.html in reports/ directory
4. WHEN displaying metrics THEN the system SHALL show ROC-AUC, PR-AUC, F1@0.5, Order-F1@top-k, training time, and feature counts
5. IF no metrics files exist THEN the system SHALL display appropriate error message

### Requirement 4: Three-Command Reproducibility

**User Story:** As a developer, I want to reproduce the entire pipeline with exactly 3 commands, so that results are consistent and deployment is simple.

#### Acceptance Criteria

1. WHEN executing the pipeline THEN the system SHALL complete with: python src/build_dataset.py, python src/train.py --model [logreg|xgb|lgbm], python src/report.py
2. WHEN running build_dataset.py THEN the system SHALL execute SQL and create data/features.parquet
3. WHEN running train.py THEN the system SHALL accept --model, --out, --topk, --seed parameters and output model + metrics files
4. WHEN running report.py THEN the system SHALL generate leaderboard and HTML report from existing metrics files
5. WHEN pipeline completes THEN the system SHALL produce deterministic results with the same random seed

### Requirement 5: Project Structure and Documentation

**User Story:** As a developer, I want a well-organized project structure with clear documentation, so that the pipeline is maintainable and easy to understand.

#### Acceptance Criteria

1. WHEN setting up the project THEN the system SHALL create a Python virtual environment and directory structure: data/, src/, src/sql/, reports/, with appropriate Python modules
2. WHEN managing dependencies THEN the system SHALL use a virtual environment with requirements.txt containing all dependencies (duckdb, pandas, scikit-learn, xgboost, lightgbm, joblib, pyarrow)
3. WHEN creating documentation THEN the system SHALL provide a README.md with project description, reproduction steps, and limitations
4. WHEN organizing code THEN the system SHALL separate concerns: build_dataset.py for data processing, train.py for modeling, report.py for output generation
5. IF users need to understand the approach THEN the system SHALL include clear comments explaining data leakage prevention and modeling choices
6. WHEN writing code THEN the system SHALL follow a "studentisches Lernprojekt" approach with extensive educational comments like "Ich mache hier bewusst X, weil...", verbose descriptive variable names, step-by-step explanations, and beginner-friendly structure prioritizing learning clarity over optimization

### Requirement 6: Technical Architecture Requirements

**User Story:** As a developer, I want the system to use specified technologies efficiently, so that it integrates well with existing infrastructure and performs optimally.

#### Acceptance Criteria

1. WHEN processing data THEN the system SHALL use DuckDB for SQL operations on CSV files
2. WHEN implementing ML components THEN the system SHALL use Python with pandas, scikit-learn, XGBoost, LightGBM, and joblib
3. WHEN handling large datasets THEN the system SHALL use pyarrow for efficient data serialization when needed
4. WHEN optimizing performance THEN the system SHALL keep the implementation lean and prioritize getting a working solution before advanced tuning
5. IF memory constraints are encountered THEN the system SHALL implement chunked processing or data sampling strategies