"""
Data quality checks and monitoring for Instacart dataset.
Implements row count validation, duplicate detection, null rate monitoring,
and business rule validation.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
from datetime import datetime


class DataQualityChecker:
    """
    Comprehensive data quality checker for Instacart dataset.
    
    Performs various data quality checks including:
    - Row count validation
    - Duplicate detection
    - Null rate monitoring
    - Business rule validation
    - Data quality report generation
    """
    
    def __init__(self, output_dir: str = "data/quality_reports"):
        """
        Initialize data quality checker.
        
        Args:
            output_dir: Directory to save quality reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quality_metrics = {}
        self.violations = []
        
    def check_row_counts(self, file_paths: Dict[str, str], expected_ranges: Optional[Dict[str, Tuple[int, int]]] = None) -> Dict[str, Any]:
        """
        Validate row counts for CSV files.
        
        Args:
            file_paths: Dictionary mapping dataset names to file paths
            expected_ranges: Optional dictionary with expected min/max row counts
            
        Returns:
            Dictionary with row count metrics
        """
        logging.info("Checking row counts...")
        
        row_count_metrics = {}
        
        for dataset_name, file_path in file_paths.items():
            try:
                # Count rows efficiently without loading full dataset
                row_count = sum(1 for _ in open(file_path)) - 1  # Subtract header
                row_count_metrics[dataset_name] = {
                    "row_count": row_count,
                    "file_path": file_path,
                    "status": "ok"
                }
                
                # Check against expected ranges if provided
                if expected_ranges and dataset_name in expected_ranges:
                    min_expected, max_expected = expected_ranges[dataset_name]
                    if row_count < min_expected:
                        row_count_metrics[dataset_name]["status"] = "too_few_rows"
                        self.violations.append({
                            "type": "row_count",
                            "dataset": dataset_name,
                            "issue": f"Too few rows: {row_count} < {min_expected}",
                            "severity": "high"
                        })
                    elif row_count > max_expected:
                        row_count_metrics[dataset_name]["status"] = "too_many_rows"
                        self.violations.append({
                            "type": "row_count", 
                            "dataset": dataset_name,
                            "issue": f"Too many rows: {row_count} > {max_expected}",
                            "severity": "medium"
                        })
                
                logging.info(f"  {dataset_name}: {row_count:,} rows")
                
            except Exception as e:
                row_count_metrics[dataset_name] = {
                    "row_count": None,
                    "file_path": file_path,
                    "status": "error",
                    "error": str(e)
                }
                self.violations.append({
                    "type": "row_count",
                    "dataset": dataset_name,
                    "issue": f"Cannot count rows: {e}",
                    "severity": "high"
                })
                logging.error(f"  {dataset_name}: Error counting rows - {e}")
        
        self.quality_metrics["row_counts"] = row_count_metrics
        return row_count_metrics
    
    def check_duplicates(self, file_paths: Dict[str, str], key_columns: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Detect duplicate records in datasets.
        
        Args:
            file_paths: Dictionary mapping dataset names to file paths
            key_columns: Dictionary mapping dataset names to their key columns
            
        Returns:
            Dictionary with duplicate detection metrics
        """
        logging.info("Checking for duplicates...")
        
        duplicate_metrics = {}
        
        for dataset_name, file_path in file_paths.items():
            try:
                # Load dataset (sample for large files)
                df = pd.read_csv(file_path, nrows=50000)  # Sample for performance
                
                duplicate_info = {
                    "total_rows": len(df),
                    "duplicate_rows": 0,
                    "duplicate_rate": 0.0,
                    "status": "ok"
                }
                
                # Check for duplicates based on key columns
                if dataset_name in key_columns:
                    key_cols = key_columns[dataset_name]
                    
                    # Check if key columns exist
                    missing_cols = [col for col in key_cols if col not in df.columns]
                    if missing_cols:
                        duplicate_info["status"] = "error"
                        duplicate_info["error"] = f"Missing key columns: {missing_cols}"
                        self.violations.append({
                            "type": "duplicate_check",
                            "dataset": dataset_name,
                            "issue": f"Missing key columns: {missing_cols}",
                            "severity": "high"
                        })
                    else:
                        # Count duplicates
                        duplicate_mask = df.duplicated(subset=key_cols, keep=False)
                        duplicate_count = duplicate_mask.sum()
                        duplicate_rate = (duplicate_count / len(df)) * 100
                        
                        duplicate_info["duplicate_rows"] = duplicate_count
                        duplicate_info["duplicate_rate"] = duplicate_rate
                        
                        # Flag high duplicate rates
                        if duplicate_rate > 5.0:  # More than 5% duplicates
                            duplicate_info["status"] = "high_duplicates"
                            self.violations.append({
                                "type": "duplicates",
                                "dataset": dataset_name,
                                "issue": f"High duplicate rate: {duplicate_rate:.1f}%",
                                "severity": "medium"
                            })
                        
                        logging.info(f"  {dataset_name}: {duplicate_count} duplicates ({duplicate_rate:.1f}%)")
                else:
                    # Check for exact row duplicates
                    duplicate_count = df.duplicated().sum()
                    duplicate_rate = (duplicate_count / len(df)) * 100
                    
                    duplicate_info["duplicate_rows"] = duplicate_count
                    duplicate_info["duplicate_rate"] = duplicate_rate
                    
                    logging.info(f"  {dataset_name}: {duplicate_count} exact duplicates ({duplicate_rate:.1f}%)")
                
                duplicate_metrics[dataset_name] = duplicate_info
                
            except Exception as e:
                duplicate_metrics[dataset_name] = {
                    "status": "error",
                    "error": str(e)
                }
                self.violations.append({
                    "type": "duplicate_check",
                    "dataset": dataset_name,
                    "issue": f"Cannot check duplicates: {e}",
                    "severity": "high"
                })
                logging.error(f"  {dataset_name}: Error checking duplicates - {e}")
        
        self.quality_metrics["duplicates"] = duplicate_metrics
        return duplicate_metrics
    
    def check_null_rates(self, file_paths: Dict[str, str], max_null_rates: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Monitor null rates in datasets.
        
        Args:
            file_paths: Dictionary mapping dataset names to file paths
            max_null_rates: Optional dictionary with maximum allowed null rates per column
            
        Returns:
            Dictionary with null rate metrics
        """
        logging.info("Checking null rates...")
        
        null_rate_metrics = {}
        
        for dataset_name, file_path in file_paths.items():
            try:
                # Load dataset (sample for large files)
                df = pd.read_csv(file_path, nrows=50000)
                
                # Calculate null rates for each column
                null_counts = df.isnull().sum()
                null_rates = (null_counts / len(df)) * 100
                
                column_metrics = {}
                for column in df.columns:
                    null_rate = null_rates[column]
                    column_metrics[column] = {
                        "null_count": int(null_counts[column]),
                        "null_rate": float(null_rate),
                        "status": "ok"
                    }
                    
                    # Check against maximum allowed null rates
                    if max_null_rates and dataset_name in max_null_rates and column in max_null_rates[dataset_name]:
                        max_allowed = max_null_rates[dataset_name][column]
                        if null_rate > max_allowed:
                            column_metrics[column]["status"] = "high_null_rate"
                            self.violations.append({
                                "type": "null_rate",
                                "dataset": dataset_name,
                                "column": column,
                                "issue": f"High null rate: {null_rate:.1f}% > {max_allowed}%",
                                "severity": "medium"
                            })
                
                null_rate_metrics[dataset_name] = {
                    "total_rows": len(df),
                    "columns": column_metrics,
                    "overall_null_rate": float(null_rates.mean())
                }
                
                # Log columns with high null rates
                high_null_cols = null_rates[null_rates > 10].sort_values(ascending=False)
                if len(high_null_cols) > 0:
                    logging.info(f"  {dataset_name}: Columns with >10% nulls:")
                    for col, rate in high_null_cols.items():
                        logging.info(f"    {col}: {rate:.1f}%")
                else:
                    logging.info(f"  {dataset_name}: All columns have <10% null rates")
                
            except Exception as e:
                null_rate_metrics[dataset_name] = {
                    "status": "error",
                    "error": str(e)
                }
                self.violations.append({
                    "type": "null_rate_check",
                    "dataset": dataset_name,
                    "issue": f"Cannot check null rates: {e}",
                    "severity": "high"
                })
                logging.error(f"  {dataset_name}: Error checking null rates - {e}")
        
        self.quality_metrics["null_rates"] = null_rate_metrics
        return null_rate_metrics
    
    def check_business_rules(self, file_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate business rules specific to Instacart dataset.
        
        Args:
            file_paths: Dictionary mapping dataset names to file paths
            
        Returns:
            Dictionary with business rule validation results
        """
        logging.info("Checking business rules...")
        
        business_rule_metrics = {}
        
        try:
            # Load datasets for business rule validation
            datasets = {}
            for name, path in file_paths.items():
                try:
                    datasets[name] = pd.read_csv(path, nrows=10000)  # Sample for performance
                except Exception as e:
                    logging.warning(f"Cannot load {name} for business rule validation: {e}")
                    continue
            
            # Rule 1: Order sequence validation
            if "orders" in datasets:
                orders_df = datasets["orders"]
                rule_violations = []
                
                # Check if order_number sequences are reasonable
                if "order_number" in orders_df.columns and "user_id" in orders_df.columns:
                    user_order_stats = orders_df.groupby("user_id")["order_number"].agg(["min", "max", "count"])
                    
                    # Check for users with unreasonable order sequences
                    unreasonable_sequences = user_order_stats[
                        (user_order_stats["min"] != 1) |  # Should start from 1
                        (user_order_stats["max"] != user_order_stats["count"])  # Should be consecutive
                    ]
                    
                    if len(unreasonable_sequences) > 0:
                        violation_rate = (len(unreasonable_sequences) / len(user_order_stats)) * 100
                        rule_violations.append({
                            "rule": "order_sequence_integrity",
                            "violations": len(unreasonable_sequences),
                            "violation_rate": violation_rate,
                            "description": "Users with non-consecutive order numbers"
                        })
                        
                        if violation_rate > 10:  # More than 10% of users
                            self.violations.append({
                                "type": "business_rule",
                                "dataset": "orders",
                                "rule": "order_sequence_integrity",
                                "issue": f"High rate of non-consecutive order sequences: {violation_rate:.1f}%",
                                "severity": "medium"
                            })
                
                business_rule_metrics["orders"] = {
                    "rules_checked": ["order_sequence_integrity"],
                    "violations": rule_violations
                }
            
            # Rule 2: Product-Aisle-Department consistency
            if "products" in datasets and "aisles" in datasets and "departments" in datasets:
                products_df = datasets["products"]
                aisles_df = datasets["aisles"]
                departments_df = datasets["departments"]
                
                rule_violations = []
                
                # Check for orphaned product references
                if all(col in products_df.columns for col in ["aisle_id", "department_id"]):
                    orphaned_aisles = ~products_df["aisle_id"].isin(aisles_df["aisle_id"])
                    orphaned_departments = ~products_df["department_id"].isin(departments_df["department_id"])
                    
                    if orphaned_aisles.sum() > 0:
                        rule_violations.append({
                            "rule": "product_aisle_consistency",
                            "violations": int(orphaned_aisles.sum()),
                            "violation_rate": (orphaned_aisles.sum() / len(products_df)) * 100,
                            "description": "Products referencing non-existent aisles"
                        })
                    
                    if orphaned_departments.sum() > 0:
                        rule_violations.append({
                            "rule": "product_department_consistency", 
                            "violations": int(orphaned_departments.sum()),
                            "violation_rate": (orphaned_departments.sum() / len(products_df)) * 100,
                            "description": "Products referencing non-existent departments"
                        })
                
                business_rule_metrics["products"] = {
                    "rules_checked": ["product_aisle_consistency", "product_department_consistency"],
                    "violations": rule_violations
                }
            
            # Rule 3: Order-Product relationship validation
            if "order_products__train" in datasets and "orders" in datasets:
                order_products_df = datasets["order_products__train"]
                orders_df = datasets["orders"]
                
                rule_violations = []
                
                # Check for order products referencing non-existent orders
                if "order_id" in order_products_df.columns and "order_id" in orders_df.columns:
                    orphaned_orders = ~order_products_df["order_id"].isin(orders_df["order_id"])
                    
                    if orphaned_orders.sum() > 0:
                        violation_rate = (orphaned_orders.sum() / len(order_products_df)) * 100
                        rule_violations.append({
                            "rule": "order_product_consistency",
                            "violations": int(orphaned_orders.sum()),
                            "violation_rate": violation_rate,
                            "description": "Order products referencing non-existent orders"
                        })
                        
                        if violation_rate > 1:  # More than 1% orphaned
                            self.violations.append({
                                "type": "business_rule",
                                "dataset": "order_products__train",
                                "rule": "order_product_consistency",
                                "issue": f"High rate of orphaned order products: {violation_rate:.1f}%",
                                "severity": "high"
                            })
                
                business_rule_metrics["order_products"] = {
                    "rules_checked": ["order_product_consistency"],
                    "violations": rule_violations
                }
            
            logging.info(f"  Checked {len(business_rule_metrics)} datasets for business rules")
            
        except Exception as e:
            business_rule_metrics = {
                "status": "error",
                "error": str(e)
            }
            self.violations.append({
                "type": "business_rule_check",
                "issue": f"Cannot check business rules: {e}",
                "severity": "high"
            })
            logging.error(f"  Error checking business rules: {e}")
        
        self.quality_metrics["business_rules"] = business_rule_metrics
        return business_rule_metrics
    
    def generate_quality_report(self, report_name: str = None) -> str:
        """
        Generate comprehensive data quality report.
        
        Args:
            report_name: Optional custom report name
            
        Returns:
            Path to generated report file
        """
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"data_quality_report_{timestamp}"
        
        # Generate JSON report
        json_report_path = self.output_dir / f"{report_name}.json"
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_violations": len(self.violations),
                "high_severity_violations": len([v for v in self.violations if v.get("severity") == "high"]),
                "medium_severity_violations": len([v for v in self.violations if v.get("severity") == "medium"]),
                "overall_status": "fail" if len([v for v in self.violations if v.get("severity") == "high"]) > 0 else "pass"
            },
            "metrics": self.quality_metrics,
            "violations": self.violations
        }
        
        with open(json_report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate human-readable report
        txt_report_path = self.output_dir / f"{report_name}.txt"
        
        with open(txt_report_path, 'w') as f:
            f.write("INSTACART DATA QUALITY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            f.write("SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Violations: {len(self.violations)}\n")
            f.write(f"High Severity: {len([v for v in self.violations if v.get('severity') == 'high'])}\n")
            f.write(f"Medium Severity: {len([v for v in self.violations if v.get('severity') == 'medium'])}\n")
            f.write(f"Overall Status: {report_data['summary']['overall_status'].upper()}\n\n")
            
            # Violations
            if self.violations:
                f.write("VIOLATIONS\n")
                f.write("-" * 20 + "\n")
                for i, violation in enumerate(self.violations, 1):
                    f.write(f"{i}. [{violation.get('severity', 'unknown').upper()}] {violation.get('type', 'unknown')}\n")
                    f.write(f"   Dataset: {violation.get('dataset', 'N/A')}\n")
                    f.write(f"   Issue: {violation.get('issue', 'N/A')}\n")
                    if 'rule' in violation:
                        f.write(f"   Rule: {violation['rule']}\n")
                    f.write("\n")
            
            # Metrics summary
            f.write("METRICS SUMMARY\n")
            f.write("-" * 20 + "\n")
            
            if "row_counts" in self.quality_metrics:
                f.write("Row Counts:\n")
                for dataset, metrics in self.quality_metrics["row_counts"].items():
                    if metrics.get("row_count") is not None:
                        f.write(f"  {dataset}: {metrics['row_count']:,} rows\n")
                f.write("\n")
            
            if "duplicates" in self.quality_metrics:
                f.write("Duplicate Rates:\n")
                for dataset, metrics in self.quality_metrics["duplicates"].items():
                    if "duplicate_rate" in metrics:
                        f.write(f"  {dataset}: {metrics['duplicate_rate']:.1f}%\n")
                f.write("\n")
            
            if "null_rates" in self.quality_metrics:
                f.write("Average Null Rates:\n")
                for dataset, metrics in self.quality_metrics["null_rates"].items():
                    if "overall_null_rate" in metrics:
                        f.write(f"  {dataset}: {metrics['overall_null_rate']:.1f}%\n")
                f.write("\n")
        
        logging.info(f"Data quality report generated:")
        logging.info(f"  JSON: {json_report_path}")
        logging.info(f"  Text: {txt_report_path}")
        
        return str(json_report_path)
    
    def log_structured_metrics(self) -> None:
        """
        Log data quality metrics in structured format for monitoring systems.
        """
        logging.info("DATA_QUALITY_METRICS: " + json.dumps({
            "timestamp": datetime.now().isoformat(),
            "total_violations": len(self.violations),
            "high_severity_violations": len([v for v in self.violations if v.get("severity") == "high"]),
            "medium_severity_violations": len([v for v in self.violations if v.get("severity") == "medium"]),
            "metrics_summary": {
                "datasets_checked": len(set(
                    list(self.quality_metrics.get("row_counts", {}).keys()) +
                    list(self.quality_metrics.get("duplicates", {}).keys()) +
                    list(self.quality_metrics.get("null_rates", {}).keys())
                )),
                "checks_performed": list(self.quality_metrics.keys())
            }
        }, default=str))


def run_comprehensive_quality_checks(data_dir: str = "data", output_dir: str = "data/quality_reports") -> str:
    """
    Run comprehensive data quality checks on Instacart dataset.
    
    Args:
        data_dir: Directory containing CSV files
        output_dir: Directory to save quality reports
        
    Returns:
        Path to generated quality report
    """
    # Initialize quality checker
    checker = DataQualityChecker(output_dir)
    
    # Define file paths
    file_paths = {
        "orders": f"{data_dir}/orders.csv",
        "products": f"{data_dir}/products.csv",
        "aisles": f"{data_dir}/aisles.csv", 
        "departments": f"{data_dir}/departments.csv",
        "order_products__prior": f"{data_dir}/order_products__prior.csv",
        "order_products__train": f"{data_dir}/order_products__train.csv"
    }
    
    # Define key columns for duplicate detection
    key_columns = {
        "orders": ["order_id"],
        "products": ["product_id"],
        "aisles": ["aisle_id"],
        "departments": ["department_id"],
        "order_products__prior": ["order_id", "product_id"],
        "order_products__train": ["order_id", "product_id"]
    }
    
    # Define expected row count ranges (rough estimates)
    expected_ranges = {
        "orders": (1000, 5000000),  # 1K to 5M orders
        "products": (1000, 100000),  # 1K to 100K products
        "aisles": (50, 500),  # 50 to 500 aisles
        "departments": (5, 50),  # 5 to 50 departments
        "order_products__prior": (10000, 50000000),  # 10K to 50M order products
        "order_products__train": (1000, 5000000)  # 1K to 5M order products
    }
    
    # Define maximum allowed null rates
    max_null_rates = {
        "orders": {
            "order_id": 0.0,
            "user_id": 0.0,
            "eval_set": 0.0,
            "order_number": 0.0,
            "order_dow": 0.0,
            "order_hour_of_day": 0.0,
            "days_since_prior_order": 50.0  # Can be null for first orders
        },
        "products": {
            "product_id": 0.0,
            "product_name": 0.0,
            "aisle_id": 0.0,
            "department_id": 0.0
        }
    }
    
    # Run all quality checks
    checker.check_row_counts(file_paths, expected_ranges)
    checker.check_duplicates(file_paths, key_columns)
    checker.check_null_rates(file_paths, max_null_rates)
    checker.check_business_rules(file_paths)
    
    # Log structured metrics
    checker.log_structured_metrics()
    
    # Generate comprehensive report
    report_path = checker.generate_quality_report()
    
    return report_path