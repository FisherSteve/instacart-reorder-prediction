#!/usr/bin/env python3
"""
Pipeline Validation Script

Als Lernprojekt: Dieses Skript validiert die komplette Pipeline und prüft
auf häufige Fehlerquellen wie Data Leakage, fehlende Dateien, und 
deterministische Reproduzierbarkeit.

Ich implementiere hier bewusst umfassende Checks, weil Pipeline-Validierung
in der Praxis oft übersehen wird, aber kritisch für zuverlässige ML-Systeme ist.
"""

import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Dict, List, Tuple, Optional

def log_step(message: str) -> None:
    """Hilfsfunktion für klare Logging-Ausgaben"""
    print(f"[VALIDATION] {message}")

def check_required_files() -> bool:
    """
    Prüft ob alle benötigten Input-Dateien vorhanden sind.
    
    Ich checke hier explizit alle CSV-Dateien, weil fehlende Daten
    der häufigste Grund für Pipeline-Failures sind.
    """
    log_step("Checking required input files...")
    
    required_files = [
        "data/raw/orders.csv",
        "data/raw/order_products__prior.csv", 
        "data/raw/order_products__train.csv",
        "data/raw/products.csv",
        "data/raw/aisles.csv",
        "data/raw/departments.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        log_step(f"ERROR: Missing required files: {missing_files}")
        return False
    
    log_step("✓ All required input files found")
    return True

def check_data_leakage() -> bool:
    """
    Validiert dass keine Data Leakage zwischen prior/train Daten existiert.
    
    Kritischer Check: Ich stelle sicher, dass train orders nicht für 
    Feature Engineering verwendet werden. Das ist ein häufiger Fehler
    der zu unrealistisch guten Modell-Performance führt.
    """
    log_step("Checking for data leakage...")
    
    try:
        # Lade die relevanten Dateien
        orders = pd.read_csv("data/raw/orders.csv")
        prior_products = pd.read_csv("data/raw/order_products__prior.csv")
        train_products = pd.read_csv("data/raw/order_products__train.csv")
        
        # Identifiziere train vs prior orders
        train_orders = set(orders[orders['eval_set'] == 'train']['order_id'])
        prior_orders = set(orders[orders['eval_set'] == 'prior']['order_id'])
        
        # Check 1: Keine Überschneidung zwischen train/prior order_ids
        overlap = train_orders.intersection(prior_orders)
        if overlap:
            log_step(f"ERROR: Found {len(overlap)} orders in both train and prior sets")
            return False
        
        # Check 2: prior_products sollte nur prior orders enthalten
        prior_product_orders = set(prior_products['order_id'])
        train_in_prior = prior_product_orders.intersection(train_orders)
        if train_in_prior:
            log_step(f"ERROR: Found {len(train_in_prior)} train orders in prior products")
            return False
        
        # Check 3: train_products sollte nur train orders enthalten  
        train_product_orders = set(train_products['order_id'])
        prior_in_train = train_product_orders.intersection(prior_orders)
        if prior_in_train:
            log_step(f"ERROR: Found {len(prior_in_train)} prior orders in train products")
            return False
        
        log_step("✓ No data leakage detected between train/prior sets")
        return True
        
    except Exception as e:
        log_step(f"ERROR: Failed to check data leakage: {e}")
        return False

def run_build_dataset() -> bool:
    """
    Führt build_dataset.py aus und validiert das Ergebnis.
    
    Ich teste hier den ersten Schritt der Pipeline und prüfe
    ob features.parquet korrekt generiert wird.
    """
    log_step("Running build_dataset.py...")
    
    try:
        # Lösche alte features.parquet falls vorhanden
        if os.path.exists("data/features/features.parquet"):
            os.remove("data/features/features.parquet")
        
        # Führe build_dataset.py aus
        result = subprocess.run([
            sys.executable, "src/build_dataset.py"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            log_step(f"ERROR: build_dataset.py failed with return code {result.returncode}")
            log_step(f"STDERR: {result.stderr}")
            return False
        
        # Prüfe ob features.parquet erstellt wurde
        if not os.path.exists("data/features/features.parquet"):
            log_step("ERROR: features.parquet was not created")
            return False
        
        log_step("✓ build_dataset.py completed successfully")
        return True
        
    except subprocess.TimeoutExpired:
        log_step("ERROR: build_dataset.py timed out after 5 minutes")
        return False
    except Exception as e:
        log_step(f"ERROR: Failed to run build_dataset.py: {e}")
        return False

def validate_features_dataset() -> bool:
    """
    Validiert die generierte features.parquet Datei.
    
    Ich prüfe hier Schema, Datenqualität und logische Konsistenz
    der generierten Features.
    """
    log_step("Validating features dataset...")
    
    try:
        df = pd.read_parquet("data/features/features.parquet")
        
        # Check 1: Erwartete Spalten vorhanden
        expected_columns = [
            'user_id', 'product_id', 'y',
            'times_bought', 'times_reordered', 'user_prod_reorder_rate',
            'last_prior_ordnum', 'orders_since_last', 'avg_add_to_cart_pos',
            'avg_days_since_prior', 'aisle_id', 'department_id',
            'prod_cnt', 'prod_users'
        ]
        
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            log_step(f"ERROR: Missing expected columns: {missing_cols}")
            return False
        
        # Check 2: Keine leeren Daten
        if df.empty:
            log_step("ERROR: Features dataset is empty")
            return False
        
        # Check 3: Label distribution sinnvoll (nicht alle 0 oder alle 1)
        label_dist = df['y'].value_counts()
        if len(label_dist) != 2:
            log_step(f"ERROR: Invalid label distribution: {label_dist}")
            return False
        
        reorder_rate = label_dist[1] / len(df)
        if reorder_rate < 0.1 or reorder_rate > 0.9:
            log_step(f"WARNING: Unusual reorder rate: {reorder_rate:.3f}")
        
        # Check 4: Feature-Werte in sinnvollen Bereichen
        if df['times_bought'].min() < 0:
            log_step("ERROR: Negative times_bought values found")
            return False
        
        if df['user_prod_reorder_rate'].min() < 0 or df['user_prod_reorder_rate'].max() > 1:
            log_step("ERROR: user_prod_reorder_rate outside [0,1] range")
            return False
        
        log_step(f"✓ Features dataset validated: {len(df)} rows, {len(df.columns)} columns")
        log_step(f"  Reorder rate: {reorder_rate:.3f}")
        log_step(f"  Users: {df['user_id'].nunique()}, Products: {df['product_id'].nunique()}")
        
        return True
        
    except Exception as e:
        log_step(f"ERROR: Failed to validate features dataset: {e}")
        return False

def run_model_training(model_name: str, seed: int = 42) -> bool:
    """
    Führt Modell-Training aus und validiert das Ergebnis.
    
    Ich teste hier mit einem festen Seed für deterministische Ergebnisse.
    """
    log_step(f"Training {model_name} model with seed {seed}...")
    
    try:
        # Lösche alte Modell-Dateien
        model_file = f"reports/model_test_{model_name}.joblib"
        metrics_file = f"reports/metrics_test_{model_name}.json"
        
        for file_path in [model_file, metrics_file]:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Führe Training aus
        result = subprocess.run([
            sys.executable, "src/train.py",
            "--model", model_name,
            "--out", f"test_{model_name}",
            "--seed", str(seed),
            "--topk", "10"
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            log_step(f"ERROR: Training {model_name} failed with return code {result.returncode}")
            log_step(f"STDERR: {result.stderr}")
            return False
        
        # Prüfe ob Output-Dateien erstellt wurden
        if not os.path.exists(model_file):
            log_step(f"ERROR: Model file {model_file} was not created")
            return False
        
        if not os.path.exists(metrics_file):
            log_step(f"ERROR: Metrics file {metrics_file} was not created")
            return False
        
        log_step(f"✓ {model_name} training completed successfully")
        return True
        
    except subprocess.TimeoutExpired:
        log_step(f"ERROR: Training {model_name} timed out after 10 minutes")
        return False
    except Exception as e:
        log_step(f"ERROR: Failed to train {model_name}: {e}")
        return False

def validate_model_outputs(model_name: str) -> bool:
    """
    Validiert die Modell-Outputs (Metriken und gespeichertes Modell).
    
    Ich prüfe hier ob die Metriken sinnvolle Werte haben und
    das Modell korrekt serialisiert wurde.
    """
    log_step(f"Validating {model_name} model outputs...")
    
    try:
        metrics_file = f"reports/metrics_test_{model_name}.json"
        
        # Lade und validiere Metriken
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Check 1: Erwartete Metriken vorhanden
        expected_metrics = ['roc_auc', 'pr_auc', 'f1_at_0.5', 'order_f1_topk']
        missing_metrics = set(expected_metrics) - set(metrics.keys())
        if missing_metrics:
            log_step(f"ERROR: Missing metrics: {missing_metrics}")
            return False
        
        # Check 2: Metriken in sinnvollen Bereichen
        roc_auc = metrics['roc_auc']
        if roc_auc < 0.5 or roc_auc > 1.0:
            log_step(f"ERROR: ROC-AUC out of range: {roc_auc}")
            return False
        
        if roc_auc < 0.7:
            log_step(f"WARNING: Low ROC-AUC performance: {roc_auc:.3f}")
        
        # Check 3: Training time reasonable
        train_time = metrics.get('train_secs', 0)
        if train_time > 3600:  # 1 Stunde
            log_step(f"WARNING: Very long training time: {train_time:.1f} seconds")
        
        log_step(f"✓ {model_name} metrics validated: ROC-AUC={roc_auc:.3f}")
        return True
        
    except Exception as e:
        log_step(f"ERROR: Failed to validate {model_name} outputs: {e}")
        return False

def test_deterministic_behavior() -> bool:
    """
    Testet ob die Pipeline deterministische Ergebnisse mit festen Seeds liefert.
    
    Kritischer Test: Ich führe das gleiche Training zweimal aus und
    prüfe ob identische Ergebnisse entstehen.
    """
    log_step("Testing deterministic behavior...")
    
    try:
        # Führe Training zweimal mit gleichem Seed aus
        seed = 12345
        model = "logreg"  # Schnellstes Modell für Test
        
        results = []
        for run in [1, 2]:
            log_step(f"  Determinism test run {run}/2...")
            
            # Lösche alte Dateien
            metrics_file = f"reports/metrics_determ_test_{run}.json"
            if os.path.exists(metrics_file):
                os.remove(metrics_file)
            
            # Training ausführen
            result = subprocess.run([
                sys.executable, "src/train.py",
                "--model", model,
                "--out", f"determ_test_{run}",
                "--seed", str(seed)
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                log_step(f"ERROR: Determinism test run {run} failed")
                return False
            
            # Metriken laden
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            results.append(metrics['roc_auc'])
        
        # Vergleiche Ergebnisse
        if abs(results[0] - results[1]) > 1e-10:
            log_step(f"ERROR: Non-deterministic results: {results[0]} vs {results[1]}")
            return False
        
        log_step(f"✓ Deterministic behavior confirmed: ROC-AUC={results[0]:.6f}")
        return True
        
    except Exception as e:
        log_step(f"ERROR: Failed determinism test: {e}")
        return False

def run_report_generation() -> bool:
    """
    Testet die Report-Generierung.
    
    Ich prüfe hier ob report.py korrekt läuft und die erwarteten
    Output-Dateien generiert.
    """
    log_step("Testing report generation...")
    
    try:
        # Lösche alte Report-Dateien
        report_files = ["reports/report.html", "reports/metrics_leaderboard.csv"]
        for file_path in report_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Führe report.py aus
        result = subprocess.run([
            sys.executable, "src/report.py"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            log_step(f"ERROR: report.py failed with return code {result.returncode}")
            log_step(f"STDERR: {result.stderr}")
            return False
        
        # Prüfe ob Report-Dateien erstellt wurden
        for file_path in report_files:
            if not os.path.exists(file_path):
                log_step(f"ERROR: {file_path} was not created")
                return False
        
        log_step("✓ Report generation completed successfully")
        return True
        
    except Exception as e:
        log_step(f"ERROR: Failed to generate reports: {e}")
        return False

def cleanup_test_files() -> None:
    """
    Räumt Test-Dateien auf.
    
    Ich lösche hier alle temporären Dateien die während der Validierung
    erstellt wurden, um das Repository sauber zu halten.
    """
    log_step("Cleaning up test files...")
    
    test_patterns = [
        "reports/model_test_*.joblib",
        "reports/metrics_test_*.json", 
        "reports/model_determ_test_*.joblib",
        "reports/metrics_determ_test_*.json"
    ]
    
    import glob
    for pattern in test_patterns:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                log_step(f"  Removed {file_path}")
            except Exception as e:
                log_step(f"  Warning: Could not remove {file_path}: {e}")

def main() -> int:
    """
    Hauptfunktion: Führt alle Validierungstests aus.
    
    Als Lernprojekt: Ich strukturiere die Tests logisch und gebe
    klares Feedback über jeden Schritt. Bei Fehlern wird sofort
    abgebrochen um schnelles Debugging zu ermöglichen.
    """
    log_step("Starting end-to-end pipeline validation")
    log_step("=" * 50)
    
    # Schritt 1: Grundlegende Checks
    if not check_required_files():
        return 1
    
    if not check_data_leakage():
        return 1
    
    # Schritt 2: Dataset Building
    if not run_build_dataset():
        return 1
    
    if not validate_features_dataset():
        return 1
    
    # Schritt 3: Model Training (teste nur LogReg für Geschwindigkeit)
    if not run_model_training("logreg"):
        return 1
    
    if not validate_model_outputs("logreg"):
        return 1
    
    # Schritt 4: Determinismus-Test
    if not test_deterministic_behavior():
        return 1
    
    # Schritt 5: Report Generation
    if not run_report_generation():
        return 1
    
    # Cleanup
    cleanup_test_files()
    
    log_step("=" * 50)
    log_step("✓ All validation tests passed successfully!")
    log_step("Pipeline is ready for production use.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())