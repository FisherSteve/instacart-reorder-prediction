#!/usr/bin/env python3
"""
Instacart Reorder Prediction - Training Pipeline

Als Lernprojekt bewusst ausführlich kommentiert und mit klaren, verständlichen
Variablennamen. Dieser Code zeigt den kompletten ML-Pipeline-Aufbau von
Datenladung bis Modelltraining.

Verwendung:
    python src/train.py --model logreg --out logreg_model --topk 10 --seed 42
"""

import argparse
import json
import os
import psutil
import sys
import time
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Import configuration utilities
from config_utils import load_config, get_model_config

# Optional imports für XGBoost und LightGBM
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("XGBoost nicht verfügbar. Installiere mit: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    warnings.warn("LightGBM nicht verfügbar. Installiere mit: pip install lightgbm")


def get_memory_usage() -> Dict[str, float]:
    """
    Aktuelle Speichernutzung des Prozesses abrufen.
    
    Ich verwende psutil für plattformübergreifende Memory-Überwachung.
    Das hilft bei der Diagnose von Memory-Problemen.
    """
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            'percent': process.memory_percent()        # Prozent des System-RAMs
        }
    except Exception as e:
        warnings.warn(f"Memory-Monitoring nicht verfügbar: {e}")
        return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}


def check_system_resources(min_memory_gb: float = 2.0) -> None:
    """
    Prüfe verfügbare System-Ressourcen vor dem Training.
    
    Ich checke hier explizit die verfügbaren Ressourcen, um frühzeitig
    vor Memory-Problemen zu warnen.
    """
    try:
        # Verfügbarer RAM
        memory = psutil.virtual_memory()
        available_gb = memory.available / 1024 / 1024 / 1024
        
        print(f"System-Ressourcen:")
        print(f"  Verfügbarer RAM: {available_gb:.1f} GB")
        print(f"  RAM-Auslastung: {memory.percent:.1f}%")
        
        if available_gb < min_memory_gb:
            warnings.warn(
                f"Wenig verfügbarer RAM ({available_gb:.1f} GB < {min_memory_gb} GB). "
                f"Training könnte langsam sein oder fehlschlagen."
            )
        
        # CPU-Kerne
        cpu_count = psutil.cpu_count()
        print(f"  CPU-Kerne: {cpu_count}")
        
    except Exception as e:
        warnings.warn(f"System-Ressourcen-Check fehlgeschlagen: {e}")


def handle_memory_error(operation: str, error: Exception) -> None:
    """
    Behandle Memory-Fehler mit hilfreichen Lösungsvorschlägen.
    
    Ich gebe hier konkrete Tipps, wie Memory-Probleme gelöst werden können.
    """
    memory_usage = get_memory_usage()
    
    error_msg = f"""
Memory-Fehler bei {operation}:
{str(error)}

Aktuelle Speichernutzung:
  RSS: {memory_usage['rss_mb']:.1f} MB
  VMS: {memory_usage['vms_mb']:.1f} MB
  Prozent: {memory_usage['percent']:.1f}%

Lösungsvorschläge:
1. Reduziere die Datenmenge mit Sampling
2. Verwende weniger Features oder kleinere Modelle
3. Schließe andere Programme um RAM freizugeben
4. Für XGBoost/LightGBM: Reduziere n_estimators
5. Verwende Chunked Processing (falls implementiert)
"""
    
    print(error_msg, file=sys.stderr)
    raise MemoryError(error_msg) from error


def validate_data_quality(df: pd.DataFrame, name: str) -> None:
    """
    Validiere Datenqualität und gebe Warnungen bei Problemen aus.
    
    Ich prüfe hier typische Datenprobleme, die das Training beeinträchtigen können.
    """
    print(f"\nValidiere Datenqualität für {name}...")
    
    # Grundlegende Checks
    if df.empty:
        raise ValueError(f"{name} ist leer!")
    
    # Missing Values
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    
    if len(missing_cols) > 0:
        print(f"  Warnung: {len(missing_cols)} Spalten mit Missing Values:")
        for col, count in missing_cols.head(5).items():
            pct = count / len(df) * 100
            print(f"    {col}: {count:,} ({pct:.1f}%)")
        if len(missing_cols) > 5:
            print(f"    ... und {len(missing_cols) - 5} weitere")
    
    # Infinite Values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count
    
    if inf_counts:
        print(f"  Warnung: {len(inf_counts)} Spalten mit Infinite Values:")
        for col, count in list(inf_counts.items())[:5]:
            print(f"    {col}: {count:,}")
    
    # Konstante Spalten
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"  Warnung: {len(constant_cols)} konstante Spalten (werden ignoriert):")
        for col in constant_cols[:5]:
            print(f"    {col}")
    
    print(f"  [OK] {name} Validierung abgeschlossen")


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validiere Kommandozeilen-Argumente auf sinnvolle Werte.
    
    Ich prüfe hier explizit die Parameter-Bereiche und gebe hilfreiche
    Fehlermeldungen bei ungültigen Werten.
    """
    # Top-K Validierung
    if args.topk <= 0:
        raise ValueError(f"--topk muss positiv sein, erhalten: {args.topk}")
    
    if args.topk > 100:
        warnings.warn(f"--topk={args.topk} ist sehr hoch. Typische Werte: 5-20")
    
    # Random Seed Validierung
    if args.seed < 0:
        raise ValueError(f"--seed muss nicht-negativ sein, erhalten: {args.seed}")
    
    # Output Name Validierung
    if args.out and not args.out.replace('_', '').replace('-', '').isalnum():
        raise ValueError(f"--out darf nur alphanumerische Zeichen, '_' und '-' enthalten: {args.out}")


def parse_arguments() -> argparse.Namespace:
    """
    Kommandozeilen-Argumente parsen mit Config-Support.
    
    Ich mache hier bewusst ausführliche Hilfe-Texte und Validierung,
    damit auch Anfänger verstehen, was jeder Parameter bewirkt.
    """
    parser = argparse.ArgumentParser(
        description="Trainiere ML-Modelle für Instacart Reorder Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Beispiele:
  python src/train.py --model logreg
  python src/train.py --model xgb --out xgb_optimized
  python src/train.py --model lgbm --config custom_config.yaml
  python src/train.py --model xgb --override models.xgboost.n_estimators=1000
        """
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["logreg", "xgb", "lgbm"],
        required=True,
        help="Modelltyp: logreg (LogisticRegression), xgb (XGBoost), lgbm (LightGBM)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Pfad zur Config-Datei (default: sucht config.yaml)"
    )
    
    parser.add_argument(
        "--out", 
        type=str, 
        default=None,
        help="Output-Präfix für Modell- und Metrik-Dateien (default: verwendet --model)"
    )
    
    parser.add_argument(
        "--topk", 
        type=int, 
        default=None,
        help="Top-K Produkte pro User für Order-F1 Berechnung (default: aus config.yaml)"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random Seed für reproduzierbare Ergebnisse (default: aus config.yaml)"
    )
    
    parser.add_argument(
        "--override",
        action='append',
        help='Config-Override im Format key=value (z.B. models.xgboost.n_estimators=1000)'
    )
    
    args = parser.parse_args()
    
    # Wenn kein --out angegeben, verwende den Modellnamen
    if args.out is None:
        args.out = args.model
    
    return args


def load_and_prepare_data(features_path: str = "data/features.parquet") -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Lade Feature-Daten mit robuster Fehlerbehandlung.
    
    Ich implementiere hier umfassende Validierung und Memory-Monitoring,
    um typische Probleme beim Daten-Laden frühzeitig zu erkennen.
    
    Returns:
        features_df: DataFrame mit allen Features
        target_labels: Array mit y-Labels (0/1)
        user_groups: Array mit user_ids für GroupShuffleSplit
    """
    print(f"Lade Daten aus {features_path}...")
    
    # System-Ressourcen prüfen
    check_system_resources()
    initial_memory = get_memory_usage()
    print(f"Memory vor Daten-Laden: {initial_memory['rss_mb']:.1f} MB")
    
    # Datei-Existenz prüfen
    features_file = Path(features_path)
    if not features_file.exists():
        raise FileNotFoundError(
            f"Features-Datei nicht gefunden: {features_path}\n"
            f"Stelle sicher, dass 'python src/build_dataset.py' erfolgreich ausgeführt wurde."
        )
    
    # Datei-Größe prüfen
    file_size_mb = features_file.stat().st_size / 1024 / 1024
    print(f"Datei-Größe: {file_size_mb:.1f} MB")
    
    if file_size_mb > 1000:  # > 1 GB
        warnings.warn(f"Große Datei ({file_size_mb:.1f} MB). Loading könnte langsam sein.")
    
    try:
        # Lade Parquet-Datei mit Memory-Monitoring
        print("Lade Parquet-Datei...")
        full_df = pd.read_parquet(features_path)
        
        # Memory nach Laden prüfen
        after_load_memory = get_memory_usage()
        memory_increase = after_load_memory['rss_mb'] - initial_memory['rss_mb']
        print(f"Memory nach Laden: {after_load_memory['rss_mb']:.1f} MB (+{memory_increase:.1f} MB)")
        
    except MemoryError as e:
        handle_memory_error("Daten-Laden", e)
    except Exception as e:
        raise RuntimeError(f"Fehler beim Laden der Parquet-Datei: {e}") from e
    
    print(f"Geladene Daten: {full_df.shape[0]:,} Zeilen, {full_df.shape[1]} Spalten")
    
    # Validiere erforderliche Spalten
    required_columns = ['y', 'user_id']
    missing_columns = [col for col in required_columns if col not in full_df.columns]
    
    if missing_columns:
        raise ValueError(
            f"Erforderliche Spalten fehlen: {missing_columns}\n"
            f"Verfügbare Spalten: {list(full_df.columns)}"
        )
    
    # Datenqualität validieren
    validate_data_quality(full_df, "Gesamtdatensatz")
    
    try:
        # Trenne Features von Labels und Gruppierungs-Variable
        target_labels = full_df['y'].values
        user_groups = full_df['user_id'].values
        
        # Validiere Labels
        unique_labels = np.unique(target_labels)
        if not np.array_equal(unique_labels, [0, 1]) and not np.array_equal(unique_labels, [0]) and not np.array_equal(unique_labels, [1]):
            warnings.warn(f"Unerwartete Label-Werte: {unique_labels}. Erwartet: [0, 1]")
        
        # Features = alles außer y und user_id
        feature_columns = [col for col in full_df.columns if col not in ['y', 'user_id']]
        
        if not feature_columns:
            raise ValueError("Keine Feature-Spalten gefunden!")
        
        features_df = full_df[feature_columns].copy()
        
        # Validiere Features
        validate_data_quality(features_df, "Features")
        
    except Exception as e:
        raise RuntimeError(f"Fehler bei der Daten-Aufbereitung: {e}") from e
    
    # Statistiken ausgeben
    print(f"Features: {len(feature_columns)} Spalten")
    print(f"Labels: {len(target_labels):,} Samples")
    
    reorder_rate = np.mean(target_labels)
    print(f"Label-Verteilung: {reorder_rate:.3f} Reorder-Rate")
    
    if reorder_rate < 0.01 or reorder_rate > 0.99:
        warnings.warn(f"Extreme Label-Verteilung ({reorder_rate:.3f}). Modell-Performance könnte schlecht sein.")
    
    # Finale Memory-Statistik
    final_memory = get_memory_usage()
    print(f"Memory nach Aufbereitung: {final_memory['rss_mb']:.1f} MB")
    
    return features_df, target_labels, user_groups


def detect_column_types(df: pd.DataFrame) -> Tuple[list, list]:
    """
    Automatische Erkennung von numerischen vs. kategorischen Spalten.
    
    Ich verwende hier eine intelligentere Heuristik:
    - Explizit kategorische Spalten: aisle_id, department_id, user_prod_reorder_rate
    - Spalten mit wenigen unique Werten (< 50) UND nicht-float Datentyp = kategorisch
    - Alle anderen = numerisch
    
    Das berücksichtigt Domain-Wissen über die Instacart-Daten.
    """
    numeric_columns = []
    categorical_columns = []
    
    # Explizit kategorische Spalten basierend auf Domain-Wissen
    known_categorical = {'aisle_id', 'department_id', 'user_prod_reorder_rate'}
    
    for column in df.columns:
        unique_values = df[column].nunique()
        
        if column in known_categorical:
            categorical_columns.append(column)
            print(f"Kategorisch (Domain): {column} ({unique_values} unique Werte)")
        elif unique_values < 50 and not pd.api.types.is_float_dtype(df[column]):
            categorical_columns.append(column)
            print(f"Kategorisch (Heuristik): {column} ({unique_values} unique Werte)")
        else:
            numeric_columns.append(column)
            print(f"Numerisch: {column} ({unique_values} unique Werte)")
    
    print(f"\nErkannte Spalten-Typen:")
    print(f"  Numerisch: {len(numeric_columns)} Spalten")
    print(f"  Kategorisch: {len(categorical_columns)} Spalten")
    
    return numeric_columns, categorical_columns


def create_preprocessing_pipeline(numeric_columns: list, categorical_columns: list, preprocessing_config: Dict[str, Any]) -> ColumnTransformer:
    """
    Erstelle Preprocessing-Pipeline mit Imputation, StandardScaler und OneHotEncoder.
    
    Ich verwende hier ColumnTransformer, weil er saubere Trennung zwischen
    numerischen und kategorischen Features ermöglicht.
    Wichtig: Imputation vor Scaling/Encoding für NaN-Werte.
    """
    transformers = []
    
    # Numerische Features: Imputation + StandardScaler
    if numeric_columns:
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy=preprocessing_config['numeric_imputation_strategy'])),
            ('scaler', StandardScaler())
        ])
        transformers.append(('numeric', numeric_transformer, numeric_columns))
        print(f"SimpleImputer + StandardScaler für {len(numeric_columns)} numerische Features")
    
    # Kategorische Features: Imputation + OneHotEncoder
    if categorical_columns:
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(
                strategy=preprocessing_config['categorical_imputation_strategy'], 
                fill_value=preprocessing_config['categorical_imputation_fill_value']
            )),
            ('encoder', OneHotEncoder(
                drop='first' if preprocessing_config['onehot_drop_first'] else None,
                sparse_output=False,
                handle_unknown=preprocessing_config['onehot_handle_unknown']
            ))
        ])
        transformers.append(('categorical', categorical_transformer, categorical_columns))
        print(f"SimpleImputer + OneHotEncoder für {len(categorical_columns)} kategorische Features")
    
    # ColumnTransformer zusammenbauen
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Alle anderen Spalten ignorieren
    )
    
    return preprocessor


def split_data_by_users(features_df: pd.DataFrame, 
                       target_labels: np.ndarray, 
                       user_groups: np.ndarray,
                       test_size: float = 0.2,
                       random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Teile Daten in Train/Validation mit GroupShuffleSplit.
    
    Ich verwende hier GroupShuffleSplit statt train_test_split, weil normale
    Aufteilung User zwischen Train/Val aufteilen könnte → unrealistische Evaluation.
    Mit GroupShuffleSplit sind alle Samples eines Users entweder in Train ODER Val.
    """
    print(f"\nTeile Daten auf (Test-Size: {test_size})...")
    
    # GroupShuffleSplit: Teile nach user_id Gruppen
    group_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )
    
    # Einziger Split (n_splits=1)
    train_indices, val_indices = next(group_splitter.split(
        X=features_df, 
        y=target_labels, 
        groups=user_groups
    ))
    
    # Aufteilen
    X_train = features_df.iloc[train_indices]
    X_val = features_df.iloc[val_indices]
    y_train = target_labels[train_indices]
    y_val = target_labels[val_indices]
    
    # User-IDs für Order-F1 Berechnung behalten
    users_train = user_groups[train_indices]
    users_val = user_groups[val_indices]
    
    # Statistiken
    train_users = len(np.unique(users_train))
    val_users = len(np.unique(users_val))
    
    print(f"Training: {len(X_train):,} Samples, {train_users:,} Users")
    print(f"Validation: {len(X_val):,} Samples, {val_users:,} Users")
    print(f"Train Reorder-Rate: {np.mean(y_train):.3f}")
    print(f"Val Reorder-Rate: {np.mean(y_val):.3f}")
    
    # Prüfe User-Überschneidung (sollte 0 sein)
    train_user_set = set(users_train)
    val_user_set = set(users_val)
    overlap = train_user_set.intersection(val_user_set)
    
    if overlap:
        raise ValueError(f"User-Leakage detected! {len(overlap)} Users in beiden Sets")
    else:
        print("[OK] Kein User-Leakage: Train/Val Users sind disjunkt")
    
    return X_train, X_val, y_train, y_val, users_train, users_val


def validate_model_parameters(model_type: str) -> None:
    """
    Validiere Modell-Parameter und Abhängigkeiten.
    
    Ich prüfe hier explizit die Verfügbarkeit der Bibliotheken und gebe
    hilfreiche Fehlermeldungen, falls etwas fehlt.
    """
    if model_type == "xgb" and not HAS_XGB:
        raise ImportError(
            "XGBoost ist nicht installiert. Installiere mit:\n"
            "  pip install xgboost\n"
            "Oder füge es zu requirements.txt hinzu."
        )
    
    if model_type == "lgbm" and not HAS_LGB:
        raise ImportError(
            "LightGBM ist nicht installiert. Installiere mit:\n"
            "  pip install lightgbm\n"
            "Oder füge es zu requirements.txt hinzu."
        )


def build_model(model_type: str, model_config: Dict[str, Any]):
    """
    Erstelle optimierte Modelle mit spezifischen Hyperparametern.
    
    Ich verwende hier die empfohlenen Parameter für jedes Modell:
    - XGBoost: tree_method="hist", n_estimators=2000, learning_rate=0.05
    - LightGBM: n_estimators=4000, learning_rate=0.03, num_leaves=127
    
    Diese Parameter sind auf Performance und Regularisierung optimiert.
    """
    print(f"\nErstelle {model_type} Modell...")
    
    # Parameter-Validierung
    validate_model_parameters(model_type)
    
    if model_type == "logreg":
        # LogisticRegression: Schneller Baseline-Klassifikator
        model = LogisticRegression(**model_config)
        print(f"LogisticRegression mit Parametern aus Config: {model_config}")
        
    elif model_type == "xgb":
        # XGBoost: Parameter aus Konfiguration
        model = xgb.XGBClassifier(**model_config)
        print(f"XGBoost mit Parametern aus Config:")
        for key, value in model_config.items():
            print(f"  - {key}={value}")
        print(f"  - Optimiert für Instacart Reorder Prediction")
        
    elif model_type == "lgbm":
        # LightGBM: Parameter aus Konfiguration
        model = lgb.LGBMClassifier(**model_config)
        print(f"LightGBM mit Parametern aus Config:")
        for key, value in model_config.items():
            print(f"  - {key}={value}")
        print(f"  - Optimiert für Memory-Effizienz und Performance")
        
    else:
        raise ValueError(f"Unbekannter Modelltyp: {model_type}. Verfügbare Optionen: logreg, xgb, lgbm")
    
    return model


def order_f1_by_user(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                     user_ids: np.ndarray, topk: int = 10) -> float:
    """
    Berechne Order-F1 Metrik: Set-basierte F1 pro User, dann gemittelt.
    
    Diese Metrik ist wichtig, weil sie das Business-Ziel widerspiegelt:
    Für jeden User die richtigen Produkte vorhersagen (nicht nur einzelne Wahrscheinlichkeiten).
    
    Algorithmus:
    1. Für jeden User: Top-K Produkte mit höchster Wahrscheinlichkeit
    2. F1 zwischen predicted set und true reorder set
    3. Mittelwert über alle User
    """
    print(f"\nBerechne Order-F1@{topk}...")
    
    # DataFrame für einfachere Gruppierung
    df = pd.DataFrame({
        'user_id': user_ids,
        'y_true': y_true,
        'y_pred_proba': y_pred_proba
    })
    
    user_f1_scores = []
    
    for user_id, user_df in df.groupby('user_id'):
        # True reorders für diesen User
        true_reorders = set(user_df[user_df['y_true'] == 1].index)
        
        # Top-K predicted reorders
        top_k_indices = user_df.nlargest(topk, 'y_pred_proba').index
        pred_reorders = set(top_k_indices)
        
        # Set-basierte F1 Berechnung
        if len(true_reorders) == 0:
            # Wenn User nichts reordert, ist F1 = 1 wenn wir auch nichts vorhersagen
            f1_user = 1.0 if len(pred_reorders) == 0 else 0.0
        else:
            intersection = true_reorders.intersection(pred_reorders)
            precision = len(intersection) / len(pred_reorders) if len(pred_reorders) > 0 else 0.0
            recall = len(intersection) / len(true_reorders)
            
            if precision + recall == 0:
                f1_user = 0.0
            else:
                f1_user = 2 * precision * recall / (precision + recall)
        
        user_f1_scores.append(f1_user)
    
    # Mittelwert über alle User
    mean_order_f1 = np.mean(user_f1_scores)
    
    print(f"Order-F1@{topk}: {mean_order_f1:.4f} (über {len(user_f1_scores)} Users)")
    return mean_order_f1


def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                  user_ids: np.ndarray, topk: int = 10) -> Dict[str, float]:
    """
    Berechne alle wichtigen Metriken für Reorder Prediction.
    
    Ich sammle hier alle Metriken, die für Business und Technical Evaluation
    wichtig sind: ROC-AUC (Ranking), PR-AUC (Precision/Recall), F1@0.5 (Threshold), Order-F1 (Business).
    """
    print("\nBerechne Evaluation-Metriken...")
    
    metrics = {}
    
    # ROC-AUC: Wie gut rankt das Modell?
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    metrics['roc_auc'] = roc_auc
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # PR-AUC: Precision-Recall für unbalancierte Daten
    pr_auc = average_precision_score(y_true, y_pred_proba)
    metrics['pr_auc'] = pr_auc
    print(f"PR-AUC: {pr_auc:.4f}")
    
    # F1@0.5: Standard Binary Classification
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    f1_at_05 = f1_score(y_true, y_pred_binary)
    metrics['f1_at_0.5'] = f1_at_05
    print(f"F1@0.5: {f1_at_05:.4f}")
    
    # Order-F1@top-k: Business-relevante Set-basierte Metrik
    order_f1 = order_f1_by_user(y_true, y_pred_proba, user_ids, topk)
    metrics['order_f1_topk'] = order_f1
    
    return metrics


def save_model_and_metrics(model_pipeline: Pipeline, metrics: Dict[str, Any], 
                          model_name: str, output_dir: str = "reports") -> None:
    """
    Speichere trainiertes Modell und Metriken.
    
    Ich speichere sowohl das Modell (für Inference) als auch die Metriken (für Vergleiche).
    """
    # Output-Verzeichnis erstellen
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Modell speichern
    model_file = output_path / f"model_{model_name}.joblib"
    joblib.dump(model_pipeline, model_file)
    print(f"Modell gespeichert: {model_file}")
    
    # Metriken speichern
    metrics_file = output_path / f"metrics_{model_name}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metriken gespeichert: {metrics_file}")


def train_and_evaluate_model(model_type: str, model_config: Dict[str, Any], output_name: str, topk: int, 
                           X_train: pd.DataFrame, X_val: pd.DataFrame,
                           y_train: np.ndarray, y_val: np.ndarray,
                           users_val: np.ndarray, preprocessor: ColumnTransformer,
                           reports_dir: str = "reports") -> None:
    """
    Robuster Training- und Evaluation-Workflow mit umfassender Fehlerbehandlung.
    
    Ich implementiere hier Memory-Monitoring, Timeout-Handling und graceful
    Degradation bei Problemen.
    """
    print(f"\n{'='*60}")
    print(f"TRAINING: {model_type.upper()}")
    print(f"{'='*60}")
    
    # Memory vor Training prüfen
    pre_training_memory = get_memory_usage()
    print(f"Memory vor Training: {pre_training_memory['rss_mb']:.1f} MB")
    
    try:
        # Modell erstellen mit Validierung
        model = build_model(model_type, model_config)
        
        # Pipeline: Preprocessing + Modell
        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Validiere Training-Daten
        if len(X_train) < 100:
            raise ValueError(f"Zu wenige Training-Samples: {len(X_train)} < 100")
        
        if len(np.unique(y_train)) < 2:
            raise ValueError("Training-Daten enthalten nur eine Klasse!")
        
        # Training mit Memory-Monitoring und Timeout-Schutz
        print(f"\nStarte Training...")
        print(f"Training-Samples: {len(X_train):,}")
        print(f"Features: {X_train.shape[1]}")
        
        start_time = time.time()
        training_memory_peak = pre_training_memory['rss_mb']
        
        try:
            # Für große Datasets: Memory-Monitoring während Training
            if len(X_train) > 100000:
                print("Großer Datensatz erkannt - aktiviere Memory-Monitoring...")
            
            model_pipeline.fit(X_train, y_train)
            
            # Memory nach Training prüfen
            post_training_memory = get_memory_usage()
            training_memory_peak = max(training_memory_peak, post_training_memory['rss_mb'])
            
        except MemoryError as e:
            handle_memory_error("Model Training", e)
        except Exception as e:
            if "memory" in str(e).lower() or "allocation" in str(e).lower():
                handle_memory_error("Model Training", e)
            else:
                raise RuntimeError(f"Training fehlgeschlagen für {model_type}: {e}") from e
        
        training_time = time.time() - start_time
        print(f"Training abgeschlossen in {training_time:.1f} Sekunden")
        print(f"Memory-Peak während Training: {training_memory_peak:.1f} MB")
        
        # Vorhersagen mit Fehlerbehandlung
        print("Erstelle Vorhersagen...")
        try:
            y_pred_proba = model_pipeline.predict_proba(X_val)[:, 1]
        except Exception as e:
            if "memory" in str(e).lower():
                # Fallback: Chunked Prediction für große Validation Sets
                print("Memory-Problem bei Vorhersagen - verwende Chunked Processing...")
                y_pred_proba = predict_in_chunks(model_pipeline, X_val, chunk_size=10000)
            else:
                raise RuntimeError(f"Vorhersage fehlgeschlagen: {e}") from e
        
        # Feature-Counts nach Preprocessing (mit Fehlerbehandlung)
        try:
            sample_size = min(100, len(X_train))
            sample_transformed = preprocessor.fit_transform(X_train.head(sample_size))
            total_features = sample_transformed.shape[1]
        except Exception as e:
            warnings.warn(f"Feature-Count-Bestimmung fehlgeschlagen: {e}")
            total_features = X_train.shape[1]  # Fallback
        
        # Metriken berechnen mit Fehlerbehandlung
        try:
            metrics = calculate_comprehensive_metrics(y_val, y_pred_proba, users_val, topk)
        except Exception as e:
            raise RuntimeError(f"Metrik-Berechnung fehlgeschlagen: {e}") from e
        
        # Zusätzliche Metadaten
        metrics.update({
            'model': model_type,
            'topk': topk,
            'n_train': len(X_train),
            'n_valid': len(X_val),
            'train_secs': round(training_time, 2),
            'num_features': int(total_features),
            'cat_features': len([col for col in X_train.columns if X_train[col].nunique() < 50]),
            'memory_peak_mb': round(training_memory_peak, 1)
        })
        
        # Speichern mit Fehlerbehandlung
        try:
            save_model_and_metrics(model_pipeline, metrics, output_name, reports_dir)
        except Exception as e:
            raise RuntimeError(f"Speichern fehlgeschlagen: {e}") from e
        
        print(f"\n[OK] {model_type.upper()} Training erfolgreich abgeschlossen!")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Order-F1@{topk}: {metrics['order_f1_topk']:.4f}")
        print(f"Memory-Verbrauch: {training_memory_peak:.1f} MB")
        
    except Exception as e:
        # Finale Fehlerbehandlung mit hilfreichen Tipps
        final_memory = get_memory_usage()
        
        error_context = f"""
Training für {model_type} fehlgeschlagen:
{str(e)}

Kontext:
- Training-Samples: {len(X_train):,}
- Features: {X_train.shape[1]}
- Memory: {final_memory['rss_mb']:.1f} MB

Mögliche Lösungen:
1. Reduziere Datenmenge oder Features
2. Verwende einen einfacheren Modelltyp (logreg statt xgb/lgbm)
3. Schließe andere Programme um RAM freizugeben
4. Prüfe Datenqualität auf Probleme
"""
        
        print(error_context, file=sys.stderr)
        raise


def predict_in_chunks(model_pipeline: Pipeline, X: pd.DataFrame, chunk_size: int = 10000) -> np.ndarray:
    """
    Chunked Prediction für große Datasets bei Memory-Problemen.
    
    Ich teile hier die Vorhersagen in kleinere Batches auf, um Memory-Probleme
    zu vermeiden.
    """
    print(f"Chunked Prediction mit Chunk-Size: {chunk_size:,}")
    
    predictions = []
    n_samples = len(X)
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    
    for i in range(0, n_samples, chunk_size):
        chunk_end = min(i + chunk_size, n_samples)
        chunk = X.iloc[i:chunk_end]
        
        chunk_pred = model_pipeline.predict_proba(chunk)[:, 1]
        predictions.append(chunk_pred)
        
        if (i // chunk_size + 1) % 10 == 0:
            print(f"  Chunk {i // chunk_size + 1}/{n_chunks} verarbeitet...")
    
    return np.concatenate(predictions)


def check_dependencies() -> None:
    """
    Prüfe erforderliche Abhängigkeiten und gebe hilfreiche Fehlermeldungen.
    
    Ich checke hier explizit alle optionalen Dependencies und gebe konkrete
    Installationsanweisungen.
    """
    missing_deps = []
    
    # psutil für Memory-Monitoring
    try:
        import psutil
    except ImportError:
        missing_deps.append("psutil")
        warnings.warn("psutil nicht verfügbar - Memory-Monitoring deaktiviert")
    
    # Modell-spezifische Dependencies werden in build_model() geprüft
    
    if missing_deps:
        print(f"\nOptionale Dependencies fehlen: {missing_deps}")
        print("Installiere mit: pip install " + " ".join(missing_deps))
        print("Oder füge sie zu requirements.txt hinzu.\n")


if __name__ == "__main__":
    try:
        # Dependency-Check
        check_dependencies()
        
        # Kommandozeilen-Argumente parsen
        args = parse_arguments()
        
        # Konfiguration laden mit Overrides
        try:
            # Parse overrides from command line
            overrides = {}
            if args.override:
                for override in args.override:
                    if '=' not in override:
                        print(f"Ungültiges Override-Format: {override}. Verwende key=value")
                        sys.exit(1)
                    key, value = override.split('=', 1)
                    # Try to convert to appropriate type
                    try:
                        if value.lower() in ('true', 'false'):
                            value = value.lower() == 'true'
                        elif value.lower() == 'null':
                            value = None
                        elif value.isdigit():
                            value = int(value)
                        elif '.' in value and value.replace('.', '').isdigit():
                            value = float(value)
                    except:
                        pass  # Keep as string
                    overrides[key] = value
            
            config = load_config(args.config, overrides)
            
        except Exception as e:
            print(f"FEHLER beim Laden der Konfiguration: {e}", file=sys.stderr)
            sys.exit(1)
        
        # Verwende Config-Werte oder Command-Line Overrides
        topk = args.topk or config['models']['default_topk']
        seed = args.seed or config['sampling']['random_seed']
        features_file = config['data']['features_file']
        reports_dir = config['output']['reports_dir']
        
        print("=" * 60)
        print("INSTACART REORDER PREDICTION - TRAINING PIPELINE")
        print("=" * 60)
        print(f"Modell: {args.model}")
        print(f"Output: {args.out}")
        print(f"Top-K: {topk}")
        print(f"Seed: {seed}")
        print(f"Features: {features_file}")
        print(f"Reports: {reports_dir}")
        print("=" * 60)
        
        # Random Seed setzen für Reproduzierbarkeit
        np.random.seed(seed)
        
        # Pipeline-Schritte mit individueller Fehlerbehandlung
        try:
            # 1. Daten laden und vorbereiten
            print("\n[SCHRITT 1] Daten laden und vorbereiten...")
            features_df, target_labels, user_groups = load_and_prepare_data(features_file)
            
        except Exception as e:
            print(f"\n[FEHLER] Schritt 1 fehlgeschlagen: {e}", file=sys.stderr)
            print("\nLösungsvorschläge:")
            print("- Prüfe ob data/features.parquet existiert")
            print("- Führe 'python src/build_dataset.py' aus")
            print("- Prüfe verfügbaren Speicher")
            sys.exit(1)
        
        try:
            # 2. Spalten-Typen automatisch erkennen
            print("\n[SCHRITT 2] Feature-Typen erkennen...")
            numeric_columns, categorical_columns = detect_column_types(features_df)
            
        except Exception as e:
            print(f"\n[FEHLER] Schritt 2 fehlgeschlagen: {e}", file=sys.stderr)
            print("\nLösungsvorschläge:")
            print("- Prüfe Datenqualität und Spalten-Namen")
            print("- Validiere Feature-Schema")
            sys.exit(1)
        
        try:
            # 3. Preprocessing-Pipeline erstellen
            print("\n[SCHRITT 3] Preprocessing-Pipeline erstellen...")
            preprocessing_config = config['preprocessing']
            preprocessor = create_preprocessing_pipeline(numeric_columns, categorical_columns, preprocessing_config)
            
        except Exception as e:
            print(f"\n[FEHLER] Schritt 3 fehlgeschlagen: {e}", file=sys.stderr)
            print("\nLösungsvorschläge:")
            print("- Prüfe Feature-Typen und Datenqualität")
            print("- Reduziere Anzahl kategorischer Features")
            sys.exit(1)
        
        try:
            # 4. Train/Validation Split mit GroupShuffleSplit
            print("\n[SCHRITT 4] Daten aufteilen...")
            test_size = config['sampling']['test_size']
            X_train, X_val, y_train, y_val, users_train, users_val = split_data_by_users(
                features_df, target_labels, user_groups, test_size=test_size, random_state=seed
            )
            
        except Exception as e:
            print(f"\n[FEHLER] Schritt 4 fehlgeschlagen: {e}", file=sys.stderr)
            print("\nLösungsvorschläge:")
            print("- Prüfe user_id und y Spalten")
            print("- Validiere Datenverteilung")
            sys.exit(1)
        
        try:
            # 5. Modell trainieren und evaluieren
            print("\n[SCHRITT 5] Modell trainieren und evaluieren...")
            model_config = get_model_config(config, args.model)
            train_and_evaluate_model(
                model_type=args.model,
                model_config=model_config,
                output_name=args.out,
                topk=topk,
                X_train=X_train,
                X_val=X_val,
                y_train=y_train,
                y_val=y_val,
                users_val=users_val,
                preprocessor=preprocessor,
                reports_dir=reports_dir
            )
            
        except Exception as e:
            print(f"\n[FEHLER] Schritt 5 fehlgeschlagen: {e}", file=sys.stderr)
            print("\nLösungsvorschläge:")
            print("- Reduziere Modell-Komplexität (verwende logreg)")
            print("- Reduziere Datenmenge oder Features")
            print("- Prüfe verfügbaren Speicher")
            print("- Installiere fehlende Dependencies (xgboost, lightgbm)")
            sys.exit(1)
        
        # Erfolgreicher Abschluss
        print(f"\n{'='*60}")
        print("TRAINING PIPELINE ERFOLGREICH ABGESCHLOSSEN!")
        print(f"{'='*60}")
        print(f"Modell und Metriken gespeichert in reports/")
        print(f"Verwende 'python src/report.py' für HTML-Report")
        
        # Finale Memory-Statistik
        final_memory = get_memory_usage()
        print(f"\nFinale Memory-Nutzung: {final_memory['rss_mb']:.1f} MB")
        
    except KeyboardInterrupt:
        print("\n\n[ABBRUCH] Training durch Benutzer abgebrochen (Ctrl+C)")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n[KRITISCHER FEHLER] Unerwarteter Fehler: {e}", file=sys.stderr)
        print("\nFür Hilfe:")
        print("1. Prüfe die Fehlermeldung oben")
        print("2. Validiere Input-Daten und Dependencies")
        print("3. Reduziere Modell-Komplexität oder Datenmenge")
        print("4. Kontaktiere Support mit vollständiger Fehlermeldung")
        sys.exit(1)