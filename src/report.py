#!/usr/bin/env python3
"""
Instacart Reorder Prediction - Automated Reporting System

Als Lernprojekt bewusst ausführlich kommentiert und mit klaren Variablennamen.
Dieses Skript sammelt alle Modell-Metriken und generiert einen HTML-Report.

Funktionalität:
1. Sammelt alle metrics_*.json Dateien aus reports/ Verzeichnis
2. Erstellt Leaderboard sortiert nach ROC-AUC
3. Generiert HTML-Report mit Modellvergleich und Projektbeschreibung

Verwendung:
    python src/report.py
"""

import json
import os
import glob
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import warnings
import argparse

# Import configuration utilities
from config_utils import load_config
from logging_utils import (
    configure_logging_from_config, log_execution_time, log_memory_usage,
    log_structured_metrics, log_file_operation
)

# Für bessere Lesbarkeit der Pandas-Ausgabe
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def discover_metrics_files(reports_dir: str = "reports", logger=None) -> List[str]:
    """
    Findet alle metrics_*.json Dateien im reports/ Verzeichnis.
    
    Als Lernprojekt: Ich verwende glob für Pattern-Matching, weil es
    robuster ist als manuelles Durchsuchen der Verzeichnisse.
    
    Args:
        reports_dir: Pfad zum reports Verzeichnis
        
    Returns:
        Liste der gefundenen Metrics-Dateien
    """
    # Pattern für alle metrics_*.json Dateien
    pattern = os.path.join(reports_dir, "metrics_*.json")
    
    # Glob findet alle Dateien die dem Pattern entsprechen
    metrics_files = glob.glob(pattern)
    
    if logger:
        logger.info(f"Gefundene Metrics-Dateien: {len(metrics_files)}")
        for file_path in metrics_files:
            logger.debug(f"  - {file_path}")
    else:
        print(f"Gefundene Metrics-Dateien: {len(metrics_files)}")
        for file_path in metrics_files:
            print(f"  - {file_path}")
    
    return metrics_files


def load_metrics_file(file_path: str) -> Dict[str, Any]:
    """
    Lädt eine einzelne Metrics-JSON-Datei mit Fehlerbehandlung.
    
    Als Lernprojekt: Ich implementiere explizite Fehlerbehandlung für
    häufige Probleme wie korrupte JSON-Dateien oder fehlende Schlüssel.
    
    Args:
        file_path: Pfad zur JSON-Datei
        
    Returns:
        Dictionary mit Metrics oder None bei Fehlern
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)
        
        # Validierung der erwarteten Schlüssel
        required_keys = ['model', 'roc_auc', 'pr_auc', 'f1_at_0.5', 'order_f1_topk']
        missing_keys = [key for key in required_keys if key not in metrics_data]
        
        if missing_keys:
            print(f"Warnung: {file_path} fehlen Schlüssel: {missing_keys}")
            return None
            
        return metrics_data
        
    except json.JSONDecodeError as e:
        print(f"Fehler beim Parsen von {file_path}: {e}")
        return None
    except FileNotFoundError:
        print(f"Datei nicht gefunden: {file_path}")
        return None
    except Exception as e:
        print(f"Unerwarteter Fehler bei {file_path}: {e}")
        return None


def collect_all_metrics(reports_dir: str = "reports", logger = None) -> List[Dict[str, Any]]:
    """
    Sammelt alle Metrics aus allen gefundenen JSON-Dateien.
    
    Als Lernprojekt: Ich trenne die Logik für Datei-Discovery und
    Daten-Loading für bessere Testbarkeit und Klarheit.
    
    Args:
        reports_dir: Pfad zum reports Verzeichnis
        
    Returns:
        Liste aller erfolgreich geladenen Metrics
    """
    metrics_files = discover_metrics_files(reports_dir, logger)
    
    if not metrics_files:
        message = f"Keine metrics_*.json Dateien in {reports_dir}/ gefunden!"
        instruction = "Führe zuerst 'python src/train.py --model [logreg|xgb|lgbm]' aus."
        if logger:
            logger.warning(message)
            logger.info(instruction)
        else:
            print(message)
            print(instruction)
        return []
    
    all_metrics = []
    
    for file_path in metrics_files:
        metrics_data = load_metrics_file(file_path)
        if metrics_data is not None:
            all_metrics.append(metrics_data)
    
    print(f"Erfolgreich geladen: {len(all_metrics)} von {len(metrics_files)} Dateien")
    return all_metrics


def create_leaderboard_dataframe(all_metrics: List[Dict[str, Any]], logger = None) -> pd.DataFrame:
    """
    Erstellt pandas DataFrame aus Metrics und sortiert nach ROC-AUC.
    
    Als Lernprojekt: Ich verwende pandas für strukturierte Datenverarbeitung,
    weil es die Sortierung und CSV-Export vereinfacht.
    
    Args:
        all_metrics: Liste aller Metrics-Dictionaries
        
    Returns:
        Sortiertes DataFrame mit allen Modell-Metriken
    """
    if not all_metrics:
        print("Keine gültigen Metrics gefunden - leeres DataFrame erstellt")
        return pd.DataFrame()
    
    # DataFrame aus Liste von Dictionaries erstellen
    leaderboard_df = pd.DataFrame(all_metrics)
    
    # Nach ROC-AUC absteigend sortieren (beste Modelle zuerst)
    leaderboard_df = leaderboard_df.sort_values('roc_auc', ascending=False)
    
    # Index zurücksetzen für saubere Nummerierung
    leaderboard_df = leaderboard_df.reset_index(drop=True)
    
    # Für bessere Lesbarkeit: Metriken auf 4 Dezimalstellen runden
    numeric_columns = ['roc_auc', 'pr_auc', 'f1_at_0.5', 'order_f1_topk', 'train_secs']
    for col in numeric_columns:
        if col in leaderboard_df.columns:
            leaderboard_df[col] = leaderboard_df[col].round(4)
    
    print(f"Leaderboard erstellt mit {len(leaderboard_df)} Modellen")
    print("\nTop 3 Modelle nach ROC-AUC:")
    if len(leaderboard_df) > 0:
        top_models = leaderboard_df[['model', 'roc_auc', 'pr_auc', 'order_f1_topk']].head(3)
        print(top_models.to_string(index=False))
    
    return leaderboard_df


def save_leaderboard_csv(leaderboard_df: pd.DataFrame, output_path: str = "reports/metrics_leaderboard.csv", logger = None):
    """
    Speichert Leaderboard als CSV-Datei.
    
    Als Lernprojekt: Ich verwende UTF-8 Encoding und explizite Pfad-Behandlung
    für Kompatibilität zwischen verschiedenen Betriebssystemen.
    
    Args:
        leaderboard_df: DataFrame mit Modell-Metriken
        output_path: Pfad für CSV-Output
    """
    try:
        # Verzeichnis erstellen falls es nicht existiert
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # CSV mit UTF-8 Encoding speichern
        leaderboard_df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"Leaderboard gespeichert: {output_path}")
        print(f"Anzahl Modelle: {len(leaderboard_df)}")
        
    except Exception as e:
        print(f"Fehler beim Speichern der CSV: {e}")


def generate_html_report(leaderboard_df: pd.DataFrame, output_path: str = "reports/report.html", logger = None):
    """
    Generiert HTML-Report mit Modellvergleich und Projektbeschreibung.
    
    Als Lernprojekt: Ich verwende ein einfaches HTML-Template mit embedded CSS
    für bessere Portabilität und Verständlichkeit.
    
    Args:
        leaderboard_df: DataFrame mit sortierten Modell-Metriken
        output_path: Pfad für HTML-Output
    """
    if leaderboard_df.empty:
        print("Leeres DataFrame - kann keinen HTML-Report generieren")
        return
    
    # HTML-Template mit embedded CSS für saubere Formatierung
    html_template = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Instacart Reorder Prediction - Performance Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }}
        .metrics-table th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        .metrics-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }}
        .metrics-table tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .metrics-table tr:hover {{
            background-color: #e8f4f8;
        }}
        .best-model {{
            background-color: #d4edda !important;
            font-weight: bold;
        }}
        .metric-highlight {{
            color: #27ae60;
            font-weight: bold;
        }}
        .section {{
            margin: 25px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .limitation {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
        }}
        .methodology {{
            background-color: #d1ecf1;
            border-left: 4px solid #17a2b8;
            padding: 15px;
            margin: 15px 0;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #6c757d;
            font-size: 12px;
        }}
        .lineage-diagram {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }}
        .flow-stage {{
            margin: 15px 0;
            text-align: center;
        }}
        .flow-stage h4 {{
            margin: 10px 0;
            color: #495057;
            font-size: 16px;
        }}
        .flow-arrow {{
            text-align: center;
            font-size: 24px;
            color: #6c757d;
            margin: 10px 0;
        }}
        .data-sources, .output-files {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 8px;
            margin: 10px 0;
        }}
        .data-file, .output-file {{
            background-color: #e3f2fd;
            color: #1565c0;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-family: monospace;
            border: 1px solid #bbdefb;
        }}
        .output-file {{
            background-color: #e8f5e8;
            color: #2e7d32;
            border: 1px solid #c8e6c9;
        }}
        .process-box, .dataset-box, .training-box {{
            background-color: white;
            border: 2px solid #dee2e6;
            border-radius: 6px;
            padding: 12px;
            margin: 10px auto;
            max-width: 400px;
            font-size: 13px;
            line-height: 1.4;
        }}
        .process-box {{
            border-color: #ffc107;
            background-color: #fff8e1;
        }}
        .dataset-box {{
            border-color: #17a2b8;
            background-color: #e0f7fa;
        }}
        .training-box {{
            border-color: #28a745;
            background-color: #e8f5e8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🛒 Instacart Reorder Prediction - Performance Report</h1>
        
        <div class="section">
            <h2>📊 Projekt Übersicht</h2>
            <p><strong>Ziel:</strong> Vorhersage welche Produkte ein Kunde in seinem nächsten Einkaufskorb wiederholt bestellen wird.</p>
            <p><strong>Ansatz:</strong> End-to-End ML Pipeline mit SQL-basierter Feature Engineering (DuckDB), Multi-Model Training (LogReg/XGBoost/LightGBM) und automatisierter Berichtserstellung.</p>
            <p><strong>Evaluation:</strong> Fokus auf Set-basierte Metriken (Order-F1) die das Business-Ziel widerspiegeln: die richtigen Produkte pro Nutzer vorherzusagen.</p>
        </div>

        <h2>🏆 Modell Leaderboard</h2>
        <p>Sortiert nach ROC-AUC (höher = besser). Ziel: ROC-AUC ≥ 0.83</p>
        
        {model_table}
        
        <div class="section">
            <h2>🔄 Data Lineage & Pipeline Flow</h2>
            <div class="lineage-diagram">
                <div class="flow-stage">
                    <h4>📁 Raw Data Sources</h4>
                    <div class="data-sources">
                        <span class="data-file">orders.csv</span>
                        <span class="data-file">order_products__prior.csv</span>
                        <span class="data-file">order_products__train.csv</span>
                        <span class="data-file">products.csv</span>
                        <span class="data-file">aisles.csv</span>
                        <span class="data-file">departments.csv</span>
                    </div>
                </div>
                
                <div class="flow-arrow">⬇️</div>
                
                <div class="flow-stage">
                    <h4>🔧 Feature Engineering</h4>
                    <div class="process-box">
                        <strong>src/sql/01_build.sql</strong><br>
                        • User-product aggregations<br>
                        • Temporal recency features<br>
                        • Product popularity metrics<br>
                        • Categorical lookups<br>
                        • Label generation (train only)
                    </div>
                </div>
                
                <div class="flow-arrow">⬇️</div>
                
                <div class="flow-stage">
                    <h4>📊 ML-Ready Dataset</h4>
                    <div class="dataset-box">
                        <strong>data/features/features.parquet</strong><br>
                        Schema: user_id, product_id, y, [55+ features]<br>
                        Scale: 6.7M+ training samples
                    </div>
                </div>
                
                <div class="flow-arrow">⬇️</div>
                
                <div class="flow-stage">
                    <h4>🤖 Model Training</h4>
                    <div class="training-box">
                        <strong>src/train.py</strong><br>
                        • GroupShuffleSplit (user-based)<br>
                        • StandardScaler + OneHotEncoder<br>
                        • LogReg / XGBoost / LightGBM<br>
                        • Multi-metric evaluation
                    </div>
                </div>
                
                <div class="flow-arrow">⬇️</div>
                
                <div class="flow-stage">
                    <h4>📈 Performance Reports</h4>
                    <div class="output-files">
                        <span class="output-file">model_*.joblib</span>
                        <span class="output-file">metrics_*.json</span>
                        <span class="output-file">metrics_leaderboard.csv</span>
                        <span class="output-file">report.html</span>
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 15px; text-align: center;">
                <p><em>📋 Detailed data lineage documentation available in <code>docs/data_lineage.md</code></em></p>
                <p><strong>Key Transformation Principles:</strong></p>
                <ul style="text-align: left; max-width: 600px; margin: 0 auto;">
                    <li><strong>Temporal Separation:</strong> 'prior' orders → features, 'train' orders → labels only</li>
                    <li><strong>User-Based Splitting:</strong> GroupShuffleSplit prevents user leakage</li>
                    <li><strong>SQL-First Approach:</strong> DuckDB for efficient CSV processing</li>
                    <li><strong>Comprehensive Features:</strong> 55+ engineered features from 6 raw data sources</li>
                    <li><strong>Quality Assurance:</strong> Schema validation and data quality monitoring</li>
                </ul>
            </div>
        </div>

        <div class="methodology">
            <h2>🔬 Methodologie</h2>
            <ul>
                <li><strong>Data Leakage Prevention:</strong> Strikte Trennung - 'prior' Orders für Features, 'train' Orders nur für Labels</li>
                <li><strong>User-based Splitting:</strong> GroupShuffleSplit verhindert User-Leakage zwischen Train/Validation</li>
                <li><strong>Feature Engineering:</strong> SQL-basiert mit DuckDB für Effizienz und Transparenz</li>
                <li><strong>Order-F1 Metric:</strong> Set-basierte Evaluation - F1 zwischen vorhergesagten und tatsächlichen Produktsets pro User</li>
                <li><strong>Preprocessing:</strong> StandardScaler für numerische Features, OneHotEncoder für kategorische</li>
            </ul>
        </div>
        
        <div class="limitation">
            <h2>⚠️ Bekannte Limitationen</h2>
            <ul>
                <li><strong>Cold Start Problem:</strong> Keine Behandlung neuer User oder Produkte ohne Historie</li>
                <li><strong>Simple Thresholding:</strong> Für erste Version bewusst einfache 0.5 Schwelle statt optimierter Thresholds</li>
                <li><strong>Minimal Tuning:</strong> Fokus auf Pipeline-Aufbau, nicht auf Hyperparameter-Optimierung</li>
                <li><strong>Static Features:</strong> Keine zeitabhängigen Features wie Saisonalität oder Trends</li>
                <li><strong>Binary Classification:</strong> Vereinfachung auf Ja/Nein Reorder, keine Mengen-Vorhersage</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>📈 Key Insights</h2>
            <ul>
                <li><strong>Beste Performance:</strong> {best_model} mit ROC-AUC {best_roc_auc}</li>
                <li><strong>Training Effizienz:</strong> LogReg am schnellsten ({logreg_time}s), LightGBM guter Kompromiss</li>
                <li><strong>Set-based Evaluation:</strong> Order-F1 zeigt praktische Relevanz für Business-Anwendung</li>
                <li><strong>Feature Importance:</strong> {num_features} numerische + {cat_features} kategorische Features</li>
            </ul>
        </div>
        
        <div class="footer">
            <p>Generiert am {timestamp} | Studentisches Lernprojekt mit Fokus auf End-to-End ML Pipeline</p>
            <p>Reproduktion: python src/build_dataset.py → python src/train.py --model [logreg|xgb|lgbm] → python src/report.py</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Modell-Tabelle HTML generieren
    model_table_html = generate_model_table_html(leaderboard_df)
    
    # Best model Informationen für Insights
    best_model_row = leaderboard_df.iloc[0]
    best_model = best_model_row['model']
    best_roc_auc = best_model_row['roc_auc']
    
    # LogReg Zeit finden (falls vorhanden)
    logreg_rows = leaderboard_df[leaderboard_df['model'] == 'logreg']
    logreg_time = logreg_rows.iloc[0]['train_secs'] if len(logreg_rows) > 0 else "N/A"
    
    # Feature counts
    num_features = best_model_row['num_features']
    cat_features = best_model_row['cat_features']
    
    # Timestamp für Footer
    from datetime import datetime
    timestamp = datetime.now().strftime("%d.%m.%Y %H:%M")
    
    # Template mit Daten füllen
    html_content = html_template.format(
        model_table=model_table_html,
        best_model=best_model.upper(),
        best_roc_auc=best_roc_auc,
        logreg_time=logreg_time,
        num_features=num_features,
        cat_features=cat_features,
        timestamp=timestamp
    )
    
    # HTML-Datei schreiben
    try:
        # Verzeichnis erstellen falls es nicht existiert
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML-Report generiert: {output_path}")
        print(f"Öffne die Datei im Browser für vollständigen Report")
        
    except Exception as e:
        print(f"Fehler beim Generieren des HTML-Reports: {e}")


def generate_model_table_html(leaderboard_df: pd.DataFrame) -> str:
    """
    Generiert HTML-Tabelle für Modellvergleich.
    
    Als Lernprojekt: Ich erstelle die Tabelle manuell für bessere Kontrolle
    über Formatierung und Highlighting der besten Modelle.
    
    Args:
        leaderboard_df: DataFrame mit Modell-Metriken
        
    Returns:
        HTML-String für Modell-Tabelle
    """
    if leaderboard_df.empty:
        return "<p>Keine Modell-Daten verfügbar</p>"
    
    # Tabellen-Header
    table_html = """
    <table class="metrics-table">
        <thead>
            <tr>
                <th>Rang</th>
                <th>Modell</th>
                <th>ROC-AUC</th>
                <th>PR-AUC</th>
                <th>F1@0.5</th>
                <th>Order-F1@top-k</th>
                <th>Top-K</th>
                <th>Training Zeit (s)</th>
                <th>Features</th>
            </tr>
        </thead>
        <tbody>
    """
    
    # Zeilen für jedes Modell
    for idx, row in leaderboard_df.iterrows():
        # Beste Modell hervorheben
        row_class = "best-model" if idx == 0 else ""
        
        # Metriken formatieren
        roc_auc = f"{row['roc_auc']:.4f}"
        pr_auc = f"{row['pr_auc']:.4f}"
        f1_05 = f"{row['f1_at_0.5']:.4f}"
        order_f1 = f"{row['order_f1_topk']:.4f}"
        train_time = f"{row['train_secs']:.1f}"
        features = f"{row['num_features']}+{row['cat_features']}"
        
        # ROC-AUC highlighting wenn >= 0.83 (Ziel erreicht)
        roc_class = "metric-highlight" if row['roc_auc'] >= 0.83 else ""
        
        table_html += f"""
            <tr class="{row_class}">
                <td>{idx + 1}</td>
                <td><strong>{row['model'].upper()}</strong></td>
                <td class="{roc_class}">{roc_auc}</td>
                <td>{pr_auc}</td>
                <td>{f1_05}</td>
                <td>{order_f1}</td>
                <td>{row['topk']}</td>
                <td>{train_time}</td>
                <td>{features}</td>
            </tr>
        """
    
    table_html += """
        </tbody>
    </table>
    """
    
    return table_html


def parse_arguments() -> argparse.Namespace:
    """
    Command-Line Argumente parsen mit Config-Support.
    """
    parser = argparse.ArgumentParser(
        description='Instacart Reorder Prediction - Automated Reporting',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='''
Beispiele:
  python src/report.py
  python src/report.py --config custom_config.yaml
  python src/report.py --override output.reports_dir=custom_reports
        '''
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Pfad zur Config-Datei (default: sucht config.yaml)'
    )
    
    parser.add_argument(
        '--override',
        action='append',
        help='Config-Override im Format key=value (z.B. output.reports_dir=custom_reports)'
    )
    
    return parser.parse_args()


def main():
    """
    Hauptfunktion für Metrics-Sammlung und Report-Generierung mit Config-Support.
    
    Als Lernprojekt: Ich strukturiere die main() Funktion klar in
    logische Schritte für bessere Nachvollziehbarkeit.
    """
    # Schritt 0: Argumente parsen und Konfiguration laden
    args = parse_arguments()
    
    try:
        # Parse overrides from command line
        overrides = {}
        if args.override:
            for override in args.override:
                if '=' not in override:
                    print(f"Ungültiges Override-Format: {override}. Verwende key=value")
                    return
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
        print(f"FEHLER beim Laden der Konfiguration: {e}")
        return
    
    # Setup structured logging
    logger = configure_logging_from_config(config, __name__, "report")
    
    # Log initial memory usage
    log_memory_usage("startup", logger)
    
    # Verwende Config-Werte
    output_config = config['output']
    reports_dir = output_config['reports_dir']
    leaderboard_file = os.path.join(reports_dir, output_config['leaderboard_file'])
    html_report_file = os.path.join(reports_dir, output_config['html_report_file'])
    
    logger.info("=== Instacart Reorder Prediction - Automated Reporting ===")
    logger.info(f"Reports Verzeichnis: {reports_dir}")
    
    # Log configuration
    config_metrics = {
        "reports_dir": reports_dir,
        "leaderboard_file": leaderboard_file,
        "html_report_file": html_report_file
    }
    log_structured_metrics(config_metrics, "report_config", logger)
    
    try:
        # Schritt 1: Alle Metrics sammeln
        with log_execution_time("collect_metrics", logger):
            all_metrics = collect_all_metrics(reports_dir, logger)
        
        if not all_metrics:
            logger.warning("Keine Metrics gefunden. Beende Programm.")
            return
        
        # Schritt 2: Leaderboard DataFrame erstellen
        with log_execution_time("create_leaderboard", logger):
            leaderboard_df = create_leaderboard_dataframe(all_metrics, logger)
        
        # Schritt 3: CSV speichern
        with log_execution_time("save_csv", logger):
            log_file_operation("create", leaderboard_file, logger, 
                             overwrite=Path(leaderboard_file).exists(), config=config)
            save_leaderboard_csv(leaderboard_df, leaderboard_file, logger)
        
        # Schritt 4: HTML-Report generieren
        with log_execution_time("generate_html", logger):
            log_file_operation("create", html_report_file, logger, 
                             overwrite=Path(html_report_file).exists(), config=config)
            generate_html_report(leaderboard_df, html_report_file, logger)
        
        # Log final memory usage
        log_memory_usage("completion", logger)
        
        logger.info("=== Automated Reporting abgeschlossen ===")
        logger.info("Outputs:")
        logger.info(f"  - {leaderboard_file}")
        logger.info(f"  - {html_report_file}")
        logger.info(f"Öffne {html_report_file} im Browser für den vollständigen Report!")
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        log_memory_usage("error", logger)
        raise


if __name__ == "__main__":
    main()