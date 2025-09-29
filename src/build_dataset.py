#!/usr/bin/env python3
"""
INSTACART REORDER PREDICTION - DATASET BUILDER
===============================================

Als studentisches Lernprojekt bewusst ausführlich kommentiert.

Zweck dieses Scripts:
- SQL-basierte Feature Engineering mit DuckDB ausführen
- CSV-Dateien direkt verarbeiten ohne ETL-Pipeline
- Sauberes Parquet-Dataset für ML-Training erstellen
- Data Leakage durch strikte Prior/Train Trennung vermeiden

Warum DuckDB?
- Kann CSV-Dateien direkt lesen ohne Import
- Sehr schnell für analytische Queries
- Einfache Integration in Python
- Parquet Export out-of-the-box

Warum Parquet?
- Kompakte, spaltenorientierte Speicherung
- Schnelles Laden für ML-Training
- Behält Datentypen bei
- Standard für ML-Pipelines

Usage:
    python src/build_dataset.py [--output OUTPUT_PATH] [--sql SQL_PATH]

"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import duckdb
    import pandas as pd
    import pandera.pandas as pa
    from pandera.errors import SchemaError
except ImportError as e:
    print(f"ERROR: Required package not installed: {e}")
    print("Please install with: pip install -r requirements.txt")
    sys.exit(1)

# Import configuration utilities
from config_utils import load_config, get_model_config
from schemas.input_schemas import validate_dataframe, get_schema_info, SCHEMA_MAP
from data_quality import run_comprehensive_quality_checks


def setup_logging() -> None:
    """
    Logging Setup für bessere Nachvollziehbarkeit.
    
    Als Lernprojekt verwende ich hier verbose logging um jeden Schritt
    zu dokumentieren. In Production würde man das reduzieren.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def validate_input_files() -> None:
    """
    Validierung der Input CSV-Dateien.
    
    Ich prüfe hier explizit ob alle benötigten Dateien existieren,
    bevor ich mit der Verarbeitung beginne. Das verhindert kryptische
    Fehlermeldungen später im Prozess.
    """
    required_files = [
        'data/orders.csv',
        'data/order_products__prior.csv', 
        'data/order_products__train.csv',
        'data/products.csv',
        'data/aisles.csv',
        'data/departments.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logging.error("Fehlende CSV-Dateien gefunden:")
        for file_path in missing_files:
            logging.error(f"  - {file_path}")
        logging.error("Bitte stelle sicher, dass alle Instacart CSV-Dateien im data/ Verzeichnis liegen.")
        sys.exit(1)
    
    logging.info("[OK] Alle benötigten CSV-Dateien gefunden")


def validate_csv_schemas() -> None:
    """
    Validiert alle CSV-Dateien gegen ihre Pandera-Schemas.
    
    Diese Funktion lädt jede CSV-Datei und validiert sie gegen das entsprechende
    Schema um sicherzustellen, dass:
    - Alle erforderlichen Spalten vorhanden sind
    - Datentypen korrekt sind
    - Wertebereiche eingehalten werden
    - Null-Rate-Constraints erfüllt sind
    
    Bei Schema-Verletzungen wird eine detaillierte Fehlermeldung ausgegeben.
    """
    logging.info("Starte Schema-Validierung der CSV-Dateien...")
    
    # Mapping von Dateipfaden zu Schema-Namen
    file_schema_mapping = {
        'data/orders.csv': 'orders',
        'data/products.csv': 'products', 
        'data/aisles.csv': 'aisles',
        'data/departments.csv': 'departments',
        'data/order_products__prior.csv': 'order_products__prior',
        'data/order_products__train.csv': 'order_products__train'
    }
    
    validation_errors = []
    
    for file_path, schema_name in file_schema_mapping.items():
        try:
            logging.info(f"  Validiere {file_path} gegen {schema_name} Schema...")
            
            # CSV-Datei laden (nur erste 10000 Zeilen für Performance bei großen Dateien)
            try:
                df = pd.read_csv(file_path, nrows=10000)
                logging.info(f"    Geladen: {len(df)} Zeilen, {len(df.columns)} Spalten")
            except Exception as e:
                logging.error(f"    Fehler beim Laden der CSV-Datei: {e}")
                validation_errors.append(f"{file_path}: Kann nicht geladen werden - {e}")
                continue
            
            # Schema-Validierung durchführen
            try:
                validated_df = validate_dataframe(df, schema_name)
                logging.info(f"    [OK] Schema-Validierung erfolgreich")
                
                # Zusätzliche Statistiken für Debugging
                null_counts = df.isnull().sum()
                if null_counts.sum() > 0:
                    logging.info(f"    Null-Werte gefunden:")
                    for col, null_count in null_counts[null_counts > 0].items():
                        null_rate = (null_count / len(df)) * 100
                        logging.info(f"      {col}: {null_count} ({null_rate:.1f}%)")
                
            except SchemaError as e:
                logging.error(f"    [FEHLER] Schema-Validierung fehlgeschlagen:")
                
                # Detaillierte Fehleranalyse
                error_details = []
                
                # Parse Pandera error details
                if hasattr(e, 'failure_cases') and e.failure_cases is not None:
                    for _, failure in e.failure_cases.iterrows():
                        error_details.append(f"      Spalte '{failure.get('column', 'unknown')}': {failure.get('check', 'unknown check')} verletzt")
                
                if hasattr(e, 'schema_errors'):
                    for schema_error in e.schema_errors:
                        error_details.append(f"      {schema_error}")
                
                if not error_details:
                    error_details.append(f"      {str(e)}")
                
                for detail in error_details:
                    logging.error(detail)
                
                validation_errors.append(f"{file_path}: Schema-Validierung fehlgeschlagen")
                
        except Exception as e:
            logging.error(f"    [FEHLER] Unerwarteter Fehler bei {file_path}: {e}")
            validation_errors.append(f"{file_path}: Unerwarteter Fehler - {e}")
    
    # Zusammenfassung der Validierung
    if validation_errors:
        logging.error("=" * 60)
        logging.error("SCHEMA-VALIDIERUNG FEHLGESCHLAGEN!")
        logging.error("=" * 60)
        logging.error("Folgende Dateien haben Schema-Verletzungen:")
        for error in validation_errors:
            logging.error(f"  - {error}")
        logging.error("")
        logging.error("Mögliche Lösungen:")
        logging.error("  1. Prüfe ob die CSV-Dateien das erwartete Format haben")
        logging.error("  2. Überprüfe die Schema-Definitionen in src/schemas/input_schemas.py")
        logging.error("  3. Verwende --skip-validation um Schema-Prüfung zu überspringen (nicht empfohlen)")
        logging.error("")
        sys.exit(1)
    else:
        logging.info("[OK] Alle CSV-Dateien haben die Schema-Validierung bestanden")
        logging.info("")
        
        # Schema-Informationen für Debugging ausgeben
        logging.info("Schema-Übersicht:")
        for schema_name in set(file_schema_mapping.values()):
            try:
                schema_info = get_schema_info(schema_name)
                logging.info(f"  {schema_name}: {len(schema_info['columns'])} Spalten")
            except Exception:
                pass


def run_data_quality_checks() -> None:
    """
    Führt umfassende Datenqualitätsprüfungen durch.
    
    Diese Funktion führt verschiedene Datenqualitätschecks durch:
    - Zeilenzahl-Validierung
    - Duplikatserkennung  
    - Null-Rate-Monitoring
    - Geschäftsregel-Validierung
    - Generierung von Qualitätsberichten
    """
    logging.info("Starte umfassende Datenqualitätsprüfungen...")
    
    try:
        # Führe alle Qualitätschecks durch und generiere Bericht
        report_path = run_comprehensive_quality_checks(
            data_dir="data",
            output_dir="data/quality_reports"
        )
        
        logging.info(f"[OK] Datenqualitätsprüfungen abgeschlossen")
        logging.info(f"  Qualitätsbericht: {report_path}")
        
        # Lade Bericht-Summary für Logging
        try:
            import json
            with open(report_path, 'r') as f:
                report_data = json.load(f)
            
            summary = report_data.get("summary", {})
            total_violations = summary.get("total_violations", 0)
            high_severity = summary.get("high_severity_violations", 0)
            overall_status = summary.get("overall_status", "unknown")
            
            logging.info(f"  Qualitätsstatus: {overall_status.upper()}")
            logging.info(f"  Verstöße gesamt: {total_violations}")
            logging.info(f"  Hohe Schwere: {high_severity}")
            
            # Warnung bei kritischen Qualitätsproblemen
            if high_severity > 0:
                logging.warning("WARNUNG: Kritische Datenqualitätsprobleme gefunden!")
                logging.warning("Prüfe den Qualitätsbericht vor dem Fortfahren.")
                logging.warning("Verwende --skip-quality-checks um zu überspringen (nicht empfohlen)")
            
        except Exception as e:
            logging.warning(f"Kann Bericht-Summary nicht laden: {e}")
        
    except Exception as e:
        logging.error(f"Fehler bei Datenqualitätsprüfungen: {e}")
        logging.error("Mögliche Ursachen:")
        logging.error("  - CSV-Dateien können nicht gelesen werden")
        logging.error("  - Unzureichender Speicher für große Datasets")
        logging.error("  - Keine Schreibberechtigung für Qualitätsberichte")
        logging.error("Verwende --skip-quality-checks um zu überspringen")
        
        # Bei kritischen Fehlern stoppen, außer explizit übersprungen
        raise


def load_sql_query(sql_file_path: str) -> str:
    """
    SQL-Query aus Datei laden.
    
    Args:
        sql_file_path: Pfad zur SQL-Datei
        
    Returns:
        SQL-Query als String
        
    Raises:
        FileNotFoundError: Wenn SQL-Datei nicht existiert
        
    Warum separate SQL-Datei?
    - Bessere Trennung von SQL-Logik und Python-Code
    - SQL-Syntax-Highlighting in Editoren
    - Einfachere Wartung komplexer Queries
    """
    sql_path = Path(sql_file_path)
    
    if not sql_path.exists():
        logging.error(f"SQL-Datei nicht gefunden: {sql_file_path}")
        logging.error("Bitte stelle sicher, dass src/sql/01_build.sql existiert.")
        sys.exit(1)
    
    try:
        with open(sql_path, 'r', encoding='utf-8') as f:
            sql_query = f.read()
        
        logging.info(f"[OK] SQL-Query geladen aus {sql_file_path}")
        logging.info(f"  Query-Länge: {len(sql_query)} Zeichen")
        
        return sql_query
        
    except Exception as e:
        logging.error(f"Fehler beim Laden der SQL-Datei: {e}")
        sys.exit(1)


def execute_feature_engineering(sql_query: str, output_path: str) -> None:
    """
    Feature Engineering mit DuckDB ausführen und Ergebnis als Parquet speichern.
    
    Args:
        sql_query: SQL-Query für Feature Engineering
        output_path: Pfad für Output Parquet-Datei
        
    Hier passiert die eigentliche Magie:
    1. DuckDB Verbindung aufbauen
    2. SQL-Query ausführen (liest CSV, macht Feature Engineering)
    3. Ergebnis direkt als Parquet exportieren
    
    Warum COPY TO statt Python DataFrame?
    - Effizienter: Kein Umweg über Python Memory
    - DuckDB optimiert den Export automatisch
    - Weniger Code, weniger Fehlerquellen
    """
    logging.info("Starte Feature Engineering mit DuckDB...")
    
    # Schritt 1: DuckDB Connection aufbauen
    # Ich verwende hier eine In-Memory Database, da wir keine Persistierung brauchen
    try:
        conn = duckdb.connect(':memory:')
        logging.info("[OK] DuckDB Verbindung hergestellt")
    except Exception as e:
        logging.error(f"Fehler bei DuckDB Verbindung: {e}")
        sys.exit(1)
    
    # Schritt 2: SQL Query ausführen und Timing messen
    # Das ist der Hauptteil: Alle CSV-Dateien werden gelesen und Features berechnet
    start_time = time.time()
    
    try:
        logging.info("Führe SQL Feature Engineering aus...")
        logging.info("  - Lese CSV-Dateien aus data/ Verzeichnis")
        logging.info("  - Trenne Prior/Train Daten (Data Leakage Prevention)")
        logging.info("  - Berechne User-Product Interaction Features")
        logging.info("  - Berechne Recency Features")
        logging.info("  - Berechne Product Popularity Features")
        logging.info("  - Füge Categorical Features hinzu")
        logging.info("  - Generiere Labels aus Train Data")
        
        # SQL ausführen - das Ergebnis ist ein DuckDB Relation
        result = conn.execute(sql_query)
        
        execution_time = time.time() - start_time
        logging.info(f"[OK] SQL Feature Engineering abgeschlossen in {execution_time:.2f} Sekunden")
        
    except Exception as e:
        logging.error(f"Fehler bei SQL-Ausführung: {e}")
        logging.error("Mögliche Ursachen:")
        logging.error("  - CSV-Dateien haben unerwartetes Format")
        logging.error("  - Speicher nicht ausreichend für große Datasets")
        logging.error("  - SQL-Syntax-Fehler in 01_build.sql")
        sys.exit(1)
    
    # Schritt 3: Ergebnis als Parquet exportieren
    # Da die SQL-Datei CREATE VIEW Statements enthält, führe ich sie erst aus
    # und dann exportiere ich das finale SELECT-Result
    try:
        logging.info(f"Exportiere Ergebnis nach {output_path}...")
        
        # Erst alle Views erstellen durch Ausführung der kompletten SQL-Datei
        conn.execute(sql_query)
        
        # Dann das finale SELECT für den Export verwenden
        # Das finale SELECT ist das letzte Statement in der SQL-Datei
        final_select_query = """
        SELECT 
            -- Identifiers
            cup.user_id,
            cup.product_id,
            
            -- Label (0 wenn nicht in train_labels, 1 wenn drin)
            -- Das ist unser Zielvariable für die ML-Modelle
            COALESCE(tl.y, 0) as y,
            
            -- User-Product Interaction Features (aus Prior Data)
            upr.times_bought,
            upr.times_reordered, 
            upr.user_prod_reorder_rate,
            upr.last_prior_ordnum,
            upr.orders_since_last,
            upr.avg_add_to_cart_pos,
            upr.avg_days_since_prior,
            
            -- Product Popularity Features (aus Prior Data)
            pp.prod_cnt,
            pp.prod_users,
            pp.prod_avg_reorder_rate,
            
            -- Categorical Features (Aisle & Department)
            -- Diese werden später im Python Code mit OneHotEncoder verarbeitet
            pc.aisle_id,
            pc.department_id

        FROM candidate_user_products cup

        -- Join User-Product Features (mit Recency)
        LEFT JOIN user_product_recency upr 
            ON cup.user_id = upr.user_id AND cup.product_id = upr.product_id

        -- Join Product Popularity Features  
        LEFT JOIN product_popularity pp 
            ON cup.product_id = pp.product_id
            
        -- Join Product Categories
        LEFT JOIN product_categories pc 
            ON cup.product_id = pc.product_id

        -- Join Train Labels (nur für User-Products die tatsächlich reordered wurden)
        LEFT JOIN train_labels tl 
            ON cup.user_id = tl.user_id AND cup.product_id = tl.product_id

        -- Filter: Nur User die auch Train Orders haben
        -- Rationale: Wir können nur für User Predictions machen für die wir Labels haben
        WHERE cup.user_id IN (SELECT DISTINCT user_id FROM train_orders)

        -- Sortierung für bessere Lesbarkeit und Debugging
        ORDER BY cup.user_id, cup.product_id
        """
        
        # COPY TO ist DuckDBs nativer Export-Befehl
        # FORMAT PARQUET sorgt für optimale Kompression und Performance
        export_query = f"""
        COPY ({final_select_query}) 
        TO '{output_path}' 
        (FORMAT PARQUET)
        """
        
        conn.execute(export_query)
        
        logging.info("[OK] Parquet-Export abgeschlossen")
        
    except Exception as e:
        logging.error(f"Fehler beim Parquet-Export: {e}")
        logging.error("Mögliche Ursachen:")
        logging.error("  - Keine Schreibberechtigung für Output-Verzeichnis")
        logging.error("  - Nicht genügend Speicherplatz")
        logging.error("  - Output-Pfad ungültig")
        sys.exit(1)
    
    # Schritt 4: Validierung des Outputs
    try:
        # Kurze Validierung: Parquet-Datei laden und Grundstatistiken anzeigen
        validation_result = conn.execute(f"SELECT COUNT(*) as row_count FROM '{output_path}'").fetchone()
        row_count = validation_result[0]
        
        # Schema-Info für Debugging
        schema_info = conn.execute(f"DESCRIBE SELECT * FROM '{output_path}'").fetchall()
        
        logging.info("[OK] Output-Validierung:")
        logging.info(f"  - Anzahl Zeilen: {row_count:,}")
        logging.info(f"  - Anzahl Spalten: {len(schema_info)}")
        logging.info("  - Schema:")
        for col_name, col_type, null_allowed, key, default, extra in schema_info:
            logging.info(f"    {col_name}: {col_type}")
            
        if row_count == 0:
            logging.warning("WARNUNG: Dataset ist leer! Prüfe SQL-Query und Input-Daten.")
        
    except Exception as e:
        logging.warning(f"Validierung fehlgeschlagen (Dataset wurde trotzdem erstellt): {e}")
    
    finally:
        # Verbindung schließen
        conn.close()
        logging.info("[OK] DuckDB Verbindung geschlossen")


def parse_arguments() -> argparse.Namespace:
    """
    Command-Line Argumente parsen mit Config-Override Support.
    
    Als Lernprojekt verwende ich hier ausführliche Help-Texte und
    sinnvolle Defaults aus der Konfiguration.
    """
    parser = argparse.ArgumentParser(
        description='Instacart Reorder Prediction - Dataset Builder',
        epilog='''
Beispiel-Usage:
  python src/build_dataset.py
  python src/build_dataset.py --output data/my_features.parquet
  python src/build_dataset.py --config custom_config.yaml
  python src/build_dataset.py --override sampling.max_users=1000
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Pfad zur Config-Datei (default: sucht config.yaml)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output-Pfad für das Feature-Dataset (default: aus config.yaml)'
    )
    
    parser.add_argument(
        '--sql',
        type=str, 
        default=None,
        help='Pfad zur SQL-Datei mit Feature Engineering Logic (default: aus config.yaml)'
    )
    
    parser.add_argument(
        '--override',
        action='append',
        help='Config-Override im Format key=value (z.B. sampling.max_users=1000)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose Logging für Debugging'
    )
    
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Schema-Validierung überspringen (nicht empfohlen)'
    )
    
    parser.add_argument(
        '--skip-quality-checks',
        action='store_true',
        help='Datenqualitätsprüfungen überspringen (nicht empfohlen)'
    )
    
    return parser.parse_args()


def main() -> None:
    """
    Hauptfunktion: Orchestriert den gesamten Dataset-Building Prozess mit Config-Support.
    
    Workflow:
    1. Argumente parsen
    2. Konfiguration laden
    3. Logging Setup
    4. Input-Dateien validieren
    5. SQL-Query laden
    6. Feature Engineering ausführen
    7. Success-Message
    
    Als Lernprojekt strukturiere ich das bewusst in kleine, verständliche Schritte.
    """
    # Schritt 1: Command-Line Argumente parsen
    args = parse_arguments()
    
    # Schritt 2: Konfiguration laden mit Overrides
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
    
    # Schritt 3: Logging Setup mit Config
    logging_config = config['logging']
    logging.basicConfig(
        level=getattr(logging, logging_config['level']),
        format=logging_config['format'],
        datefmt=logging_config.get('date_format', '%Y-%m-%d %H:%M:%S')
    )
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Verbose Logging aktiviert")
    
    logging.info("=" * 60)
    logging.info("INSTACART REORDER PREDICTION - DATASET BUILDER")
    logging.info("=" * 60)
    logging.info("Als Lernprojekt bewusst ausführlich kommentiert")
    logging.info("")
    
    # Verwende Config-Werte oder Command-Line Overrides
    data_config = config['data']
    sql_file = args.sql or data_config['sql_file']
    output_file = args.output or data_config['features_file']
    
    logging.info(f"Konfiguration:")
    logging.info(f"  SQL-Datei: {sql_file}")
    logging.info(f"  Output-Datei: {output_file}")
    logging.info(f"  Raw Data Pfad: {data_config['raw_path']}")
    if config['sampling']['max_users']:
        logging.info(f"  Sampling: {config['sampling']['max_users']} Users")
    logging.info("")
    
    # Schritt 4: Input-Validierung
    logging.info("Schritt 1: Input-Validierung")
    validate_input_files()
    
    # Schritt 4.1: Schema-Validierung (optional)
    if not args.skip_validation:
        validate_csv_schemas()
    else:
        logging.warning("Schema-Validierung übersprungen (--skip-validation)")
        logging.info("")
    
    # Schritt 4.2: Datenqualitätsprüfungen (optional)
    if not args.skip_quality_checks:
        run_data_quality_checks()
        logging.info("")
    else:
        logging.warning("Datenqualitätsprüfungen übersprungen (--skip-quality-checks)")
        logging.info("")
    
    # Schritt 5: SQL-Query laden
    logging.info("Schritt 2: SQL-Query laden")
    sql_query = load_sql_query(sql_file)
    logging.info("")
    
    # Schritt 6: Feature Engineering ausführen
    logging.info("Schritt 3: Feature Engineering ausführen")
    
    # Output-Verzeichnis erstellen falls es nicht existiert
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    execute_feature_engineering(sql_query, output_file)
    logging.info("")
    
    # Schritt 7: Success-Message
    logging.info("=" * 60)
    logging.info("[SUCCESS] DATASET BUILDING ERFOLGREICH ABGESCHLOSSEN!")
    logging.info("=" * 60)
    logging.info(f"Feature-Dataset erstellt: {output_file}")
    logging.info("")
    logging.info("Nächste Schritte:")
    logging.info("  1. python src/train.py --model logreg")
    logging.info("  2. python src/train.py --model xgb") 
    logging.info("  3. python src/train.py --model lgbm")
    logging.info("  4. python src/report.py")
    logging.info("")
    logging.info("Viel Erfolg beim ML-Training! 🚀")


if __name__ == '__main__':
    main()