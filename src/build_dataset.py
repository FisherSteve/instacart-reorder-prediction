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
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

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
from logging_utils import (
    configure_logging_from_config, log_execution_time, log_memory_usage,
    log_structured_metrics, log_file_operation, get_pipeline_state_tracker,
    check_operation_idempotency, backup_existing_file
)





def validate_input_files(logger) -> None:
    """
    Validierung der Input CSV-Dateien.
    
    Ich prüfe hier explizit ob alle benötigten Dateien existieren,
    bevor ich mit der Verarbeitung beginne. Das verhindert kryptische
    Fehlermeldungen später im Prozess.
    """
    required_files = [
        'data/raw/orders.csv',
        'data/raw/order_products__prior.csv', 
        'data/raw/order_products__train.csv',
        'data/raw/products.csv',
        'data/raw/aisles.csv',
        'data/raw/departments.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error("Fehlende CSV-Dateien gefunden:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        logger.error("Bitte stelle sicher, dass alle Instacart CSV-Dateien im data/ Verzeichnis liegen.")
        sys.exit(1)
    
    logger.info("Alle benötigten CSV-Dateien gefunden")
    
    # Log file sizes for monitoring
    file_sizes = {}
    for file_path in required_files:
        try:
            size_mb = Path(file_path).stat().st_size / 1024 / 1024
            file_sizes[Path(file_path).name] = size_mb
            logger.debug(f"  {file_path}: {size_mb:.1f} MB")
        except Exception as e:
            logger.warning(f"Could not get size for {file_path}: {e}")
    
    log_structured_metrics(file_sizes, "input_file_sizes", logger)


def validate_csv_schemas(logger) -> None:
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
    logger.info("Starte Schema-Validierung der CSV-Dateien...")
    
    # Mapping von Dateipfaden zu Schema-Namen
    file_schema_mapping = {
        'data/raw/orders.csv': 'orders',
        'data/raw/products.csv': 'products', 
        'data/raw/aisles.csv': 'aisles',
        'data/raw/departments.csv': 'departments',
        'data/raw/order_products__prior.csv': 'order_products__prior',
        'data/raw/order_products__train.csv': 'order_products__train'
    }
    
    validation_errors = []
    
    for file_path, schema_name in file_schema_mapping.items():
        try:
            logger.info(f"  Validiere {file_path} gegen {schema_name} Schema...")
            
            # CSV-Datei laden (nur erste 10000 Zeilen für Performance bei großen Dateien)
            try:
                df = pd.read_csv(file_path, nrows=10000)
                logger.info(f"    Geladen: {len(df)} Zeilen, {len(df.columns)} Spalten")
            except Exception as e:
                logger.error(f"    Fehler beim Laden der CSV-Datei: {e}")
                validation_errors.append(f"{file_path}: Kann nicht geladen werden - {e}")
                continue
            
            # Schema-Validierung durchführen
            try:
                validated_df = validate_dataframe(df, schema_name)
                logger.info(f"    [OK] Schema-Validierung erfolgreich")
                
                # Zusätzliche Statistiken für Debugging
                null_counts = df.isnull().sum()
                if null_counts.sum() > 0:
                    logger.info(f"    Null-Werte gefunden:")
                    for col, null_count in null_counts[null_counts > 0].items():
                        null_rate = (null_count / len(df)) * 100
                        logger.info(f"      {col}: {null_count} ({null_rate:.1f}%)")
                
            except SchemaError as e:
                logger.error(f"    [FEHLER] Schema-Validierung fehlgeschlagen:")
                
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
                    logger.error(detail)
                
                validation_errors.append(f"{file_path}: Schema-Validierung fehlgeschlagen")
                
        except Exception as e:
            logger.error(f"    [FEHLER] Unerwarteter Fehler bei {file_path}: {e}")
            validation_errors.append(f"{file_path}: Unerwarteter Fehler - {e}")
    
    # Zusammenfassung der Validierung
    if validation_errors:
        logger.error("=" * 60)
        logger.error("SCHEMA-VALIDIERUNG FEHLGESCHLAGEN!")
        logger.error("=" * 60)
        logger.error("Folgende Dateien haben Schema-Verletzungen:")
        for error in validation_errors:
            logger.error(f"  - {error}")
        logger.error("")
        logger.error("Mögliche Lösungen:")
        logger.error("  1. Prüfe ob die CSV-Dateien das erwartete Format haben")
        logger.error("  2. Überprüfe die Schema-Definitionen in src/schemas/input_schemas.py")
        logger.error("  3. Verwende --skip-validation um Schema-Prüfung zu überspringen (nicht empfohlen)")
        logger.error("")
        sys.exit(1)
    else:
        logger.info("[OK] Alle CSV-Dateien haben die Schema-Validierung bestanden")
        logger.info("")
        
        # Schema-Informationen für Debugging ausgeben
        logger.info("Schema-Übersicht:")
        for schema_name in set(file_schema_mapping.values()):
            try:
                schema_info = get_schema_info(schema_name)
                logger.info(f"  {schema_name}: {len(schema_info['columns'])} Spalten")
            except Exception:
                pass


def run_data_quality_checks(logger) -> None:
    """
    Führt umfassende Datenqualitätsprüfungen durch.
    
    Diese Funktion führt verschiedene Datenqualitätschecks durch:
    - Zeilenzahl-Validierung
    - Duplikatserkennung  
    - Null-Rate-Monitoring
    - Geschäftsregel-Validierung
    - Generierung von Qualitätsberichten
    """
    logger.info("Starte umfassende Datenqualitätsprüfungen...")
    
    try:
        # Führe alle Qualitätschecks durch und generiere Bericht
        report_path = run_comprehensive_quality_checks(
            data_dir="data/raw",
            output_dir="data/quality_reports"
        )
        
        logger.info(f"Datenqualitätsprüfungen abgeschlossen")
        logger.info(f"  Qualitätsbericht: {report_path}")
        
        # Lade Bericht-Summary für Logging
        try:
            import json
            with open(report_path, 'r') as f:
                report_data = json.load(f)
            
            summary = report_data.get("summary", {})
            total_violations = summary.get("total_violations", 0)
            high_severity = summary.get("high_severity_violations", 0)
            overall_status = summary.get("overall_status", "unknown")
            
            logger.info(f"  Qualitätsstatus: {overall_status.upper()}")
            logger.info(f"  Verstöße gesamt: {total_violations}")
            logger.info(f"  Hohe Schwere: {high_severity}")
            
            # Warnung bei kritischen Qualitätsproblemen
            if high_severity > 0:
                logger.warning("WARNUNG: Kritische Datenqualitätsprobleme gefunden!")
                logger.warning("Prüfe den Qualitätsbericht vor dem Fortfahren.")
                logger.warning("Verwende --skip-quality-checks um zu überspringen (nicht empfohlen)")
            
        except Exception as e:
            logger.warning(f"Kann Bericht-Summary nicht laden: {e}")
        
    except Exception as e:
        logger.error(f"Fehler bei Datenqualitätsprüfungen: {e}")
        logger.error("Mögliche Ursachen:")
        logger.error("  - CSV-Dateien können nicht gelesen werden")
        logger.error("  - Unzureichender Speicher für große Datasets")
        logger.error("  - Keine Schreibberechtigung für Qualitätsberichte")
        logger.error("Verwende --skip-quality-checks um zu überspringen")
        
        # Bei kritischen Fehlern stoppen, außer explizit übersprungen
        raise


def load_sql_query(sql_file_path: str, logger) -> str:
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
        logger.error(f"SQL-Datei nicht gefunden: {sql_file_path}")
        logger.error("Bitte stelle sicher, dass src/sql/01_build.sql existiert.")
        sys.exit(1)
    
    try:
        with open(sql_path, 'r', encoding='utf-8') as f:
            sql_query = f.read()
        
        logger.info(f"SQL-Query geladen aus {sql_file_path}")
        logger.debug(f"  Query-Länge: {len(sql_query)} Zeichen")
        
        # Log structured metrics about SQL query
        sql_metrics = {
            "file_path": sql_file_path,
            "query_length_chars": len(sql_query),
            "query_lines": len(sql_query.splitlines())
        }
        log_structured_metrics(sql_metrics, "sql_query_loaded", logger)
        
        return sql_query
        
    except Exception as e:
        logger.error(f"Fehler beim Laden der SQL-Datei: {e}")
        sys.exit(1)


def execute_feature_engineering(sql_query: str, output_path: str, logger, config: Dict[str, Any] = None) -> None:
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
    logger.info("Starte Feature Engineering mit DuckDB...")
    
    # Log file operation for output with configuration
    file_exists = Path(output_path).exists()
    if not log_file_operation("create", output_path, logger, overwrite=file_exists, config=config):
        logger.error("File operation blocked by configuration")
        return
    
    # Schritt 1: DuckDB Connection aufbauen
    # Ich verwende hier eine In-Memory Database, da wir keine Persistierung brauchen
    try:
        conn = duckdb.connect(':memory:')
        logger.info("DuckDB Verbindung hergestellt")
    except Exception as e:
        logger.error(f"Fehler bei DuckDB Verbindung: {e}")
        sys.exit(1)
    
    # Schritt 2: SQL Query ausführen und Timing messen
    # Das ist der Hauptteil: Alle CSV-Dateien werden gelesen und Features berechnet
    start_time = time.time()
    
    try:
        logger.info("Führe SQL Feature Engineering aus...")
        logger.info("  - Lese CSV-Dateien aus data/ Verzeichnis")
        logger.info("  - Trenne Prior/Train Daten (Data Leakage Prevention)")
        logger.info("  - Berechne User-Product Interaction Features")
        logger.info("  - Berechne Recency Features")
        logger.info("  - Berechne Product Popularity Features")
        logger.info("  - Füge Categorical Features hinzu")
        logger.info("  - Generiere Labels aus Train Data")
        
        # SQL ausführen - das Ergebnis ist ein DuckDB Relation
        result = conn.execute(sql_query)
        
        execution_time = time.time() - start_time
        logger.info(f"[OK] SQL Feature Engineering abgeschlossen in {execution_time:.2f} Sekunden")
        
    except Exception as e:
        logger.error(f"Fehler bei SQL-Ausführung: {e}")
        logger.error("Mögliche Ursachen:")
        logger.error("  - CSV-Dateien haben unerwartetes Format")
        logger.error("  - Speicher nicht ausreichend für große Datasets")
        logger.error("  - SQL-Syntax-Fehler in 01_build.sql")
        sys.exit(1)
    
    # Schritt 3: Ergebnis als Parquet exportieren
    # Da die SQL-Datei CREATE VIEW Statements enthält, führe ich sie erst aus
    # und dann exportiere ich das finale SELECT-Result
    try:
        logger.info(f"Exportiere Ergebnis nach {output_path}...")
        
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
        
        logger.info("[OK] Parquet-Export abgeschlossen")
        
    except Exception as e:
        logger.error(f"Fehler beim Parquet-Export: {e}")
        logger.error("Mögliche Ursachen:")
        logger.error("  - Keine Schreibberechtigung für Output-Verzeichnis")
        logger.error("  - Nicht genügend Speicherplatz")
        logger.error("  - Output-Pfad ungültig")
        sys.exit(1)
    
    # Schritt 4: Validierung des Outputs
    try:
        # Kurze Validierung: Parquet-Datei laden und Grundstatistiken anzeigen
        validation_result = conn.execute(f"SELECT COUNT(*) as row_count FROM '{output_path}'").fetchone()
        row_count = validation_result[0]
        
        # Schema-Info für Debugging
        schema_info = conn.execute(f"DESCRIBE SELECT * FROM '{output_path}'").fetchall()
        
        logger.info("[OK] Output-Validierung:")
        logger.info(f"  - Anzahl Zeilen: {row_count:,}")
        logger.info(f"  - Anzahl Spalten: {len(schema_info)}")
        logger.info("  - Schema:")
        for col_name, col_type, null_allowed, key, default, extra in schema_info:
            logger.info(f"    {col_name}: {col_type}")
            
        if row_count == 0:
            logger.warning("WARNUNG: Dataset ist leer! Prüfe SQL-Query und Input-Daten.")
        
    except Exception as e:
        logger.warning(f"Validierung fehlgeschlagen (Dataset wurde trotzdem erstellt): {e}")
    
    finally:
        # Verbindung schließen
        conn.close()
        logger.info("[OK] DuckDB Verbindung geschlossen")


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
    
    # Schritt 3: Logging Setup mit neuer Infrastruktur
    logger = configure_logging_from_config(config, __name__, "build_dataset")
    
    if args.verbose:
        logger.setLevel("DEBUG")
        logger.debug("Verbose Logging aktiviert")
    
    # Log initial memory usage
    log_memory_usage("startup", logger)
    
    logger.info("=" * 60)
    logger.info("INSTACART REORDER PREDICTION - DATASET BUILDER")
    logger.info("=" * 60)
    logger.info("Als Lernprojekt bewusst ausführlich kommentiert")
    
    # Verwende Config-Werte oder Command-Line Overrides
    data_config = config['data']
    sql_file = args.sql or data_config['sql_file']
    output_file = args.output or data_config['features_file']
    
    # Log configuration
    config_metrics = {
        "sql_file": sql_file,
        "output_file": output_file,
        "raw_data_path": data_config['raw_path'],
        "max_users_sampling": config['sampling']['max_users'],
        "skip_validation": args.skip_validation,
        "skip_quality_checks": args.skip_quality_checks
    }
    log_structured_metrics(config_metrics, "build_dataset_config", logger)
    
    try:
        # Schritt 4: Input-Validierung mit Timing
        with log_execution_time("input_validation", logger):
            validate_input_files(logger)
        
        # Schritt 4.1: Schema-Validierung (optional)
        if not args.skip_validation:
            with log_execution_time("schema_validation", logger):
                validate_csv_schemas(logger)
        else:
            logger.warning("Schema-Validierung übersprungen (--skip-validation)")
        
        # Schritt 4.2: Datenqualitätsprüfungen (optional)
        if not args.skip_quality_checks:
            with log_execution_time("data_quality_checks", logger):
                run_data_quality_checks(logger)
        else:
            logger.warning("Datenqualitätsprüfungen übersprungen (--skip-quality-checks)")
        
        # Schritt 5: SQL-Query laden
        with log_execution_time("sql_query_loading", logger):
            sql_query = load_sql_query(sql_file, logger)
        
        # Schritt 6: Feature Engineering ausführen
        # Output-Verzeichnis erstellen falls es nicht existiert
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check idempotency before executing feature engineering
        input_files = [
            'data/raw/orders.csv',
            'data/raw/order_products__prior.csv', 
            'data/raw/order_products__train.csv',
            'data/raw/products.csv',
            'data/raw/aisles.csv',
            'data/raw/departments.csv',
            sql_file
        ]
        output_files = [output_file]
        
        idempotency_result = check_operation_idempotency(
            "feature_engineering", input_files, output_files, logger, config
        )
        
        if not idempotency_result["needs_execution"]:
            logger.info(f"Skipping feature engineering: {idempotency_result['reason']}")
            logger.info(f"Existing output file: {output_file}")
        else:
            logger.info(f"Executing feature engineering: {idempotency_result['reason']}")
            
            # Create backup if configured and file exists
            if config.get('pipeline', {}).get('backup_existing_files', False) and Path(output_file).exists():
                backup_existing_file(output_file, logger)
            
            with log_execution_time("feature_engineering", logger):
                execute_feature_engineering(sql_query, output_file, logger, config)
        
        # Log final memory usage
        log_memory_usage("completion", logger)
        
        # Schritt 7: Success-Message
        logger.info("=" * 60)
        logger.info("DATASET BUILDING ERFOLGREICH ABGESCHLOSSEN!")
        logger.info("=" * 60)
        logger.info(f"Feature-Dataset erstellt: {output_file}")
        logger.info("Nächste Schritte:")
        logger.info("  1. python src/train.py --model logreg")
        logger.info("  2. python src/train.py --model xgb") 
        logger.info("  3. python src/train.py --model lgbm")
        logger.info("  4. python src/report.py")
        logger.info("Viel Erfolg beim ML-Training! 🚀")
        
    except Exception as e:
        logger.error(f"Dataset building failed: {e}")
        log_memory_usage("error", logger)
        raise


if __name__ == '__main__':
    main()
