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

Author: Lernprojekt Instacart Reorder Prediction
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import duckdb
except ImportError:
    print("ERROR: DuckDB ist nicht installiert!")
    print("Bitte installieren mit: pip install duckdb")
    sys.exit(1)


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
    
    logging.info("✓ Alle benötigten CSV-Dateien gefunden")


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
        
        logging.info(f"✓ SQL-Query geladen aus {sql_file_path}")
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
        logging.info("✓ DuckDB Verbindung hergestellt")
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
        logging.info(f"✓ SQL Feature Engineering abgeschlossen in {execution_time:.2f} Sekunden")
        
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
        
        logging.info("✓ Parquet-Export abgeschlossen")
        
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
        
        logging.info("✓ Output-Validierung:")
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
        logging.info("✓ DuckDB Verbindung geschlossen")


def parse_arguments() -> argparse.Namespace:
    """
    Command-Line Argumente parsen.
    
    Als Lernprojekt verwende ich hier ausführliche Help-Texte und
    sinnvolle Defaults. Das macht das Script benutzerfreundlicher.
    """
    parser = argparse.ArgumentParser(
        description='Instacart Reorder Prediction - Dataset Builder',
        epilog='''
Beispiel-Usage:
  python src/build_dataset.py
  python src/build_dataset.py --output data/my_features.parquet
  python src/build_dataset.py --sql src/sql/custom_build.sql
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/features.parquet',
        help='Output-Pfad für das Feature-Dataset (default: data/features.parquet)'
    )
    
    parser.add_argument(
        '--sql',
        type=str, 
        default='src/sql/01_build.sql',
        help='Pfad zur SQL-Datei mit Feature Engineering Logic (default: src/sql/01_build.sql)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose Logging für Debugging'
    )
    
    return parser.parse_args()


def main() -> None:
    """
    Hauptfunktion: Orchestriert den gesamten Dataset-Building Prozess.
    
    Workflow:
    1. Logging Setup
    2. Argumente parsen
    3. Input-Dateien validieren
    4. SQL-Query laden
    5. Feature Engineering ausführen
    6. Success-Message
    
    Als Lernprojekt strukturiere ich das bewusst in kleine, verständliche Schritte.
    """
    # Schritt 1: Logging Setup
    setup_logging()
    
    logging.info("=" * 60)
    logging.info("INSTACART REORDER PREDICTION - DATASET BUILDER")
    logging.info("=" * 60)
    logging.info("Als Lernprojekt bewusst ausführlich kommentiert")
    logging.info("")
    
    # Schritt 2: Command-Line Argumente parsen
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Verbose Logging aktiviert")
    
    logging.info(f"Konfiguration:")
    logging.info(f"  SQL-Datei: {args.sql}")
    logging.info(f"  Output-Datei: {args.output}")
    logging.info("")
    
    # Schritt 3: Input-Validierung
    logging.info("Schritt 1: Input-Validierung")
    validate_input_files()
    logging.info("")
    
    # Schritt 4: SQL-Query laden
    logging.info("Schritt 2: SQL-Query laden")
    sql_query = load_sql_query(args.sql)
    logging.info("")
    
    # Schritt 5: Feature Engineering ausführen
    logging.info("Schritt 3: Feature Engineering ausführen")
    
    # Output-Verzeichnis erstellen falls es nicht existiert
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    execute_feature_engineering(sql_query, args.output)
    logging.info("")
    
    # Schritt 6: Success-Message
    logging.info("=" * 60)
    logging.info("✓ DATASET BUILDING ERFOLGREICH ABGESCHLOSSEN!")
    logging.info("=" * 60)
    logging.info(f"Feature-Dataset erstellt: {args.output}")
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