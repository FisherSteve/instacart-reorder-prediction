#!/usr/bin/env python3
"""
Configuration Utilities for Instacart Reorder Prediction

Als Lernprojekt bewusst ausführlich kommentiert. Dieses Modul stellt
zentrale Konfigurationsverwaltung für alle Pipeline-Komponenten bereit.

Funktionalität:
- Laden und Validieren von config.yaml
- Command-line Argument Override Support
- Konfigurationsfehler mit hilfreichen Meldungen
- Type-safe Zugriff auf Konfigurationswerte

Usage:
    from src.config_utils import load_config, validate_config
    config = load_config()
    model_params = config['models']['xgboost']
"""

import os
import sys
import yaml
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union


def find_config_file(config_path: Optional[str] = None) -> str:
    """
    Findet die config.yaml Datei mit intelligenter Pfad-Suche.
    
    Als Lernprojekt: Ich implementiere robuste Pfad-Suche, die sowohl
    absolute als auch relative Pfade und verschiedene Arbeitsverzeichnisse
    unterstützt.
    
    Args:
        config_path: Optionaler expliziter Pfad zur Config-Datei
        
    Returns:
        Absoluter Pfad zur gefundenen Config-Datei
        
    Raises:
        FileNotFoundError: Wenn keine Config-Datei gefunden wird
    """
    # Wenn expliziter Pfad angegeben, verwende diesen
    if config_path:
        if Path(config_path).exists():
            return str(Path(config_path).resolve())
        else:
            raise FileNotFoundError(f"Explizit angegebene Config-Datei nicht gefunden: {config_path}")
    
    # Standard-Suchpfade für config.yaml
    search_paths = [
        "config.yaml",                    # Aktuelles Verzeichnis
        "config/config.yaml",             # Config-Unterverzeichnis
        "../config.yaml",                 # Ein Verzeichnis höher
        str(Path(__file__).parent.parent / "config.yaml"),  # Projekt-Root
    ]
    
    for path in search_paths:
        if Path(path).exists():
            found_path = str(Path(path).resolve())
            print(f"Config-Datei gefunden: {found_path}")
            return found_path
    
    # Wenn nichts gefunden, hilfreiche Fehlermeldung
    current_dir = Path.cwd()
    raise FileNotFoundError(
        f"Keine config.yaml Datei gefunden!\n"
        f"Aktuelles Verzeichnis: {current_dir}\n"
        f"Gesuchte Pfade:\n" + 
        "\n".join(f"  - {path}" for path in search_paths) +
        f"\n\nErstelle eine config.yaml Datei oder verwende --config parameter."
    )


def load_yaml_file(config_file_path: str) -> Dict[str, Any]:
    """
    Lädt YAML-Datei mit robuster Fehlerbehandlung.
    
    Als Lernprojekt: Ich implementiere explizite Behandlung häufiger
    YAML-Parsing-Fehler mit hilfreichen Lösungsvorschlägen.
    
    Args:
        config_file_path: Pfad zur YAML-Datei
        
    Returns:
        Dictionary mit Konfigurationsdaten
        
    Raises:
        yaml.YAMLError: Bei YAML-Parsing-Fehlern
        FileNotFoundError: Wenn Datei nicht existiert
    """
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        if config_data is None:
            raise ValueError("Config-Datei ist leer oder enthält nur Kommentare")
        
        if not isinstance(config_data, dict):
            raise ValueError("Config-Datei muss ein YAML-Dictionary sein")
        
        print(f"[OK] Config erfolgreich geladen: {len(config_data)} Hauptsektionen")
        return config_data
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"YAML-Parsing-Fehler in {config_file_path}:\n{e}\n\n"
            f"Häufige Ursachen:\n"
            f"- Falsche Einrückung (YAML ist einrückungsabhängig)\n"
            f"- Fehlende Anführungszeichen bei Strings mit Sonderzeichen\n"
            f"- Ungültige YAML-Syntax\n"
            f"Validiere die YAML-Syntax online oder mit einem YAML-Validator."
        ) from e
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Config-Datei nicht gefunden: {config_file_path}")
    
    except Exception as e:
        raise RuntimeError(f"Unerwarteter Fehler beim Laden der Config: {e}") from e


def validate_config_structure(config: Dict[str, Any]) -> None:
    """
    Validiert die Struktur der Konfiguration auf erforderliche Sektionen.
    
    Als Lernprojekt: Ich prüfe hier explizit die erwartete Struktur und
    gebe hilfreiche Fehlermeldungen bei fehlenden oder falschen Sektionen.
    
    Args:
        config: Geladene Konfigurationsdaten
        
    Raises:
        ValueError: Bei fehlenden oder ungültigen Konfigurationssektionen
    """
    # Erforderliche Hauptsektionen
    required_sections = [
        'data', 'sampling', 'models', 'preprocessing', 
        'output', 'logging', 'system', 'validation'
    ]
    
    missing_sections = [section for section in required_sections if section not in config]
    
    if missing_sections:
        raise ValueError(
            f"Fehlende Konfigurationssektionen: {missing_sections}\n"
            f"Verfügbare Sektionen: {list(config.keys())}\n"
            f"Stelle sicher, dass die config.yaml alle erforderlichen Sektionen enthält."
        )
    
    # Validiere data Sektion
    data_config = config['data']
    required_data_keys = ['features_file', 'sql_file', 'raw_path']
    missing_data_keys = [key for key in required_data_keys if key not in data_config]
    
    if missing_data_keys:
        raise ValueError(
            f"Fehlende Schlüssel in 'data' Sektion: {missing_data_keys}\n"
            f"Verfügbare Schlüssel: {list(data_config.keys())}"
        )
    
    # Validiere models Sektion
    models_config = config['models']
    required_model_types = ['logreg', 'xgboost', 'lightgbm']
    missing_models = [model for model in required_model_types if model not in models_config]
    
    if missing_models:
        raise ValueError(
            f"Fehlende Modell-Konfigurationen: {missing_models}\n"
            f"Verfügbare Modelle: {list(models_config.keys())}"
        )
    
    print("[OK] Konfigurationsstruktur validiert")


def validate_config_values(config: Dict[str, Any]) -> None:
    """
    Validiert die Werte in der Konfiguration auf sinnvolle Bereiche.
    
    Als Lernprojekt: Ich prüfe hier typische Konfigurationsfehler und
    gebe Warnungen bei ungewöhnlichen aber nicht fatalen Werten.
    
    Args:
        config: Geladene Konfigurationsdaten
    """
    # Sampling Validierung
    sampling_config = config['sampling']
    
    if sampling_config['test_size'] <= 0 or sampling_config['test_size'] >= 1:
        raise ValueError(f"test_size muss zwischen 0 und 1 liegen: {sampling_config['test_size']}")
    
    if sampling_config['random_seed'] < 0:
        raise ValueError(f"random_seed muss nicht-negativ sein: {sampling_config['random_seed']}")
    
    # Models Validierung
    models_config = config['models']
    
    if models_config['default_topk'] <= 0:
        raise ValueError(f"default_topk muss positiv sein: {models_config['default_topk']}")
    
    if models_config['default_topk'] > 100:
        warnings.warn(f"default_topk ist sehr hoch ({models_config['default_topk']}). Typische Werte: 5-20")
    
    # XGBoost Parameter Validierung
    xgb_config = models_config['xgboost']
    if xgb_config['n_estimators'] <= 0:
        raise ValueError(f"XGBoost n_estimators muss positiv sein: {xgb_config['n_estimators']}")
    
    if xgb_config['learning_rate'] <= 0 or xgb_config['learning_rate'] > 1:
        raise ValueError(f"XGBoost learning_rate muss zwischen 0 und 1 liegen: {xgb_config['learning_rate']}")
    
    # LightGBM Parameter Validierung
    lgb_config = models_config['lightgbm']
    if lgb_config['n_estimators'] <= 0:
        raise ValueError(f"LightGBM n_estimators muss positiv sein: {lgb_config['n_estimators']}")
    
    if lgb_config['learning_rate'] <= 0 or lgb_config['learning_rate'] > 1:
        raise ValueError(f"LightGBM learning_rate muss zwischen 0 und 1 liegen: {lgb_config['learning_rate']}")
    
    # System Validierung
    system_config = config['system']
    if system_config['min_memory_gb'] <= 0:
        raise ValueError(f"min_memory_gb muss positiv sein: {system_config['min_memory_gb']}")
    
    print("[OK] Konfigurationswerte validiert")


def apply_command_line_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wendet Command-Line Overrides auf die Konfiguration an.
    
    Als Lernprojekt: Ich implementiere hier einen flexiblen Override-Mechanismus
    der nested dictionary keys mit dot-notation unterstützt.
    
    Args:
        config: Basis-Konfiguration
        overrides: Dictionary mit Override-Werten (dot-notation für nested keys)
        
    Returns:
        Konfiguration mit angewendeten Overrides
        
    Example:
        overrides = {'models.xgboost.n_estimators': 1000, 'sampling.test_size': 0.3}
    """
    if not overrides:
        return config
    
    # Deep copy um Original nicht zu modifizieren
    import copy
    config_copy = copy.deepcopy(config)
    
    for key, value in overrides.items():
        # Dot-notation in nested dictionary path umwandeln
        keys = key.split('.')
        
        # Navigiere zum parent dictionary
        current_dict = config_copy
        for k in keys[:-1]:
            if k not in current_dict:
                current_dict[k] = {}
            current_dict = current_dict[k]
        
        # Setze den finalen Wert
        final_key = keys[-1]
        old_value = current_dict.get(final_key, "NOT_SET")
        current_dict[final_key] = value
        
        print(f"Override angewendet: {key} = {value} (vorher: {old_value})")
    
    return config_copy


def load_config(config_path: Optional[str] = None, 
                overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Hauptfunktion zum Laden und Validieren der Konfiguration.
    
    Als Lernprojekt: Diese Funktion orchestriert den gesamten Config-Loading
    Prozess mit umfassender Validierung und Override-Support.
    
    Args:
        config_path: Optionaler Pfad zur Config-Datei
        overrides: Optionale Command-Line Overrides
        
    Returns:
        Vollständig validierte Konfiguration
        
    Raises:
        FileNotFoundError: Wenn Config-Datei nicht gefunden wird
        ValueError: Bei ungültiger Konfiguration
        yaml.YAMLError: Bei YAML-Parsing-Fehlern
    """
    print("Lade Konfiguration...")
    
    try:
        # Schritt 1: Config-Datei finden
        config_file_path = find_config_file(config_path)
        
        # Schritt 2: YAML laden
        config = load_yaml_file(config_file_path)
        
        # Schritt 3: Struktur validieren
        validate_config_structure(config)
        
        # Schritt 4: Werte validieren
        validate_config_values(config)
        
        # Schritt 5: Command-Line Overrides anwenden
        if overrides:
            config = apply_command_line_overrides(config, overrides)
            # Nach Overrides nochmal validieren
            validate_config_values(config)
        
        print("[OK] Konfiguration erfolgreich geladen und validiert")
        return config
        
    except Exception as e:
        print(f"FEHLER beim Laden der Konfiguration: {e}", file=sys.stderr)
        raise


def get_model_config(config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """
    Extrahiert Modell-spezifische Konfiguration.
    
    Args:
        config: Vollständige Konfiguration
        model_type: Modelltyp ('logreg', 'xgboost', 'lightgbm')
        
    Returns:
        Modell-spezifische Konfiguration
        
    Raises:
        ValueError: Bei unbekanntem Modelltyp
    """
    models_config = config['models']
    
    # Mapping von CLI-Namen zu Config-Namen
    model_mapping = {
        'logreg': 'logreg',
        'xgb': 'xgboost', 
        'lgbm': 'lightgbm'
    }
    
    config_key = model_mapping.get(model_type, model_type)
    
    if config_key not in models_config:
        available_models = list(models_config.keys())
        raise ValueError(
            f"Unbekannter Modelltyp: {model_type}\n"
            f"Verfügbare Modelle: {available_models}\n"
            f"CLI-Namen: {list(model_mapping.keys())}"
        )
    
    return models_config[config_key]


def create_example_config(output_path: str = "config.yaml.example") -> None:
    """
    Erstellt eine Beispiel-Konfigurationsdatei.
    
    Als Lernprojekt: Hilfreiche Funktion für neue Nutzer um eine
    vollständige Beispiel-Konfiguration zu generieren.
    
    Args:
        output_path: Pfad für die Beispiel-Config-Datei
    """
    example_config = """# Instacart Reorder Prediction - Example Configuration
# Copy this file to config.yaml and adjust values as needed

data:
  raw_path: "data/raw"
  features_file: "data/features/features.parquet"
  sql_file: "src/sql/01_build.sql"

sampling:
  max_users: null  # null = no sampling
  random_seed: 42
  test_size: 0.2

models:
  default_topk: 10
  
  logreg:
    max_iter: 1000
    random_state: 42
  
  xgboost:
    n_estimators: 2000
    learning_rate: 0.05
    random_state: 42
  
  lightgbm:
    n_estimators: 4000
    learning_rate: 0.03
    random_state: 42

output:
  reports_dir: "reports"

logging:
  level: "INFO"

system:
  min_memory_gb: 2.0
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(example_config)
    
    print(f"Beispiel-Konfiguration erstellt: {output_path}")


if __name__ == "__main__":
    # Test der Konfiguration
    try:
        config = load_config()
        print("\nKonfiguration erfolgreich getestet!")
        print(f"Verfügbare Sektionen: {list(config.keys())}")
        
        # Test Modell-Config Extraktion
        for model_type in ['logreg', 'xgb', 'lgbm']:
            model_config = get_model_config(config, model_type)
            print(f"{model_type} Config: {len(model_config)} Parameter")
            
    except Exception as e:
        print(f"Konfigurationstest fehlgeschlagen: {e}")
        sys.exit(1)