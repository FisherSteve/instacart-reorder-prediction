#!/usr/bin/env python3
"""
Comprehensive Logging Infrastructure for Instacart Reorder Prediction

Als Lernprojekt bewusst ausführlich kommentiert. Dieses Modul stellt
eine zentrale Logging-Infrastruktur für alle Pipeline-Komponenten bereit.

Funktionalität:
- Strukturiertes Logging mit konfigurierbaren Formattern
- Execution Time Logging mit Context Managern
- Memory Usage Monitoring und Resource Utilization Tracking
- Konsistente Log-Ausgabe über alle Scripts hinweg
- Performance-Metriken und Pipeline-State Tracking

Usage:
    from src.logging_utils import setup_logging, log_execution_time, log_memory_usage
    
    logger = setup_logging(__name__)
    
    with log_execution_time("data_loading", logger):
        # Your code here
        pass
    
    log_memory_usage("after_training", logger)
"""

import logging
import time
import psutil
import os
import sys
import json
from contextlib import contextmanager
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter für strukturierte Log-Ausgaben.
    
    Als Lernprojekt: Ich erstelle einen benutzerdefinierten Formatter,
    der sowohl menschenlesbare als auch maschinenlesbare Logs unterstützt.
    """
    
    def __init__(self, include_memory: bool = True, include_process_info: bool = True):
        """
        Initialize structured formatter.
        
        Args:
            include_memory: Whether to include memory usage in logs
            include_process_info: Whether to include process information
        """
        super().__init__()
        self.include_memory = include_memory
        self.include_process_info = include_process_info
        
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with structured information.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log string
        """
        # Base message
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Build structured log entry
        log_parts = [
            f"{timestamp}",
            f"[{record.levelname}]",
            f"{record.name}",
        ]
        
        # Add process info if enabled
        if self.include_process_info:
            log_parts.append(f"PID:{os.getpid()}")
        
        # Add memory info if enabled and available
        if self.include_memory:
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                log_parts.append(f"MEM:{memory_mb:.1f}MB")
            except Exception:
                pass  # Skip memory info if not available
        
        # Add the actual message
        log_parts.append(f"- {record.getMessage()}")
        
        # Add exception info if present
        if record.exc_info:
            log_parts.append(f"\n{self.formatException(record.exc_info)}")
        
        return " ".join(log_parts)


class PipelineStateTracker:
    """
    Tracks pipeline execution state and enables resumption from failed steps.
    
    Als Lernprojekt: Ich implementiere hier ein einfaches State-Tracking-System,
    das bei Pipeline-Fehlern hilft und Wiederaufnahme ermöglicht.
    """
    
    def __init__(self, state_file: str = "pipeline_state.json"):
        """
        Initialize pipeline state tracker.
        
        Args:
            state_file: Path to state tracking file
        """
        self.state_file = Path(state_file)
        self.state = self._load_state()
        
    def _load_state(self) -> Dict[str, Any]:
        """Load existing pipeline state or create new one."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass  # Start with fresh state if loading fails
        
        return {
            "created_at": datetime.now().isoformat(),
            "steps": {},
            "last_update": None
        }
    
    def _save_state(self) -> None:
        """Save current state to file."""
        self.state["last_update"] = datetime.now().isoformat()
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logging.warning(f"Could not save pipeline state: {e}")
    
    def mark_step_started(self, step_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark a pipeline step as started.
        
        Args:
            step_name: Name of the pipeline step
            metadata: Optional metadata about the step
        """
        self.state["steps"][step_name] = {
            "status": "started",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "metadata": metadata or {}
        }
        self._save_state()
    
    def mark_step_completed(self, step_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark a pipeline step as completed.
        
        Args:
            step_name: Name of the pipeline step
            metadata: Optional metadata about the step completion
        """
        if step_name in self.state["steps"]:
            step = self.state["steps"][step_name]
            end_time = datetime.now()
            step["status"] = "completed"
            step["end_time"] = end_time.isoformat()
            
            # Calculate duration if start time exists
            if step.get("start_time"):
                try:
                    start_time = datetime.fromisoformat(step["start_time"])
                    step["duration_seconds"] = (end_time - start_time).total_seconds()
                except Exception:
                    pass
            
            # Update metadata
            if metadata:
                step["metadata"].update(metadata)
        else:
            # Step wasn't marked as started, create completed entry
            self.state["steps"][step_name] = {
                "status": "completed",
                "start_time": None,
                "end_time": datetime.now().isoformat(),
                "duration_seconds": None,
                "metadata": metadata or {}
            }
        
        self._save_state()
    
    def mark_step_failed(self, step_name: str, error: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark a pipeline step as failed.
        
        Args:
            step_name: Name of the pipeline step
            error: Error message or description
            metadata: Optional metadata about the failure
        """
        if step_name in self.state["steps"]:
            step = self.state["steps"][step_name]
            end_time = datetime.now()
            step["status"] = "failed"
            step["end_time"] = end_time.isoformat()
            step["error"] = error
            
            # Calculate duration if start time exists
            if step.get("start_time"):
                try:
                    start_time = datetime.fromisoformat(step["start_time"])
                    step["duration_seconds"] = (end_time - start_time).total_seconds()
                except Exception:
                    pass
            
            # Update metadata
            if metadata:
                step["metadata"].update(metadata)
        else:
            # Step wasn't marked as started, create failed entry
            self.state["steps"][step_name] = {
                "status": "failed",
                "start_time": None,
                "end_time": datetime.now().isoformat(),
                "duration_seconds": None,
                "error": error,
                "metadata": metadata or {}
            }
        
        self._save_state()
    
    def get_step_status(self, step_name: str) -> Optional[str]:
        """
        Get the status of a pipeline step.
        
        Args:
            step_name: Name of the pipeline step
            
        Returns:
            Step status ('started', 'completed', 'failed') or None if not found
        """
        return self.state["steps"].get(step_name, {}).get("status")
    
    def get_completed_steps(self) -> list:
        """Get list of completed step names."""
        return [name for name, step in self.state["steps"].items() 
                if step.get("status") == "completed"]
    
    def get_failed_steps(self) -> list:
        """Get list of failed step names."""
        return [name for name, step in self.state["steps"].items() 
                if step.get("status") == "failed"]
    
    def clear_state(self) -> None:
        """Clear all pipeline state."""
        self.state = {
            "created_at": datetime.now().isoformat(),
            "steps": {},
            "last_update": None
        }
        self._save_state()


# Global pipeline state tracker instance
_pipeline_state_tracker = None


def get_pipeline_state_tracker(state_file: str = "pipeline_state.json") -> PipelineStateTracker:
    """
    Get global pipeline state tracker instance.
    
    Args:
        state_file: Path to state tracking file
        
    Returns:
        PipelineStateTracker instance
    """
    global _pipeline_state_tracker
    if _pipeline_state_tracker is None:
        _pipeline_state_tracker = PipelineStateTracker(state_file)
    return _pipeline_state_tracker


def setup_logging(name: str, 
                 level: str = "INFO",
                 format_string: Optional[str] = None,
                 include_memory: bool = True,
                 include_process_info: bool = True,
                 log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup comprehensive logging for a module with structured formatting.
    
    Als Lernprojekt: Diese Funktion konfiguriert ein vollständiges Logging-System
    mit strukturierten Ausgaben, Memory-Monitoring und optionalem File-Logging.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Optional custom format string
        include_memory: Whether to include memory usage in logs
        include_process_info: Whether to include process information
        log_file: Optional file path for file logging
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    # Set logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Create console handler with structured formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if format_string:
        formatter = logging.Formatter(format_string)
    else:
        formatter = StructuredFormatter(
            include_memory=include_memory,
            include_process_info=include_process_info
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        try:
            # Ensure log directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"File logging enabled: {log_file}")
        except Exception as e:
            logger.warning(f"Could not setup file logging to {log_file}: {e}")
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    
    return logger


@contextmanager
def log_execution_time(operation_name: str, 
                      logger: logging.Logger,
                      level: str = "INFO",
                      include_memory: bool = True,
                      track_state: bool = True):
    """
    Context manager for logging execution times with memory monitoring.
    
    Als Lernprojekt: Ich verwende einen Context Manager für automatisches
    Timing und Memory-Monitoring von Pipeline-Operationen.
    
    Args:
        operation_name: Name of the operation being timed
        logger: Logger instance to use
        level: Logging level for timing messages
        include_memory: Whether to include memory usage information
        track_state: Whether to track this operation in pipeline state
        
    Usage:
        with log_execution_time("data_loading", logger):
            # Your code here
            data = load_data()
    """
    # Get logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Get initial memory usage
    initial_memory = None
    if include_memory:
        try:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        except Exception:
            include_memory = False
    
    # Track pipeline state if enabled
    state_tracker = None
    if track_state:
        try:
            state_tracker = get_pipeline_state_tracker()
            state_tracker.mark_step_started(operation_name, {
                "initial_memory_mb": initial_memory
            })
        except Exception as e:
            logger.warning(f"Could not track pipeline state for {operation_name}: {e}")
    
    # Log start
    start_message = f"Starting {operation_name}"
    if include_memory and initial_memory is not None:
        start_message += f" (initial memory: {initial_memory:.1f} MB)"
    
    logger.log(numeric_level, start_message)
    
    start_time = time.time()
    error_occurred = False
    
    try:
        yield
        
    except Exception as e:
        error_occurred = True
        duration = time.time() - start_time
        
        # Log failure
        error_message = f"Failed {operation_name} after {duration:.2f}s: {e}"
        if include_memory and initial_memory is not None:
            try:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_change = current_memory - initial_memory
                error_message += f" (memory change: {memory_change:+.1f} MB)"
            except Exception:
                pass
        
        logger.error(error_message)
        
        # Track failure in pipeline state
        if track_state and state_tracker:
            try:
                state_tracker.mark_step_failed(operation_name, str(e), {
                    "duration_seconds": duration,
                    "final_memory_mb": current_memory if include_memory else None
                })
            except Exception:
                pass
        
        raise
    
    else:
        # Calculate duration and memory change
        duration = time.time() - start_time
        
        # Log success
        success_message = f"Completed {operation_name} in {duration:.2f}s"
        
        final_memory = None
        if include_memory and initial_memory is not None:
            try:
                final_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_change = final_memory - initial_memory
                success_message += f" (memory change: {memory_change:+.1f} MB)"
            except Exception:
                pass
        
        logger.log(numeric_level, success_message)
        
        # Track completion in pipeline state
        if track_state and state_tracker:
            try:
                state_tracker.mark_step_completed(operation_name, {
                    "duration_seconds": duration,
                    "final_memory_mb": final_memory,
                    "memory_change_mb": final_memory - initial_memory if final_memory and initial_memory else None
                })
            except Exception:
                pass


def log_memory_usage(context: str, 
                    logger: logging.Logger,
                    level: str = "INFO",
                    include_system_info: bool = True) -> Dict[str, float]:
    """
    Log current memory usage with optional system information.
    
    Als Lernprojekt: Ich implementiere detailliertes Memory-Monitoring
    für bessere Diagnose von Memory-Problemen in der Pipeline.
    
    Args:
        context: Context description for the memory measurement
        logger: Logger instance to use
        level: Logging level for memory messages
        include_system_info: Whether to include system-wide memory info
        
    Returns:
        Dictionary with memory usage metrics
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    try:
        # Process memory info
        process = psutil.Process()
        memory_info = process.memory_info()
        
        process_metrics = {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()        # Percent of system RAM
        }
        
        # System memory info if requested
        system_metrics = {}
        if include_system_info:
            system_memory = psutil.virtual_memory()
            system_metrics = {
                'total_gb': system_memory.total / 1024 / 1024 / 1024,
                'available_gb': system_memory.available / 1024 / 1024 / 1024,
                'used_percent': system_memory.percent
            }
        
        # Build log message
        log_message = f"Memory usage at {context}: "
        log_message += f"Process RSS={process_metrics['rss_mb']:.1f}MB "
        log_message += f"({process_metrics['percent']:.1f}% of system RAM)"
        
        if include_system_info:
            log_message += f", System: {system_metrics['available_gb']:.1f}GB available "
            log_message += f"({system_metrics['used_percent']:.1f}% used)"
        
        logger.log(numeric_level, log_message)
        
        # Return combined metrics
        return {**process_metrics, **system_metrics}
        
    except Exception as e:
        logger.warning(f"Could not get memory usage for {context}: {e}")
        return {}


def log_structured_metrics(metrics: Dict[str, Any], 
                          context: str,
                          logger: logging.Logger,
                          level: str = "INFO") -> None:
    """
    Log structured metrics in JSON format for monitoring systems.
    
    Als Lernprojekt: Ich implementiere strukturiertes Metrics-Logging
    das von Monitoring-Systemen automatisch geparst werden kann.
    
    Args:
        metrics: Dictionary with metrics to log
        context: Context description for the metrics
        logger: Logger instance to use
        level: Logging level for metrics messages
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    try:
        # Add timestamp and context to metrics
        structured_metrics = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "metrics": metrics
        }
        
        # Log as structured JSON
        logger.log(numeric_level, f"STRUCTURED_METRICS: {json.dumps(structured_metrics, default=str)}")
        
    except Exception as e:
        logger.warning(f"Could not log structured metrics for {context}: {e}")


def log_file_operation(operation: str,
                      file_path: str,
                      logger: logging.Logger,
                      overwrite: bool = False,
                      level: str = "INFO",
                      config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Log file operations with overwrite behavior tracking and control.
    
    Als Lernprojekt: Ich implementiere explizites Logging von Datei-Operationen
    für bessere Nachvollziehbarkeit und Idempotenz-Tracking mit konfigurierbarem
    Overwrite-Verhalten.
    
    Args:
        operation: Type of file operation (create, overwrite, append, delete)
        file_path: Path to the file being operated on
        logger: Logger instance to use
        overwrite: Whether this operation overwrites existing data
        level: Logging level for file operation messages
        config: Optional configuration dictionary with pipeline settings
        
    Returns:
        True if operation should proceed, False if it should be skipped
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    try:
        file_path_obj = Path(file_path)
        file_exists = file_path_obj.exists()
        
        # Get file metadata if it exists
        file_size = None
        file_mtime = None
        if file_exists:
            try:
                stat_info = file_path_obj.stat()
                file_size = stat_info.st_size
                file_mtime = datetime.fromtimestamp(stat_info.st_mtime)
            except Exception:
                pass
        
        # Handle overwrite behavior based on configuration
        if file_exists and overwrite and config:
            pipeline_config = config.get('pipeline', {})
            overwrite_behavior = pipeline_config.get('overwrite_behavior', 'warn')
            
            if overwrite_behavior == 'prevent':
                logger.error(f"File operation BLOCKED: {operation.upper()} {file_path} - overwrite prevented by configuration")
                return False
            elif overwrite_behavior == 'prompt':
                # In a real implementation, this would prompt the user
                # For now, we'll treat it as 'warn'
                logger.warning(f"File operation requires confirmation: {operation.upper()} {file_path}")
            elif overwrite_behavior == 'warn':
                logger.warning(f"File operation will overwrite existing data: {operation.upper()} {file_path}")
        
        # Build detailed log message
        log_message = f"File operation: {operation.upper()} {file_path}"
        
        if file_exists:
            log_message += f" (existing file"
            if file_size is not None:
                log_message += f", {file_size:,} bytes"
            if file_mtime is not None:
                log_message += f", modified {file_mtime.strftime('%Y-%m-%d %H:%M:%S')}"
            log_message += ")"
            
            if overwrite:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_message += f" - OVERWRITING EXISTING DATA at {timestamp}"
        else:
            log_message += " (new file)"
        
        # Log the operation
        if overwrite and file_exists:
            logger.log(logging.WARNING, log_message)
        else:
            logger.log(numeric_level, log_message)
        
        # Log structured metrics for monitoring
        operation_metrics = {
            "operation": operation,
            "file_path": str(file_path),
            "file_exists": file_exists,
            "overwrite": overwrite,
            "file_size_bytes": file_size,
            "timestamp": datetime.now().isoformat()
        }
        
        if file_mtime:
            operation_metrics["file_modified_time"] = file_mtime.isoformat()
        
        log_structured_metrics(operation_metrics, "file_operation", logger, "DEBUG")
        
        return True
        
    except Exception as e:
        logger.warning(f"Could not log file operation {operation} for {file_path}: {e}")
        return True  # Default to allowing the operation


def check_operation_idempotency(operation_name: str,
                               input_files: List[str],
                               output_files: List[str],
                               logger: logging.Logger,
                               config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Check if an operation needs to be executed based on file timestamps and state.
    
    Als Lernprojekt: Ich implementiere hier intelligente Idempotenz-Prüfung,
    die basierend auf Input/Output-Dateien und Pipeline-State entscheidet,
    ob eine Operation übersprungen werden kann.
    
    Args:
        operation_name: Name of the operation to check
        input_files: List of input file paths
        output_files: List of output file paths
        logger: Logger instance to use
        config: Optional configuration dictionary
        
    Returns:
        Dictionary with idempotency check results
    """
    try:
        # Initialize result
        result = {
            "operation": operation_name,
            "needs_execution": True,
            "reason": "initial_run",
            "input_files_exist": True,
            "output_files_exist": True,
            "output_newer_than_input": False,
            "pipeline_state": None
        }
        
        # Check if all input files exist
        missing_inputs = []
        input_timestamps = []
        
        for input_file in input_files:
            input_path = Path(input_file)
            if not input_path.exists():
                missing_inputs.append(input_file)
            else:
                try:
                    input_timestamps.append(input_path.stat().st_mtime)
                except Exception:
                    pass
        
        if missing_inputs:
            result["input_files_exist"] = False
            result["reason"] = f"missing_inputs: {missing_inputs}"
            logger.warning(f"Idempotency check for {operation_name}: Missing input files: {missing_inputs}")
            return result
        
        # Check if all output files exist
        missing_outputs = []
        output_timestamps = []
        
        for output_file in output_files:
            output_path = Path(output_file)
            if not output_path.exists():
                missing_outputs.append(output_file)
            else:
                try:
                    output_timestamps.append(output_path.stat().st_mtime)
                except Exception:
                    pass
        
        if missing_outputs:
            result["output_files_exist"] = False
            result["reason"] = f"missing_outputs: {missing_outputs}"
            logger.info(f"Idempotency check for {operation_name}: Missing output files: {missing_outputs}")
            return result
        
        # Check if outputs are newer than inputs
        if input_timestamps and output_timestamps:
            newest_input = max(input_timestamps)
            oldest_output = min(output_timestamps)
            
            if oldest_output > newest_input:
                result["output_newer_than_input"] = True
                result["needs_execution"] = False
                result["reason"] = "outputs_newer_than_inputs"
                logger.info(f"Idempotency check for {operation_name}: Outputs are newer than inputs, skipping")
            else:
                result["reason"] = "inputs_newer_than_outputs"
                logger.info(f"Idempotency check for {operation_name}: Inputs are newer than outputs, execution needed")
        
        # Check pipeline state if available
        if config and config.get('pipeline', {}).get('enable_state_tracking', False):
            try:
                state_tracker = get_pipeline_state_tracker(
                    config.get('pipeline', {}).get('state_file', 'pipeline_state.json')
                )
                step_status = state_tracker.get_step_status(operation_name)
                result["pipeline_state"] = step_status
                
                if step_status == "completed" and result["output_newer_than_input"]:
                    result["needs_execution"] = False
                    result["reason"] = "completed_and_outputs_current"
                    logger.info(f"Idempotency check for {operation_name}: Previously completed and outputs current")
                elif step_status == "failed":
                    result["reason"] = "previous_failure_retry"
                    logger.info(f"Idempotency check for {operation_name}: Previous failure detected, retrying")
                
            except Exception as e:
                logger.warning(f"Could not check pipeline state for {operation_name}: {e}")
        
        # Log structured metrics
        log_structured_metrics(result, "idempotency_check", logger, "DEBUG")
        
        return result
        
    except Exception as e:
        logger.error(f"Idempotency check failed for {operation_name}: {e}")
        return {
            "operation": operation_name,
            "needs_execution": True,
            "reason": f"check_failed: {e}",
            "error": str(e)
        }


def backup_existing_file(file_path: str, 
                        logger: logging.Logger,
                        backup_suffix: str = None) -> Optional[str]:
    """
    Create a backup of an existing file before overwriting.
    
    Args:
        file_path: Path to the file to backup
        logger: Logger instance to use
        backup_suffix: Optional suffix for backup file (default: timestamp)
        
    Returns:
        Path to backup file if created, None if no backup needed or failed
    """
    try:
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            return None
        
        # Generate backup filename
        if backup_suffix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_suffix = f"backup_{timestamp}"
        
        backup_path = file_path_obj.with_suffix(f".{backup_suffix}{file_path_obj.suffix}")
        
        # Create backup
        import shutil
        shutil.copy2(file_path, backup_path)
        
        logger.info(f"Created backup: {file_path} -> {backup_path}")
        return str(backup_path)
        
    except Exception as e:
        logger.warning(f"Could not create backup for {file_path}: {e}")
        return None


def configure_logging_from_config(config: Dict[str, Any], 
                                 module_name: str,
                                 log_file_suffix: Optional[str] = None) -> logging.Logger:
    """
    Configure logging from configuration dictionary.
    
    Als Lernprojekt: Diese Funktion integriert das Logging-System
    mit der zentralen Konfigurationsverwaltung.
    
    Args:
        config: Configuration dictionary with logging section
        module_name: Name of the module requesting logging
        log_file_suffix: Optional suffix for log file name
        
    Returns:
        Configured logger instance
    """
    logging_config = config.get('logging', {})
    
    # Extract logging configuration
    level = logging_config.get('level', 'INFO')
    format_string = logging_config.get('format')
    
    # Determine log file path if suffix provided
    log_file = None
    if log_file_suffix:
        reports_dir = config.get('output', {}).get('reports_dir', 'reports')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{reports_dir}/logs/{module_name}_{log_file_suffix}_{timestamp}.log"
    
    # Setup logger
    logger = setup_logging(
        name=module_name,
        level=level,
        format_string=format_string,
        include_memory=config.get('system', {}).get('enable_memory_monitoring', True),
        include_process_info=True,
        log_file=log_file
    )
    
    return logger


if __name__ == "__main__":
    # Test the logging infrastructure
    logger = setup_logging(__name__, level="DEBUG")
    
    logger.info("Testing logging infrastructure")
    
    # Test execution time logging
    with log_execution_time("test_operation", logger):
        time.sleep(1)
        logger.info("Inside timed operation")
    
    # Test memory logging
    memory_metrics = log_memory_usage("test_context", logger)
    logger.info(f"Memory metrics: {memory_metrics}")
    
    # Test structured metrics logging
    test_metrics = {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88
    }
    log_structured_metrics(test_metrics, "test_model_evaluation", logger)
    
    # Test file operation logging
    log_file_operation("create", "test_file.txt", logger, overwrite=False)
    log_file_operation("overwrite", "existing_file.txt", logger, overwrite=True)
    
    # Test pipeline state tracking
    state_tracker = get_pipeline_state_tracker("test_pipeline_state.json")
    state_tracker.mark_step_started("test_step")
    time.sleep(0.5)
    state_tracker.mark_step_completed("test_step", {"test_metric": 42})
    
    logger.info("Logging infrastructure test completed")