@echo off
REM Instacart Reorder Prediction - Windows Batch Orchestration
REM Als Lernprojekt bewusst ausführlich kommentiert und strukturiert
REM
REM Usage:
REM   pipeline.bat build          - Run data validation and feature engineering
REM   pipeline.bat train xgb      - Train specified model with configuration
REM   pipeline.bat report         - Generate leaderboard and HTML reports
REM   pipeline.bat validate       - Run end-to-end pipeline testing
REM   pipeline.bat clean          - Remove intermediate and output files

setlocal enabledelayedexpansion

REM Default configuration
set PYTHON=python
set CONFIG=config.yaml

REM Colors (Windows doesn't support ANSI colors in batch by default, so we use echo)
set "GREEN=[32m"
set "YELLOW=[33m"
set "RED=[31m"
set "NC=[0m"

REM Check arguments
if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="build" goto build
if "%1"=="train" goto train
if "%1"=="report" goto report
if "%1"=="validate" goto validate
if "%1"=="clean" goto clean
if "%1"=="status" goto status
if "%1"=="quickstart" goto quickstart
if "%1"=="workflow" goto workflow

echo ERROR: Unknown command '%1'
goto help

:help
echo.
echo Instacart Reorder Prediction - Windows Pipeline Commands
echo.
echo Main Commands:
echo   pipeline.bat build          - Run data validation and feature engineering
echo   pipeline.bat train MODEL    - Train model (logreg, xgb, lgbm)
echo   pipeline.bat report         - Generate performance reports
echo   pipeline.bat validate       - Run end-to-end pipeline validation
echo   pipeline.bat clean          - Clean intermediate and output files
echo.
echo Examples:
echo   pipeline.bat build
echo   pipeline.bat train logreg
echo   pipeline.bat train xgb
echo   pipeline.bat train lgbm
echo   pipeline.bat report
echo.
echo Other Commands:
echo   pipeline.bat status         - Show current pipeline status
echo   pipeline.bat quickstart     - Run complete workflow
echo   pipeline.bat workflow       - Show workflow steps
echo   pipeline.bat help           - Show this help
echo.
goto end

:check_config
if not exist "%CONFIG%" (
    echo ERROR: Configuration file %CONFIG% not found!
    echo Create config.yaml or specify CONFIG=path\to\config.yaml
    exit /b 1
)
goto :eof

:check_python
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Make sure Python is installed and in PATH.
    exit /b 1
)
goto :eof

:build
echo.
echo Building feature dataset...
call :check_config
call :check_python
echo Step 1: Data validation and feature engineering
%PYTHON% src\build_dataset.py --config %CONFIG%
if errorlevel 1 (
    echo ERROR: Feature building failed!
    exit /b 1
)
echo ✓ Feature dataset built successfully
goto end

:train
if "%2"=="" (
    echo ERROR: Model type required. Use: pipeline.bat train [logreg^|xgb^|lgbm]
    exit /b 1
)
set MODEL=%2
echo.
echo Training model: %MODEL%
call :check_config
call :check_python
if not exist "data\features\features.parquet" (
    echo Features not found. Running build first...
    call :build
)
echo Training %MODEL% model with configuration...
%PYTHON% src\train.py --model %MODEL% --config %CONFIG%
if errorlevel 1 (
    echo ERROR: Model training failed!
    exit /b 1
)
echo ✓ Model %MODEL% trained successfully
goto end

:report
echo.
echo Generating performance reports...
call :check_config
call :check_python
if not exist "reports" mkdir reports
dir /b reports\metrics_*.json >nul 2>&1
if errorlevel 1 (
    echo No model metrics found. Training a model first...
    call :train logreg
)
echo Collecting metrics and generating reports...
%PYTHON% src\report.py --config %CONFIG%
if errorlevel 1 (
    echo ERROR: Report generation failed!
    exit /b 1
)
echo ✓ Reports generated successfully
echo Open reports\report.html in your browser
goto end

:validate
echo.
echo Running end-to-end pipeline validation...
call :check_config
call :check_python
if exist "src\validate_pipeline.py" (
    echo Running pipeline validation script...
    %PYTHON% src\validate_pipeline.py --config %CONFIG%
) else (
    echo Running basic validation workflow...
    echo 1. Building features...
    call :build
    echo 2. Training test model...
    call :train logreg
    echo 3. Generating report...
    call :report
)
if errorlevel 1 (
    echo ERROR: Pipeline validation failed!
    exit /b 1
)
echo ✓ Pipeline validation completed
goto end

:clean
echo.
echo Cleaning intermediate and output files...
echo Removing feature datasets...
if exist "data\features\features.parquet" del /q "data\features\features.parquet"
if exist "data\intermediate" rmdir /s /q "data\intermediate"
echo Removing model outputs...
if exist "reports\model_*.joblib" del /q "reports\model_*.joblib"
if exist "reports\metrics_*.json" del /q "reports\metrics_*.json"
if exist "reports\metrics_leaderboard.csv" del /q "reports\metrics_leaderboard.csv"
if exist "reports\report.html" del /q "reports\report.html"
echo Removing Python cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
del /s /q *.pyc >nul 2>&1
echo ✓ Cleanup completed
goto end

:status
echo.
echo Pipeline Status:
echo.
echo Configuration:
if exist "%CONFIG%" (
    echo   ✓ Config file: %CONFIG%
) else (
    echo   ✗ Config file: %CONFIG% ^(missing^)
)
echo.
echo Data:
if exist "data\features\features.parquet" (
    echo   ✓ Features: data\features\features.parquet
) else (
    echo   ✗ Features: data\features\features.parquet ^(run 'pipeline.bat build'^)
)
echo.
echo Models:
if exist "reports\model_logreg.joblib" (
    echo   ✓ Model: logreg
) else (
    echo   ✗ Model: logreg ^(run 'pipeline.bat train logreg'^)
)
if exist "reports\model_xgb.joblib" (
    echo   ✓ Model: xgb
) else (
    echo   ✗ Model: xgb ^(run 'pipeline.bat train xgb'^)
)
if exist "reports\model_lgbm.joblib" (
    echo   ✓ Model: lgbm
) else (
    echo   ✗ Model: lgbm ^(run 'pipeline.bat train lgbm'^)
)
echo.
echo Reports:
if exist "reports\report.html" (
    echo   ✓ HTML Report: reports\report.html
) else (
    echo   ✗ HTML Report: reports\report.html ^(run 'pipeline.bat report'^)
)
goto end

:quickstart
echo.
echo Running complete pipeline workflow...
echo This will: build features → train all models → generate reports
set /p confirm="Continue? (y/N): "
if /i not "%confirm%"=="y" goto end
call :build
call :train logreg
call :train xgb
call :train lgbm
call :report
echo ✓ Complete pipeline finished!
echo Open reports\report.html to see results
goto end

:workflow
echo.
echo Instacart Reorder Prediction - Pipeline Workflow
echo.
echo 1. Setup (one-time):
echo    python -m venv venv           # Create virtual environment
echo    venv\Scripts\activate         # Activate venv (Windows)
echo    pip install -r requirements.txt  # Install dependencies
echo.
echo 2. Build Features:
echo    pipeline.bat build            # SQL-based feature engineering
echo.
echo 3. Train Models:
echo    pipeline.bat train logreg     # Train LogisticRegression
echo    pipeline.bat train xgb        # Train XGBoost
echo    pipeline.bat train lgbm       # Train LightGBM
echo.
echo 4. Generate Reports:
echo    pipeline.bat report           # Create HTML performance report
echo.
echo 5. Validation ^& Cleanup:
echo    pipeline.bat validate         # End-to-end pipeline test
echo    pipeline.bat clean            # Remove intermediate files
echo.
echo Quick Start:
echo    pipeline.bat quickstart       # Complete workflow in one command
goto end

:end
endlocal