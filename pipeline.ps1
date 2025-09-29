# Instacart Reorder Prediction - PowerShell Orchestration
param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    
    [Parameter(Position=1)]
    [string]$Model = "logreg",
    
    [string]$Config = "config.yaml"
)

function Show-Help {
    Write-Host ""
    Write-Host "Instacart Reorder Prediction - PowerShell Pipeline Commands" -ForegroundColor Green
    Write-Host ""
    Write-Host "Main Commands:" -ForegroundColor Yellow
    Write-Host "  .\pipeline.ps1 build          - Run data validation and feature engineering"
    Write-Host "  .\pipeline.ps1 train MODEL    - Train model (logreg, xgb, lgbm)"
    Write-Host "  .\pipeline.ps1 report         - Generate performance reports"
    Write-Host "  .\pipeline.ps1 validate       - Run end-to-end pipeline validation"
    Write-Host "  .\pipeline.ps1 clean          - Clean intermediate and output files"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\pipeline.ps1 build"
    Write-Host "  .\pipeline.ps1 train logreg"
    Write-Host "  .\pipeline.ps1 train xgb"
    Write-Host "  .\pipeline.ps1 train lgbm"
    Write-Host "  .\pipeline.ps1 report"
    Write-Host "  .\pipeline.ps1 validate"
    Write-Host "  .\pipeline.ps1 clean"
    Write-Host ""
}

function Test-Config {
    if (-not (Test-Path $Config)) {
        Write-Host "ERROR: Configuration file $Config not found!" -ForegroundColor Red
        exit 1
    }
}

function Invoke-Build {
    Write-Host "Building feature dataset..." -ForegroundColor Green
    Test-Config
    python src\build_dataset.py --config $Config
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Feature building failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "Feature dataset built successfully" -ForegroundColor Green
}

function Invoke-Train {
    param([string]$ModelType)
    Write-Host "Training model: $ModelType" -ForegroundColor Green
    Test-Config
    if (-not (Test-Path "data\features\features.parquet")) {
        Write-Host "Features not found. Running build first..." -ForegroundColor Yellow
        Invoke-Build
    }
    python src\train.py --model $ModelType --config $Config
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Model training failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "Model $ModelType trained successfully" -ForegroundColor Green
}

function Invoke-Report {
    Write-Host "Generating performance reports..." -ForegroundColor Green
    Test-Config
    python src\report.py --config $Config
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Report generation failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "Reports generated successfully" -ForegroundColor Green
}

function Invoke-Validate {
    Write-Host "Running end-to-end pipeline validation..." -ForegroundColor Green
    Test-Config
    if (Test-Path "src\validate_pipeline.py") {
        Write-Host "Running pipeline validation script..." -ForegroundColor Yellow
        python src\validate_pipeline.py --config $Config
    } else {
        Write-Host "Running basic validation workflow..." -ForegroundColor Yellow
        Write-Host "1. Building features..." -ForegroundColor Yellow
        Invoke-Build
        Write-Host "2. Training test model..." -ForegroundColor Yellow
        Invoke-Train "logreg"
        Write-Host "3. Generating report..." -ForegroundColor Yellow
        Invoke-Report
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Pipeline validation failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "Pipeline validation completed" -ForegroundColor Green
}

function Invoke-Clean {
    Write-Host "Cleaning intermediate and output files..." -ForegroundColor Green
    Write-Host "Removing feature datasets..." -ForegroundColor Yellow
    if (Test-Path "data\features\features.parquet") {
        Remove-Item "data\features\features.parquet" -Force
    }
    if (Test-Path "data\intermediate") {
        Remove-Item "data\intermediate" -Recurse -Force
    }
    Write-Host "Removing model outputs..." -ForegroundColor Yellow
    Get-ChildItem -Path "reports" -Filter "model_*.joblib" -ErrorAction SilentlyContinue | Remove-Item -Force
    Get-ChildItem -Path "reports" -Filter "metrics_*.json" -ErrorAction SilentlyContinue | Remove-Item -Force
    if (Test-Path "reports\metrics_leaderboard.csv") {
        Remove-Item "reports\metrics_leaderboard.csv" -Force
    }
    if (Test-Path "reports\report.html") {
        Remove-Item "reports\report.html" -Force
    }
    Write-Host "Removing Python cache..." -ForegroundColor Yellow
    Get-ChildItem -Path . -Recurse -Directory -Name "__pycache__" -ErrorAction SilentlyContinue | 
        ForEach-Object { Remove-Item $_ -Recurse -Force }
    Get-ChildItem -Path . -Recurse -Filter "*.pyc" -ErrorAction SilentlyContinue | Remove-Item -Force
    Write-Host "Cleanup completed" -ForegroundColor Green
}

function Show-Status {
    Write-Host "Pipeline Status:" -ForegroundColor Green
    Write-Host ""
    Write-Host "Configuration:" -ForegroundColor Yellow
    if (Test-Path $Config) {
        Write-Host "  Config file: $Config" -ForegroundColor Green
    } else {
        Write-Host "  Config file: $Config (missing)" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "Data:" -ForegroundColor Yellow
    if (Test-Path "data\features\features.parquet") {
        Write-Host "  Features: data\features\features.parquet" -ForegroundColor Green
    } else {
        Write-Host "  Features: data\features\features.parquet (run build)" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "Models:" -ForegroundColor Yellow
    $models = @("logreg", "xgb", "lgbm")
    foreach ($model in $models) {
        if (Test-Path "reports\model_$model.joblib") {
            Write-Host "  Model: $model" -ForegroundColor Green
        } else {
            Write-Host "  Model: $model (not trained)" -ForegroundColor Red
        }
    }
}

# Main execution
switch ($Command.ToLower()) {
    "help" { Show-Help }
    "build" { Invoke-Build }
    "train" { Invoke-Train $Model }
    "report" { Invoke-Report }
    "validate" { Invoke-Validate }
    "clean" { Invoke-Clean }
    "status" { Show-Status }
    default { Show-Help }
}