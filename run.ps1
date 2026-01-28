# SDSS-Tox 애플리케이션 실행 스크립트
# Python 백엔드와 Java GUI를 동시에 실행합니다

$ErrorActionPreference = "Stop"

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "SDSS-Tox 애플리케이션 시작 중..." -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $RootDir "backend"

try {
    # Python FastAPI 백엔드 시작
    Write-Host "`n[1/2] Python FastAPI 백엔드 시작..." -ForegroundColor Green
    $BackendProcess = Start-Process -FilePath "python" `
        -ArgumentList "-m", "uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000" `
        -WorkingDirectory $BackendDir `
        -PassThru `
        -NoNewWindow

    Write-Host "✓ 백엔드 시작 (포트 8000)" -ForegroundColor Green
    Start-Sleep -Seconds 3

    # Java GUI 시작
    Write-Host "`n[2/2] Java JavaFX GUI 시작..." -ForegroundColor Green
    $MvnExe = Join-Path $RootDir "apache-maven-3.9.12\bin\mvn.cmd"
    $GuiProcess = Start-Process -FilePath $MvnExe `
        -ArgumentList "clean", "javafx:run" `
        -WorkingDirectory $RootDir `
        -PassThru

    Write-Host "✓ GUI 시작 중..." -ForegroundColor Green

    Write-Host "`n" -NoNewline
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host "애플리케이션 실행 중..." -ForegroundColor Yellow
    Write-Host "백엔드: http://0.0.0.0:8000" -ForegroundColor Yellow
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host "종료하려면 Ctrl+C를 누르세요.`n" -ForegroundColor Yellow

    # 프로세스가 끝날 때까지 대기
    Get-Process | Where-Object { $_.Id -eq $BackendProcess.Id -or $_.Id -eq $GuiProcess.Id } | Wait-Process
}
catch {
    Write-Host "에러 발생: $_" -ForegroundColor Red
    exit 1
}
