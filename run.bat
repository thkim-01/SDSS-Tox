@echo off
REM SDSS-Tox 애플리케이션 실행 스크립트
REM Python 백엔드와 Java GUI를 동시에 실행합니다

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo SDSS-Tox 애플리케이션 시작 중...
echo ============================================================
echo.

REM 현재 디렉토리
set ROOT_DIR=%~dp0
set BACKEND_DIR=%ROOT_DIR%backend
set MVN_EXE=%ROOT_DIR%apache-maven-3.9.12\bin\mvn.cmd

REM Python FastAPI 백엔드 시작
echo [1/2] Python FastAPI 백엔드 시작...
start "SDSS-Tox Backend" cmd /k "cd /d %BACKEND_DIR% && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo 백엔드 시작 (포트 8000)

REM 백엔드 준비 대기
timeout /t 3 /nobreak

REM Java GUI 시작
echo.
echo [2/2] Java JavaFX GUI 시작...
start "SDSS-Tox GUI" cmd /k "cd /d %ROOT_DIR% && %MVN_EXE% clean javafx:run"
echo GUI 시작 중...

echo.
echo ============================================================
echo 애플리케이션 실행 중...
echo 백엔드: http://0.0.0.0:8000
echo ============================================================
echo.
echo 모든 창을 종료하여 애플리케이션을 종료합니다.
pause
