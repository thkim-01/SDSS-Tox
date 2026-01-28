#!/usr/bin/env python3
"""
SDSS-Tox 애플리케이션 실행 스크립트
Java GUI와 Python FastAPI 백엔드를 동시에 실행합니다.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def run_application():
    """Java GUI와 Python 백엔드를 동시에 실행"""
    
    root_dir = Path(__file__).parent
    backend_dir = root_dir / "backend"
    
    # 프로세스 리스트
    processes = []
    
    try:
        print("=" * 60)
        print("SDSS-Tox 애플리케이션 시작 중...")
        print("=" * 60)
        
        # Python FastAPI 백엔드 시작
        print("\n[1/2] Python FastAPI 백엔드 시작...")
        backend_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
            cwd=str(backend_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(("Backend", backend_process))
        print("✓ 백엔드 시작 (포트 8000)")
        
        # 백엔드 준비 대기
        time.sleep(3)
        
        # Java GUI 시작
        print("\n[2/2] Java JavaFX GUI 시작...")
        gui_process = subprocess.Popen(
            [str(root_dir / "apache-maven-3.9.12" / "bin" / "mvn"), "clean", "javafx:run"],
            cwd=str(root_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(("GUI", gui_process))
        print("✓ GUI 시작 중...")
        
        print("\n" + "=" * 60)
        print("애플리케이션 실행 중...")
        print("백엔드: http://0.0.0.0:8000")
        print("=" * 60)
        print("\n종료하려면 Ctrl+C를 누르세요.\n")
        
        # 모든 프로세스가 끝날 때까지 대기
        while True:
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"\n⚠️ {name} 프로세스가 종료되었습니다.")
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n애플리케이션 종료 중...")
        for name, proc in processes:
            if proc.poll() is None:
                proc.terminate()
                print(f"✓ {name} 종료됨")
        
        # 강제 종료 대기
        for name, proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"✓ {name} 강제 종료됨")
        
        print("애플리케이션 종료 완료")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n에러 발생: {e}")
        for name, proc in processes:
            if proc.poll() is None:
                proc.kill()
        sys.exit(1)

if __name__ == "__main__":
    run_application()
