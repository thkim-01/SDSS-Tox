#!/usr/bin/env python3
"""
SDSS-Tox   
Java GUI Python FastAPI   .
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def run_application():
    """Java GUI Python   """
    
    root_dir = Path(__file__).parent
    backend_dir = root_dir / "backend"
    
    #  
    processes = []
    
    try:
        print("=" * 60)
        print("SDSS-Tox   ...")
        print("=" * 60)
        
        # Python FastAPI  
        print("\n[1/2] Python FastAPI  ...")
        backend_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
            cwd=str(backend_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(("Backend", backend_process))
        print("   ( 8000)")
        
        #   
        time.sleep(3)
        
        # Java GUI 
        print("\n[2/2] Java JavaFX GUI ...")
        gui_process = subprocess.Popen(
            [str(root_dir / "apache-maven-3.9.12" / "bin" / "mvn"), "clean", "javafx:run"],
            cwd=str(root_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(("GUI", gui_process))
        print(" GUI  ...")
        
        print("\n" + "=" * 60)
        print("  ...")
        print(": http://0.0.0.0:8000")
        print("=" * 60)
        print("\n Ctrl+C .\n")
        
        #     
        while True:
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"\n {name}  .")
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n  ...")
        for name, proc in processes:
            if proc.poll() is None:
                proc.terminate()
                print(f" {name} ")
        
        #   
        for name, proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f" {name}  ")
        
        print("  ")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n : {e}")
        for name, proc in processes:
            if proc.poll() is None:
                proc.kill()
        sys.exit(1)

if __name__ == "__main__":
    run_application()
