import time
import subprocess
import sys
import os
from datetime import datetime

def run_loop(interval_seconds=300):
    """
    Run streaming pipeline every interval_seconds.
    """
    print(f"🚀 Starting Autocloud Streaming (Interval: {interval_seconds}s)")
    print(f"Target: Latest Data (2026-01-02 for test)")
    
    # Ensure env vars are set (passed from parent process or set here if needed)
    # We rely on parent process env vars for R2 keys
    
    count = 1
    while True:
        now = datetime.now().strftime("%H:%M:%S")
        print(f"\n[Run #{count} - {now}] Starting streaming pipeline...")
        
        try:
            # Run pipeline for Friday 2026-01-02 (latest data) to ensure we have data to process
            # In real usage, this would be datetime.now().strftime('%Y-%m-%d')
            # But for this weekend test, we force the last trading day
            cmd = [sys.executable, "scripts/run_pipeline.py", "--mode", "stream", "--date", "2026-01-02"]
            
            result = subprocess.run(cmd, capture_output=False, env=os.environ)
            
            if result.returncode == 0:
                print(f"✅ Run #{count} completed successfully")
            else:
                print(f"❌ Run #{count} failed with code {result.returncode}")
                
        except Exception as e:
            print(f"❌ Error executing pipeline: {e}")
            
        print(f"💤 Sleeping {interval_seconds}s...")
        time.sleep(interval_seconds)
        count += 1

if __name__ == "__main__":
    # Default to 60s (1 min) for this test to see frequent updates
    run_loop(interval_seconds=60)
