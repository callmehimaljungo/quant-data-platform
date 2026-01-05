"""
Realtime Scheduler - Runs Incremental Pipeline Every 15 Minutes

Manages periodic execution of the incremental pipeline with:
- Graceful shutdown on Ctrl+C
- Error handling and logging
- Configurable interval
"""

import sys
import time
import signal
import logging
from pathlib import Path
from datetime import datetime
import subprocess

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
INTERVAL_MINUTES = 15
PIPELINE_SCRIPT = PROJECT_ROOT / 'scripts' / 'incremental_pipeline.py'

# Shutdown flag
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.info("\n⚠ Shutdown signal received, stopping after current run...")
    shutdown_requested = True


def run_pipeline() -> bool:
    """
    Run the incremental pipeline script.
    
    Returns:
        True if pipeline succeeded, False otherwise
    """
    try:
        result = subprocess.run(
            [sys.executable, str(PIPELINE_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300  # 5 min timeout
        )
        
        # Log output
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(f"  {line}")
        
        if result.returncode == 0:
            logger.info("✓ Pipeline completed successfully")
            return True
        else:
            logger.error(f"✗ Pipeline failed with code {result.returncode}")
            if result.stderr:
                logger.error(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("✗ Pipeline timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"✗ Failed to run pipeline: {e}")
        return False


def main():
    """Main scheduler loop."""
    global shutdown_requested
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("=" * 70)
    logger.info("REALTIME SCHEDULER STARTED")
    logger.info(f"Interval: {INTERVAL_MINUTES} minutes")
    logger.info(f"Pipeline: {PIPELINE_SCRIPT}")
    logger.info("Press Ctrl+C to stop gracefully")
    logger.info("=" * 70)
    
    run_count = 0
    
    while not shutdown_requested:
        run_count += 1
        logger.info(f"\n[Run #{run_count}] Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run pipeline
        success = run_pipeline()
        
        if shutdown_requested:
            break
        
        # Wait for next interval
        logger.info(f"⏳ Waiting {INTERVAL_MINUTES} minutes until next run...")
        
        # Sleep in small chunks to allow quick shutdown
        sleep_seconds = INTERVAL_MINUTES * 60
        for _ in range(sleep_seconds):
            if shutdown_requested:
                break
            time.sleep(1)
    
    logger.info("\n" + "=" * 70)
    logger.info(f"SCHEDULER STOPPED | Total runs: {run_count}")
    logger.info("=" * 70)
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)
