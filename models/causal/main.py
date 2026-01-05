
"""
Main entry point for Causal Analysis Pipeline
Run with: python -m models.causal.main
"""
import logging
from models.causal.analyzer import CausalAnalyzer
from models.causal.data import load_unified_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Causal Analysis Pipeline")
    
    # 1. Load Data
    df = load_unified_data()
    if df.empty:
        logger.error("No data loaded. Exiting.")
        return
        
    # 2. Analyze
    analyzer = CausalAnalyzer(df)
    results = analyzer.run_full_analysis()
    
    # 3. Report
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(results)
    
    # 4. Save results (Placeholder)
    # save_to_r2(results)

if __name__ == "__main__":
    main()
