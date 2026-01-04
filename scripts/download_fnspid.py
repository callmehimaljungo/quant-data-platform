"""
Download FNSPID Dataset from Hugging Face
15.7 million financial news articles (1999-2023)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from config import BRONZE_DIR
import pandas as pd

print("=" * 80)
print("📚 DOWNLOADING FNSPID DATASET FROM HUGGING FACE")
print("=" * 80)
print(f"Start time: {datetime.now()}")
print()
print("Dataset: Zihan1004/FNSPID")
print("Size: 15.7 million financial news articles (1999-2023)")
print()

try:
    from datasets import load_dataset
    
    print("Loading dataset from Hugging Face...")
    print("(This may take a while - downloading several GB of data)")
    print()
    
    # Load FNSPID dataset
    # Note: This is a large dataset, we'll load a subset first
    dataset = load_dataset("Zihan1004/FNSPID", split="train", streaming=True)
    
    # Collect sample (first 100,000 rows to start)
    print("Collecting first 100,000 articles...")
    records = []
    for i, item in enumerate(dataset):
        if i >= 100000:
            break
        records.append({
            'news_id': i,
            'ticker': item.get('ticker'),
            'title': item.get('title'),
            'content': item.get('content'),
            'published_at': item.get('date'),
            'source': 'FNSPID',
            'sentiment': item.get('sentiment')
        })
        if (i + 1) % 10000 == 0:
            print(f"   Processed {i + 1:,} articles...")
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    df['fetched_at'] = datetime.now()
    
    # Save to Bronze
    output_dir = BRONZE_DIR / 'fnspid_news_lakehouse'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'fnspid_sample_100k.parquet'
    df.to_parquet(output_file, index=False)
    
    print()
    print("=" * 80)
    print("✅ FNSPID DOWNLOAD COMPLETE!")
    print("=" * 80)
    print(f"Records: {len(df):,}")
    print(f"Date Range: {df['published_at'].min()} to {df['published_at'].max()}")
    print(f"Output: {output_file}")
    print()
    print("Note: This is a sample. Full dataset has 15.7 million articles.")
    print("To download full dataset, run again with streaming=False (requires ~10GB RAM)")
    
except ImportError as e:
    print(f"❌ Missing library: {e}")
    print("Run: pip install datasets huggingface_hub")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
