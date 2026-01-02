"""
Airflow DAG: Quant Data Pipeline
Orchestrates Bronze → Silver → Gold data processing
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# Default arguments
default_args = {
    'owner': 'quant-platform',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# =============================================================================
# TASK FUNCTIONS
# =============================================================================

def fetch_stock_prices(**context):
    """Bronze: Fetch latest stock prices from yfinance"""
    import sys
    sys.path.insert(0, '/opt/airflow/project')
    
    from bronze.ingest import main as ingest_main
    result = ingest_main()
    return f"Ingested stock prices: {result}"


def fetch_economic_data(**context):
    """Bronze: Fetch economic indicators from FRED"""
    import sys
    sys.path.insert(0, '/opt/airflow/project')
    
    from bronze.economic_loader import main as economic_main
    result = economic_main()
    return f"Loaded economic data: {result}"


def fetch_news(**context):
    """Bronze: Fetch market news"""
    import sys
    sys.path.insert(0, '/opt/airflow/project')
    
    from bronze.news_loader import main as news_main
    result = news_main(num_articles=200, test=False)
    return f"Loaded news: {result}"


def clean_and_enrich(**context):
    """Silver: Clean and enrich stock data"""
    import sys
    sys.path.insert(0, '/opt/airflow/project')
    
    from silver.clean import main as clean_main
    result = clean_main()
    return f"Cleaned data: {result}"


def process_news(**context):
    """Silver: Process news sentiment"""
    import sys
    sys.path.insert(0, '/opt/airflow/project')
    
    from silver.process_news import main as process_news_main
    result = process_news_main()
    return f"Processed news: {result}"


def calculate_risk_metrics(**context):
    """Gold: Calculate risk metrics per ticker"""
    import sys
    sys.path.insert(0, '/opt/airflow/project')
    
    from gold.risk_metrics import main as risk_main
    result = risk_main()
    return f"Risk metrics calculated: {result}"


def run_strategies(**context):
    """Gold: Run investment strategies"""
    import sys
    sys.path.insert(0, '/opt/airflow/project')
    
    from gold.run_all_strategies import main as strategies_main
    result = strategies_main()
    return f"Strategies executed: {result}"


def upload_to_r2(**context):
    """Upload processed data to R2 cloud storage"""
    import sys
    sys.path.insert(0, '/opt/airflow/project')
    
    from upload_r2 import upload_data_to_r2
    upload_data_to_r2()
    return "Uploaded to R2"


# =============================================================================
# DAG DEFINITION
# =============================================================================

with DAG(
    dag_id='quant_data_pipeline',
    default_args=default_args,
    description='Daily data pipeline: Bronze → Silver → Gold',
    schedule_interval='0 8 * * *',  # Run at 8:00 AM daily
    start_date=days_ago(1),
    catchup=False,
    tags=['quant', 'data-pipeline', 'medallion'],
) as dag:
    
    # =========================================================================
    # BRONZE LAYER TASKS (Data Ingestion)
    # =========================================================================
    
    task_fetch_prices = PythonOperator(
        task_id='bronze_fetch_stock_prices',
        python_callable=fetch_stock_prices,
        doc='Fetch stock prices from yfinance',
    )
    
    task_fetch_economic = PythonOperator(
        task_id='bronze_fetch_economic',
        python_callable=fetch_economic_data,
        doc='Fetch economic data from FRED',
    )
    
    task_fetch_news = PythonOperator(
        task_id='bronze_fetch_news',
        python_callable=fetch_news,
        doc='Fetch market news',
    )
    
    # =========================================================================
    # SILVER LAYER TASKS (Data Processing)
    # =========================================================================
    
    task_clean_enrich = PythonOperator(
        task_id='silver_clean_and_enrich',
        python_callable=clean_and_enrich,
        doc='Clean data and add technical indicators',
    )
    
    task_process_news = PythonOperator(
        task_id='silver_process_news',
        python_callable=process_news,
        doc='Process news sentiment',
    )
    
    # =========================================================================
    # GOLD LAYER TASKS (Analytics)
    # =========================================================================
    
    task_risk_metrics = PythonOperator(
        task_id='gold_risk_metrics',
        python_callable=calculate_risk_metrics,
        doc='Calculate risk metrics per ticker',
    )
    
    task_strategies = PythonOperator(
        task_id='gold_run_strategies',
        python_callable=run_strategies,
        doc='Run investment strategies',
    )
    
    # =========================================================================
    # UPLOAD TO CLOUD
    # =========================================================================
    
    task_upload_r2 = PythonOperator(
        task_id='upload_to_r2',
        python_callable=upload_to_r2,
        doc='Upload processed data to Cloudflare R2',
    )
    
    # =========================================================================
    # TASK DEPENDENCIES (DAG Structure)
    # =========================================================================
    
    # Bronze layer (parallel ingestion)
    [task_fetch_prices, task_fetch_economic, task_fetch_news]
    
    # Bronze → Silver
    task_fetch_prices >> task_clean_enrich
    task_fetch_economic >> task_clean_enrich
    task_fetch_news >> task_process_news
    
    # Silver → Gold
    [task_clean_enrich, task_process_news] >> task_risk_metrics
    task_risk_metrics >> task_strategies
    
    # Gold → Upload
    task_strategies >> task_upload_r2
