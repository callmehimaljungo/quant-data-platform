"""
R2 Data Loader for Streamlit Dashboard
Load parquet data from Cloudflare R2 storage
"""

import os
import io
import pandas as pd
import streamlit as st

# Check if boto3 is available
try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


def get_r2_client():
    """Get R2 client using Streamlit secrets or env vars"""
    if not BOTO3_AVAILABLE:
        return None
    
    # Try Streamlit secrets first, then env vars
    try:
        endpoint = st.secrets.get("R2_ENDPOINT") or os.environ.get("R2_ENDPOINT")
        access_key = st.secrets.get("R2_ACCESS_KEY") or os.environ.get("R2_ACCESS_KEY")
        secret_key = st.secrets.get("R2_SECRET_KEY") or os.environ.get("R2_SECRET_KEY")
    except Exception:
        endpoint = os.environ.get("R2_ENDPOINT")
        access_key = os.environ.get("R2_ACCESS_KEY")
        secret_key = os.environ.get("R2_SECRET_KEY")
    
    if not all([endpoint, access_key, secret_key]):
        return None
    
    return boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version='s3v4')
    )


def get_bucket_name():
    """Get bucket name from secrets or env"""
    try:
        return st.secrets.get("R2_BUCKET") or os.environ.get("R2_BUCKET", "datn")
    except Exception:
        return os.environ.get("R2_BUCKET", "datn")


@st.cache_data(ttl=300)
def load_parquet_from_r2(r2_key: str) -> pd.DataFrame:
    """Load a parquet file from R2 and return as DataFrame"""
    client = get_r2_client()
    if client is None:
        return None
    
    bucket = get_bucket_name()
    
    try:
        response = client.get_object(Bucket=bucket, Key=r2_key)
        parquet_data = response['Body'].read()
        return pd.read_parquet(io.BytesIO(parquet_data))
    except ClientError as e:
        # st.warning(f"Could not load {r2_key}: {e}")
        return None
    except Exception as e:
        # st.warning(f"Error loading {r2_key}: {e}")
        return None


@st.cache_data(ttl=300)
def get_r2_object_last_modified(r2_key: str):
    """Get last modified timestamp of an R2 object"""
    client = get_r2_client()
    if client is None:
        return None
        
    bucket = get_bucket_name()
    try:
        response = client.head_object(Bucket=bucket, Key=r2_key)
        return response.get('LastModified')
    except Exception:
        return None


@st.cache_data(ttl=3600)
def list_r2_files(prefix: str) -> list:
    """List files in R2 bucket with given prefix"""
    client = get_r2_client()
    if client is None:
        return []
    
    bucket = get_bucket_name()
    
    try:
        response = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        files = []
        for obj in response.get('Contents', []):
            files.append(obj['Key'])
        return files
    except Exception:
        return []


@st.cache_data(ttl=3600)
def load_latest_from_lakehouse(lakehouse_prefix: str) -> pd.DataFrame:
    """Load the latest parquet file from a lakehouse folder in R2"""
    files = list_r2_files(lakehouse_prefix)
    parquet_files = [f for f in files if f.endswith('.parquet')]
    
    if not parquet_files:
        return None
    
    # Get latest file (sorted by name, which includes timestamp)
    latest_file = sorted(parquet_files)[-1]
    return load_parquet_from_r2(latest_file)


def is_r2_available() -> bool:
    """Check if R2 is configured and accessible"""
    client = get_r2_client()
    if client is None:
        return False
    
    try:
        bucket = get_bucket_name()
        client.head_bucket(Bucket=bucket)
        return True
    except Exception:
        return False
