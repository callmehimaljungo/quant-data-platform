# Dev Notes / Scratchpad

## 2024-12-28: Ingestion Issues

- Finnhub API keeps timing out on batch requests.
- Switched to serial requests for now, but it's slow.
- Need to look into `aiohttp` for async fetching later.

## 2024-12-30: R2 Sync

- Access Key ID was wrong in `.env`, fixed it.
- Bucket policy was blocking uploads from non-US IP? No, just my VPN.
- Remember: `boto3` needs `endpoint_url` for Cloudflare R2.

## 2025-01-02: Dashboard

- Streamlit cache is awesome but memory usage is climbing.
- `ttl=300` seems okay for now.
- Users asking for "Realtime" - added scheduled task (15 mins).
- Need to fix the layout on 13" screens, charts get squished.

## 2025-01-05: Incident

- Dashboard crashed when running `realtime_scheduler.py`.
- Cause: Metrics calculation was overwriting historical data with empty DF.
- Fix: Implemented "Merge" logic.
- **Lesson**: Don't run risky scripts in prod without backup.

## Ideas for V2

- migrate to Next.js? Streamlit is getting limited for custom UI.
- Use DuckDB for everything instead of Pandas/Parquet?
- Add sentiment analysis from Reddit/Twitter?
