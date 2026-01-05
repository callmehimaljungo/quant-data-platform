
# TODO List

## 🔴 Critical / High Priority

- [ ] **Fix Weekend Data Gap**: Bronze layer missing data for weekends. Need backfill strategy.
- [ ] **API Rate Limits**: Finnhub is limiting us. Add exponential backoff in `ingest.py`.
- [ ] **Data Validation**: `avg_daily_return` sometimes shows huge numbers (millions %). Need filter in pipeline.
- [ ] **Security**: Revoke old keys that were committed to git (oops).

## 🟡 Medium Priority

- [ ] **Dashboard Performance**: `app.py` is getting too big (>1000 lines). Load time is slow (~3s).
  - [ ] Split into pages
  - [ ] Optimize Parquet loading
- [ ] **Sector Analysis**: Add GICS Level 2/3 drill-down.
- [ ] **Mobile View**: Dashboard looks broken on phone (sidebar too wide).
- [ ] **Logs**: Add rotating log files, current print statements are messy.

## 🟢 Low Priority / Nice to Have

- [ ] Dark/Light mode toggle (Streamlit handles this but custom charts look weird in light mode).
- [ ] "Explain AI" tooltip for Causal Model results.
- [ ] Export to CSV button for all tables.
- [ ] **Refactor**: Remove `utils/r2_sync.py` duplication? (Double check dependencies).

## 🗑️ Tech Debt

- [ ] `risk_metrics.py` failing on new numpy version?
- [ ] Config is scattered between `.env` and `config.py`. Centralize it.
- [ ] Hardcoded paths in some test files (e:\GitHub...).
