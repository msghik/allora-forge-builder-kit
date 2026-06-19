# Graph Report - .  (2026-06-19)

## Corpus Check
- 53 files · ~66,957 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 609 nodes · 1090 edges · 39 communities (23 shown, 16 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 53 edges (avg confidence: 0.62)
- Token cost: 129,021 input · 22,768 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Worker Manager & Identity|Worker Manager & Identity]]
- [[_COMMUNITY_Base Data Manager (Storage)|Base Data Manager (Storage)]]
- [[_COMMUNITY_Web Dashboard & Worker Monitor|Web Dashboard & Worker Monitor]]
- [[_COMMUNITY_Binance Data Manager|Binance Data Manager]]
- [[_COMMUNITY_Performance Evaluator (Metrics)|Performance Evaluator (Metrics)]]
- [[_COMMUNITY_Topic Discovery & Evaluation|Topic Discovery & Evaluation]]
- [[_COMMUNITY_Data Manager Tests|Data Manager Tests]]
- [[_COMMUNITY_Agent Guide & README Docs|Agent Guide & README Docs]]
- [[_COMMUNITY_Model Creation Skills & Methodology|Model Creation Skills & Methodology]]
- [[_COMMUNITY_Bitcoin Walkthrough Examples|Bitcoin Walkthrough Examples]]
- [[_COMMUNITY_ML Workflow Tests|ML Workflow Tests]]
- [[_COMMUNITY_Evaluation Metric Tests|Evaluation Metric Tests]]
- [[_COMMUNITY_Atlas Data Manager|Atlas Data Manager]]
- [[_COMMUNITY_Data Manager Factory & Workflow|Data Manager Factory & Workflow]]
- [[_COMMUNITY_Polars Feature Extraction|Polars Feature Extraction]]
- [[_COMMUNITY_Worker Manager Tests|Worker Manager Tests]]
- [[_COMMUNITY_Atlas Dataset-ID Resolution Tests|Atlas Dataset-ID Resolution Tests]]
- [[_COMMUNITY_Worker Runtime|Worker Runtime]]
- [[_COMMUNITY_Atlas REST & Bulk Download|Atlas REST & Bulk Download]]
- [[_COMMUNITY_Feature Integrity Tests|Feature Integrity Tests]]
- [[_COMMUNITY_Dummy Remote DataManager|Dummy Remote DataManager]]
- [[_COMMUNITY_Live Features Consistency Tests|Live Features Consistency Tests]]
- [[_COMMUNITY_Target Integrity Tests|Target Integrity Tests]]
- [[_COMMUNITY_Resilient Chunked Backfill|Resilient Chunked Backfill]]
- [[_COMMUNITY_Base Features Visualization|Base Features Visualization]]
- [[_COMMUNITY_Feature Engineering Example|Feature Engineering Example]]
- [[_COMMUNITY_Pytest Configuration|Pytest Configuration]]
- [[_COMMUNITY_Worker Deployment Script|Worker Deployment Script]]
- [[_COMMUNITY_Live Prediction Streaming|Live Prediction Streaming]]
- [[_COMMUNITY_PyPI Release Workflow|PyPI Release Workflow]]
- [[_COMMUNITY_Atlas Default Directory Test|Atlas Default Directory Test]]
- [[_COMMUNITY_Workflow Explicit Manager Test|Workflow Explicit Manager Test]]
- [[_COMMUNITY_Workflow Invalid Manager Test|Workflow Invalid Manager Test]]
- [[_COMMUNITY_Atlas Live Snapshot Test|Atlas Live Snapshot Test]]
- [[_COMMUNITY_Atlas Live Window Coverage Test|Atlas Live Window Coverage Test]]
- [[_COMMUNITY_Binance Live Features E2E Test|Binance Live Features E2E Test]]
- [[_COMMUNITY_Allora Workflow E2E Test|Allora Workflow E2E Test]]
- [[_COMMUNITY_Graphify Project Instructions|Graphify Project Instructions]]
- [[_COMMUNITY_Project Root|Project Root]]

## God Nodes (most connected - your core abstractions)
1. `WorkerManager` - 65 edges
2. `AlloraMLWorkflow` - 46 edges
3. `AtlasDataManager` - 38 edges
4. `BaseDataManager` - 37 edges
5. `BinanceDataManager` - 35 edges
6. `WorkerMonitor` - 34 edges
7. `PerformanceEvaluator` - 23 edges
8. `AlloraTopicDiscovery` - 20 edges
9. `Nine Principles for Robust Financial Prediction` - 17 edges
10. `AlloraSDKEventFetcher` - 14 edges

## Surprising Connections (you probably didn't know these)
- `test_atlas_bulk_download_chunked_gives_up_at_min_chunk()` --calls--> `AtlasDataManager`  [EXTRACTED]
  tests/test_data_managers.py → allora_forge_builder_kit/atlas_data_manager.py
- `test_atlas_bulk_download_chunked_splits_on_failure()` --calls--> `AtlasDataManager`  [EXTRACTED]
  tests/test_data_managers.py → allora_forge_builder_kit/atlas_data_manager.py
- `TestResolveDatasetId` --uses--> `AtlasDataManager`  [INFERRED]
  tests/test_data_managers.py → allora_forge_builder_kit/atlas_data_manager.py
- `DummyRemoteOnlyDM` --uses--> `BaseDataManager`  [INFERRED]
  tests/test_get_live_features_remote_only.py → allora_forge_builder_kit/base_data_manager.py
- `DataFrame` --uses--> `BaseDataManager`  [INFERRED]
  tests/test_get_live_features_remote_only.py → allora_forge_builder_kit/base_data_manager.py

## Import Cycles
- 1-file cycle: `allora_forge_builder_kit/atlas_data_manager.py -> allora_forge_builder_kit/atlas_data_manager.py`
- 1-file cycle: `allora_forge_builder_kit/base_data_manager.py -> allora_forge_builder_kit/base_data_manager.py`
- 1-file cycle: `allora_forge_builder_kit/binance_data_manager.py -> allora_forge_builder_kit/binance_data_manager.py`
- 1-file cycle: `allora_forge_builder_kit/worker_monitor.py -> allora_forge_builder_kit/worker_monitor.py`

## Hyperedges (group relationships)
- **Three Forge Model Creation Skills (Diverse Entry Points)** — hypothesis_driven_skill, signal_discovery_skill, robustness_first_skill, methodology_nine_principles [EXTRACTED 0.95]
- **Three-Stage Validation Flow (Optimize, Evaluate, Deploy)** — methodology_three_stage_separation, validation_framework_purged_walk_forward_cv, validation_framework_quality_gates, validation_framework_deployment_validation [EXTRACTED 0.95]
- **Agent Task Routing to Three Skill Packages** — skills_task_router, allora_data_exploration_skill, allora_model_builder_skill, allora_worker_manager_skill [EXTRACTED 0.95]

## Communities (39 total, 16 thin omitted)

### Community 0 - "Worker Manager & Identity"
Cohesion: 0.06
Nodes (21): Identity, Any, Path, Return last N stdout log lines for a worker slot., Refresh all worker topic descriptions from resolver, when configured., Attach monitor and optionally bootstrap existing workers into monitoring., Lightweight local worker registry + lifecycle manager.      Designed for one w, Write or copy a mnemonic into worker_keys/<alias>.key with mode 0o600. (+13 more)

### Community 1 - "Base Data Manager (Storage)"
Cohesion: 0.07
Nodes (26): ABC, BaseDataManager, day_str(), from_ms(), DataFrame, datetime, Get path for partitioned Parquet file., Append a bar to Parquet storage and cache.         Shared implementation for al (+18 more)

### Community 2 - "Web Dashboard & Worker Monitor"
Cohesion: 0.08
Nodes (19): DashboardApp, main(), make_handler(), EventFetcherProtocol, _extract_nonce(), MonitorTarget, _parse_dt(), _parse_json() (+11 more)

### Community 3 - "Binance Data Manager"
Cohesion: 0.08
Nodes (23): Convert datetime to milliseconds timestamp., to_ms(), BinanceDataManager, DataFrame, datetime, Parse Binance kline array into standardized bar format.                  Binan, Backfill historical data for a single symbol via REST API., Fill only from the last stored bar forward (overwrite last bar). (+15 more)

### Community 4 - "Performance Evaluator (Metrics)"
Cohesion: 0.10
Nodes (19): PerformanceEvaluator, Calculate Pearson correlation and related metrics.                  Args:, Calculate Weighted RMSE improvement vs. zero-returns baseline., Calculate Cumulative Z-scored Absolute Return (CZAR) improvement.          Rep, Comprehensive performance metrics calculator for financial time-series predictio, Calculate Z-transformed Power-Tanh Absolute Error improvement., Calculate log aspect ratio: log10(std(predicted) / std(actual))., Calculate naive annualized return from a simple trading strategy. (+11 more)

### Community 5 - "Topic Discovery & Evaluation"
Cohesion: 0.10
Nodes (16): AlloraTopicDiscovery, Any, Filter topics that predict log returns., Extract a TopicInfo from the SDK topic object., Execute an async coroutine from synchronous code.      Handles Jupyter/IPython, Parsed metadata for a single Allora topic., Discover and inspect Allora Network topics.      Uses the allora_sdk ``AlloraA, Return every active topic on the network (cached after first call). (+8 more)

### Community 6 - "Data Manager Tests"
Cohesion: 0.06
Nodes (31): allora_api_key(), integration_check(), Comprehensive tests for DataManager implementations and Workflow integration., Test BinanceDataManager initialization., Test Binance kline parsing to standardized format., Test AtlasDataManager initialization., Test that different sources use different directories., Binance test symbols (small set for speed). (+23 more)

### Community 7 - "Agent Guide & README Docs"
Cohesion: 0.09
Nodes (31): File-Based Key Management Security Fix, Agent Session Notes (PR #14 Review), API Key as Human-Confirmed Input, Base Feature Schema (Normalized OHLCV Ratios), Agent Operating Guide, Topic Prediction Format Correctness Rule, Canonical Starter Flows (Notebook, Python API, Worker Ops), AtlasDataManager (+23 more)

### Community 8 - "Model Creation Skills & Methodology"
Cohesion: 0.12
Nodes (31): Allora Forge Model Creation Skills Bundle, Diversity by Design Rationale, Standard Pipeline Artifacts, Feature Engineering Guide, Estimation Goals as Organizing Principle, Common Leakage Patterns, Lookahead Prevention by Construction, forge-hypothesis-driven Skill (+23 more)

### Community 9 - "Bitcoin Walkthrough Examples"
Cohesion: 0.11
Nodes (20): get_api_key(), Resolve the Allora API key.      Resolution order:       1. ``ALLORA_API_KEY`, engineer_returns(), predict(), Add log return features over multiple horizons (no data leakage - same row only), # NOTE: Base features are already normalized (z-scored) by the workflow, Predict Bitcoin price 24 hours into the future.          Args:         nonce:, Convert numpy/pandas objects into JSON-serializable Python types. (+12 more)

### Community 10 - "ML Workflow Tests"
Cohesion: 0.10
Nodes (19): AlloraMLWorkflow, Backfill OHLCV data for all workflow tickers., Load raw OHLCV data for all workflow tickers as a Pandas DataFrame., Test AlloraMLWorkflow with Binance using string API., Test AlloraMLWorkflow with Allora using string API (now Atlas backend)., Test full workflow with Binance: backfill + feature extraction., Test workflow get_live_features with Binance., Test full workflow with Allora: backfill + feature extraction. (+11 more)

### Community 12 - "Atlas Data Manager"
Cohesion: 0.15
Nodes (8): AtlasDataManager, Convert interval string (e.g. '5m', '1h', '1d') to minutes., Resolve a ticker symbol to an Atlas dataset ID (cached)., Atlas data service manager for Allora Network.      Uses the Atlas timeseries, List datasets available on Atlas for a given source and frequency.          Us, Free-text search across Atlas dataset names and descriptions., Acquire the 'public' tag if not already held.          The Atlas UI does this, Series

### Community 13 - "Data Manager Factory & Workflow"
Cohesion: 0.12
Nodes (16): _check_unknown_kwargs(), DataManager(), list_data_sources(), List all available data sources and their parameters., Factory function to create data managers from a source string.      Args:, _extract_features_numba(), JIT-compiled core loop for feature extraction.     Data is already resampled, s, High-level ML workflow built on top of DataManager.                  Args: (+8 more)

### Community 14 - "Polars Feature Extraction"
Cohesion: 0.20
Nodes (7): DataFrame, Convert interval string (e.g., '5m', '1h', '15m') to bars per hour., Get live features for a single ticker (data-source agnostic)., Shared function to extract features from 1-minute bars.                  This, General OHLCV resampling for polars DataFrames.                  Args:, Compute log return to future close for polars DataFrame.                  Args, Numba-optimized version of feature extraction (polars).         Extracts featur

### Community 15 - "Worker Manager Tests"
Cohesion: 0.40
Nodes (12): WorkerSpec, _new_manager(), Path, WorkerManager, test_add_worker_enforces_unique_topic_address(), test_attach_monitor_bootstraps_existing_workers(), test_conflict_without_replace_auto_assigns_alternate_address(), test_deploy_without_address_creates_new_when_all_used_for_topic() (+4 more)

### Community 17 - "Worker Runtime"
Cohesion: 0.36
Nodes (8): _build_network(), _load_api_key(), main(), _run(), AlloraNetworkConfig, test_build_network_mainnet_does_not_set_faucet_url(), test_build_network_no_faucet_clears_url(), test_build_network_testnet_overrides_faucet_url()

### Community 18 - "Atlas REST & Bulk Download"
Cohesion: 0.31
Nodes (5): DataFrame, Fetch rows from Atlas via paginated REST calls., Efficient bulk download using Atlas streaming endpoint., Convert Atlas row objects into a flat DataFrame., Fetch the most recent 1-minute bars available for a symbol.          Atlas dat

### Community 19 - "Feature Integrity Tests"
Cohesion: 0.22
Nodes (7): Test to verify feature extraction integrity.  This test ensures that extracted, Test feature integrity with Allora 1-hour bars - validates ALL features., Test feature integrity across multiple assets - validates ALL features., Test feature integrity with Allora 5-minute bars - validates ALL features., test_feature_integrity_allora_1hour(), test_feature_integrity_allora_5min(), test_feature_integrity_multi_asset()

### Community 21 - "Live Features Consistency Tests"
Cohesion: 0.22
Nodes (7): Test to verify live features match historical features for the same timestamp., Test that live features match historical features for 1-hour bars., Test live vs historical features across multiple assets., Test that live features match historical features for 5-minute bars., test_live_features_multi_asset(), test_live_vs_historical_features_1hour(), test_live_vs_historical_features_5min()

### Community 22 - "Target Integrity Tests"
Cohesion: 0.22
Nodes (7): Test to verify target values are correctly computed.  This ensures that the ta, Test that targets match manually calculated log returns for 1-hour bars., Test target integrity across multiple assets., Test that targets match manually calculated log returns for 5-min bars., test_target_integrity_1hour(), test_target_integrity_5min(), test_target_integrity_multi_asset()

### Community 24 - "Base Features Visualization"
Cohesion: 0.53
Nodes (6): Hourly Candlesticks + SMA Overlay (Panel 1), Base Features Visualization (4-Panel Financial Chart), Time-Series Feature Engineering (Concept), MACD Indicator (12, 26, 9) (Panel 4), Bar-to-Bar Returns (Panel 3), Volume Profile (Panel 2)

### Community 25 - "Feature Engineering Example"
Cohesion: 0.50
Nodes (4): calculate_ema(), engineer_features(), Calculate Exponential Moving Average, Engineer features from base OHLCV features.     All features are functions of t

### Community 26 - "Pytest Configuration"
Cohesion: 0.50
Nodes (3): pytest_configure(), Pytest configuration for Allora Forge Builder Kit tests.  This file automatica, Called before test run starts.     Auto-loads ALLORA_API_KEY from .allora_api_k

## Knowledge Gaps
- **18 isolated node(s):** `Path`, `AlloraNetworkConfig`, `allora-forge-builder-kit`, `Release to PyPI GitHub Workflow`, `PyPI Trusted Publishing (OIDC)` (+13 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **16 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AlloraMLWorkflow` connect `ML Workflow Tests` to `Workflow Invalid Manager Test`, `Base Data Manager (Storage)`, `Binance Live Features E2E Test`, `Allora Workflow E2E Test`, `Topic Discovery & Evaluation`, `Data Manager Tests`, `Bitcoin Walkthrough Examples`, `Data Manager Factory & Workflow`, `Polars Feature Extraction`, `Feature Integrity Tests`, `Dummy Remote DataManager`, `Live Features Consistency Tests`, `Target Integrity Tests`, `Feature Engineering Example`, `Live Prediction Streaming`, `Workflow Explicit Manager Test`?**
  _High betweenness centrality (0.208) - this node is a cross-community bridge._
- **Why does `WorkerManager` connect `Worker Manager & Identity` to `Web Dashboard & Worker Monitor`, `Topic Discovery & Evaluation`, `Worker Manager Tests`?**
  _High betweenness centrality (0.168) - this node is a cross-community bridge._
- **Why does `PerformanceEvaluator` connect `Performance Evaluator (Metrics)` to `Bitcoin Walkthrough Examples`, `Evaluation Metric Tests`, `Topic Discovery & Evaluation`?**
  _High betweenness centrality (0.150) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `WorkerManager` (e.g. with `DashboardApp` and `AlloraTopicDiscovery`) actually correct?**
  _`WorkerManager` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `AlloraMLWorkflow` (e.g. with `BaseDataManager` and `DummyRemoteOnlyDM`) actually correct?**
  _`AlloraMLWorkflow` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `AtlasDataManager` (e.g. with `BaseDataManager` and `TestResolveDatasetId`) actually correct?**
  _`AtlasDataManager` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `BaseDataManager` (e.g. with `AtlasDataManager` and `DataFrame`) actually correct?**
  _`BaseDataManager` has 11 INFERRED edges - model-reasoned connections that need verification._