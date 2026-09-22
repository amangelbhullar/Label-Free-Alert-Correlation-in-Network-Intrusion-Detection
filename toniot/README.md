# TON_IoT scripts

The exact TON_IoT KG-build and evaluation scripts were run via inline
heredocs during development and were not saved as standalone files.
Reconstruct following the same pattern as cicids/build_cicids_kg.py and
cicids/test_cicids_downstream.py, with two differences specific to
TON_IoT:

1. No temporal-proximity edges: the TON_IoT distribution used
   (codymlewis/TON_IoT_network on HuggingFace) has no timestamp field,
   so the KG uses only IP (src/dst) + protocol + service edges.
2. NORMAL_LABEL = 'normal' (lowercase) per this dataset's label column,
   rather than 'BENIGN' (CICIDS2017) or 'Normal' (UNSW-NB15/BoT-IoT).

Everything else (edge weights, FANOUT_CAP=15, selective_correlation with
alpha in 0.005/0.01/0.02/0.05/0.1/0.2, trivial baseline, ML
baseline with 5 features instead of 7 since the two timestamp-based
features are dropped) is identical to the CICIDS2017 pipeline.
