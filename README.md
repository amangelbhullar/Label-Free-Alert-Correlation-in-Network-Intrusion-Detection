# Structural Selectivity for Efficient, Label-Free Alert Correlation in Network Intrusion Detection

Code accompanying the AI4DEMONS'26 (CIKM 2026 workshop) short research statement by
Amangel (Daisy) Bhullar and Ziad Kobti, University of Windsor.

## Overview

This repository builds alert-correlation knowledge graphs from real network intrusion
datasets and evaluates a simple, unsupervised selective correlation procedure
(top-alpha weighted-neighbor pruning during BFS) against full-graph correlation, a
trivial attribute-matching baseline, and a supervised Random Forest baseline, on an
Attack-Type Recovery task (retrieve alerts sharing a query alert's true attack
category).

## Repository structure

    unsw/       KG construction + evaluation for UNSW-NB15 (the exact scripts used for the paper's numbers)
    cicids/     KG construction (raw-port and named-service variants) + evaluation for CICIDS2017
    toniot/     Notes for reconstructing the TON_IoT pipeline (see toniot/README.md)
    figures/    Script to regenerate the precision/speedup trade-off figure
    botiot_exploratory/   Exploratory-only: BoT-IoT KG + eval (NOT included in the paper's
                          main results; the trivial baseline saturates at 100% precision
                          on this dataset due to extreme class imbalance and low feature
                          diversity in the 5,000-row sample, so it cannot discriminate
                          between methods -- kept here for transparency/reproducibility only)

## Method summary

KG construction: nodes are alerts; edge weight accumulates additively from four
signals: shared source IP (0.3), shared destination IP (0.3), shared protocol (0.15),
shared service (0.25), and temporal proximity within 60s (0.3). Each grouping signal's
fan-out is capped at 15 alerts to bound edge count.

Selective correlation: weighted BFS to 2 hops, edge weight decayed by 0.8 per hop;
at each node, only the top k = ceil(alpha * num_neighbors) highest-weight neighbors
are expanded, for alpha in 0.005/0.01/0.02/0.05/0.1/0.2. alpha = 1
recovers full (unselective) correlation.

Baselines: (1) trivial non-graph attribute matching (proto/service/time-window
overlap, no traversal); (2) a Random Forest over pairwise features, trained on
sampled query-alert pairs and scored against all candidates at inference time.

## Datasets

Three independent, real, labeled network intrusion datasets are used in the paper:

- UNSW-NB15 (Moustafa & Slay, 2015) -- native named service field.
- CICIDS2017 (Sharafaldin, Lashkari & Ghorbani, 2018) -- evaluated under both a
  raw-destination-port service signal and a port-to-named-service mapping (an ablation
  isolating the effect of categorical feature granularity).
- TON_IoT (Alsaedi, Moustafa, Tari, Mahmood & Anwar, 2020) -- IoT/IIoT traffic;
  the distribution used here lacks a timestamp field, so the KG omits the
  temporal-proximity edge type entirely (a robustness check on a missing structural
  signal).

Datasets are downloaded via huggingface_hub/nids-datasets (UNSW-NB15, CICIDS2017)
and a HuggingFace CSV mirror (TON_IoT); see each subdirectory's script for the exact
loading path. Raw dataset files are not redistributed here.

## Reproducing results

Each dataset directory contains a build_*_kg.py script (constructs the KG from the
raw dataset and saves it as JSON) and a test_*.py script (runs the full
Attack-Type Recovery evaluation). Run the build script first, then the eval script.

    pip install -r requirements.txt
    cd unsw && python3 build_unsw_kg_v3.py && python3 test_downstream_expanded.py

## License

MIT (see LICENSE).
