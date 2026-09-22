#!/usr/bin/env python3
"""
ML baseline: train a classifier to predict "same attack_type" from
pairwise features (proto match, service match, time gap, IP overlap).
Higher bar than the trivial heuristic since it learns optimal weights.
"""
import json
import time
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("Loading UNSW-NB15 data...")
with open('unsw_alerts.json') as f:
    data = json.load(f)

alerts = data['alerts']
alert_by_id = {a['id']: a for a in alerts}
attack_type = {a['id']: a['attack_type'] for a in alerts}
type_groups = defaultdict(set)
for a in alerts:
    if a['attack_type'] != 'Normal':
        type_groups[a['attack_type']].add(a['id'])

all_ids = [a['id'] for a in alerts]
all_attack_ids = [a['id'] for a in alerts if a['attack_type'] != 'Normal'
                  and len(type_groups[a['attack_type']]) >= 2]

def pair_features(qid, cid):
    q = alert_by_id[qid]
    c = alert_by_id[cid]
    return [
        1.0 if q['proto'] == c['proto'] else 0.0,
        1.0 if q['service'] == c['service'] else 0.0,
        1.0 if q['src'] == c['src'] else 0.0,
        1.0 if q['dst'] == c['dst'] else 0.0,
        1.0 if q['src'] == c['dst'] or q['dst'] == c['src'] else 0.0,
        abs(q['stime'] - c['stime']),
        min(abs(q['stime'] - c['stime']) / 60.0, 10.0),  # capped time-gap feature
    ]

# --- Build training set: sample pairs, label = same attack_type ---
print("Building training pairs...")
np.random.seed(0)
train_query_ids = np.random.choice(all_attack_ids, size=300, replace=False)
X, y = [], []
for qid in train_query_ids:
    gt = type_groups[attack_type[qid]] - {qid}
    # positive pairs (up to 10)
    pos = list(gt)[:10]
    for cid in pos:
        X.append(pair_features(qid, cid))
        y.append(1)
    # negative pairs (random non-matching, same count)
    neg_candidates = [i for i in all_ids if i != qid and i not in gt]
    neg = np.random.choice(neg_candidates, size=len(pos) if pos else 5, replace=False)
    for cid in neg:
        X.append(pair_features(qid, cid))
        y.append(0)

X, y = np.array(X), np.array(y)
print(f"Training set: {len(X)} pairs, {y.sum()} positive, {len(y)-y.sum()} negative")

clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
clf.fit(X, y)
print("Trained RandomForestClassifier")

# --- Evaluate on held-out query alerts (disjoint from training queries) ---
remaining_ids = [i for i in all_attack_ids if i not in set(train_query_ids)]
np.random.seed(42)
test_queries = list(np.random.choice(remaining_ids, size=min(500, len(remaining_ids)), replace=False))
print(f"\nTesting on {len(test_queries)} held-out query alerts")

def ml_topk(qid, all_ids, k):
    feats = [pair_features(qid, cid) for cid in all_ids if cid != qid]
    cand_ids = [cid for cid in all_ids if cid != qid]
    if not feats:
        return []
    probs = clf.predict_proba(np.array(feats))[:, 1]
    ranked = sorted(zip(cand_ids, probs), key=lambda x: -x[1])
    return [cid for cid, p in ranked[:k]]

def precision_at_k_list(top_ids, gt_ids, query_id, k):
    top = [i for i in top_ids if i != query_id][:k]
    if not top:
        return 0.0
    gt = gt_ids - {query_id}
    tp = len(set(top) & gt)
    return tp / len(top)

def ci95(arr):
    arr = np.array(arr)
    m = np.mean(arr)
    se = np.std(arr, ddof=1) / np.sqrt(len(arr))
    return m, 1.96 * se

ks = [5, 10, 20]
ml_p_at_k = {k: [] for k in ks}
ml_times = []

for q in test_queries:
    t0 = time.time()
    top20 = ml_topk(q, all_ids, max(ks))
    ml_times.append(time.time() - t0)
    gt = type_groups[attack_type[q]]
    for k in ks:
        ml_p_at_k[k].append(precision_at_k_list(top20, gt, q, k))

print("\n" + "="*70)
print("ML BASELINE (Random Forest on pairwise features)")
print("="*70)
for k in ks:
    m, h = ci95(ml_p_at_k[k])
    print(f"  Precision@{k}: {m*100:.1f}% ± {h*100:.1f}")
print(f"  Avg time/query: {np.mean(ml_times)*1000:.2f} ms (scores against all {len(all_ids)} candidates, O(N))")

print("\n" + "="*70)
print("REFERENCE — SELECTIVE KG METHOD (from previous run)")
print("="*70)
print("  α=0.1:  Precision@5=65.4%  @10=64.3%  @20=62.4%  time=0.03ms")
print("  α=0.2:  Precision@5=56.2%  @10=57.4%  @20=58.1%  time=0.07ms")
