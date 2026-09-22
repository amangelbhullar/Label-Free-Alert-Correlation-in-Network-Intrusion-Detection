#!/usr/bin/env python3
"""
Expanded downstream task test: Attack-Type Recovery
- 500 queries (vs 100)
- Precision@k for k=5,10,20 (fixed-size shortlist, more realistic for triage)
- Mean + std + 95% CI for significance
"""
import json
import time
import numpy as np
from collections import defaultdict

print("Loading UNSW-NB15 data...")
with open('unsw_alerts.json') as f:
    data = json.load(f)

alerts = data['alerts']
kg = data['kg']
print(f"Loaded {len(alerts)} alerts, {len(kg['edges'])} edges")

adj = defaultdict(list)
for edge in kg['edges']:
    adj[edge['src']].append((edge['dst'], edge['weight']))
for src in adj:
    adj[src].sort(key=lambda x: -x[1])

attack_type = {a['id']: a['attack_type'] for a in alerts}
type_groups = defaultdict(set)
for a in alerts:
    if a['attack_type'] != 'Normal':
        type_groups[a['attack_type']].add(a['id'])

def full_correlation(query_id, max_hops=2):
    if query_id not in adj:
        return {}
    correlations = {query_id: 1.0}
    visited = {query_id}
    queue = [(query_id, 0)]
    while queue:
        u, hop = queue.pop(0)
        if hop >= max_hops or u not in adj:
            continue
        for v, w in adj[u]:
            if v not in visited:
                visited.add(v)
                correlations[v] = w * (0.8 ** hop)
                queue.append((v, hop + 1))
    return correlations

def selective_correlation(query_id, alpha, max_hops=2):
    if query_id not in adj:
        return {}
    correlations = {query_id: 1.0}
    visited = {query_id}
    queue = [(query_id, 0)]
    while queue:
        u, hop = queue.pop(0)
        if hop >= max_hops or u not in adj:
            continue
        neighbors_u = adj[u]
        k_u = max(1, int(np.ceil(alpha * len(neighbors_u))))
        for v, w in neighbors_u[:k_u]:
            if v not in visited:
                visited.add(v)
                correlations[v] = w * (0.8 ** hop)
                queue.append((v, hop + 1))
    return correlations

all_attack_ids = [a['id'] for a in alerts if a['attack_type'] != 'Normal'
                  and len(type_groups[a['attack_type']]) >= 2]
np.random.seed(42)
test_queries = list(np.random.choice(all_attack_ids, size=min(500, len(all_attack_ids)), replace=False))
print(f"Testing on {len(test_queries)} attack-labeled query alerts")

def precision_at_k(scored_dict, gt_ids, query_id, k):
    # scored_dict: {id: score}, ranked by score desc, excluding query itself
    ranked = sorted([(v, s) for v, s in scored_dict.items() if v != query_id],
                     key=lambda x: -x[1])
    top_k = [v for v, s in ranked[:k]]
    if not top_k:
        return 0.0
    gt = gt_ids - {query_id}
    tp = len(set(top_k) & gt)
    return tp / len(top_k)

def precision_recall(found_ids, gt_ids, query_id):
    found = set(found_ids) - {query_id}
    gt = gt_ids - {query_id}
    if not found or not gt:
        return 0.0, 0.0
    tp = len(found & gt)
    return tp / len(found), tp / len(gt)

def ci95(arr):
    arr = np.array(arr)
    m = np.mean(arr)
    se = np.std(arr, ddof=1) / np.sqrt(len(arr))
    return m, 1.96 * se

ks = [5, 10, 20]

# BASELINE
base_prec, base_rec, base_times = [], [], []
base_p_at_k = {k: [] for k in ks}
for q in test_queries:
    t0 = time.time()
    corr = full_correlation(q)
    base_times.append(time.time() - t0)
    gt = type_groups[attack_type[q]]
    p, r = precision_recall(corr.keys(), gt, q)
    base_prec.append(p)
    base_rec.append(r)
    for k in ks:
        base_p_at_k[k].append(precision_at_k(corr, gt, q, k))

print("\n" + "="*70)
print("BASELINE (full BFS)")
print("="*70)
m, h = ci95(base_prec)
print(f"  Precision: {m*100:.1f}% ± {h*100:.1f}  (95% CI)")
m, h = ci95(base_rec)
print(f"  Recall:    {m*100:.1f}% ± {h*100:.1f}")
for k in ks:
    m, h = ci95(base_p_at_k[k])
    print(f"  Precision@{k}: {m*100:.1f}% ± {h*100:.1f}")

alphas = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
results = {'baseline': {
    'precision': ci95(base_prec),
    'recall': ci95(base_rec),
    'precision_at_k': {k: ci95(base_p_at_k[k]) for k in ks}
}}

for alpha in alphas:
    sel_prec, sel_rec, sel_times = [], [], []
    sel_p_at_k = {k: [] for k in ks}
    for q in test_queries:
        t0 = time.time()
        corr = selective_correlation(q, alpha)
        sel_times.append(time.time() - t0)
        gt = type_groups[attack_type[q]]
        p, r = precision_recall(corr.keys(), gt, q)
        sel_prec.append(p)
        sel_rec.append(r)
        for k in ks:
            sel_p_at_k[k].append(precision_at_k(corr, gt, q, k))

    speedup = np.mean(base_times) / np.mean(sel_times)
    print("\n" + "="*70)
    print(f"α={alpha}  (speedup: {speedup:.1f}×)")
    print("="*70)
    m, h = ci95(sel_prec)
    print(f"  Precision: {m*100:.1f}% ± {h*100:.1f}")
    m, h = ci95(sel_rec)
    print(f"  Recall:    {m*100:.1f}% ± {h*100:.1f}")
    for k in ks:
        m, h = ci95(sel_p_at_k[k])
        print(f"  Precision@{k}: {m*100:.1f}% ± {h*100:.1f}")

    results[str(alpha)] = {
        'speedup': speedup,
        'precision': ci95(sel_prec),
        'recall': ci95(sel_rec),
        'precision_at_k': {k: ci95(sel_p_at_k[k]) for k in ks}
    }

# Serialize (convert tuples/np types to plain floats)
def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [float(obj[0]), float(obj[1])]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj

with open('attack_type_recovery_expanded.json', 'w') as f:
    json.dump(clean(results), f, indent=2)
print("\nSaved attack_type_recovery_expanded.json")
