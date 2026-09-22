#!/usr/bin/env python3
"""
Trivial non-graph baseline: score candidates purely by shared attributes
(proto match, service match, temporal proximity) - no graph traversal at all.
Tests whether the KG/selectivity machinery adds anything over simple filtering.
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

alert_by_id = {a['id']: a for a in alerts}
attack_type = {a['id']: a['attack_type'] for a in alerts}
type_groups = defaultdict(set)
for a in alerts:
    if a['attack_type'] != 'Normal':
        type_groups[a['attack_type']].add(a['id'])

# Also rebuild the selective-correlation machinery for direct comparison
adj = defaultdict(list)
for edge in kg['edges']:
    adj[edge['src']].append((edge['dst'], edge['weight']))
for src in adj:
    adj[src].sort(key=lambda x: -x[1])

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

# TRIVIAL BASELINE: score by shared attributes, no graph at all
TIME_WINDOW = 60.0

def trivial_score(query_id, candidate_id):
    q = alert_by_id[query_id]
    c = alert_by_id[candidate_id]
    score = 0.0
    if q['proto'] == c['proto']:
        score += 1.0
    if q['service'] == c['service']:
        score += 1.0
    if abs(q['stime'] - c['stime']) <= TIME_WINDOW:
        score += 1.0
    return score

def trivial_topk(query_id, all_ids, k):
    scored = []
    for cid in all_ids:
        if cid == query_id:
            continue
        s = trivial_score(query_id, cid)
        if s > 0:
            scored.append((cid, s))
    scored.sort(key=lambda x: -x[1])
    return [cid for cid, s in scored[:k]]

all_ids = [a['id'] for a in alerts]

all_attack_ids = [a['id'] for a in alerts if a['attack_type'] != 'Normal'
                  and len(type_groups[a['attack_type']]) >= 2]
np.random.seed(42)
test_queries = list(np.random.choice(all_attack_ids, size=min(500, len(all_attack_ids)), replace=False))
print(f"Testing on {len(test_queries)} attack-labeled query alerts")

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

# Trivial baseline timing + precision@k
trivial_times = []
trivial_p_at_k = {k: [] for k in ks}
for q in test_queries:
    t0 = time.time()
    top20 = trivial_topk(q, all_ids, max(ks))
    trivial_times.append(time.time() - t0)
    gt = type_groups[attack_type[q]]
    for k in ks:
        trivial_p_at_k[k].append(precision_at_k_list(top20, gt, q, k))

print("\n" + "="*70)
print("TRIVIAL NON-GRAPH BASELINE (attribute matching only, no traversal)")
print("="*70)
for k in ks:
    m, h = ci95(trivial_p_at_k[k])
    print(f"  Precision@{k}: {m*100:.1f}% ± {h*100:.1f}")
print(f"  Avg time/query: {np.mean(trivial_times)*1000:.2f} ms")

# Selective KG method for direct comparison (best-performing alphas from before)
for alpha in [0.1, 0.2]:
    sel_times = []
    sel_p_at_k = {k: [] for k in ks}
    for q in test_queries:
        t0 = time.time()
        corr = selective_correlation(q, alpha)
        sel_times.append(time.time() - t0)
        gt = type_groups[attack_type[q]]
        ranked = sorted([(v, s) for v, s in corr.items() if v != q], key=lambda x: -x[1])
        top_ids = [v for v, s in ranked]
        for k in ks:
            sel_p_at_k[k].append(precision_at_k_list(top_ids, gt, q, k))

    print("\n" + "="*70)
    print(f"SELECTIVE KG METHOD (α={alpha})")
    print("="*70)
    for k in ks:
        m, h = ci95(sel_p_at_k[k])
        print(f"  Precision@{k}: {m*100:.1f}% ± {h*100:.1f}")
    print(f"  Avg time/query: {np.mean(sel_times)*1000:.2f} ms")

print("\n" + "="*70)
print(f"Trivial baseline avg time: {np.mean(trivial_times)*1000:.2f} ms/query "
      f"(scans all {len(all_ids)} alerts per query, O(N))")
