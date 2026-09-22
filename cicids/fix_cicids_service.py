#!/usr/bin/env python3
"""
Test whether CICIDS2017's weak result was caused by using raw destination_port
(high cardinality, noisy) as the 'service' signal instead of a named service
category (low cardinality, like UNSW-NB15's real 'service' field).
Rebuilds the KG's service-based edges using a port->named-service mapping,
keeping IP/protocol/temporal edges identical, then reruns the same eval.
"""
import json
import time
import numpy as np
from collections import defaultdict

PORT_TO_SERVICE = {
    80: 'http', 8080: 'http', 8000: 'http',
    443: 'https', 8443: 'https',
    22: 'ssh',
    21: 'ftp', 20: 'ftp-data',
    25: 'smtp', 587: 'smtp',
    53: 'dns',
    110: 'pop3', 995: 'pop3s',
    143: 'imap', 993: 'imaps',
    23: 'telnet',
    445: 'smb', 139: 'netbios', 137: 'netbios', 138: 'netbios',
    135: 'rpc',
    3389: 'rdp',
    3306: 'mysql',
    5432: 'postgresql',
    123: 'ntp',
    161: 'snmp',
    389: 'ldap',
}

def port_to_service(port_str):
    try:
        port = int(port_str)
    except ValueError:
        return 'other'
    return PORT_TO_SERVICE.get(port, 'other')

print("Loading cached CICIDS2017 alerts...")
with open('cicids_alerts.json') as f:
    data = json.load(f)

alerts = data['alerts']
print(f"Loaded {len(alerts)} alerts")

service_counts = defaultdict(int)
for a in alerts:
    a['service'] = port_to_service(a['service'])
    service_counts[a['service']] += 1

print("Named service distribution after remapping:")
for svc, cnt in sorted(service_counts.items(), key=lambda x: -x[1]):
    print(f"  {svc}: {cnt}")

src_index = defaultdict(list)
dst_index = defaultdict(list)
proto_index = defaultdict(list)
service_index = defaultdict(list)
for a in alerts:
    src_index[a['src']].append(a['id'])
    dst_index[a['dst']].append(a['id'])
    proto_index[a['proto']].append(a['id'])
    service_index[a['service']].append(a['id'])

by_time = sorted(alerts, key=lambda a: a['stime'])
TIME_WINDOW = 60.0
FANOUT_CAP = 15

edge_weight = defaultdict(float)

def add_group_edges(index_dict, weight):
    for key, ids in index_dict.items():
        if len(ids) < 2:
            continue
        for i in range(len(ids)):
            for j in range(i+1, min(i+1+FANOUT_CAP, len(ids))):
                a, b = ids[i], ids[j]
                pair = (min(a,b), max(a,b))
                edge_weight[pair] += weight

add_group_edges(src_index, 0.3)
add_group_edges(dst_index, 0.3)
add_group_edges(proto_index, 0.15)
add_group_edges(service_index, 0.25)

for i in range(len(by_time)):
    for j in range(i+1, min(i+1+FANOUT_CAP, len(by_time))):
        if by_time[j]['stime'] - by_time[i]['stime'] > TIME_WINDOW:
            break
        a, b = by_time[i]['id'], by_time[j]['id']
        pair = (min(a,b), max(a,b))
        edge_weight[pair] += 0.3

edges = []
for (a, b), w in edge_weight.items():
    edges.append({'src': a, 'dst': b, 'weight': w})
    edges.append({'src': b, 'dst': a, 'weight': w})

print(f"\nRebuilt KG: {len(alerts)} nodes, {len(edges)} edges")

with open('cicids_alerts_v2.json', 'w') as f:
    json.dump({'alerts': alerts, 'kg': {'nodes': [a['id'] for a in alerts], 'edges': edges}}, f)
print("Saved cicids_alerts_v2.json")

adj = defaultdict(list)
for edge in edges:
    adj[edge['src']].append((edge['dst'], edge['weight']))
for src in adj:
    adj[src].sort(key=lambda x: -x[1])

attack_type = {a['id']: a['attack_type'] for a in alerts}
type_groups = defaultdict(set)
for a in alerts:
    if a['attack_type'] != 'BENIGN':
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

all_attack_ids = [a['id'] for a in alerts if a['attack_type'] != 'BENIGN'
                  and len(type_groups[a['attack_type']]) >= 2]
np.random.seed(42)
test_queries = list(np.random.choice(all_attack_ids, size=min(500, len(all_attack_ids)), replace=False))
print(f"\nTesting on {len(test_queries)} attack-labeled query alerts")

def precision_at_k(scored_dict, gt_ids, query_id, k):
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
print("BASELINE (full BFS) - CICIDS2017 v2 (named services)")
print("="*70)
m, h = ci95(base_prec)
print(f"  Precision: {m*100:.1f}% +/- {h*100:.1f}")
m, h = ci95(base_rec)
print(f"  Recall:    {m*100:.1f}% +/- {h*100:.1f}")
for k in ks:
    m, h = ci95(base_p_at_k[k])
    print(f"  Precision@{k}: {m*100:.1f}% +/- {h*100:.1f}")

alphas = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
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
    print(f"alpha={alpha}  (speedup: {speedup:.1f}x)")
    print("="*70)
    m, h = ci95(sel_prec)
    print(f"  Precision: {m*100:.1f}% +/- {h*100:.1f}")
    m, h = ci95(sel_rec)
    print(f"  Recall:    {m*100:.1f}% +/- {h*100:.1f}")
    for k in ks:
        m, h = ci95(sel_p_at_k[k])
        print(f"  Precision@{k}: {m*100:.1f}% +/- {h*100:.1f}")
