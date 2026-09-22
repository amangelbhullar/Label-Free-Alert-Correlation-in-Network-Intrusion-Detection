#!/usr/bin/env python3
"""
Build enriched KG for CICIDS2017, mirroring build_unsw_kg_v3.py's method exactly:
shared src IP, shared dst IP, shared "service" (here: destination_port as proxy),
shared protocol, temporal proximity -- weighted, accumulated edges.
Run from /storage/bhull113/alert-correlation-kg/
"""
import pandas as pd
import json
import numpy as np
from collections import defaultdict
from datetime import datetime

print("Loading CICIDS2017 Network-Flows parquet...")
df = pd.read_parquet('CIC-IDS2017/Network-Flows/CICIDS_Flow.parquet')
print(f"Full shape: {df.shape}")

df = df[['source_ip', 'destination_ip', 'source_port', 'destination_port',
         'protocol', 'Timestamp', 'attack_label']].copy()

df = df.sample(n=min(5000, len(df)), random_state=42).reset_index(drop=True)
print(f"Sampled to {len(df)}")
print("Attack label distribution:")
print(df['attack_label'].value_counts())

def parse_ts(s):
    try:
        return datetime.strptime(str(s).strip(), '%d/%m/%Y %H:%M:%S').timestamp()
    except Exception:
        try:
            return datetime.strptime(str(s).strip(), '%m/%d/%Y %H:%M:%S').timestamp()
        except Exception:
            return 0.0

df['stime'] = df['Timestamp'].apply(parse_ts)

alerts = []
for idx, row in df.iterrows():
    alerts.append({
        'id': idx,
        'src': str(row['source_ip']),
        'dst': str(row['destination_ip']),
        'proto': str(row['protocol']),
        'service': str(row['destination_port']),
        'stime': float(row['stime']),
        'attack_type': str(row['attack_label'])
    })

print("Building enriched KG edges (shared IP + proto/port + temporal proximity)...")

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

edge_weight = defaultdict(float)
FANOUT_CAP = 15

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

print("Adding temporal proximity edges...")
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

nodes = [a['id'] for a in alerts]
print(f"Built enriched KG: {len(nodes)} nodes, {len(edges)} edges")
if edges:
    print(f"Weight range: min={min(e['weight'] for e in edges):.2f}, max={max(e['weight'] for e in edges):.2f}")

output = {'alerts': alerts, 'kg': {'nodes': nodes, 'edges': edges}}
with open('cicids_alerts.json', 'w') as f:
    json.dump(output, f)
print("Saved cicids_alerts.json")
