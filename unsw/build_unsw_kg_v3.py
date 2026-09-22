#!/usr/bin/env python3
"""
Richer KG construction: combine shared-IP with proto/service match
and temporal proximity, weighted, to create more meaningful edges.
"""
import pandas as pd
import json
import glob
import numpy as np
from collections import defaultdict

COLUMNS = [
    'srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes',
    'sttl','dttl','sloss','dloss','service','Sload','Dload','Spkts','Dpkts',
    'swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len',
    'Sjit','Djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat',
    'is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd',
    'ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ltm','ct_src_dport_ltm',
    'ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat','Label'
]

print("Loading UNSW-NB15 dataset...")
csv_files = sorted(glob.glob('UNSW-NB15_[0-9].csv'))
dfs = [pd.read_csv(f, header=None, names=COLUMNS, low_memory=False) for f in csv_files]
df = pd.concat(dfs, ignore_index=True)

df['attack_cat'] = df['attack_cat'].fillna('Normal').astype(str).str.strip()
df.loc[df['attack_cat'].isin(['', 'nan']), 'attack_cat'] = 'Normal'

df = df.sample(n=min(5000, len(df)), random_state=42).reset_index(drop=True)
print(f"Sampled to {len(df)}")

alerts = []
for idx, row in df.iterrows():
    alerts.append({
        'id': idx,
        'src': str(row['srcip']),
        'dst': str(row['dstip']),
        'proto': str(row['proto']),
        'service': str(row['service']),
        'stime': float(row['Stime']) if pd.notna(row['Stime']) else 0.0,
        'attack_type': str(row['attack_cat'])
    })

print("Building enriched KG edges (shared IP + proto/service + temporal proximity)...")

src_index = defaultdict(list)
dst_index = defaultdict(list)
proto_index = defaultdict(list)
service_index = defaultdict(list)
for a in alerts:
    src_index[a['src']].append(a['id'])
    dst_index[a['dst']].append(a['id'])
    proto_index[a['proto']].append(a['id'])
    service_index[a['service']].append(a['id'])

# sort by stime for temporal proximity edges
by_time = sorted(alerts, key=lambda a: a['stime'])
TIME_WINDOW = 60.0  # seconds; alerts within this window get a temporal edge

edge_weight = defaultdict(float)  # (a,b) -> accumulated weight
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

# temporal proximity edges (sliding window over sorted-by-time list)
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
print(f"Weight range: min={min(e['weight'] for e in edges):.2f}, max={max(e['weight'] for e in edges):.2f}")

output = {'alerts': alerts, 'kg': {'nodes': nodes, 'edges': edges}}
with open('unsw_alerts.json', 'w') as f:
    json.dump(output, f)
print("Saved unsw_alerts.json (enriched)")
