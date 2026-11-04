"""Standalone Q9 re-runner. Loads cached embeddings.npz, rebuilds the
heterograph + feature matrices, then runs the updated Q9 logic (now with
Kernel PCA via the connected-user mapping). Saves the new neighborhoods.png
and prints the cosine-similarity tables (also patched into hw2.ipynb as the
Q9 cell output)."""
import os, sys, time, warnings, json
warnings.filterwarnings("ignore")
os.chdir("/home/wertheizer/structeredML/HW2")

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import HeteroData
from relbench.datasets import get_dataset
from relbench.tasks import get_task

VAL_TIMESTAMP = None  # set below
EMB_DIM = 128

# ---------------- 1. Load data & feature matrices (fast) ----------------
print("Loading data...")
dataset = get_dataset("rel-stack", download=False)
db = dataset.get_db()
task = get_task("rel-stack", "user-engagement", download=False)
train_df = task.get_table("train").df.copy()
val_df   = task.get_table("val").df.copy()
VAL_TIMESTAMP = pd.Timestamp(dataset.val_timestamp)
print(f"VAL_TIMESTAMP={VAL_TIMESTAMP} | train={train_df.shape} val={val_df.shape}")

# We don't need to recompute auto_features — we only need OwnerUserId from train/val
# rows in their original order, which we already have. Skip auto_features entirely.

# ---------------- 2. Rebuild the heterograph (same as hw2.py) ----------------
print("Building heterograph...")
t0 = time.time()

def build_hetero(db, cutoff_ts):
    hd = HeteroData()
    id_to_idx = {}
    for name, t in db.table_dict.items():
        df = t.df
        if t.time_col is not None and t.time_col in df.columns:
            df = df[df[t.time_col].isna() | (df[t.time_col] <= cutoff_ts)]
        pk = t.pkey_col
        if pk is None:
            id_to_idx[name] = (df, dict(enumerate(np.arange(len(df)))), None)
        else:
            ids = df[pk].values
            id_to_idx[name] = (df, {int(v): i for i, v in enumerate(ids)}, pk)
        hd[name].num_nodes = len(df)
    for child, t in db.table_dict.items():
        df_child, child_map, _ = id_to_idx[child]
        for fk_col, parent in t.fkey_col_to_pkey_table.items():
            df_parent, parent_map, _ = id_to_idx[parent]
            if fk_col not in df_child.columns: continue
            fk_vals = df_child[fk_col].values
            src_local = np.arange(len(df_child))
            mask = ~pd.isna(fk_vals)
            src_local = src_local[mask]
            tgt_ids = fk_vals[mask]
            tgt_local = np.fromiter(
                (parent_map.get(int(v), -1) for v in tgt_ids),
                dtype=np.int64, count=len(tgt_ids))
            keep = tgt_local >= 0
            src_local = src_local[keep].astype(np.int64)
            tgt_local = tgt_local[keep]
            if len(src_local) == 0: continue
            edge_index = torch.tensor(np.stack([src_local, tgt_local]), dtype=torch.long)
            rel = f"{child}__{fk_col}__{parent}"
            hd[child, rel, parent].edge_index = edge_index
    hd._id_to_idx = {k: v[1] for k, v in id_to_idx.items()}
    hd._dfs = {k: v[0] for k, v in id_to_idx.items()}
    return hd

hetero = build_hetero(db, VAL_TIMESTAMP)
homo = hetero.to_homogeneous()
ei = homo.edge_index
N = homo.num_nodes
src = ei[0].numpy(); dst = ei[1].numpy()
lo = np.minimum(src, dst); hi = np.maximum(src, dst)
mask = lo != hi
lo, hi = lo[mask], hi[mask]
und_pairs = np.unique(np.stack([lo, hi], axis=1), axis=0)
print(f"Heterograph built in {time.time()-t0:.1f}s | N={N:,} und_E={len(und_pairs):,}")

# offsets, degrees, neighbor lists, types
node_types = list(hetero.node_types)
offsets = {}
cum = 0
for nt in node_types:
    offsets[nt] = cum
    cum += hetero[nt].num_nodes

u_off = offsets["users"]
users_local_idx = hetero._id_to_idx["users"]

deg = np.zeros(N, dtype=np.int64)
np.add.at(deg, und_pairs[:, 0], 1)
np.add.at(deg, und_pairs[:, 1], 1)

print("Building neighbor list...")
t0 = time.time()
neigh = [list() for _ in range(N)]
for s_, t_ in und_pairs:
    neigh[s_].append(t_); neigh[t_].append(s_)
print(f"  done in {time.time()-t0:.1f}s")

type_starts = [(offsets[nt], offsets[nt] + hetero[nt].num_nodes, nt) for nt in node_types]
def node_type(idx):
    for s_, e_, nt in type_starts:
        if s_ <= idx < e_: return nt
    return "?"


# ---------------- 3. Load cached embeddings ----------------
print("Loading cached embeddings...")
t0 = time.time()
npz = np.load("/home/wertheizer/structeredML/HW2/embeddings.npz")
emb_kpca_train = npz["kpca_train"]
emb_kpca_val   = npz["kpca_val"]
emb_lap        = npz["emb_lap_raw"]
emb_n2v        = npz["emb_n2v_raw"]
emb_m2v_users  = npz["emb_m2v_users"]
# m2v posts/comments not cached separately — need to derive
# Actually they ARE cached as emb_m2v_users (only). For posts/comments we need to
# reload from the model or skip. Let me check.
print("  keys:", list(npz.files))
print(f"Embeddings loaded in {time.time()-t0:.1f}s")
print(f"  kpca_train={emb_kpca_train.shape} kpca_val={emb_kpca_val.shape}")
print(f"  lap={emb_lap.shape} n2v={emb_n2v.shape} m2v_users={emb_m2v_users.shape}")


# ---------------- 4. KPCA user-map and connected-user lookup ----------------
print("Building user -> KPCA map...")
t0 = time.time()
user_to_kpca = {}
for i, oid in enumerate(train_df["OwnerUserId"].values):
    if pd.notna(oid):
        oid = int(oid)
        if oid not in user_to_kpca:
            user_to_kpca[oid] = emb_kpca_train[i]
for i, oid in enumerate(val_df["OwnerUserId"].values):
    if pd.notna(oid):
        oid = int(oid)
        if oid not in user_to_kpca:
            user_to_kpca[oid] = emb_kpca_val[i]
print(f"  {len(user_to_kpca):,} users have a KPCA embedding ({time.time()-t0:.1f}s)")

posts_id_to_owner = dict(zip(db.table_dict["posts"].df["Id"].astype("int64"),
                              db.table_dict["posts"].df["OwnerUserId"]))

def homo_to_user_id(homo_idx):
    nt = node_type(homo_idx)
    local = homo_idx - offsets[nt]
    df = hetero._dfs[nt]
    if local < 0 or local >= len(df): return None
    row = df.iloc[local]
    if nt == "users": return int(row["Id"])
    user_fk_col = {"posts": "OwnerUserId", "comments": "UserId",
                    "badges": "UserId", "votes": "UserId",
                    "postHistory": "UserId"}.get(nt)
    if user_fk_col is not None:
        v = row[user_fk_col]
        return int(v) if pd.notna(v) else None
    if nt == "postLinks":
        pid = row["PostId"]
        if pd.isna(pid): return None
        owner = posts_id_to_owner.get(int(pid))
        return int(owner) if (owner is not None and pd.notna(owner)) else None
    return None

def kpca_emb_for(homo_idx):
    uid = homo_to_user_id(homo_idx)
    if uid is None: return None
    return user_to_kpca.get(uid)


def cos_sim(a, b, eps=1e-9):
    if a is None or b is None: return float("nan")
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < eps or nb < eps: return 0.0
    return float(np.dot(a, b) / (na * nb))


def m2v_emb_for(homo_idx):
    nt = node_type(homo_idx)
    if nt == "users":
        return emb_m2v_users[homo_idx - offsets["users"]]
    # posts/comments not cached separately — would need to reload model
    return None


def two_hop_subgraph(center, max_per_hop=8):
    hop1 = neigh[center][:max_per_hop]
    seen = {center, *hop1}
    hop2 = []
    for v in hop1:
        for w in neigh[v]:
            if w in seen: continue
            seen.add(w); hop2.append((v, w))
            if len([1 for _, x in hop2 if _ == v]) >= max_per_hop:
                break
    return hop1, hop2


# ---------------- 5. Pick the same 3 users (deterministic) ----------------
candidate_users = []
for oid in val_df["OwnerUserId"].dropna().unique():
    loc = users_local_idx.get(int(oid), -1)
    if loc < 0: continue
    homo_idx = u_off + loc
    d_ = deg[homo_idx]
    if 5 <= d_ <= 50:
        candidate_users.append((int(oid), homo_idx, int(d_)))
    if len(candidate_users) >= 50: break

rng2 = np.random.default_rng(0)
sel = rng2.choice(len(candidate_users), size=3, replace=False)
chosen = [candidate_users[i] for i in sel]
print(f"\nChosen focal users (OwnerUserId, homo_idx, degree): {chosen}")


# ---------------- 6. Plot + tables ----------------
captured_output = []

fig, axes = plt.subplots(1, 3, figsize=(24, 8))
for ax_i, (oid, center, d_) in enumerate(chosen):
    hop1, hop2 = two_hop_subgraph(center, max_per_hop=8)
    nodes_used = [center] + hop1 + [w for _, w in hop2]
    nodes_used = list(dict.fromkeys(nodes_used))
    g = nx.Graph()
    for v in nodes_used:
        g.add_node(v)
    for v in hop1:
        g.add_edge(center, v)
    for u_, w in hop2:
        g.add_edge(u_, w)
    pos = nx.spring_layout(g, seed=42)
    type_to_color = {"users":"#1f77b4", "posts":"#ff7f0e", "comments":"#2ca02c",
                     "badges":"#d62728", "votes":"#9467bd", "postHistory":"#8c564b",
                     "postLinks":"#e377c2"}
    colors = [type_to_color.get(node_type(v), "#7f7f7f") for v in g.nodes]
    nx.draw_networkx_nodes(g, pos, node_color=colors, node_size=400, ax=axes[ax_i])
    nx.draw_networkx_edges(g, pos, alpha=0.4, ax=axes[ax_i])
    labels = {v: f"{node_type(v)[:4]}\n{v}" for v in g.nodes}
    nx.draw_networkx_labels(g, pos, labels, font_size=7, ax=axes[ax_i])
    axes[ax_i].set_title(f"User {oid} (idx {center}, deg {d_})\n"
                          f"|N1|={len(hop1)}, |N2-only|={len([w for _,w in hop2])}",
                          fontsize=10)
    axes[ax_i].axis("off")

    sim_rows = []
    for v in nodes_used[1:]:
        row = {"node": v, "type": node_type(v),
               "Kernel PCA": cos_sim(kpca_emb_for(center), kpca_emb_for(v)),
               "Laplacian":  cos_sim(emb_lap[center], emb_lap[v]),
               "Node2Vec":   cos_sim(emb_n2v[center], emb_n2v[v]),
               "Metapath2Vec": cos_sim(m2v_emb_for(center), m2v_emb_for(v))}
        sim_rows.append(row)
    sim_df = pd.DataFrame(sim_rows).round(3)
    header = f"\n=== User {oid} (idx {center}, deg {d_}) - 2-hop neighbor similarities ==="
    body = sim_df.to_string(index=False)
    print(header); print(body)
    captured_output.append(header + "\n" + body)

plt.tight_layout()
plt.savefig("/home/wertheizer/structeredML/HW2/neighborhoods.png", dpi=110, bbox_inches="tight")
print("\nSaved neighborhoods.png")

# Write captured output to a file so we can paste into the ipynb cell
with open("/home/wertheizer/structeredML/HW2/_q9_output.txt", "w") as f:
    f.write(f"Chosen focal users (OwnerUserId, homo_idx, degree): {chosen}\n")
    for s in captured_output:
        f.write(s + "\n")
print("Wrote _q9_output.txt for ipynb injection.")
