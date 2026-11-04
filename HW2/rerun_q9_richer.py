"""Re-run Q9 with a richer 2-hop sampler. Use cached embeddings.npz (which now
includes all 3 m2v slices). Goal: tables with interesting variation across
embedding methods — specifically 2-hop neighbors that map to different users
(so KPCA != 1.0) and a mix of node types per row."""
import os, time, warnings
warnings.filterwarnings("ignore")
os.chdir("/home/wertheizer/structeredML/HW2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import HeteroData
from relbench.datasets import get_dataset
from relbench.tasks import get_task

dataset = get_dataset("rel-stack", download=False)
db = dataset.get_db()
task = get_task("rel-stack", "user-engagement", download=False)
train_df = task.get_table("train").df.copy()
val_df   = task.get_table("val").df.copy()
VAL_TIMESTAMP = pd.Timestamp(dataset.val_timestamp)

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
        df_child, _, _ = id_to_idx[child]
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

neigh = [list() for _ in range(N)]
for s_, t_ in und_pairs:
    neigh[s_].append(t_); neigh[t_].append(s_)

type_starts = [(offsets[nt], offsets[nt] + hetero[nt].num_nodes, nt) for nt in node_types]
def node_type(idx):
    for s_, e_, nt in type_starts:
        if s_ <= idx < e_: return nt
    return "?"

print("Loading embeddings...")
npz = np.load("/home/wertheizer/structeredML/HW2/embeddings.npz")
emb_kpca_train = npz["kpca_train"]; emb_kpca_val = npz["kpca_val"]
emb_lap = npz["emb_lap_raw"]; emb_n2v = npz["emb_n2v_raw"]
m2v_user = npz["emb_m2v_users"]
m2v_post = npz["m2v_posts"]
m2v_comment = npz["m2v_comments"]
print(f"  m2v users={m2v_user.shape} posts={m2v_post.shape} comments={m2v_comment.shape}")

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

posts_id_to_owner = dict(zip(db.table_dict["posts"].df["Id"].astype("int64"),
                              db.table_dict["posts"].df["OwnerUserId"]))

def homo_to_user_id(homo_idx):
    nt = node_type(homo_idx)
    local = homo_idx - offsets[nt]
    df = hetero._dfs[nt]
    if local < 0 or local >= len(df): return None
    row = df.iloc[local]
    if nt == "users": return int(row["Id"])
    user_fk = {"posts": "OwnerUserId", "comments": "UserId",
                "badges": "UserId", "votes": "UserId",
                "postHistory": "UserId"}.get(nt)
    if user_fk is not None:
        v = row[user_fk]
        return int(v) if pd.notna(v) else None
    if nt == "postLinks":
        pid = row["PostId"]
        if pd.isna(pid): return None
        owner = posts_id_to_owner.get(int(pid))
        return int(owner) if (owner is not None and pd.notna(owner)) else None
    return None

def kpca_emb_for(h):
    uid = homo_to_user_id(h)
    return user_to_kpca.get(uid) if uid is not None else None

def m2v_emb_for(h):
    nt = node_type(h)
    if nt == "users":    return m2v_user[h - offsets["users"]]
    if nt == "posts":    return m2v_post[h - offsets["posts"]]
    if nt == "comments": return m2v_comment[h - offsets["comments"]]
    return None

def cos(a, b, eps=1e-9):
    if a is None or b is None: return float("nan")
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < eps or nb < eps: return 0.0
    return float(np.dot(a, b) / (na*nb))


def richer_subgraph(center, max_per_type_hop1=2, max_hop2_diff_user=10, max_hop2_same_user=4):
    """Pick a diverse 1-hop sample (max N per node-type) and prefer 2-hop
    expansion through neighbors that map to a DIFFERENT user (so the KPCA
    column has variation)."""
    center_uid = homo_to_user_id(center)
    by_type = {}
    for v in neigh[center]:
        by_type.setdefault(node_type(v), []).append(v)
    hop1 = []
    for nt in ["badges", "comments", "posts", "votes", "postHistory"]:
        hop1.extend(by_type.get(nt, [])[:max_per_type_hop1])
    seen = {center, *hop1}
    hop2_diff = []
    hop2_same = []
    for v in hop1:
        for w in neigh[v]:
            if w in seen: continue
            seen.add(w)
            wid = homo_to_user_id(w)
            if wid is not None and wid != center_uid:
                hop2_diff.append((v, w))
            else:
                hop2_same.append((v, w))
            if len(hop2_diff) >= max_hop2_diff_user:
                break
    hop2 = hop2_diff[:max_hop2_diff_user] + hop2_same[:max_hop2_same_user]
    return hop1, hop2


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
print(f"\nChosen focal users: {chosen}")

captured = []
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
for ax_i, (oid, center, d_) in enumerate(chosen):
    hop1, hop2 = richer_subgraph(center)
    nodes_used = [center] + hop1 + [w for _, w in hop2]
    nodes_used = list(dict.fromkeys(nodes_used))
    g = nx.Graph()
    for v in nodes_used: g.add_node(v)
    for v in hop1: g.add_edge(center, v)
    for u_, w in hop2: g.add_edge(u_, w)
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
                          f"|N1|={len(hop1)}, |N2|={len(hop2)}", fontsize=10)
    axes[ax_i].axis("off")

    center_uid = homo_to_user_id(center)
    rows = []
    for v in hop1:
        rows.append({"hop": 1, "node": v, "type": node_type(v),
                     "same_user": "yes",
                     "Kernel PCA": cos(kpca_emb_for(center), kpca_emb_for(v)),
                     "Laplacian":  cos(emb_lap[center], emb_lap[v]),
                     "Node2Vec":   cos(emb_n2v[center], emb_n2v[v]),
                     "Metapath2Vec": cos(m2v_emb_for(center), m2v_emb_for(v))})
    for _, w in hop2:
        wid = homo_to_user_id(w)
        rows.append({"hop": 2, "node": w, "type": node_type(w),
                     "same_user": "yes" if wid == center_uid else "no",
                     "Kernel PCA": cos(kpca_emb_for(center), kpca_emb_for(w)),
                     "Laplacian":  cos(emb_lap[center], emb_lap[w]),
                     "Node2Vec":   cos(emb_n2v[center], emb_n2v[w]),
                     "Metapath2Vec": cos(m2v_emb_for(center), m2v_emb_for(w))})
    df = pd.DataFrame(rows).round(3)
    h = f"=== User {oid} (idx {center}, deg {d_}) - 2-hop neighbor similarities ==="
    print(h); print(df.to_string(index=False))
    captured.append(h + "\n" + df.to_string(index=False))

plt.tight_layout()
plt.savefig("/home/wertheizer/structeredML/HW2/neighborhoods.png", dpi=110, bbox_inches="tight")
print("\nSaved neighborhoods.png")

with open("/home/wertheizer/structeredML/HW2/_q9_output.txt", "w") as f:
    f.write(f"Chosen focal users (OwnerUserId, homo_idx, degree): {chosen}\n\n")
    for s in captured:
        f.write(s + "\n\n")
print("Wrote _q9_output.txt")
