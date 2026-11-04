"""Re-train Metapath2Vec (25 epochs), save all three node-type embeddings,
then re-run Q9 with the full embedding set so the Metapath2Vec column gets
real values instead of NaN.

Loads the cached kpca / lap / n2v embeddings from embeddings.npz so we don't
re-do any of the expensive non-m2v work.
"""
import os, sys, time, warnings, datetime
warnings.filterwarnings("ignore")
os.chdir("/home/wertheizer/structeredML/HW2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import MetaPath2Vec
from relbench.datasets import get_dataset
from relbench.tasks import get_task

EMB_DIM = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

HB_PATH = "/home/wertheizer/structeredML/HW2/_progress_m2v.log"
def hb(msg):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(HB_PATH, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass

hb(f"START m2v-only retrain | torch={torch.__version__} device={DEVICE}")

# --- 1. Load data ---
hb("Loading relbench rel-stack ...")
dataset = get_dataset("rel-stack", download=False)
db = dataset.get_db()
task = get_task("rel-stack", "user-engagement", download=False)
train_df = task.get_table("train").df.copy()
val_df   = task.get_table("val").df.copy()
VAL_TIMESTAMP = pd.Timestamp(dataset.val_timestamp)
hb(f"data loaded | val_ts={VAL_TIMESTAMP}")

# --- 2. Build heterograph ---
hb("Building heterograph ...")
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
hb(f"heterograph built in {time.time()-t0:.1f}s")

# --- 3. Metapath subgraph + training ---
def get_or_make_edge(hd, src_type, rel_substr, dst_type):
    for et in hd.edge_types:
        s, r, d = et
        if s == src_type and d == dst_type and rel_substr in r:
            return hd[et].edge_index
    return None

mp_edges = {}
posts_owner   = get_or_make_edge(hetero, "posts",    "OwnerUserId", "users")
comments_post = get_or_make_edge(hetero, "comments", "PostId",      "posts")
comments_user = get_or_make_edge(hetero, "comments", "UserId",      "users")
mp_edges[("users",    "owns",       "posts")]    = posts_owner.flip(0)
mp_edges[("posts",    "ownedBy",    "users")]    = posts_owner
mp_edges[("posts",    "hasComment", "comments")] = comments_post.flip(0)
mp_edges[("comments", "onPost",     "posts")]    = comments_post
mp_edges[("comments", "byUser",     "users")]    = comments_user
mp_edges[("users",    "wrote",      "comments")] = comments_user.flip(0)

metapath = [
    ("users",    "owns",       "posts"),
    ("posts",    "hasComment", "comments"),
    ("comments", "byUser",     "users"),
]

hd_mp = HeteroData()
for nt in ["users", "posts", "comments"]:
    hd_mp[nt].num_nodes = hetero[nt].num_nodes
for et, ei in mp_edges.items():
    hd_mp[et].edge_index = ei

hb(f"m2v subgraph: users={hd_mp['users'].num_nodes:,} posts={hd_mp['posts'].num_nodes:,} comments={hd_mp['comments'].num_nodes:,}")

# --- Train ---
EPOCHS = 25
hb(f"Training Metapath2Vec (epochs={EPOCHS}, batch=128, lr=0.01) on {DEVICE} ...")
model = MetaPath2Vec(hd_mp.edge_index_dict, embedding_dim=EMB_DIM, metapath=metapath,
                     walk_length=24, context_size=5, walks_per_node=10,
                     num_negative_samples=5, sparse=True).to(DEVICE)
loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
opt = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
for ep in range(EPOCHS):
    model.train()
    t_ep = time.time()
    total = 0.0; n = 0
    for pos_rw, neg_rw in loader:
        opt.zero_grad()
        loss = model.loss(pos_rw.to(DEVICE), neg_rw.to(DEVICE))
        loss.backward(); opt.step()
        total += float(loss); n += 1
    hb(f"m2v ep {ep+1}/{EPOCHS} loss={total/max(n,1):.4f} ({time.time()-t_ep:.1f}s)")

model.eval()
emb_m2v_by_type = {}
for nt in ["users", "posts", "comments"]:
    with torch.no_grad():
        emb_m2v_by_type[nt] = model(nt).detach().cpu().numpy().astype("float32")
hb(f"m2v trained | users={emb_m2v_by_type['users'].shape} posts={emb_m2v_by_type['posts'].shape} comments={emb_m2v_by_type['comments'].shape}")

# --- 4. Merge with cached npz and save ---
hb("Merging with cached embeddings ...")
cache = np.load("/home/wertheizer/structeredML/HW2/embeddings.npz")
keep = {k: cache[k] for k in cache.files}
keep["emb_m2v_users"]   = emb_m2v_by_type["users"]
keep["m2v_posts"]       = emb_m2v_by_type["posts"]
keep["m2v_comments"]    = emb_m2v_by_type["comments"]
np.savez_compressed("/home/wertheizer/structeredML/HW2/embeddings.npz", **keep)
hb("embeddings.npz updated with all 3 m2v slices")

# --- 5. Re-run Q9 with full embeddings ---
hb("Re-running Q9 ...")
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

emb_kpca_train = keep["kpca_train"]; emb_kpca_val = keep["kpca_val"]
emb_lap = keep["emb_lap_raw"]; emb_n2v = keep["emb_n2v_raw"]
m2v_user, m2v_post, m2v_comment = emb_m2v_by_type["users"], emb_m2v_by_type["posts"], emb_m2v_by_type["comments"]

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

def m2v_emb_for(homo_idx):
    nt = node_type(homo_idx)
    if nt == "users":    return m2v_user[homo_idx - offsets["users"]]
    if nt == "posts":    return m2v_post[homo_idx - offsets["posts"]]
    if nt == "comments": return m2v_comment[homo_idx - offsets["comments"]]
    return None

def cos_sim(a, b, eps=1e-9):
    if a is None or b is None: return float("nan")
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < eps or nb < eps: return 0.0
    return float(np.dot(a, b) / (na * nb))

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
hb(f"Chosen focal users: {chosen}")

captured = []
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
    header = f"=== User {oid} (idx {center}, deg {d_}) - 2-hop neighbor similarities ==="
    body = sim_df.to_string(index=False)
    print(header); print(body)
    captured.append(header + "\n" + body)

plt.tight_layout()
plt.savefig("/home/wertheizer/structeredML/HW2/neighborhoods.png", dpi=110, bbox_inches="tight")
hb("Saved neighborhoods.png")
with open("/home/wertheizer/structeredML/HW2/_q9_output.txt", "w") as f:
    f.write(f"Chosen focal users (OwnerUserId, homo_idx, degree): {chosen}\n\n")
    for s in captured:
        f.write(s + "\n\n")
hb("DONE - wrote _q9_output.txt")
