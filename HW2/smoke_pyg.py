"""Smoke test: instantiate Node2Vec and MetaPath2Vec on a TINY graph,
run 1 epoch on a few batches each. Verifies the PyG wiring works
before we ship to the GPU.
"""
import torch
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.nn import Node2Vec, MetaPath2Vec

print("PyG smoke...")

# --- Tiny homogeneous graph: 50 nodes, ~150 undirected edges ---
N = 50
rng = np.random.default_rng(42)
src = rng.integers(0, N, size=150)
dst = rng.integers(0, N, size=150)
m = src != dst
src, dst = src[m], dst[m]
und_src = np.concatenate([src, dst])
und_dst = np.concatenate([dst, src])
ei = torch.tensor(np.stack([und_src, und_dst]), dtype=torch.long)
print(f"Tiny graph: N={N}, E(directed)={ei.shape[1]}")

# Node2Vec
n2v = Node2Vec(ei, embedding_dim=32, walk_length=24, context_size=5,
               walks_per_node=10, num_negative_samples=1, p=1.0, q=1.0,
               sparse=True, num_nodes=N)
loader = n2v.loader(batch_size=8, shuffle=True, num_workers=0)
opt = torch.optim.SparseAdam(list(n2v.parameters()), lr=0.01)
n2v.train()
losses = []
for i, (pos, neg) in enumerate(loader):
    opt.zero_grad()
    loss = n2v.loss(pos, neg)
    loss.backward()
    opt.step()
    losses.append(float(loss))
    if i >= 4: break
print(f"Node2Vec: 5 batches, losses {[round(l,4) for l in losses]}")
with torch.no_grad():
    print(f"  emb shape: {n2v.embedding.weight.shape}")

# MetaPath2Vec  --- tiny heterograph: 20 users, 30 posts, 40 comments ---
Nu, Np, Nc = 20, 30, 40
e_up = torch.tensor(np.stack([rng.integers(0, Nu, 80), rng.integers(0, Np, 80)]), dtype=torch.long)
e_pc = torch.tensor(np.stack([rng.integers(0, Np, 120), rng.integers(0, Nc, 120)]), dtype=torch.long)
e_cu = torch.tensor(np.stack([rng.integers(0, Nc, 80), rng.integers(0, Nu, 80)]), dtype=torch.long)

eidict = {
    ("users", "owns", "posts"): e_up,
    ("posts", "hasComment", "comments"): e_pc,
    ("comments", "byUser", "users"): e_cu,
}
metapath = [("users", "owns", "posts"),
            ("posts", "hasComment", "comments"),
            ("comments", "byUser", "users")]

m2v = MetaPath2Vec(eidict, embedding_dim=32, metapath=metapath,
                   walk_length=24, context_size=5, walks_per_node=10,
                   num_negative_samples=5, sparse=True,
                   num_nodes_dict={"users": Nu, "posts": Np, "comments": Nc})
loader = m2v.loader(batch_size=4, shuffle=True, num_workers=0)
opt = torch.optim.SparseAdam(list(m2v.parameters()), lr=0.01)
m2v.train()
losses = []
for i, (pos, neg) in enumerate(loader):
    opt.zero_grad()
    loss = m2v.loss(pos, neg)
    loss.backward()
    opt.step()
    losses.append(float(loss))
    if i >= 4: break
print(f"Metapath2Vec: 5 batches, losses {[round(l,4) for l in losses]}")
print(f"  user emb shape: {m2v('users').shape}")
print(f"  post emb shape: {m2v('posts').shape}")
print(f"  comment emb shape: {m2v('comments').shape}")
print("\nPyG smoke PASSED")
