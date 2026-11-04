# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python 3 (structml1)
#     language: python
#     name: python3
# ---

# %% [markdown] id="6f57e2e8"
# # StructML (236605) — Assignment 1: Feature Extraction
#
# Database: **rel-stack** (relbench). Task: **user-engagement** — predict if a user makes any
# vote / post / comment in the next 3 months (binary classification).
#
# This notebook walks through Q1–Q16 from the assignment in order. Heavy code is kept inline
# (single-notebook deliverable, as requested).

# %% [markdown] id="509ee1ab"
# ## Part 0 — Setup & data loading
#
# **If running on Google Colab**, run the cell below first to install `relbench` (and a
# matching `pyarrow`/`huggingface-hub`). Other dependencies (`torch`, `xgboost`,
# `networkx`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `tqdm`) ship with Colab.
# A free-tier Colab runtime (~12 GB RAM) should fit the workload but is tight; if you
# hit OOM, switch to a high-RAM runtime.

# %% id="5fbc1a8e"
# --- Colab-only install (skip if running locally with the conda env already prepared) ---
import sys, subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "relbench[full]"])

# %% id="a7c4b862" outputId="9327ea78-bf44-4e5c-c99d-7a3835f276d9" colab={"base_uri": "https://localhost:8080/"}
import warnings
warnings.filterwarnings("ignore")

import time
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from tqdm.auto import tqdm

from relbench.datasets import get_dataset
from relbench.tasks import get_task

import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
)
from sklearn.impute import SimpleImputer
import xgboost as xgb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 220)

RNG = np.random.default_rng(42)
torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "device =", DEVICE)
print("sklearn", sklearn.__version__, " | xgboost", xgb.__version__)

# %% id="8e63db60" outputId="1a602d34-e268-4694-f1c5-f4fa2b32b4c9" colab={"base_uri": "https://localhost:8080/"}
dataset = get_dataset("rel-stack", download=False)
db = dataset.get_db()
task = get_task("rel-stack", "user-engagement", download=False)

train_table = task.get_table("train")
val_table = task.get_table("val")
test_table = task.get_table("test")

train_df = train_table.df.copy()
val_df   = val_table.df.copy()
test_df  = test_table.df.copy()

print("Tables in db :", list(db.table_dict.keys()))
for name, t in db.table_dict.items():
    print(f"  {name:12s} rows={len(t.df):>10,} pkey={t.pkey_col!s:<6} time={t.time_col!s:<14} fkeys={t.fkey_col_to_pkey_table}")

print()
print("Target tables:")
print(f"  train: shape={train_df.shape}, unique users={train_df.OwnerUserId.nunique():,}, "
      f"timestamps {train_df.timestamp.min().date()}..{train_df.timestamp.max().date()}")
print(f"  val  : shape={val_df.shape},  unique users={val_df.OwnerUserId.nunique():,}, "
      f"timestamps {val_df.timestamp.min().date()}..{val_df.timestamp.max().date()}")
print(f"  test : shape={test_df.shape},  unique users={test_df.OwnerUserId.nunique():,}, "
      f"timestamps {test_df.timestamp.min().date()}..{test_df.timestamp.max().date()}")

print("\nClass balance:")
print(" train:", train_df["contribution"].value_counts(normalize=True).round(4).to_dict())
print(" val  :", val_df["contribution"].value_counts(normalize=True).round(4).to_dict())

# %% [markdown] id="9330c006"
# ## Part 1 — Tabular Feature Extraction
#
# ### Q1. Three hand-crafted join-aggregate features
#
#
# Here are 3 join-aggragate features we consider as helpful for the prediction task:
#
# 1. **`posts_total`** - count of posts owned by the user (`target ⟕ users ⟕ posts`,
#    count). Users who posted many posts before os more likely to keep posting.
#
# 2. **`comments_total`** - count of comments made by each user (`target ⟕ users ⟕ comments`,
#    count). Since our goal is to predict user activity over the next three months, relying solely on post counts (posts_total) could cause us to miss different 'type' of users. There are plenty of highly active users who rarely publish their own posts but engage heavily in the comment sections instead.
#
# 3. **`badge_count`** - number of badges (`target ⟕ users ⟕ badges`, count).
#    Since badges are achieved by the user's total activity, users with larger count of badges are more likely be active in the next 3 months.

# %% id="ac990fbc" outputId="185a90ea-28a3-4a4f-ee15-8c70627d2787" colab={"base_uri": "https://localhost:8080/"}
def build_q1_manual_features(target_df: pd.DataFrame) -> pd.DataFrame:
    """Return target_df with the three manual features appended."""
    out = target_df.copy()

    # posts count
    posts = db.table_dict["posts"].df[["OwnerUserId"]].dropna()
    posts_per_user = (
        posts.groupby("OwnerUserId").size().rename("q1_posts_total").astype("float64")
    )
    out = out.merge(posts_per_user, left_on="OwnerUserId", right_index=True, how="left")
    out["q1_posts_total"] = out["q1_posts_total"].fillna(0.0)


    # comments count
    comments = db.table_dict["comments"].df[["UserId"]].dropna()
    comments_per_user = (
        comments.groupby("UserId").size().rename("q1_comments_total").astype("float64")
    )
    out = out.merge(comments_per_user, left_on="OwnerUserId", right_index=True, how="left")
    out["q1_comments_total"] = out["q1_comments_total"].fillna(0.0)


    # badges count
    badges = db.table_dict["badges"].df[["UserId"]].dropna()
    badges_per_user = (
        badges.groupby("UserId").size().rename("q1_badge_count").astype("float64")
    )
    out = out.merge(badges_per_user, left_on="OwnerUserId", right_index=True, how="left")
    out["q1_badge_count"] = out["q1_badge_count"].fillna(0.0)

    return out

_q1_sample = build_q1_manual_features(train_df.head(2000))
print("Q1 sample stats:")
print(_q1_sample[["q1_posts_total", "q1_comments_total", "q1_badge_count"]].describe().round(2))



# %% [markdown] id="cfac201f"
# ### Q2. Automatic join-aggregate feature generation
#
# we start from the target and walk the PK-FK schema with BFS. each path up to
# `max_depth` hops becomes a join — left join into users first so we keep every
# target row, then natural/inner joins for the rest. for each leaf table we
# aggregate every column per target row: numeric → mean/count/sum, categorical
# or text → count, timestamps → min/max.
#
# before aggregating we drop PK and FK columns (ACORA — those are just ids, no
# signal), plus `Users.AccountId` since the course instructions say to drop it.
#
# temporal stuff: a prediction for row x can only see data from before
# `x.timestamp`. for train we check this per row — after each join we drop rows
# whose joined table's time column is later than the target timestamp. for val
# we just slice every source table to `time_col ≤ val_timestamp` (2020-10-01)
# before joining. for test we don't need to do anything; the db is already
# pre-cut at `test_timestamp`.

# %% id="1513a549" outputId="ba34fdbb-5a02-4afe-e9f7-93af6b35a782" colab={"base_uri": "https://localhost:8080/"}
# ---- Build the PK-FK relationship graph -------------------------------------------------
# Nodes are tables. Each FK column gives a (parent, child, fk_col) edge.
SCHEMA_FKEYS: List[Tuple[str, str, str]] = []   # (parent_table, child_table, fk_col_in_child)
for child, t in db.table_dict.items():
    for fk_col, parent in t.fkey_col_to_pkey_table.items():
        SCHEMA_FKEYS.append((parent, child, fk_col))

print("PK-FK edges in db:")
for p, c, f in SCHEMA_FKEYS:
    print(f"  {p:12s} <--({f})-- {c}")

# The target table has its own FK: target.OwnerUserId -> users.Id .
# We model it as an extra edge so BFS can start from "target".
TARGET_FK_COL = "OwnerUserId"      # column in target that links to users.Id
TARGET_FK_PARENT = "users"
SCHEMA_FKEYS.append((TARGET_FK_PARENT, "_target_", TARGET_FK_COL))

# %% id="d90fb4c4"
def column_dtype_kind(series: pd.Series) -> str:
    """Map a pandas dtype to one of {'numeric','categorical','timestamp'}."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "timestamp"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"

AGGS_FOR_KIND = {
    "numeric":     ["mean", "count", "sum"],
    "categorical": ["count"],
    "timestamp":   ["min", "max"],
}

# %% id="f4733eed"
def enumerate_join_paths(max_depth: int) -> List[List[Tuple[str, str, str]]]:
    """Enumerate all join paths of length 1..max_depth starting from the (synthetic)
    target table. Each path is a list of edges  (from_table, to_table, join_col_pair).

    We treat the relationship graph as undirected for navigation, but remember the
    direction of each FK so we know which column joins which.
    """
    # Build adjacency: table -> list of (neighbour_table, fk_parent, fk_child, fk_col)
    adj: Dict[str, List[Tuple[str, str, str, str]]] = defaultdict(list)
    for parent, child, fk_col in SCHEMA_FKEYS:
        # forward: parent --(pk -> fk_col on child)-- child
        adj[parent].append((child, parent, child, fk_col))
        # backward: child --(fk_col -> pk on parent)-- parent
        adj[child].append((parent, parent, child, fk_col))

    paths: List[List[Tuple[str, str, str]]] = []
    # BFS storing (current_table, visited_set, path_so_far)
    frontier = [("_target_", {"_target_"}, [])]
    for depth in range(max_depth):
        next_frontier = []
        for cur, visited, path in frontier:
            for nbr, parent, child, fk_col in adj[cur]:
                if nbr in visited:
                    continue
                edge = (cur, nbr, f"{parent}.Id={child}.{fk_col}")
                new_path = path + [edge]
                paths.append(new_path)
                next_frontier.append((nbr, visited | {nbr}, new_path))
        frontier = next_frontier
    return paths

def describe_path(path) -> str:
    return " -> ".join([path[0][0]] + [e[1] for e in path])

# %% id="c7bedbd3"
VAL_TIMESTAMP = pd.Timestamp("2020-10-01")  # dataset.val_timestamp

def _filter_by_time(df: pd.DataFrame, time_col, cutoff_ts) -> pd.DataFrame:
    """Keep rows where time_col <= cutoff_ts. No-op if cutoff_ts is None or time_col missing."""
    if cutoff_ts is None or time_col is None or time_col not in df.columns:
        return df
    return df[df[time_col] <= cutoff_ts]

def auto_features(target_df: pd.DataFrame,
                  max_depth: int,
                  split: str = "train",
                  verbose: bool = True):
    """Generate join-aggregate features automatically.

    For every join path of length 1..`max_depth`, we materialise the join
    `target ⟕ T_1 ⋈ ... ⋈ T_d` (left join on the first hop, natural/inner
    afterwards) and aggregate every column of the leaf table T_d, grouped by
    the originating target row.

    Temporal constraint (Q2a, edited):
      split='train': after every join, drop joined rows where the source-table
                     time_col > the target tuple's timestamp (per-row cutoff).
      split='val'  : pre-filter every source table to time_col <= val_timestamp
                     (2020-10-01) before joining.
      split='test' : no filtering (db is already pre-cut at test_timestamp).

    Returns (feature_matrix, feature_meta).
    """
    paths = enumerate_join_paths(max_depth)
    if verbose:
        print(f"Enumerated {len(paths)} join paths up to depth {max_depth}:")
        for p in paths:
            print("  ", describe_path(p))

    # Tag every target row with a unique id so we can group back to it after joining.
    target = target_df.copy()
    target["_row_id"] = np.arange(len(target))

    # For val: pre-filter every source table at the global val_timestamp.
    val_cut = VAL_TIMESTAMP if split == "val" else None
    def src_df(name: str) -> pd.DataFrame:
        return _filter_by_time(db.table_dict[name].df, db.table_dict[name].time_col, val_cut)

    users_df = src_df("users")
    users_time_col = db.table_dict["users"].time_col
    feature_meta_rows = []
    per_path_features: List[pd.DataFrame] = []

    for path in tqdm(paths, desc="paths", disable=not verbose):
        tables_seq = [edge[1] for edge in path]   # e.g. ['users'] or ['users','posts']
        depth = len(tables_seq)
        leaf_table = tables_seq[-1]
        leaf_df = src_df(leaf_table)
        leaf_time_col = db.table_dict[leaf_table].time_col
        prefix = f"d{depth}__" + "_".join(tables_seq)

        # ---- materialise the joined table for this path ----
        if depth == 1:
            # target ⟕ users   (1:1 because users.Id is a primary key)
            joined = target[["_row_id", "OwnerUserId", "timestamp"]].merge(
                users_df, left_on="OwnerUserId", right_on="Id", how="left",
            )
            # train: drop rows where users.CreationDate > target.timestamp
            if split == "train" and users_time_col is not None and users_time_col in joined.columns:
                mask = joined[users_time_col].isna() | (joined[users_time_col] <= joined["timestamp"])
                joined = joined[mask]
        elif depth == 2:
            # target ⟕ users ⋈ leaf_table
            # FK column on leaf_table that points back to users.Id:
            user_fk = next(c for c, t in db.table_dict[leaf_table].fkey_col_to_pkey_table.items()
                           if t == "users")
            joined = (
                target[["_row_id", "OwnerUserId", "timestamp"]]
                .merge(users_df[["Id"]], left_on="OwnerUserId", right_on="Id", how="left")
                .merge(leaf_df, left_on="Id", right_on=user_fk, how="inner",
                       suffixes=("", f"_{leaf_table}"))
            )
            # train: drop rows where leaf.time_col > target.timestamp
            if split == "train" and leaf_time_col is not None:
                cand_time = f"{leaf_time_col}_{leaf_table}"
                if cand_time not in joined.columns:
                    cand_time = leaf_time_col
                if cand_time in joined.columns:
                    mask = joined[cand_time] <= joined["timestamp"]
                    joined = joined[mask]
        else:
            raise NotImplementedError("Direct implementation supports max_depth ∈ {1, 2}")

        # ---- aggregate the leaf table's columns per target row ----
        # Following ACORA: skip PK and FK columns (they are keys, not features).
        # Also drop Users.AccountId per course instructions.
        leaf_pk = db.table_dict[leaf_table].pkey_col
        leaf_fks = set(db.table_dict[leaf_table].fkey_col_to_pkey_table.keys())
        skip_cols = {leaf_pk} | leaf_fks | {"AccountId"}

        grp = joined.groupby("_row_id", sort=False)
        agg_cols: Dict[str, pd.Series] = {}
        for col in leaf_df.columns:
            if col in skip_cols:
                continue
            # Find the column in `joined`. If merge added a suffix because of a name
            # collision, the leaf-table version lives under "{col}_{leaf_table}" and
            # the suffixed name takes priority. Otherwise fall back to the plain name.
            suffixed = f"{col}_{leaf_table}"
            cand = suffixed if suffixed in joined.columns else col
            if cand not in joined.columns:
                continue
            kind = column_dtype_kind(joined[cand])
            # bool columns are detected as numeric but mean/sum on them is not
            # well-defined in this context — skip and only keep count.
            if joined[cand].dtype == bool:
                kind = "categorical"
            for aggn in AGGS_FOR_KIND[kind]:
                try:
                    res = grp[cand].count() if aggn == "count" else grp[cand].agg(aggn)
                except Exception:
                    continue
                fname = f"{prefix}__{col}__{aggn}"
                agg_cols[fname] = res
                feature_meta_rows.append({
                    "feature_name": fname, "depth": depth, "path": describe_path(path),
                    "leaf_table": leaf_table, "source_column": col, "aggregation": aggn,
                })
        if agg_cols:
            per_path_features.append(pd.DataFrame(agg_cols))

    # Concatenate all per-path feature blocks (indexed by _row_id) and merge to target
    if per_path_features:
        all_feats = pd.concat(per_path_features, axis=1)
    else:
        all_feats = pd.DataFrame(index=pd.Index([], name="_row_id"))

    feature_matrix = target.merge(all_feats, left_on="_row_id", right_index=True, how="left")
    feature_matrix = feature_matrix.drop(columns=["_row_id"])

    # NaN counts -> 0 (no related rows == zero count); leave numeric/timestamp NaN as-is.
    count_cols = [c for c in feature_matrix.columns if c.endswith("__count")]
    feature_matrix[count_cols] = feature_matrix[count_cols].fillna(0)

    return feature_matrix, pd.DataFrame(feature_meta_rows)

# %% id="6b93af43" outputId="490cd989-6a29-42aa-d632-53e8a1b7f0ee" colab={"base_uri": "https://localhost:8080/", "height": 230, "referenced_widgets": ["bb802d112028442682d2f70dbfa1d4a4", "ee3b994c145e456cbe40f224afd0e9ed", "6764cc603c4d4025b7112b5e35076b94", "d7bf81b2fe25476497909854b22ad77f", "bf82e9b98de747f98f3ed9e926b3b8f2", "cc030037c65448ae8bb4eeecbb662387", "f21252c2eb0c43309d1468d7ef15c092", "a5f0be55ba5441b2b705d110c0a3aabc", "66ea32abc06c49a8b9519001baac0f5d", "9f8c4402b1454917b366ab5414f8fc65", "4a84d8bfb4ed4100bca963a8caee1206"]}
t0 = time.time()
feature_matrix, feature_meta = auto_features(train_df, max_depth=2, split="train", verbose=True)
print(f"\nGenerated in {time.time()-t0:.1f}s")
print("feature_matrix shape:", feature_matrix.shape)

# %% [markdown] id="34bd9d8c"
# ### Q3. Feature counts at `max_depth` ∈ {1, 2, 3}
#
# We re-run the auto-generator at depths 1 and 3 to count features programmatically (a
# closed-form derivation follows the table).

# %% id="163663b4" outputId="611834b8-d378-4b99-b19d-f2475c6bf597"
def count_features_for_depth(max_depth: int) -> tuple:
    """Return (n_paths, n_features, paths_by_depth_dict)."""
    paths = enumerate_join_paths(max_depth)
    by_depth = Counter()
    n_features = 0
    for p in paths:
        d = len(p)
        by_depth[d] += 1
        # leaf table = end of path
        leaf_table = p[-1][1] if d >= 1 else None
        leaf_df = db.table_dict[leaf_table].df if leaf_table != "_target_" else None
        if leaf_df is None:
            continue
        leaf_pk = db.table_dict[leaf_table].pkey_col
        leaf_fks = set(db.table_dict[leaf_table].fkey_col_to_pkey_table.keys())
        skip_cols = {leaf_pk} | leaf_fks | {"AccountId"}
        for col in leaf_df.columns:
            if col in skip_cols:
                continue
            kind = column_dtype_kind(leaf_df[col])
            if leaf_df[col].dtype == bool:
                kind = "categorical"
            n_features += len(AGGS_FOR_KIND[kind])
    return len(paths), n_features, dict(by_depth)

q3_rows = []
for d in (1, 2, 3):
    n_paths, n_feats, by = count_features_for_depth(d)
    q3_rows.append({"max_depth": d, "n_paths": n_paths, "n_features": n_feats,
                    "paths_per_depth": by})
q3_table = pd.DataFrame(q3_rows)
print(q3_table.to_string(index=False))

# %% [markdown] id="91f6ecac"
# for each leaf table we'd get `3·n_numeric + 1·n_categorical + 2·n_timestamp`
# features. without any filtering that'd give max 15 features per path through
# users, 96 total at depth 2, 228 at depth 3.
#
# but we drop PK/FK columns (ACORA — just row ids, no signal), drop
# `Users.AccountId` per the instructions, and treat `badges.TagBased` (bool) as
# categorical so it only gets a count. that knocks the totals down quite a bit:
#
# - depth 1 is just `target → users`. after filtering we keep `ProfileImageUrl`
#   (numeric, 3 aggs), 4 text/categorical fields (`DisplayName`, `Location`,
#   `WebsiteUrl`, `AboutMe`, 1 agg each), and `CreationDate` (timestamp, 2 aggs).
#   total = 9.
#
# - depth 2 adds 5 more paths, one per leaf:
#
#   | leaf | num | cat | ts | features |
#   |---|---|---|---|---|
#   | badges      | 1 | 2 | 1 | 7 |
#   | comments    | 0 | 3 | 1 | 5 |
#   | postHistory | 1 | 5 | 1 | 10 |
#   | posts       | 1 | 5 | 1 | 10 |
#   | votes       | 1 | 0 | 1 | 5 |
#
#   depth-2 sums to 37; plus depth-1's 9 → 46 features (vs the 96 max).
#
# - depth 3 adds 8 more paths (posts and the post-related tables fan back into
#   each other): comments→posts (10), postHistory→posts (10), posts→comments (5),
#   posts→postHistory (10), posts→postLinks via PostId (5), posts→postLinks via
#   RelatedPostId (5), posts→votes (5), votes→posts (10). that's 60, so the total
#   is 9 + 37 + 60 = 106 (vs 228 max).
#
# the count grows fast with depth because every hop fans out into several
# leaves. the PK/FK filter roughly halves the count at every level.

# %% [markdown] id="9fc14017"
# ### Q4. Shape of `feature_matrix` and how it differs from the original target table

# %% id="621af761" outputId="2e218379-b6de-4825-bead-f53cd043cdc6"
print(f"Original target (train) shape: {train_df.shape}")
print(f"feature_matrix          shape: {feature_matrix.shape}")
print(f"  added columns          : {feature_matrix.shape[1] - train_df.shape[1]}")
print(f"  rows unchanged because left-outer join on (single) PK side preserves all "
      f"target rows.")

# %% [markdown] id="9d1c8f45"
# Same rows as the target table since left join preserves all tuples from the original target table.
# Columns grew by the number of generated features, since the feature matrix is a table combining all the aggragated features.

# %% [markdown] id="763a213b"
# ### Q5.
#
# We present 3 of the newly extracted features from feature table:
#
# 1. **users_votes__CreationDate__max** - This columns contains the date of the last vote casted by each user.
#
# 2. **users_comments__Text__count** - This column counts how many non-empty comments a user has written.
#
# 3. **users_posts__Score__sum** - This column sums the total score a user achieved acroos all of his posts.
#

# %% id="d94d2853" outputId="1bdbcee4-be81-4137-f84c-be1132c9efc0" colab={"base_uri": "https://localhost:8080/"}
examples = [
    ("d2__users_comments__Text__count",
     "Count of non-null text entries in `comments` joined to the user — i.e., **how many comments the user has written**."),
    ("d2__users_votes__CreationDate__max",
     "Maximum (latest) `CreationDate` over all `votes` cast by the user — i.e. **the timestamp of the user’s most-recent vote**."),
    ("d2__users_posts__Score__sum",
     "Sum of all `Score` values across all `posts` owned by the user — i.e., **the total net upvotes/reputation score accumulated by the user's posts**."),
]

ex_table = pd.DataFrame(
    [{"feature": n,
      "non-null %": round(feature_matrix[n].notna().mean()*100, 1) if n in feature_matrix else None,
      "mean":       round(feature_matrix[n].astype("float64").mean(), 2) if n in feature_matrix and feature_matrix[n].dtype.kind in "fi" else None,
      "meaning": m}
     for n, m in examples])

print(ex_table.to_string(index=False))

# %% [markdown] id="81994d67"
# ### Q6. Missing values in extracted features
#
# yes, plenty. couple of reasons.
#
# the first join (target → users) is a left join, so we keep every target row
# even if its user isn't in the users table — those rows get NaN on all user
# columns. at depth 2 we use an inner join, so users with no rows in the leaf
# table just fall out of the join. we then merge the aggregated result back to
# the target with another left join (we want every target row to survive), and
# users with no leaf rows end up NaN for that whole depth-2 block. for count
# aggregations we fill those with 0 — a user with no comments has 0 comments,
# not "unknown". for mean/sum/min/max we leave the NaN, because aggregating over
# nothing is genuinely undefined.
#
# a couple of other sources too: some source columns are already NaN-heavy.
# `posts.ParentId` is null for top-level questions, which propagates into
# mean/sum. and `users.ProfileImageUrl` is entirely null in this dataset, so
# every aggregation built from it comes out NaN.

# %% id="c8aac2a7" outputId="087f2372-4688-43f1-ec5f-6a9d1b045225"
new_cols = [c for c in feature_matrix.columns if c.startswith("d1__") or c.startswith("d2__")]
miss = feature_matrix[new_cols].isna().mean().sort_values(ascending=False)
print("Top 10 missingness rates among new features:")
print(miss.head(10).round(3).to_string())
print(f"\nMean missingness across all {len(new_cols)} new features: {miss.mean()*100:.1f}%")
print(f"Fraction of new features that are 100% missing: {(miss == 1).mean()*100:.1f}%")

# %% [markdown] id="e02a0b00"
# ## Q7 — Train DecisionTree, XGBoost, Logistic Regression and a PyTorch NN
#
# We compute the auto-features for `val` and `test`, then train on the depth-2 feature
# matrix.

# %% id="aba3a53f" outputId="7a78235f-0618-48b2-88ad-1dc65203eeda"
print("Computing features for val/test ...")
val_feat,  _ = auto_features(val_df,  max_depth=2, split="val",  verbose=False)
test_feat, _ = auto_features(test_df, max_depth=2, split="test", verbose=False)
train_feat = feature_matrix  # already computed above

print("train_feat:", train_feat.shape, "| val_feat:", val_feat.shape, "| test_feat:", test_feat.shape)

# %% id="704f99a9" outputId="b626dc4b-c7ed-493f-b1c3-e7d0986508d9"
def to_xy(df: pd.DataFrame, label_col: str = "contribution"):
    """Convert a feature-matrix-style df to (X, y).
    Drops non-feature columns; encodes timestamp aggregations to int64 nanoseconds.
    """
    drop_cols = {"OwnerUserId", "timestamp", "contribution"}
    feat_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feat_cols].copy()
    # encode timestamp-typed columns
    for c in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[c]):
            # convert to seconds since epoch (float to allow NaN)
            X[c] = X[c].astype("int64").where(X[c].notna(), np.nan).astype("float64") / 1e9
        elif X[c].dtype == "object":
            # rare: object columns left over -> drop
            X = X.drop(columns=[c])
            feat_cols.remove(c)
    X = X.astype("float64")
    y = df[label_col].astype("int64").values if label_col in df.columns else None
    return X, y, feat_cols

X_tr, y_tr, feat_cols = to_xy(train_feat)
X_va, y_va, _         = to_xy(val_feat)
X_te, _,    _         = to_xy(test_feat)
print("Shapes:", X_tr.shape, X_va.shape, X_te.shape)
print("Dtypes ok:", X_tr.dtypes.value_counts().to_dict())
print("Train missingness:", X_tr.isna().mean().mean().round(3))

# %% [markdown] id="0860872c"
# XGBoost gets raw NaNs; DT gets median-imputed; LR and NN get imputed + scaled.

# %% id="4c60ec97" outputId="66392dab-65d4-4cbf-e56b-58662683d101"
imputer = SimpleImputer(strategy="median").fit(X_tr)
X_tr_imp = imputer.transform(X_tr)
X_va_imp = imputer.transform(X_va)
X_te_imp = imputer.transform(X_te)

scaler = StandardScaler().fit(X_tr_imp)
X_tr_sc = scaler.transform(X_tr_imp)
X_va_sc = scaler.transform(X_va_imp)
X_te_sc = scaler.transform(X_te_imp)

print("Preprocessing done")

# %% id="25f35969"
def evaluate(y_true, y_pred, y_prob):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "roc_auc":   roc_auc_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }

def eval_split(model, X, y, threshold=0.5):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    return evaluate(y, pred, prob)

# %% [markdown] id="fd40a4af"
# ### Decision Tree — hyperparameter sweep

# %% id="46d21fbe" outputId="eb15bbcc-0706-4c57-e100-25db0097be6a"
dt_results = []
for max_depth in [3, 5, 10, 20]:
    for min_samples_leaf in [50, 500]:
        clf = DecisionTreeClassifier(max_depth=max_depth,
                                     min_samples_leaf=min_samples_leaf,
                                     class_weight="balanced",
                                     random_state=42)
        clf.fit(X_tr_imp, y_tr)
        tr = eval_split(clf, X_tr_imp, y_tr)
        va = eval_split(clf, X_va_imp, y_va)
        dt_results.append({"max_depth": str(max_depth), "min_samples_leaf": min_samples_leaf,
                           **{f"val_{k}": round(v, 4) for k, v in va.items()},
                           **{f"tr_{k}": round(v, 4) for k, v in tr.items()}})
dt_df = pd.DataFrame(dt_results).sort_values("val_roc_auc", ascending=False)
print(dt_df.to_string(index=False))
dt_best_cfg = dt_df.iloc[0][["max_depth", "min_samples_leaf"]].to_dict()
dt_best = DecisionTreeClassifier(
    max_depth=None if dt_best_cfg["max_depth"] == "None" else int(dt_best_cfg["max_depth"]),
    min_samples_leaf=int(dt_best_cfg["min_samples_leaf"]),
    class_weight="balanced", random_state=42).fit(X_tr_imp, y_tr)
dt_metrics = {"train": eval_split(dt_best, X_tr_imp, y_tr),
              "val":   eval_split(dt_best, X_va_imp, y_va)}
print("Best DT cfg:", dt_best_cfg)
print("Best DT metrics:", dt_metrics)

# %% [markdown] id="e10574ed"
# ### XGBoost — hyperparameter sweep

# %% id="e8c60121" outputId="d0a73708-fe3a-4bab-d5ca-ebdbea0ee754"
neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
scale_pos_weight = neg / pos
xgb_results = []
for max_depth in [4, 6]:
    for n_estimators in [300]:
        for lr in [0.05, 0.1]:
            clf = xgb.XGBClassifier(
                max_depth=max_depth, n_estimators=n_estimators, learning_rate=lr,
                subsample=0.9, colsample_bytree=0.9,
                scale_pos_weight=scale_pos_weight, n_jobs=-1,
                eval_metric="auc", random_state=42,
                tree_method="hist",
            )
            clf.fit(X_tr, y_tr)
            tr = eval_split(clf, X_tr, y_tr)
            va = eval_split(clf, X_va, y_va)
            xgb_results.append({"max_depth": max_depth, "n_estimators": n_estimators, "lr": lr,
                                **{f"val_{k}": round(v, 4) for k, v in va.items()},
                                **{f"tr_{k}": round(v, 4) for k, v in tr.items()}})
xgb_df = pd.DataFrame(xgb_results).sort_values("val_roc_auc", ascending=False)
print(xgb_df.to_string(index=False))
xgb_best_cfg = xgb_df.iloc[0][["max_depth", "n_estimators", "lr"]].to_dict()
xgb_best = xgb.XGBClassifier(
    max_depth=int(xgb_best_cfg["max_depth"]),
    n_estimators=int(xgb_best_cfg["n_estimators"]),
    learning_rate=float(xgb_best_cfg["lr"]),
    subsample=0.9, colsample_bytree=0.9,
    scale_pos_weight=scale_pos_weight, n_jobs=-1,
    eval_metric="auc", random_state=42, tree_method="hist",
).fit(X_tr, y_tr)
xgb_metrics = {"train": eval_split(xgb_best, X_tr, y_tr),
               "val":   eval_split(xgb_best, X_va, y_va)}
print("Best XGB cfg:", xgb_best_cfg)
print("Best XGB metrics:", xgb_metrics)

# %% [markdown] id="cab9d3e3"
# ### Logistic Regression — hyperparameter sweep

# %% id="27fae236" outputId="17e7ec54-410f-4eb5-fd71-aafd35d63784"
lr_results = []
for C in [0.01, 0.1, 1.0, 10.0]:
    clf = LogisticRegression(C=C, class_weight="balanced",
                             max_iter=2000, solver="lbfgs", n_jobs=-1)
    clf.fit(X_tr_sc, y_tr)
    tr = eval_split(clf, X_tr_sc, y_tr)
    va = eval_split(clf, X_va_sc, y_va)
    lr_results.append({"C": C,
                       **{f"val_{k}": round(v, 4) for k, v in va.items()},
                       **{f"tr_{k}": round(v, 4) for k, v in tr.items()}})
lr_df = pd.DataFrame(lr_results).sort_values("val_roc_auc", ascending=False)
print(lr_df.to_string(index=False))
lr_best_C = float(lr_df.iloc[0]["C"])
lr_best = LogisticRegression(C=lr_best_C, class_weight="balanced",
                             max_iter=2000, solver="lbfgs", n_jobs=-1).fit(X_tr_sc, y_tr)
lr_metrics = {"train": eval_split(lr_best, X_tr_sc, y_tr),
              "val":   eval_split(lr_best, X_va_sc, y_va)}
print("Best LR C:", lr_best_C)
print("Best LR metrics:", lr_metrics)

# %% [markdown] id="e1be0923"
# ### NN in PyTorch
#
# 3-layer MLP (128→64→1), ReLU + Dropout(0.3). BCEWithLogitsLoss with pos_weight.
# Adam lr=1e-3, batch_size=4096, 5 epochs, best val-AUC checkpoint.

# %% id="d2d954ae" outputId="69a30fc9-0bf3-4af3-ac22-fb05910a7dd7"
class MLP(nn.Module):
    def __init__(self, in_dim, hid=(128, 64), p=0.3):
        super().__init__()
        layers = []
        last = in_dim
        for h in hid:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(p)]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_mlp(X_tr_sc, y_tr, X_va_sc, y_va, epochs=5, batch_size=4096, lr=1e-3, p=0.3,
              hid=(128,64), seed=42, verbose=True):
    torch.manual_seed(seed)
    Xtr = torch.tensor(X_tr_sc, dtype=torch.float32)
    ytr = torch.tensor(y_tr,   dtype=torch.float32)
    Xva = torch.tensor(X_va_sc, dtype=torch.float32)
    yva = torch.tensor(y_va,   dtype=torch.float32)
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                              dtype=torch.float32)
    model = MLP(Xtr.shape[1], hid=hid, p=p).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    ds = TensorDataset(Xtr, ytr)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0,
                    pin_memory=False)
    best_auc = 0.0
    best_state = None
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va_logits = model(Xva.to(DEVICE)).cpu().numpy()
        va_prob = 1 / (1 + np.exp(-va_logits))
        auc = roc_auc_score(y_va, va_prob)
        if verbose:
            print(f"  epoch {ep+1}: val AUC = {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_auc

def mlp_predict(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    prob = 1 / (1 + np.exp(-logits))
    pred = (prob >= 0.5).astype(int)
    return pred, prob

# Small architecture sweep
nn_results = []
nn_models = {}
for hid in [(64,), (128, 64), (256, 128, 64)]:
    print(f"\nNN hid={hid}")
    model, _ = train_mlp(X_tr_sc, y_tr, X_va_sc, y_va, epochs=5, hid=hid)
    pred_tr, prob_tr = mlp_predict(model, X_tr_sc)
    pred_va, prob_va = mlp_predict(model, X_va_sc)
    tr = evaluate(y_tr, pred_tr, prob_tr)
    va = evaluate(y_va, pred_va, prob_va)
    nn_results.append({"hid": str(hid),
                       **{f"val_{k}": round(v,4) for k,v in va.items()},
                       **{f"tr_{k}": round(v,4) for k,v in tr.items()}})
    nn_models[str(hid)] = model
nn_df = pd.DataFrame(nn_results).sort_values("val_roc_auc", ascending=False)
print("\nNN sweep:")
print(nn_df.to_string(index=False))
nn_best = nn_models[nn_df.iloc[0]["hid"]]
nn_best_hid = nn_df.iloc[0]["hid"]
pred_tr, prob_tr = mlp_predict(nn_best, X_tr_sc)
pred_va, prob_va = mlp_predict(nn_best, X_va_sc)
nn_metrics = {"train": evaluate(y_tr, pred_tr, prob_tr),
              "val":   evaluate(y_va, pred_va, prob_va)}
print("Best NN hid:", nn_best_hid, "metrics:", nn_metrics)

# %% [markdown] id="SBHGS9WurtYH"
# # Hyper Parameters Discussion
#
# **Decision Tree Hyperparameters** - We evaluated the performance of a Decision Tree model while varying the hyperparameters max_depth and min_samples_leaf. We tested max_depth values of 3, 5, 10, and 20, and min_samples_leaf values of 50 and 500. We chose these ranges for two main reasons:
#
# Feature amount - Since our feature table contains 46 features (after the ACORA-style PK/FK + AccountId filter, down from a 96-feature maximum), we wanted to see whether deeper trees can better capture information spread across many features, or whether a smaller depth is already enough to describe user behavior and generalizes better to new samples without overfitting.
#
# Number of examples - The dataset contains a large number of training examples, so individual leaves can easily become too specific to small subsets of data. The min_samples_leaf parameter controls this by forcing each leaf to contain at least a certain number of samples. Using larger values like 50 or 500 helps prevent the tree from creating overfited patterns, and instead encourages more stable and general rules that better reflect overall trends in the data.
#
# **XGBoost Hyperparametes** - We evaluated the performance of an XGBoost model while varying the hyperparameters max_depth (4 and 6) and learning rate (lr) values of 0.05 and 0.1. Since XGBoost is an ensemble of decision trees built sequentially, we avoided using very large tree depths to reduce the risk of overfitting and keep the individual trees relatively simple. We also tested two learning rates to check how sensitive the model is to step size- whether a larger learning rate leads to faster convergence but possibly more overfitting, or whether it still generalizes well compared to the smaller value of 0.05.
#
# **Logistic Regression Hyperparameters** - We evaluated the performance of a Logistic Regression model while varying the hyperparameter C with values 0.01, 0.1, 1 and 10. Our goal here was to see the affect of smaller\bigger regularization on the performance of the model, and if it tends to overfit.
#
# **NN model** - We evaluated the performance of a simple NN model while varying the hyperparameter hid with values(64,), (128, 64) and (256, 128, 64). Our goal here was to see the affect of varying hidden layers on the performance of the model.
#
#
# All the relevant results of all the hyperparameters variations arepresented in the above tables.

# %% [markdown] id="b997d328"
# ### Combined results table (Q7)

# %% id="b0c15b53" outputId="e336c34e-edf5-4287-d315-d845b53bbf46"
def fmt_pair(tr, va, k):
    return f"{tr[k]:.3f} / {va[k]:.3f}"

q7_table = pd.DataFrame(
    [
        {"model": "Decision Tree",       **{m: fmt_pair(dt_metrics["train"],  dt_metrics["val"],  m) for m in ["accuracy","roc_auc","precision","recall","f1"]}},
        {"model": "XGBoost",             **{m: fmt_pair(xgb_metrics["train"], xgb_metrics["val"], m) for m in ["accuracy","roc_auc","precision","recall","f1"]}},
        {"model": "Logistic Regression", **{m: fmt_pair(lr_metrics["train"],  lr_metrics["val"],  m) for m in ["accuracy","roc_auc","precision","recall","f1"]}},
        {"model": "MLP (PyTorch)",       **{m: fmt_pair(nn_metrics["train"],  nn_metrics["val"],  m) for m in ["accuracy","roc_auc","precision","recall","f1"]}},
    ]
)
print("Q7 — Train / Val metrics (best variant per model):")
print(q7_table.to_string(index=False))


# %% [markdown] id="c7bf40de"
# ## Q8 — Evaluation measures
#
# * **Accuracy** = Accuracy is defined as (TP+TN) / total. This is the rate of how many predictions our model was correct. This meassure can be misleading when trained on an imbalanced data. With 97% negatives, a trivial all-zero classifier scores 0.97, so accuracy is nearly useless is this imbalanced data.
#
#
# * **AUROC** = probability a random positive ranks above a random negative. A model that achieves a high probability, is a model that seperates the negative and positive examples well. AUROC is threshold-free, class-prior invariant.
#
#
# * **Precision** = Precision is defined as TP/(TP+FP) This measures, out of all the examples the model predicted as positive, how many were actually correct. So it reflects how “trustworthy” positive predictions are. A high precision means that when the model says "positive" it is usually right. However, precision ignores false negatives, so the model could still miss many actual positives without hurting precision.
#
# * **Recall** = Recall is defined as TP/(TP+FN). This measures how many of the actual positive cases the model manages to find. A high recall means the model misses very few positives, but it might also produce more false positives.
#
# * **F1** = F1 is defined as 2(Presicion*Recall)/(Precision+Recall). This METHOD combines precision and recall into a single number, balancing both. It becomes high only when both precision and recall are reasonably high. It is useful when you want a single metric that reflects performance on the positive class.
#
#
#
#
# **Tradeoffs** = If we increase the classification threshold, the model becomes more conservative in predicting positives. As a result, precision increases because fewer examples are labeled as positive, and those that do pass the higher threshold are more likely to be truly positive. On the other hand, recall decreases since the model misses more "weak" positive cases that fall below the threshold.
#
# AUROC evaluates performance across all possible thresholds, so it is independent of any specific decision cutoff. In contrast, F1 depends on a chosen threshold and reflects performance at that specific operating point. Accuracy is less informative in imbalanced settings, since it can remain high even for trivial models that mostly predict the majority class.
#
# With around ~3% positive examples, AUROC and F1 are more meaningful metrics. AUROC measures how well the model separates positive from negative cases overall, while F1 focuses specifically on performance for the positive class by balancing the ability to find true positives (recall) and avoid false positives (precision), which is especially important in imbalanced datasets and captured by F1.

# %% [markdown] id="584e9331"
# ## Q9 — Discussion of results
#
# on the training set, the Decision Tree gets the best AUROC (0.855), with
# XGBoost right behind (0.845). XGBoost takes the best train F1 (0.259). LR is
# last on both — its linear boundary just can't fit the patterns the trees
# pick up. the trees model the feature interactions directly, which shows up
# in their higher recall and AUROC on train.
#
# on the validation set the ordering shifts a bit. XGBoost has the best val
# AUROC (0.881), MLP is right next to it (0.877), and DT is just behind
# (0.872). LR wins on F1 (0.164) — its precision/recall balance at the default
# threshold beats the trees even though its AUROC is lower.
#
#
# * **Why these differences.** The MLP performs well because it can learn non-linear patterns in the standardized features, which matches the way user activity and engagement usually behave. Logistic Regression also works well since many features, such as post counts or recent activity, have a fairly direct relationship with the probability of engagement after scaling. XGBoost is able to capture more complex interactions between features using decision trees, but on this feature set it does not improve much over Logistic Regression, maybe the XGBoost overfitted the training a bit more than LR and our NN. The single Decision Tree performs worse because one tree alone has difficulty combining many correlated activity features, unlike ensemble methods or smoother models such as LR and MLP.
#
# * **Effect of evaluation measure.** Accuracy ranks the models very differently from AUROC — with 97% negatives, a model that mostly predicts 0 scores high on accuracy but low on AUROC and F1. AUROC and F1 are the meaningful metrics here.
#
# * **Effect of schema and task.** Because the task predicts user engagement, most of the useful information comes from the depth-2 paths target → users → {posts, comments, votes, postHistory, badges}. The dominant features across all models are counts and recency aggregations from these tables — essentially, how active the user was in the past and how recently they interacted with the platform. Features such as the number of posts, comments, votes, and badges capture the overall level of user activity, while timestamp-based features indicate whether the user has been active recently. In the REL-Stack dataset, recent and frequent activity is strongly correlated with future engagement, which is why simple aggregations like counts and maximum timestamps become highly informative features for the prediction task.

# %% [markdown] id="333aeb3b"
# ## Q10 — How feature importance is computed in each model
#
# - Decision Tree — Gini importance. for every feature we sum the Gini impurity
#   it reduced at each node it split on, weighted by how many samples reached
#   that node. then normalise so importances add up to 1.
# - XGBoost — average gain. for every feature, the mean loss-improvement it gave
#   across all the splits on that feature across all boosting rounds.
# - Logistic Regression — we just take `|β_j|`. since the features are
#   standardised the coefficients are directly comparable, and a bigger
#   `|β_j|` means the feature has more pull on the predicted log-odds.

# %% [markdown] id="a6ca6b8e"
# ## Q11 — Top-10 features per model

# %% id="bf10cea7" outputId="1f50cec9-6ec6-4934-943a-a6514a6795b9"
def top_features(model, feature_names, k=10, attr="auto"):
    """Return top-k (feature_name, importance) pairs."""
    if attr == "auto":
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_)
        elif hasattr(model, "coef_"):
            imp = np.abs(np.asarray(model.coef_)).ravel()
        else:
            raise ValueError("no importance attribute")
    else:
        imp = np.abs(np.asarray(getattr(model, attr))).ravel()
    order = np.argsort(imp)[::-1][:k]
    return [(feature_names[i], float(imp[i])) for i in order]

dt_top = top_features(dt_best, feat_cols)
xgb_top = top_features(xgb_best, feat_cols)
lr_top = top_features(lr_best, feat_cols)

top_df = pd.DataFrame({
    "rank":            list(range(1, 11)),
    "DT_feature":      [f for f, _ in dt_top],
    "DT_imp":          [round(v, 4) for _, v in dt_top],
    "XGB_feature":     [f for f, _ in xgb_top],
    "XGB_imp":         [round(v, 4) for _, v in xgb_top],
    "LR_feature":      [f for f, _ in lr_top],
    "LR_|coef|":       [round(v, 4) for _, v in lr_top],
})
print("Top-10 features per model (depth-2 tabular features only):")
print(top_df.to_string(index=False))

# %% [markdown] id="fe38b99c"
# **Commonalities & differences.**
#
# all three models put activity volume and recency near the top — counts of
# posts/badges/postHistory rows, and earliest-action timestamps like
# `posts.CreationDate.min` and `postHistory.CreationDate.min`. that tracks with
# what we'd expect from a user-engagement task: "how active was this user, and
# how recently?" is basically the whole signal.
#
# the trees and the linear model differ in where they go looking for that
# signal. DT and XGBoost lean on count aggregations from the big child tables
# (`posts.PostTypeId.count`, `posts.ContentLicense.count`) — trees deal with
# raw scale fine. LR, with standardised features, ends up leaning heavily on
# the badges block (`badges.Class.count`, `badges.Date.min`,
# `badges.TagBased.count`, `badges.Name.count`). badges are kind of a clean
# activity proxy, so once scaled their coefficients dominate.
#
# between DT and XGBoost — the single tree leans on whichever feature gives the
# cleanest first split, so its top-10 is steep and concentrated. XGBoost spreads
# importance across many features because each boosting round picks up a
# different slice of the residual.


# %% [markdown] id="1c6cf728"
# ## Part 2 — Graph Feature Extraction
#
# now we treat the database as a graph. every tuple of every table becomes a
# node, identified by `(table_name, row_id)`, and every FK-PK pair becomes an
# undirected edge between the two tuples it connects.

# %% [markdown] id="f79e7427"
# ### Q12. Build the graph and compute (a) degree, (b) WL coloring, (c) two graphlet counts
#
# we build three graphs, one per split, following the edited temporal rule:
# train uses tuples strictly before `val_timestamp` (2020-10-01), val uses
# tuples up to and including `val_timestamp`, test uses everything in the db
# (already pre-cut at `test_timestamp`). for each graph we then compute the
# degree, WL colours, and graphlet counts, and merge them onto the matching
# split's feature matrix.

# %% id="c7b06f48" outputId="0769d2ea-0138-4d17-d3af-6faa9813112d"
def build_graph(cutoff_ts, inclusive: bool):
    """Build a (G, adj, filtered_tables) for tuples with time_col <= cutoff_ts
    (or < cutoff_ts when inclusive=False). cutoff_ts=None keeps all tuples."""
    G = nx.Graph()
    adj: Dict[Tuple[str,int], set] = defaultdict(set)
    filtered: Dict[str, pd.DataFrame] = {}

    for tbl_name, tbl in db.table_dict.items():
        df = tbl.df
        if cutoff_ts is not None and tbl.time_col is not None and tbl.time_col in df.columns:
            df = df[df[tbl.time_col] <= cutoff_ts] if inclusive else df[df[tbl.time_col] < cutoff_ts]
        filtered[tbl_name] = df
        pk_col = tbl.pkey_col
        ids = df[pk_col].dropna().astype("Int64").astype("int64").to_numpy()
        G.add_nodes_from(((tbl_name, int(i)) for i in ids), table=tbl_name)

    for parent, child, fk_col in SCHEMA_FKEYS:
        if child == "_target_":
            continue
        child_df = filtered[child]
        child_pk_col = db.table_dict[child].pkey_col
        parent_pk_col = db.table_dict[parent].pkey_col
        parent_ids = set(filtered[parent][parent_pk_col].dropna().astype("int64").to_numpy())
        pairs = child_df[[child_pk_col, fk_col]].dropna()
        pairs = pairs.astype({child_pk_col: "int64", fk_col: "int64"})
        pairs = pairs[pairs[fk_col].isin(parent_ids)]
        edge_list = list(zip(
            [(child, int(x))  for x in pairs[child_pk_col].to_numpy()],
            [(parent, int(x)) for x in pairs[fk_col].to_numpy()],
        ))
        G.add_edges_from(edge_list)
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
    return G, adj, filtered

print("Building 3 graphs (train < val_timestamp, val ≤ val_timestamp, test = all) ...")
t0 = time.time()
G_train, adj_train, filt_train = build_graph(cutoff_ts=VAL_TIMESTAMP, inclusive=False)
print(f"  train: nodes={G_train.number_of_nodes():,}  edges={G_train.number_of_edges():,}")
G_val,   adj_val,   filt_val   = build_graph(cutoff_ts=VAL_TIMESTAMP, inclusive=True)
print(f"  val  : nodes={G_val.number_of_nodes():,}  edges={G_val.number_of_edges():,}")
G_test,  adj_test,  filt_test  = build_graph(cutoff_ts=None,          inclusive=True)
print(f"  test : nodes={G_test.number_of_nodes():,}  edges={G_test.number_of_edges():,}")
print(f"  built all 3 in {time.time()-t0:.1f}s")

# %% [markdown] id="29ff7a34"
# **(a) Degree.** for a user this is just the count of `posts`, `badges`,
# `comments`, `postHistory`, and `votes` rows that point at them. computed
# independently per split graph.

# %% id="266cd062" outputId="67be0917-d1c8-44f9-c2a7-3cf9954f1864"
users_ids = db.table_dict["users"].df["Id"].dropna().astype("int64").unique()

def degree_per_user(adj_map) -> pd.Series:
    return pd.Series({uid: len(adj_map[("users", int(uid))]) for uid in users_ids},
                     name="graph_degree", dtype="int64")

deg_train = degree_per_user(adj_train)
deg_val   = degree_per_user(adj_val)
deg_test  = degree_per_user(adj_test)
print("Degree stats over all users (test graph):")
print(deg_test.describe().round(2))
print(f"\nmean degree per split:  train={deg_train.mean():.2f}  "
      f"val={deg_val.mean():.2f}  test={deg_test.mean():.2f}")

# %% [markdown] id="89098a8b"
# **(b) Weisfeiler–Lehman colours.** every node starts with a colour that's just
# a hash of its table name. each iteration replaces a node's colour with a hash
# of its own current colour plus the sorted list of its neighbours' colours.
# we record the colours after K = 2, 3, and 5 iterations, and run it separately
# on each split's graph.

# %% id="cfd85205" outputId="266930b8-dfba-450d-a56d-2a33f8b098c2"
def wl_compute(adj_map: dict, init_label_fn, K_max: int = 5):
    """Run K_max WL iterations, return {K: {node: color_id}} for K = 1..K_max."""
    colors = {n: hash(init_label_fn(n)) for n in adj_map}
    out_at = {}
    for it in tqdm(range(1, K_max + 1), desc="WL iter"):
        new_colors = {}
        for n, nbrs in adj_map.items():
            nbr_cols = tuple(sorted(colors[u] for u in nbrs))
            new_colors[n] = hash((colors[n], nbr_cols))
        colors = new_colors
        out_at[it] = dict(colors)
    return out_at

def wl_features_for(adj_map) -> pd.DataFrame:
    wl_at = wl_compute(adj_map, init_label_fn=lambda n: n[0], K_max=5)
    def user_colors(color_dict):
        return pd.Series({uid: color_dict.get(("users", int(uid)), 0) for uid in users_ids})
    df = pd.DataFrame({
        "wl_K2": user_colors(wl_at[2]).factorize()[0],
        "wl_K3": user_colors(wl_at[3]).factorize()[0],
        "wl_K5": user_colors(wl_at[5]).factorize()[0],
    }, index=users_ids)
    df.index.name = "OwnerUserId"
    return df

print("Running WL on train graph ...");  t0 = time.time()
wl_train = wl_features_for(adj_train); print(f"  done in {time.time()-t0:.1f}s")
print("Running WL on val graph ...");    t0 = time.time()
wl_val   = wl_features_for(adj_val);   print(f"  done in {time.time()-t0:.1f}s")
print("Running WL on test graph ...");   t0 = time.time()
wl_test  = wl_features_for(adj_test);  print(f"  done in {time.time()-t0:.1f}s")
print("\nDistinct WL colours among users (test graph):")
print(wl_test.nunique().to_dict())

# %% [markdown] id="3c2ce061"
# **(c) Graphlet counts.** two motifs to count per user:
#
# - `graphlet_self_comment` — how many times the user commented on a post they
#   themselves own (cycle `user → comment → post → user`).
# - `graphlet_linked_self_posts` — how many times two posts owned by the user
#   are linked by a `postLinks` edge (cycle `user → post → postLink → post → user`).
#
# both come out cleanest as a couple of pandas joins on the split-filtered tables.

# %% id="c05ccd81" outputId="94a82f30-bbb9-478e-9587-76f0dece60bc"
def graphlet_features_for(filtered) -> pd.DataFrame:
    posts_owner = filtered["posts"][["Id", "OwnerUserId"]].dropna()
    comments_   = filtered["comments"][["UserId", "PostId"]].dropna()
    gj = comments_.merge(posts_owner.rename(columns={"Id": "PostId"}),
                         on="PostId", how="inner")
    self_comments = gj.loc[gj["UserId"] == gj["OwnerUserId"]] \
                      .groupby("UserId").size().rename("graphlet_self_comment")

    pl = filtered["postLinks"][["PostId", "RelatedPostId"]].dropna()
    gj2 = pl.merge(posts_owner.rename(columns={"Id": "PostId",
                                               "OwnerUserId": "ownerA"}), on="PostId") \
            .merge(posts_owner.rename(columns={"Id": "RelatedPostId",
                                               "OwnerUserId": "ownerB"}), on="RelatedPostId")
    linked_self_posts = gj2[gj2["ownerA"] == gj2["ownerB"]] \
                          .groupby("ownerA").size().rename("graphlet_linked_self_posts")

    out = pd.DataFrame({"graphlet_self_comment":      self_comments,
                        "graphlet_linked_self_posts": linked_self_posts}) \
            .reindex(users_ids).fillna(0).astype("int64")
    out.index.name = "OwnerUserId"
    return out

graphlet_train = graphlet_features_for(filt_train)
graphlet_val   = graphlet_features_for(filt_val)
graphlet_test  = graphlet_features_for(filt_test)
print("Graphlet feature stats (test graph, over all users):")
print(graphlet_test.describe().round(2))

# %% [markdown] id="0fbfd6eb"
# **Adding graph features to the feature matrix.** Each split's feature matrix gets the
# graph features computed from the matching graph.

# %% id="08ca1461" outputId="d405c839-71ce-412f-d6ff-3f3b0c6db502"
def graph_features_df(deg, graphlets, wl) -> pd.DataFrame:
    df = pd.concat([deg.rename("graph_degree"), graphlets, wl], axis=1)
    df.index.name = "OwnerUserId"
    return df

graph_features_train = graph_features_df(deg_train, graphlet_train, wl_train)
graph_features_val   = graph_features_df(deg_val,   graphlet_val,   wl_val)
graph_features_test  = graph_features_df(deg_test,  graphlet_test,  wl_test)
graph_cols = list(graph_features_train.columns)

def add_graph_feats(fm: pd.DataFrame, gf: pd.DataFrame) -> pd.DataFrame:
    out = fm.merge(gf, left_on="OwnerUserId", right_index=True, how="left")
    out[graph_cols] = out[graph_cols].fillna(0)
    return out

train_feat_g = add_graph_feats(train_feat, graph_features_train)
val_feat_g   = add_graph_feats(val_feat,   graph_features_val)
test_feat_g  = add_graph_feats(test_feat,  graph_features_test)
print("Augmented shapes:", train_feat_g.shape, val_feat_g.shape, test_feat_g.shape)

# %% [markdown] id="52fe77e5"
# ### Q13 — Retrain all four models on the tabular + graph feature matrix

# %% id="fc458265" outputId="7efaa2f1-178f-4d64-cea7-ad66153a6b63"
X_tr2, y_tr2, feat_cols2 = to_xy(train_feat_g)
X_va2, y_va2, _          = to_xy(val_feat_g)

imp2 = SimpleImputer(strategy="median").fit(X_tr2)
X_tr2_imp = imp2.transform(X_tr2)
X_va2_imp = imp2.transform(X_va2)
sc2 = StandardScaler().fit(X_tr2_imp)
X_tr2_sc = sc2.transform(X_tr2_imp)
X_va2_sc = sc2.transform(X_va2_imp)
print("Augmented preprocessing done. n_features =", X_tr2.shape[1])

# DT with best-from-before cfg
dt2 = DecisionTreeClassifier(
    max_depth=None if dt_best_cfg["max_depth"]=="None" else int(dt_best_cfg["max_depth"]),
    min_samples_leaf=int(dt_best_cfg["min_samples_leaf"]),
    class_weight="balanced", random_state=42).fit(X_tr2_imp, y_tr2)
dt2_metrics = {"train": eval_split(dt2, X_tr2_imp, y_tr2),
               "val":   eval_split(dt2, X_va2_imp, y_va2)}

# XGB
xgb2 = xgb.XGBClassifier(
    max_depth=int(xgb_best_cfg["max_depth"]),
    n_estimators=int(xgb_best_cfg["n_estimators"]),
    learning_rate=float(xgb_best_cfg["lr"]),
    subsample=0.9, colsample_bytree=0.9,
    scale_pos_weight=scale_pos_weight, n_jobs=-1,
    eval_metric="auc", random_state=42, tree_method="hist",
).fit(X_tr2, y_tr2)
xgb2_metrics = {"train": eval_split(xgb2, X_tr2, y_tr2),
                "val":   eval_split(xgb2, X_va2, y_va2)}

# LR
lr2 = LogisticRegression(C=lr_best_C, class_weight="balanced",
                         max_iter=2000, solver="lbfgs", n_jobs=-1).fit(X_tr2_sc, y_tr2)
lr2_metrics = {"train": eval_split(lr2, X_tr2_sc, y_tr2),
               "val":   eval_split(lr2, X_va2_sc, y_va2)}

# NN (best hid)
nn2, _ = train_mlp(X_tr2_sc, y_tr2, X_va2_sc, y_va2, epochs=5,
                   hid=eval(nn_best_hid), verbose=False)
pred_tr2, prob_tr2 = mlp_predict(nn2, X_tr2_sc)
pred_va2, prob_va2 = mlp_predict(nn2, X_va2_sc)
nn2_metrics = {"train": evaluate(y_tr2, pred_tr2, prob_tr2),
               "val":   evaluate(y_va2, pred_va2, prob_va2)}

q13_table = pd.DataFrame([
    {"model": "Decision Tree",       **{m: fmt_pair(dt2_metrics["train"],  dt2_metrics["val"],  m) for m in ["accuracy","roc_auc","precision","recall","f1"]}},
    {"model": "XGBoost",             **{m: fmt_pair(xgb2_metrics["train"], xgb2_metrics["val"], m) for m in ["accuracy","roc_auc","precision","recall","f1"]}},
    {"model": "Logistic Regression", **{m: fmt_pair(lr2_metrics["train"],  lr2_metrics["val"],  m) for m in ["accuracy","roc_auc","precision","recall","f1"]}},
    {"model": "MLP (PyTorch)",       **{m: fmt_pair(nn2_metrics["train"],  nn2_metrics["val"],  m) for m in ["accuracy","roc_auc","precision","recall","f1"]}},
])
print("Q13 — Train / Val metrics WITH graph features:")
print(q13_table.to_string(index=False))

# %% [markdown] id="a8fbdee3"
# ### Q14 — Top-10 features per model (now with graph features available)

# %% id="698a1288" outputId="dcc1a2f4-3253-4103-c433-7ddf77c864ae"
dt2_top = top_features(dt2, feat_cols2)
xgb2_top = top_features(xgb2, feat_cols2)
lr2_top = top_features(lr2, feat_cols2)

graph_feat_set = set(graph_cols)
def annotate(rank_top, k=10):
    rows = []
    for f, v in rank_top[:k]:
        kind = "graph" if f in graph_feat_set else "tabular"
        rows.append((f, round(v,4), kind))
    return rows

top2 = pd.DataFrame({
    "rank":           list(range(1, 11)),
    "DT_feature":     [r[0] for r in annotate(dt2_top)],
    "DT_kind":        [r[2] for r in annotate(dt2_top)],
    "XGB_feature":    [r[0] for r in annotate(xgb2_top)],
    "XGB_kind":       [r[2] for r in annotate(xgb2_top)],
    "LR_feature":     [r[0] for r in annotate(lr2_top)],
    "LR_kind":        [r[2] for r in annotate(lr2_top)],
})
print("Top-10 features (graph in **bold** annotation):")
print(top2.to_string(index=False))

# How many graph features made each model's top-10?
print()
for name, lst in [("DT", dt2_top), ("XGB", xgb2_top), ("LR", lr2_top)]:
    g = sum(1 for f, _ in lst[:10] if f in graph_feat_set)
    print(f"  {name:>3}: graph features in top-10 = {g}")

# %% [markdown] id="35d9b2a9"
# **Discussion of graph features (Q14).**
#
# the picture is mixed. train AUROC jumped a lot for the trees and the MLP
# (DT 0.855 → 0.957, XGB 0.845 → 0.959, MLP 0.840 → 0.954), but val AUROC barely
# moved for XGB (0.881 → 0.873) and MLP (0.877 → 0.865), and actually *dropped*
# for DT (0.872 → 0.827). that gap is mostly `graph_degree` leaking — its train
# graph spans everything before `val_timestamp`, so the degree of a 2015 user
# already includes activity all the way to 2020-10-01, which is exactly the kind
# of future-peeking the per-row rule on the tabular features was supposed to
# block. Q12 only asks for the looser global cutoff on the graph, so we follow
# that, but the trees end up overfitting to it.
#
# as for which models actually use the graph features: XGB picked 3 into its
# top-10 (`graph_degree` at #1, `graphlet_self_comment` at #3, `wl_K2` at #9).
# LR picked 2 (`graph_degree` and `graphlet_linked_self_posts`). DT picked 0 —
# its top splits stay on the depth-2 tabular counts.
#
# the most useful graph feature was `graph_degree`. it's very correlated with
# the depth-2 count aggregations — degree is basically the sum of leaf counts —
# which is why it lands high in importance but doesn't move val much, and also
# why it drives the train overfitting. the graphlet counts add genuinely new
# structural info (cycles), but most users have zero so the lift is small. the
# WL colours add even less, since the immediate neighbourhood structure is
# already covered by the counts.

# %% [markdown] id="b531bd37"
# ## Q15 — Summary (conclusion paragraph for the report)
#
# **(a) Steps taken.**
#
# we started by loading rel-stack and looking at the schema and class balance
# (~5% positives, which set up most of the later discussion about AUROC vs
# accuracy). we then picked three manual features by hand (post count, vote
# recency, badge count) and built the automatic join-aggregate generator on top
# of that: BFS over the FK graph from the target, with the ACORA-style filter
# dropping PK/FK columns and `Users.AccountId`. at depth 2 we end up with 46
# features (down from a 96 max). with those we trained DT, XGBoost, LR, and a
# small MLP. val AUROC ends up around 0.83–0.88, well above the users-only
# relbench baseline of 0.65. finally we built one graph per split (with the
# temporal cutoffs Q12 asks for), computed degree, WL colours at K = 2, 3, 5,
# and two graphlet counts, and merged those back into the feature matrices.
#
# **(b) Auto vs manual features.**
#
# auto-features are great for coverage and reproducibility — you get a lot of
# features without thinking too hard about each one — but a chunk of them are
# weak or redundant (sums of mostly-null columns, min/max pairs that move
# together). manual features go the other way: each one is targeted and easy to
# interpret, but you're limited to whatever you happen to think of. in practice
# the best move is using them together — let the auto generator build a broad
# baseline, then add manual features for whatever domain knowledge the
# generator can't get to on its own.
#
# **(c) Graph vs tabular-only.**
#
# graph features can in principle capture structural patterns (cycles, motifs)
# that join-aggregates miss. but on this star-shaped schema they mostly re-encode
# what the count aggregations already capture, so the marginal val AUROC lift
# was small. the biggest single jump came from going users-only → depth-2
# tabular features. graph features on top of that didn't really move val, and
# because Q12 only enforces a global temporal cutoff (not per-row), the
# train-side `graph_degree` ended up leaking some future activity — which is
# why DT's val AUROC actually dropped slightly when we added the graph features.

# %% [markdown] id="6a181a8b"
# ## Q16 — AI-tool usage
#
# The code was generated entirely using AI (Claude), but the conclusions drawn from it are entirely our own. The final results were reasonable and consistent, which increased our confidence in the generated code. We validated the outputs in several ways: for example, the table in Q3 helped us verify the number of extracted features, and the training/validation metrics showed a clear and expected trade-off between underfitting and overfitting based on the selected hyperparameters. In addition, the handcrafted features were chosen by us, and it was reassuring to see that the AI independently identified similar features as the most relevant ones in Q1.
#
# As for part 2
# we could not check the the correctness of the graph but we did look at a few things that helped us gain confidence:
# 1. Graph size — 4,247,264 nodes, 5,812,887 edges. Matches sum of table rows and non-null FKs exactly.
#   2. Degree — user #352 manual count = 1,566 (129+449+725+260+3). We can verify it matches the graph.
#   3. Graphlets — user #352: 75 self-comments, 0 linked self-posts. Computed manually via pandas joins.
#   4. WL - distinct colors increases at every iteration.
#
# Overall, we carefully cross-checked the AI-generated work in these key areas. We estimate that the AI performed around 80% of the implementation work.
#

