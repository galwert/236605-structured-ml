# StructML (236605) — Assignment 1: Feature Extraction

Database: **rel-stack** (relbench). Task: **user-engagement** — predict if a user makes any
vote / post / comment in the next 3 months (binary classification).

This notebook walks through Q1–Q16 from the assignment in order. Heavy code is kept inline
(single-notebook deliverable, as requested).

## Part 0 — Setup & data loading

**If running on Google Colab**, run the cell below first to install `relbench` (and a
matching `pyarrow`/`huggingface-hub`). Other dependencies (`torch`, `xgboost`,
`networkx`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `tqdm`) ship with Colab.
A free-tier Colab runtime (~12 GB RAM) should fit the workload but is tight; if you
hit OOM, switch to a high-RAM runtime.

```text
torch 2.11.0 device = cpu
sklearn 1.6.1  | xgboost 3.2.0
```

```text
Loading Database object from /home/wertheizer/.cache/relbench/rel-stack/db...

Done in 14.28 seconds.

Tables in db : ['badges', 'comments', 'postHistory', 'postLinks', 'posts', 'users', 'votes']
  badges       rows=   463,463 pkey=Id     time=Date           fkeys={'UserId': 'users'}
  comments     rows=   623,967 pkey=Id     time=CreationDate   fkeys={'UserId': 'users', 'PostId': 'posts'}
  postHistory  rows= 1,175,368 pkey=Id     time=CreationDate   fkeys={'PostId': 'posts', 'UserId': 'users'}
  postLinks    rows=    77,337 pkey=Id     time=CreationDate   fkeys={'PostId': 'posts', 'RelatedPostId': 'posts'}
  posts        rows=   333,893 pkey=Id     time=CreationDate   fkeys={'OwnerUserId': 'users', 'ParentId': 'posts'}
  users        rows=   255,360 pkey=Id     time=CreationDate   fkeys={}
  votes        rows= 1,317,876 pkey=Id     time=CreationDate   fkeys={'PostId': 'posts', 'UserId': 'users'}

Target tables:
  train: shape=(1360850, 3), unique users=83,531, timestamps 2009-04-16..2020-07-02
  val  : shape=(85838, 3),  unique users=85,838, timestamps 2020-10-01..2020-10-01
  test : shape=(88137, 2),  unique users=88,137, timestamps 2021-01-01..2021-01-01

Class balance:
 train: {0: 0.95, 1: 0.05}
 val  : {0: 0.9719, 1: 0.0281}
```

## Part 1 — Tabular Feature Extraction

### Q1. Three hand-crafted join-aggregate features


Here are 3 join-aggragate features we consider as helpful for the prediction task:

1. **`posts_total`** - count of posts owned by the user (`target ⟕ users ⟕ posts`,
   count). Users who posted many posts before os more likely to keep posting.

2. **`comments_total`** - count of comments made by each user (`target ⟕ users ⟕ comments`,
   count). Since our goal is to predict user activity over the next three months, relying solely on post counts (posts_total) could cause us to miss different 'type' of users. There are plenty of highly active users who rarely publish their own posts but engage heavily in the comment sections instead.

3. **`badge_count`** - number of badges (`target ⟕ users ⟕ badges`, count).
   Since badges are achieved by the user's total activity, users with larger count of badges are more likely be active in the next 3 months.

```text
Q1 sample stats:
       q1_posts_total  q1_comments_total  q1_badge_count
count         2000.00            2000.00         2000.00
mean            33.76              99.56           16.62
std            202.30             877.49           62.31
min              0.00               0.00            0.00
25%              1.00               0.00            2.00
50%              2.00               2.00            4.00
75%              7.00               9.00           10.00
max           4672.00           23771.00         1384.00
```

### Q2. Automatic join-aggregate feature generation

we start from the target and walk the PK-FK schema with BFS. each path up to
`max_depth` hops becomes a join — left join into users first so we keep every
target row, then natural/inner joins for the rest. for each leaf table we
aggregate every column per target row: numeric → mean/count/sum, categorical
or text → count, timestamps → min/max.

before aggregating we drop PK and FK columns (ACORA — those are just ids, no
signal), plus `Users.AccountId` since the course instructions say to drop it.

temporal stuff: a prediction for row x can only see data from before
`x.timestamp`. for train we check this per row — after each join we drop rows
whose joined table's time column is later than the target timestamp. for val
we just slice every source table to `time_col ≤ val_timestamp` (2020-10-01)
before joining. for test we don't need to do anything; the db is already
pre-cut at `test_timestamp`.

```text
PK-FK edges in db:
  users        <--(UserId)-- badges
  users        <--(UserId)-- comments
  posts        <--(PostId)-- comments
  posts        <--(PostId)-- postHistory
  users        <--(UserId)-- postHistory
  posts        <--(PostId)-- postLinks
  posts        <--(RelatedPostId)-- postLinks
  users        <--(OwnerUserId)-- posts
  posts        <--(ParentId)-- posts
  posts        <--(PostId)-- votes
  users        <--(UserId)-- votes
```

```text
Enumerated 6 join paths up to depth 2:
   _target_ -> users
   _target_ -> users -> badges
   _target_ -> users -> comments
   _target_ -> users -> postHistory
   _target_ -> users -> posts
   _target_ -> users -> votes

paths:  17%|█▋        | 1/6 [00:01<00:06,  1.26s/it]
paths:  33%|███▎      | 2/6 [00:06<00:14,  3.55s/it]
paths:  50%|█████     | 3/6 [00:20<00:24,  8.30s/it]
paths:  67%|██████▋   | 4/6 [00:55<00:37, 18.79s/it]
paths:  83%|████████▎ | 5/6 [01:04<00:15, 15.34s/it]
paths: 100%|██████████| 6/6 [01:05<00:00, 10.37s/it]
paths: 100%|██████████| 6/6 [01:05<00:00, 10.87s/it]

Generated in 68.3s
feature_matrix shape: (1360850, 49)
```

### Q3. Feature counts at `max_depth` ∈ {1, 2, 3}

We re-run the auto-generator at depths 1 and 3 to count features programmatically (a
closed-form derivation follows the table).

```text
 max_depth  n_paths  n_features    paths_per_depth
         1        1           9             {1: 1}
         2        6          46       {1: 1, 2: 5}
         3       14         106 {1: 1, 2: 5, 3: 8}
```

for each leaf table we'd get `3·n_numeric + 1·n_categorical + 2·n_timestamp`
features. without any filtering that'd give max 15 features per path through
users, 96 total at depth 2, 228 at depth 3.

but we drop PK/FK columns (ACORA — just row ids, no signal), drop
`Users.AccountId` per the instructions, and treat `badges.TagBased` (bool) as
categorical so it only gets a count. that knocks the totals down quite a bit:

- depth 1 is just `target → users`. after filtering we keep `ProfileImageUrl`
  (numeric, 3 aggs), 4 text/categorical fields (`DisplayName`, `Location`,
  `WebsiteUrl`, `AboutMe`, 1 agg each), and `CreationDate` (timestamp, 2 aggs).
  total = 9.

- depth 2 adds 5 more paths, one per leaf:

  | leaf | num | cat | ts | features |
  |---|---|---|---|---|
  | badges      | 1 | 2 | 1 | 7 |
  | comments    | 0 | 3 | 1 | 5 |
  | postHistory | 1 | 5 | 1 | 10 |
  | posts       | 1 | 5 | 1 | 10 |
  | votes       | 1 | 0 | 1 | 5 |

  depth-2 sums to 37; plus depth-1's 9 → 46 features (vs the 96 max).

- depth 3 adds 8 more paths (posts and the post-related tables fan back into
  each other): comments→posts (10), postHistory→posts (10), posts→comments (5),
  posts→postHistory (10), posts→postLinks via PostId (5), posts→postLinks via
  RelatedPostId (5), posts→votes (5), votes→posts (10). that's 60, so the total
  is 9 + 37 + 60 = 106 (vs 228 max).

the count grows fast with depth because every hop fans out into several
leaves. the PK/FK filter roughly halves the count at every level.

### Q4. Shape of `feature_matrix` and how it differs from the original target table

```text
Original target (train) shape: (1360850, 3)
feature_matrix          shape: (1360850, 49)
  added columns          : 46
  rows unchanged because left-outer join on (single) PK side preserves all target rows.
```

Same rows as the target table since left join preserves all tuples from the original target table.
Columns grew by the number of generated features, since the feature matrix is a table combining all the aggragated features.

### Q5.

We present 3 of the newly extracted features from feature table:

1. **users_votes__CreationDate__max** - This columns contains the date of the last vote casted by each user.

2. **users_comments__Text__count** - This column counts how many non-empty comments a user has written.

3. **users_posts__Score__sum** - This column sums the total score a user achieved acroos all of his posts.

```text
                           feature  non-null %  mean                                                                                                                                            meaning
   d2__users_comments__Text__count       100.0  7.13                                Count of non-null text entries in `comments` joined to the user — i.e., **how many comments the user has written**.
d2__users_votes__CreationDate__max         3.0   NaN                         Maximum (latest) `CreationDate` over all `votes` cast by the user — i.e. **the timestamp of the user’s most-recent vote**.
       d2__users_posts__Score__sum         NaN   NaN Sum of all `Score` values across all `posts` owned by the user — i.e., **the total net upvotes/reputation score accumulated by the user's posts**.
```

### Q6. Missing values in extracted features

yes, plenty. couple of reasons.

the first join (target → users) is a left join, so we keep every target row
even if its user isn't in the users table — those rows get NaN on all user
columns. at depth 2 we use an inner join, so users with no rows in the leaf
table just fall out of the join. we then merge the aggregated result back to
the target with another left join (we want every target row to survive), and
users with no leaf rows end up NaN for that whole depth-2 block. for count
aggregations we fill those with 0 — a user with no comments has 0 comments,
not "unknown". for mean/sum/min/max we leave the NaN, because aggregating over
nothing is genuinely undefined.

a couple of other sources too: some source columns are already NaN-heavy.
`posts.ParentId` is null for top-level questions, which propagates into
mean/sum. and `users.ProfileImageUrl` is entirely null in this dataset, so
every aggregation built from it comes out NaN.

```text
Top 10 missingness rates among new features:
d1__users__ProfileImageUrl__mean         1.000
d2__users_votes__VoteTypeId__mean        0.970
d2__users_votes__VoteTypeId__sum         0.970
d2__users_votes__CreationDate__min       0.970
d2__users_votes__CreationDate__max       0.970
d2__users_comments__CreationDate__max    0.401
d2__users_comments__CreationDate__min    0.401
d2__users_badges__Date__min              0.088
d2__users_badges__Date__max              0.088
d2__users_badges__Class__mean            0.088

Mean missingness across all 46 new features: 13.7%
Fraction of new features that are 100% missing: 2.2%
```

## Q7 — Train DecisionTree, XGBoost, Logistic Regression and a PyTorch NN

We compute the auto-features for `val` and `test`, then train on the depth-2 feature
matrix.

```text
Computing features for val/test ...

train_feat: (1360850, 49) | val_feat: (85838, 49) | test_feat: (88137, 48)
```

```text
Shapes: (1360850, 46) (85838, 46) (88137, 46)
Dtypes ok: {dtype('float64'): 46}
Train missingness: 0.137
```

XGBoost gets raw NaNs; DT gets median-imputed; LR and NN get imputed + scaled.

```text
Preprocessing done
```

### Decision Tree — hyperparameter sweep

```text
max_depth  min_samples_leaf  val_accuracy  val_roc_auc  val_precision  val_recall  val_f1  tr_accuracy  tr_roc_auc  tr_precision  tr_recall  tr_f1
       20               500        0.7133       0.8721         0.0784      0.8557  0.1436       0.7673      0.8546        0.1486     0.7729 0.2492
       10               500        0.7121       0.8715         0.0791      0.8685  0.1449       0.7782      0.8360        0.1500     0.7364 0.2492
       10                50        0.7238       0.8567         0.0795      0.8349  0.1452       0.7860      0.8405        0.1550     0.7372 0.2561
        5               500        0.6718       0.8523         0.0708      0.8814  0.1311       0.7740      0.8048        0.1417     0.6966 0.2355
        5                50        0.6718       0.8523         0.0708      0.8814  0.1311       0.7740      0.8048        0.1417     0.6966 0.2355
       20                50        0.7574       0.8283         0.0849      0.7814  0.1532       0.8023      0.9041        0.1813     0.8407 0.2983
        3                50        0.8063       0.7970         0.0918      0.6632  0.1613       0.8051      0.7842        0.1511     0.6276 0.2436
        3               500        0.8063       0.7970         0.0918      0.6632  0.1613       0.8051      0.7842        0.1511     0.6276 0.2436

Best DT cfg: {'max_depth': '20', 'min_samples_leaf': 500}
Best DT metrics: {'train': {'accuracy': 0.7672814784877099, 'roc_auc': np.float64(0.8546187598032231), 'precision': 0.14858047284597883, 'recall': 0.7728609232578654, 'f1': 0.2492443727997914}, 'val': {'accuracy': 0.713273841422214, 'roc_auc': np.float64(0.8721277806350392), 'precision': 0.07836061837657159, 'recall': 0.8556615512235587, 'f1': 0.143572969587306}}
```

### XGBoost — hyperparameter sweep

```text
 max_depth  n_estimators   lr  val_accuracy  val_roc_auc  val_precision  val_recall  val_f1  tr_accuracy  tr_roc_auc  tr_precision  tr_recall  tr_f1
         4           300 0.05        0.7315       0.8809         0.0832      0.8548  0.1517       0.7909      0.8450        0.1575     0.7316 0.2592
         4           300 0.10        0.7330       0.8800         0.0840      0.8594  0.1531       0.7930      0.8509        0.1601     0.7398 0.2633
         6           300 0.05        0.7393       0.8795         0.0853      0.8511  0.1550       0.7999      0.8587        0.1663     0.7487 0.2722
         6           300 0.10        0.7533       0.8736         0.0879      0.8299  0.1589       0.8072      0.8699        0.1740     0.7628 0.2834

Best XGB cfg: {'max_depth': 4.0, 'n_estimators': 300.0, 'lr': 0.05}
Best XGB metrics: {'train': {'accuracy': 0.7909181761399126, 'roc_auc': np.float64(0.8449765380206375), 'precision': 0.1574669109387706, 'recall': 0.7316377536018818, 'f1': 0.2591567485373417}, 'val': {'accuracy': 0.7314592604673921, 'roc_auc': np.float64(0.8809265254373371), 'precision': 0.08323237218318391, 'recall': 0.8548320199087516, 'f1': 0.15169469694181725}}
```

### Logistic Regression — hyperparameter sweep

```text
    C  val_accuracy  val_roc_auc  val_precision  val_recall  val_f1  tr_accuracy  tr_roc_auc  tr_precision  tr_recall  tr_f1
 0.01        0.7870       0.8299         0.0920      0.7424  0.1637       0.8021      0.8082        0.1529     0.6517 0.2477
 1.00        0.7877       0.8293         0.0923      0.7424  0.1642       0.8021      0.8081        0.1529     0.6515 0.2476
 0.10        0.7877       0.8292         0.0923      0.7424  0.1642       0.8021      0.8081        0.1528     0.6514 0.2475
10.00        0.7880       0.8292         0.0924      0.7420  0.1643       0.8021      0.8081        0.1528     0.6513 0.2476

Best LR C: 0.01
Best LR metrics: {'train': {'accuracy': 0.8020707645956571, 'roc_auc': np.float64(0.8081613031722064), 'precision': 0.15287107408378048, 'recall': 0.6517494854454572, 'f1': 0.24765373614587058}, 'val': {'accuracy': 0.7869708054707705, 'roc_auc': np.float64(0.8298708651310021), 'precision': 0.09200719609354921, 'recall': 0.7424305267523849, 'f1': 0.16372450379584744}}
```

### NN in PyTorch

3-layer MLP (128→64→1), ReLU + Dropout(0.3). BCEWithLogitsLoss with pos_weight.
Adam lr=1e-3, batch_size=4096, 5 epochs, best val-AUC checkpoint.

```text
NN hid=(64,)

  epoch 1: val AUC = 0.8637

  epoch 2: val AUC = 0.8635

  epoch 3: val AUC = 0.8626

  epoch 4: val AUC = 0.8630

  epoch 5: val AUC = 0.8645

NN hid=(128, 64)

  epoch 1: val AUC = 0.8636

  epoch 2: val AUC = 0.8642

  epoch 3: val AUC = 0.8706

  epoch 4: val AUC = 0.8669

  epoch 5: val AUC = 0.8712

NN hid=(256, 128, 64)

  epoch 1: val AUC = 0.8656

  epoch 2: val AUC = 0.8674

  epoch 3: val AUC = 0.8715

  epoch 4: val AUC = 0.8772

  epoch 5: val AUC = 0.8725

NN sweep:
           hid  val_accuracy  val_roc_auc  val_precision  val_recall  val_f1  tr_accuracy  tr_roc_auc  tr_precision  tr_recall  tr_f1
(256, 128, 64)        0.7129       0.8772         0.0790      0.8656  0.1449       0.7823      0.8404        0.1518     0.7317 0.2515
     (128, 64)        0.7319       0.8712         0.0827      0.8465  0.1506       0.7869      0.8399        0.1538     0.7249 0.2537
         (64,)        0.7352       0.8645         0.0822      0.8287  0.1495       0.7785      0.8334        0.1483     0.7237 0.2462

Best NN hid: (256, 128, 64) metrics: {'train': {'accuracy': 0.7822581474813536, 'roc_auc': np.float64(0.8403632439248317), 'precision': 0.1518188861571874, 'recall': 0.7317259629520729, 'f1': 0.25146390877536895}, 'val': {'accuracy': 0.7129359957128545, 'roc_auc': np.float64(0.877203321682936), 'precision': 0.07904105438569914, 'recall': 0.8656159270012443, 'f1': 0.14485511018566719}}
```

# Hyper Parameters Discussion

**Decision Tree Hyperparameters** - We evaluated the performance of a Decision Tree model while varying the hyperparameters max_depth and min_samples_leaf. We tested max_depth values of 3, 5, 10, and 20, and min_samples_leaf values of 50 and 500. We chose these ranges for two main reasons:

Feature amount - Since our feature table contains 46 features (after the ACORA-style PK/FK + AccountId filter, down from a 96-feature maximum), we wanted to see whether deeper trees can better capture information spread across many features, or whether a smaller depth is already enough to describe user behavior and generalizes better to new samples without overfitting.

Number of examples - The dataset contains a large number of training examples, so individual leaves can easily become too specific to small subsets of data. The min_samples_leaf parameter controls this by forcing each leaf to contain at least a certain number of samples. Using larger values like 50 or 500 helps prevent the tree from creating overfited patterns, and instead encourages more stable and general rules that better reflect overall trends in the data.

**XGBoost Hyperparametes** - We evaluated the performance of an XGBoost model while varying the hyperparameters max_depth (4 and 6) and learning rate (lr) values of 0.05 and 0.1. Since XGBoost is an ensemble of decision trees built sequentially, we avoided using very large tree depths to reduce the risk of overfitting and keep the individual trees relatively simple. We also tested two learning rates to check how sensitive the model is to step size- whether a larger learning rate leads to faster convergence but possibly more overfitting, or whether it still generalizes well compared to the smaller value of 0.05.

**Logistic Regression Hyperparameters** - We evaluated the performance of a Logistic Regression model while varying the hyperparameter C with values 0.01, 0.1, 1 and 10. Our goal here was to see the affect of smaller\bigger regularization on the performance of the model, and if it tends to overfit.

**NN model** - We evaluated the performance of a simple NN model while varying the hyperparameter hid with values(64,), (128, 64) and (256, 128, 64). Our goal here was to see the affect of varying hidden layers on the performance of the model.


All the relevant results of all the hyperparameters variations arepresented in the above tables.

### Combined results table (Q7)

```text
Q7 — Train / Val metrics (best variant per model):
              model      accuracy       roc_auc     precision        recall            f1
      Decision Tree 0.767 / 0.713 0.855 / 0.872 0.149 / 0.078 0.773 / 0.856 0.249 / 0.144
            XGBoost 0.791 / 0.731 0.845 / 0.881 0.157 / 0.083 0.732 / 0.855 0.259 / 0.152
Logistic Regression 0.802 / 0.787 0.808 / 0.830 0.153 / 0.092 0.652 / 0.742 0.248 / 0.164
      MLP (PyTorch) 0.782 / 0.713 0.840 / 0.877 0.152 / 0.079 0.732 / 0.866 0.251 / 0.145
```

## Q8 — Evaluation measures

* **Accuracy** = Accuracy is defined as (TP+TN) / total. This is the rate of how many predictions our model was correct. This meassure can be misleading when trained on an imbalanced data. With 97% negatives, a trivial all-zero classifier scores 0.97, so accuracy is nearly useless is this imbalanced data.


* **AUROC** = probability a random positive ranks above a random negative. A model that achieves a high probability, is a model that seperates the negative and positive examples well. AUROC is threshold-free, class-prior invariant.


* **Precision** = Precision is defined as TP/(TP+FP) This measures, out of all the examples the model predicted as positive, how many were actually correct. So it reflects how “trustworthy” positive predictions are. A high precision means that when the model says "positive" it is usually right. However, precision ignores false negatives, so the model could still miss many actual positives without hurting precision.

* **Recall** = Recall is defined as TP/(TP+FN). This measures how many of the actual positive cases the model manages to find. A high recall means the model misses very few positives, but it might also produce more false positives.

* **F1** = F1 is defined as 2(Presicion*Recall)/(Precision+Recall). This METHOD combines precision and recall into a single number, balancing both. It becomes high only when both precision and recall are reasonably high. It is useful when you want a single metric that reflects performance on the positive class.




**Tradeoffs** = If we increase the classification threshold, the model becomes more conservative in predicting positives. As a result, precision increases because fewer examples are labeled as positive, and those that do pass the higher threshold are more likely to be truly positive. On the other hand, recall decreases since the model misses more "weak" positive cases that fall below the threshold.

AUROC evaluates performance across all possible thresholds, so it is independent of any specific decision cutoff. In contrast, F1 depends on a chosen threshold and reflects performance at that specific operating point. Accuracy is less informative in imbalanced settings, since it can remain high even for trivial models that mostly predict the majority class.

With around ~3% positive examples, AUROC and F1 are more meaningful metrics. AUROC measures how well the model separates positive from negative cases overall, while F1 focuses specifically on performance for the positive class by balancing the ability to find true positives (recall) and avoid false positives (precision), which is especially important in imbalanced datasets and captured by F1.

## Q9 — Discussion of results

on the training set, the Decision Tree gets the best AUROC (0.855), with
XGBoost right behind (0.845). XGBoost takes the best train F1 (0.259). LR is
last on both — its linear boundary just can't fit the patterns the trees
pick up. the trees model the feature interactions directly, which shows up
in their higher recall and AUROC on train.

on the validation set the ordering shifts a bit. XGBoost has the best val
AUROC (0.881), MLP is right next to it (0.877), and DT is just behind
(0.872). LR wins on F1 (0.164) — its precision/recall balance at the default
threshold beats the trees even though its AUROC is lower.


* **Why these differences.** The MLP performs well because it can learn non-linear patterns in the standardized features, which matches the way user activity and engagement usually behave. Logistic Regression also works well since many features, such as post counts or recent activity, have a fairly direct relationship with the probability of engagement after scaling. XGBoost is able to capture more complex interactions between features using decision trees, but on this feature set it does not improve much over Logistic Regression, maybe the XGBoost overfitted the training a bit more than LR and our NN. The single Decision Tree performs worse because one tree alone has difficulty combining many correlated activity features, unlike ensemble methods or smoother models such as LR and MLP.

* **Effect of evaluation measure.** Accuracy ranks the models very differently from AUROC — with 97% negatives, a model that mostly predicts 0 scores high on accuracy but low on AUROC and F1. AUROC and F1 are the meaningful metrics here.

* **Effect of schema and task.** Because the task predicts user engagement, most of the useful information comes from the depth-2 paths target → users → {posts, comments, votes, postHistory, badges}. The dominant features across all models are counts and recency aggregations from these tables — essentially, how active the user was in the past and how recently they interacted with the platform. Features such as the number of posts, comments, votes, and badges capture the overall level of user activity, while timestamp-based features indicate whether the user has been active recently. In the REL-Stack dataset, recent and frequent activity is strongly correlated with future engagement, which is why simple aggregations like counts and maximum timestamps become highly informative features for the prediction task.

## Q10 — How feature importance is computed in each model

- Decision Tree — Gini importance. for every feature we sum the Gini impurity
  it reduced at each node it split on, weighted by how many samples reached
  that node. then normalise so importances add up to 1.
- XGBoost — average gain. for every feature, the mean loss-improvement it gave
  across all the splits on that feature across all boosting rounds.
- Logistic Regression — we just take `|β_j|`. since the features are
  standardised the coefficients are directly comparable, and a bigger
  `|β_j|` means the feature has more pull on the predicted log-odds.

## Q11 — Top-10 features per model

```text
Top-10 features per model (depth-2 tabular features only):
 rank                                 DT_feature  DT_imp                               XGB_feature  XGB_imp                                      LR_feature  LR_|coef|
    1          d2__users_posts__PostTypeId__mean  0.5903        d2__users_posts__PostTypeId__count   0.3107                  d2__users_badges__Class__count     1.8290
    2                d2__users_badges__Date__min  0.0617    d2__users_posts__ContentLicense__count   0.2613                     d2__users_badges__Date__min     1.0364
    3               d2__users_posts__Tags__count  0.0562          d2__users_posts__PostTypeId__sum   0.0963               d2__users_badges__TagBased__count     0.8934
    4   d2__users_postHistory__CreationDate__min  0.0430 d2__users_comments__ContentLicense__count   0.0423        d2__users_postHistory__CreationDate__min     0.6881
    5         d2__users_posts__CreationDate__min  0.0368                 d1__users__AboutMe__count   0.0362              d2__users_posts__CreationDate__min     0.6733
    6      d2__users_comments__CreationDate__min  0.0361                d1__users__Location__count   0.0265 d2__users_postHistory__PostHistoryTypeId__count     0.4911
    7               d1__users__WebsiteUrl__count  0.0359  d2__users_postHistory__CreationDate__max   0.0173                   d2__users_badges__Class__mean     0.4261
    8               d1__users__CreationDate__max  0.0250           d1__users__ProfileImageUrl__sum   0.0164                   d2__users_badges__Name__count     0.4261
    9                  d1__users__AboutMe__count  0.0168     d2__users_comments__CreationDate__max   0.0155                    d2__users_badges__Class__sum     0.4261
   10 d2__users_comments__UserDisplayName__count  0.0113        d2__users_posts__CreationDate__max   0.0153              d2__users_postHistory__Text__count     0.3755
```

**Commonalities & differences.**

all three models put activity volume and recency near the top — counts of
posts/badges/postHistory rows, and earliest-action timestamps like
`posts.CreationDate.min` and `postHistory.CreationDate.min`. that tracks with
what we'd expect from a user-engagement task: "how active was this user, and
how recently?" is basically the whole signal.

the trees and the linear model differ in where they go looking for that
signal. DT and XGBoost lean on count aggregations from the big child tables
(`posts.PostTypeId.count`, `posts.ContentLicense.count`) — trees deal with
raw scale fine. LR, with standardised features, ends up leaning heavily on
the badges block (`badges.Class.count`, `badges.Date.min`,
`badges.TagBased.count`, `badges.Name.count`). badges are kind of a clean
activity proxy, so once scaled their coefficients dominate.

between DT and XGBoost — the single tree leans on whichever feature gives the
cleanest first split, so its top-10 is steep and concentrated. XGBoost spreads
importance across many features because each boosting round picks up a
different slice of the residual.

## Part 2 — Graph Feature Extraction

now we treat the database as a graph. every tuple of every table becomes a
node, identified by `(table_name, row_id)`, and every FK-PK pair becomes an
undirected edge between the two tuples it connects.

### Q12. Build the graph and compute (a) degree, (b) WL coloring, (c) two graphlet counts

we build three graphs, one per split, following the edited temporal rule:
train uses tuples strictly before `val_timestamp` (2020-10-01), val uses
tuples up to and including `val_timestamp`, test uses everything in the db
(already pre-cut at `test_timestamp`). for each graph we then compute the
degree, WL colours, and graphlet counts, and merge them onto the matching
split's feature matrix.

```text
Building 3 graphs (train < val_timestamp, val ≤ val_timestamp, test = all) ...

  train: nodes=4,121,359  edges=5,641,899

  val  : nodes=4,121,747  edges=5,642,180

  test : nodes=4,247,264  edges=5,812,887
  built all 3 in 186.5s
```

**(a) Degree.** for a user this is just the count of `posts`, `badges`,
`comments`, `postHistory`, and `votes` rows that point at them. computed
independently per split graph.

```text
Degree stats over all users (test graph):
count    255360.00
mean          9.83
std         258.06
min           0.00
25%           0.00
50%           1.00
75%           6.00
max       70436.00
Name: graph_degree, dtype: float64

mean degree per split:  train=9.54  val=9.54  test=9.83
```

**(b) Weisfeiler–Lehman colours.** every node starts with a colour that's just
a hash of its table name. each iteration replaces a node's colour with a hash
of its own current colour plus the sorted list of its neighbours' colours.
we record the colours after K = 2, 3, and 5 iterations, and run it separately
on each split's graph.

```text
Running WL on train graph ...

WL iter:  20%|██        | 1/5 [00:14<00:59, 14.90s/it]
WL iter:  40%|████      | 2/5 [00:28<00:42, 14.22s/it]
WL iter:  60%|██████    | 3/5 [00:43<00:28, 14.49s/it]
WL iter:  80%|████████  | 4/5 [00:58<00:14, 14.57s/it]
WL iter: 100%|██████████| 5/5 [01:12<00:00, 14.67s/it]
WL iter: 100%|██████████| 5/5 [01:12<00:00, 14.60s/it]

  done in 77.4s
Running WL on val graph ...

WL iter:  20%|██        | 1/5 [00:14<00:57, 14.33s/it]
WL iter:  40%|████      | 2/5 [00:28<00:43, 14.43s/it]
WL iter:  60%|██████    | 3/5 [00:42<00:28, 14.23s/it]
WL iter:  80%|████████  | 4/5 [00:56<00:14, 14.12s/it]
WL iter: 100%|██████████| 5/5 [01:11<00:00, 14.18s/it]
WL iter: 100%|██████████| 5/5 [01:11<00:00, 14.21s/it]

  done in 75.4s
Running WL on test graph ...

WL iter:  20%|██        | 1/5 [00:14<00:56, 14.13s/it]
WL iter:  40%|████      | 2/5 [00:29<00:44, 14.78s/it]
WL iter:  60%|██████    | 3/5 [00:44<00:29, 14.81s/it]
WL iter:  80%|████████  | 4/5 [00:58<00:14, 14.53s/it]
WL iter: 100%|██████████| 5/5 [01:13<00:00, 14.81s/it]
WL iter: 100%|██████████| 5/5 [01:13<00:00, 14.72s/it]

  done in 78.0s

Distinct WL colours among users (test graph):
{'wl_K2': 53584, 'wl_K3': 72886, 'wl_K5': 84480}
```

**(c) Graphlet counts.** two motifs to count per user:

- `graphlet_self_comment` — how many times the user commented on a post they
  themselves own (cycle `user → comment → post → user`).
- `graphlet_linked_self_posts` — how many times two posts owned by the user
  are linked by a `postLinks` edge (cycle `user → post → postLink → post → user`).

both come out cleanest as a couple of pandas joins on the split-filtered tables.

```text
Graphlet feature stats (test graph, over all users):
       graphlet_self_comment  graphlet_linked_self_posts
count              255360.00                   255360.00
mean                    0.79                        0.01
std                    14.94                        0.28
min                     0.00                        0.00
25%                     0.00                        0.00
50%                     0.00                        0.00
75%                     0.00                        0.00
max                  4174.00                       59.00
```

**Adding graph features to the feature matrix.** Each split's feature matrix gets the
graph features computed from the matching graph.

```text
Augmented shapes: (1360850, 55) (85838, 55) (88137, 54)
```

### Q13 — Retrain all four models on the tabular + graph feature matrix

```text
Augmented preprocessing done. n_features = 52

Q13 — Train / Val metrics WITH graph features:
              model      accuracy       roc_auc     precision        recall            f1
      Decision Tree 0.864 / 0.916 0.957 / 0.827 0.260 / 0.167 0.934 / 0.505 0.407 / 0.251
            XGBoost 0.883 / 0.916 0.959 / 0.873 0.290 / 0.188 0.930 / 0.599 0.442 / 0.286
Logistic Regression 0.833 / 0.794 0.899 / 0.808 0.203 / 0.095 0.798 / 0.740 0.323 / 0.168
      MLP (PyTorch) 0.868 / 0.859 0.954 / 0.865 0.265 / 0.129 0.926 / 0.703 0.412 / 0.219
```

### Q14 — Top-10 features per model (now with graph features available)

```text
Top-10 features (graph in **bold** annotation):
 rank                                     DT_feature DT_kind                                     XGB_feature XGB_kind                                    LR_feature LR_kind
    1             d2__users_votes__CreationDate__max tabular                                    graph_degree    graph            d2__users_votes__CreationDate__max tabular
    2 d2__users_postHistory__PostHistoryTypeId__mean tabular      d2__users_postHistory__RevisionGUID__count  tabular                   d2__users_badges__Date__max tabular
    3   d2__users_postHistory__ContentLicense__count tabular                           graphlet_self_comment    graph    d2__users_comments__UserDisplayName__count tabular
    4     d2__users_comments__UserDisplayName__count tabular d2__users_postHistory__PostHistoryTypeId__count  tabular                                  graph_degree   graph
    5                  d2__users_badges__Name__count tabular                  d2__users_badges__Class__count  tabular                d2__users_badges__Class__count tabular
    6                   d2__users_badges__Class__sum tabular              d2__users_posts__CreationDate__min  tabular                    graphlet_linked_self_posts   graph
    7                    d2__users_badges__Date__max tabular                   d2__users_badges__Name__count  tabular d2__users_postHistory__UserDisplayName__count tabular
    8             d2__users_posts__CreationDate__min tabular                 d2__users_comments__Text__count  tabular            d2__users_postHistory__Text__count tabular
    9                   d1__users__CreationDate__max tabular                                           wl_K2    graph    d2__users_postHistory__RevisionGUID__count tabular
   10          d2__users_comments__CreationDate__min tabular              d2__users_posts__CreationDate__max  tabular  d2__users_postHistory__ContentLicense__count tabular

   DT: graph features in top-10 = 0
  XGB: graph features in top-10 = 3
   LR: graph features in top-10 = 2
```

**Discussion of graph features (Q14).**

the picture is mixed. train AUROC jumped a lot for the trees and the MLP
(DT 0.855 → 0.957, XGB 0.845 → 0.959, MLP 0.840 → 0.954), but val AUROC barely
moved for XGB (0.881 → 0.873) and MLP (0.877 → 0.865), and actually *dropped*
for DT (0.872 → 0.827). that gap is mostly `graph_degree` leaking — its train
graph spans everything before `val_timestamp`, so the degree of a 2015 user
already includes activity all the way to 2020-10-01, which is exactly the kind
of future-peeking the per-row rule on the tabular features was supposed to
block. Q12 only asks for the looser global cutoff on the graph, so we follow
that, but the trees end up overfitting to it.

as for which models actually use the graph features: XGB picked 3 into its
top-10 (`graph_degree` at #1, `graphlet_self_comment` at #3, `wl_K2` at #9).
LR picked 2 (`graph_degree` and `graphlet_linked_self_posts`). DT picked 0 —
its top splits stay on the depth-2 tabular counts.

the most useful graph feature was `graph_degree`. it's very correlated with
the depth-2 count aggregations — degree is basically the sum of leaf counts —
which is why it lands high in importance but doesn't move val much, and also
why it drives the train overfitting. the graphlet counts add genuinely new
structural info (cycles), but most users have zero so the lift is small. the
WL colours add even less, since the immediate neighbourhood structure is
already covered by the counts.

## Q15 — Summary (conclusion paragraph for the report)

**(a) Steps taken.**

we started by loading rel-stack and looking at the schema and class balance
(~5% positives, which set up most of the later discussion about AUROC vs
accuracy). we then picked three manual features by hand (post count, vote
recency, badge count) and built the automatic join-aggregate generator on top
of that: BFS over the FK graph from the target, with the ACORA-style filter
dropping PK/FK columns and `Users.AccountId`. at depth 2 we end up with 46
features (down from a 96 max). with those we trained DT, XGBoost, LR, and a
small MLP. val AUROC ends up around 0.83–0.88, well above the users-only
relbench baseline of 0.65. finally we built one graph per split (with the
temporal cutoffs Q12 asks for), computed degree, WL colours at K = 2, 3, 5,
and two graphlet counts, and merged those back into the feature matrices.

**(b) Auto vs manual features.**

auto-features are great for coverage and reproducibility — you get a lot of
features without thinking too hard about each one — but a chunk of them are
weak or redundant (sums of mostly-null columns, min/max pairs that move
together). manual features go the other way: each one is targeted and easy to
interpret, but you're limited to whatever you happen to think of. in practice
the best move is using them together — let the auto generator build a broad
baseline, then add manual features for whatever domain knowledge the
generator can't get to on its own.

**(c) Graph vs tabular-only.**

graph features can in principle capture structural patterns (cycles, motifs)
that join-aggregates miss. but on this star-shaped schema they mostly re-encode
what the count aggregations already capture, so the marginal val AUROC lift
was small. the biggest single jump came from going users-only → depth-2
tabular features. graph features on top of that didn't really move val, and
because Q12 only enforces a global temporal cutoff (not per-row), the
train-side `graph_degree` ended up leaking some future activity — which is
why DT's val AUROC actually dropped slightly when we added the graph features.

## Q16 — AI-tool usage

The code was generated entirely using AI (Claude), but the conclusions drawn from it are entirely our own. The final results were reasonable and consistent, which increased our confidence in the generated code. We validated the outputs in several ways: for example, the table in Q3 helped us verify the number of extracted features, and the training/validation metrics showed a clear and expected trade-off between underfitting and overfitting based on the selected hyperparameters. In addition, the handcrafted features were chosen by us, and it was reassuring to see that the AI independently identified similar features as the most relevant ones in Q1.

As for part 2
we could not check the the correctness of the graph but we did look at a few things that helped us gain confidence:
1. Graph size — 4,247,264 nodes, 5,812,887 edges. Matches sum of table rows and non-null FKs exactly.
  2. Degree — user #352 manual count = 1,566 (129+449+725+260+3). We can verify it matches the graph.
  3. Graphlets — user #352: 75 self-comments, 0 linked self-posts. Computed manually via pandas joins.
  4. WL - distinct colors increases at every iteration.

Overall, we carefully cross-checked the AI-generated work in these key areas. We estimate that the AI performed around 80% of the implementation work.
