# Structured Machine Learning (236605): Assignments

Two assignments on learning from relational databases, using the
[RelBench](https://relbench.stanford.edu/) `rel-stack` dataset and its
user-engagement task (predict whether a user votes, posts or comments in the
next three months).

The final project, an ablation study of relational graph neural networks, is in a
separate repository:
[structml-relational-gnn-ablations](https://github.com/galwert/structml-relational-gnn-ablations).

## `HW1/` Feature extraction

Builds a 46-column join-aggregate feature matrix (depth-2 ACORA style) with
temporal leakage guards, then trains decision trees, XGBoost, logistic regression
and a PyTorch MLP. Adds six graph features (degree, PageRank, clustering) and
compares feature-importance methods across models. Best validation AUROC 0.881
with XGBoost.

- `hw1.ipynb` / `hw1.py`: full pipeline
- `report.md`: answers to Q1 to Q16

## `HW2/` Node and tuple embeddings

Converts the database into a PyTorch Geometric heterogeneous graph, computes
Laplacian eigenmaps via randomized SVD and trains Metapath2Vec embeddings
(k = 128). Embeddings are evaluated as features for the HW1 classifier and
visualised with t-SNE.

- `hw2.ipynb` / `hw2.py`: pipeline; `hw2.sbatch`, `m2v_q9.sbatch`: SLURM launchers
- `report.md`: analysis and written answers
- `neighborhoods.png`, `tsne_grid.png`: figures

Both assignments ran on a single GPU with PyTorch, PyG, scikit-learn and XGBoost.
