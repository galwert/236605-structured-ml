"""Quick smoke test for HW2 — runs the cheap cells (data, features, kernel PCA,
graph build, randomized SVD) end-to-end. Skips Node2Vec/Metapath2Vec training.
Goal: catch obvious bugs before launching the GPU job."""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
os.chdir("/home/wertheizer/structeredML/HW2")
sys.path.insert(0, "/home/wertheizer/structeredML/HW2")

# Re-use the hw2 file by exec'ing it up to a certain point.
# Easier: just import the chunks we need by copying. For a smoke test,
# we read hw2.py and exec cells until "Q3b. Node2Vec".

with open("hw2.py") as f:
    code = f.read()

# Cells are separated by "# %%"
cells = []
cur = []
for ln in code.split("\n"):
    if ln.startswith("# %%"):
        if cur: cells.append("\n".join(cur))
        cur = [ln]
    else:
        cur.append(ln)
if cur: cells.append("\n".join(cur))

print(f"Found {len(cells)} cells.")
ns = {"__name__": "__smoke__"}
stop_after = None
for i, c in enumerate(cells):
    first_line = c.split("\n", 1)[0]
    if "Q3b" in c or "Node2Vec (PyG implementation)" in c:
        print(f"Stopping before cell {i}: {first_line[:80]}")
        break
    print(f"--- cell {i}/{len(cells)} ---  {first_line[:80]}")
    t0 = time.time()
    try:
        exec(c, ns)
    except Exception as e:
        print(f"!! ERROR in cell {i}: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
    print(f"   ({time.time()-t0:.1f}s)")
print("Smoke test PASSED through Q3a (Laplacian).")
