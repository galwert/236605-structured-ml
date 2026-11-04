"""Split executed hw1.ipynb into:
  - report.md / report.html  (markdown explanations + cell outputs, no code)
  - hw1_code.ipynb           (code cells only, minimal section headers)
"""
import json, re, html, sys, os

NB_IN = "hw1.ipynb"
REPORT_MD = "report.md"
REPORT_HTML = "report.html"
NB_CODE = "hw1_code.ipynb"

# --- ANSI / tqdm cleanup ----------------------------------------------------
ANSI = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

def clean_text(s):
    s = ANSI.sub("", s)
    # drop tqdm progress lines (mid-iteration), keep only the final 100% one
    out_lines = []
    for line in s.split("\n"):
        # tqdm lines look like: "paths:  50%|███████| 3/6 [00:16<00:20,  6.80s/it]"
        if re.search(r"\b\d+%\|", line) and "it/s]" in line and "100%|" not in line:
            continue
        out_lines.append(line)
    s = "\n".join(out_lines)
    # collapse runs of >2 blank lines
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip("\n")


def cell_text_outputs(cell):
    """Concat all stream + text/plain outputs of a code cell into one string."""
    parts = []
    for o in cell.get("outputs", []):
        t = o.get("output_type")
        if t == "stream":
            parts.append("".join(o.get("text", [])))
        elif t in ("execute_result", "display_data"):
            txt = o.get("data", {}).get("text/plain", "")
            if isinstance(txt, list):
                txt = "".join(txt)
            parts.append(txt)
        elif t == "error":
            parts.append("[ERROR]\n" + "\n".join(o.get("traceback", [])))
    return clean_text("\n".join(parts))


def src(cell):
    s = cell.get("source", [])
    if isinstance(s, list):
        s = "".join(s)
    return s

# --- build report -----------------------------------------------------------
with open(NB_IN) as f:
    nb = json.load(f)

report_parts = []
for c in nb["cells"]:
    if c["cell_type"] == "markdown":
        report_parts.append(src(c).rstrip() + "\n")
    elif c["cell_type"] == "code":
        out = cell_text_outputs(c)
        if out:
            report_parts.append("```text\n" + out + "\n```\n")

report_md = "\n".join(report_parts)
with open(REPORT_MD, "w") as f:
    f.write(report_md)
print(f"wrote {REPORT_MD}  ({len(report_md):,} bytes)")

# --- markdown → html (light, self-contained) --------------------------------
try:
    import markdown as md_lib
    md_html = md_lib.markdown(
        report_md,
        extensions=["fenced_code", "tables", "sane_lists"],
    )
except Exception:
    # fallback: very crude wrapping
    md_html = "<pre>" + html.escape(report_md) + "</pre>"

html_doc = f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>StructML_HW1_Report</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
          max-width: 850px; margin: 2em auto; padding: 0 1em; line-height: 1.5;
          color: #222; }}
  h1 {{ border-bottom: 2px solid #555; padding-bottom: .2em; }}
  h2 {{ border-bottom: 1px solid #aaa; padding-bottom: .15em; margin-top: 2em; }}
  h3 {{ margin-top: 1.5em; }}
  pre {{ background: #f4f4f4; padding: .7em 1em; border-radius: 4px;
         overflow-x: auto; font-size: 0.85em; line-height: 1.35; }}
  code {{ font-family: "SF Mono", Menlo, Consolas, monospace; }}
  table {{ border-collapse: collapse; margin: .5em 0; }}
  th, td {{ border: 1px solid #bbb; padding: 4px 8px; text-align: left; }}
  th {{ background: #eee; }}
  blockquote {{ border-left: 3px solid #ccc; margin: 0; padding-left: 1em;
                color: #555; }}
  @media print {{
    body {{ max-width: 100%; margin: 0; }}
    pre  {{ break-inside: avoid; }}
  }}
</style>
</head><body>
{md_html}
</body></html>
"""
with open(REPORT_HTML, "w") as f:
    f.write(html_doc)
print(f"wrote {REPORT_HTML} ({len(html_doc):,} bytes)")

# --- code-only ipynb --------------------------------------------------------
code_cells = []
# keep H1/H2/H3 section headers as markdown so the notebook is still navigable
HEADER_RE = re.compile(r"^\s*#{1,3}\s", re.M)

for c in nb["cells"]:
    if c["cell_type"] == "code":
        code_cells.append(c)
    elif c["cell_type"] == "markdown":
        # keep only section-header lines, drop the prose
        s = src(c)
        lines = [ln for ln in s.split("\n") if ln.lstrip().startswith("#")]
        if lines:
            code_cells.append({
                "cell_type": "markdown",
                "metadata": c.get("metadata", {}),
                "source": [ln + "\n" for ln in lines],
            })

nb_code = {k: v for k, v in nb.items() if k != "cells"}
nb_code["cells"] = code_cells
with open(NB_CODE, "w") as f:
    json.dump(nb_code, f, indent=1)
print(f"wrote {NB_CODE} ({sum(1 for c in code_cells if c['cell_type']=='code')} code cells)")
