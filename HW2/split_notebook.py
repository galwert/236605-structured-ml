"""Split executed hw2.ipynb into:
  - report.md / report.html  (markdown + cell outputs + embedded figures, no code)
  - hw2_code.ipynb           (code cells only, minimal section headers)

Same pattern as HW1 but adds support for image/png outputs (tsne_grid,
neighborhoods) which are embedded directly into the HTML report.
"""
import json, re, html, sys, os, base64

NB_IN = "hw2.ipynb"
REPORT_MD = "report.md"
REPORT_HTML = "report.html"
NB_CODE = "hw2_code.ipynb"

ANSI = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

def clean_text(s):
    s = ANSI.sub("", s)
    out_lines = []
    for line in s.split("\n"):
        if re.search(r"\b\d+%\|", line) and "it/s]" in line and "100%|" not in line:
            continue
        out_lines.append(line)
    s = "\n".join(out_lines)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip("\n")


def cell_outputs_for_report(cell):
    """Return a list of (kind, content) pairs:
       ("text", str) for stream / text/plain
       ("img",  base64 str) for image/png
    """
    items = []
    for o in cell.get("outputs", []):
        t = o.get("output_type")
        if t == "stream":
            txt = "".join(o.get("text", []))
            if txt.strip():
                items.append(("text", clean_text(txt)))
        elif t in ("execute_result", "display_data"):
            data = o.get("data", {})
            if "image/png" in data:
                items.append(("img", data["image/png"]))
            txt = data.get("text/plain", "")
            if isinstance(txt, list):
                txt = "".join(txt)
            txt = txt.strip()
            if txt and not txt.startswith("<Figure"):
                items.append(("text", clean_text(txt)))
        elif t == "error":
            items.append(("text", "[ERROR]\n" + "\n".join(o.get("traceback", []))))
    return items


def src(cell):
    s = cell.get("source", [])
    if isinstance(s, list):
        s = "".join(s)
    return s

# --- build report (markdown) ------------------------------------------------
with open(NB_IN) as f:
    nb = json.load(f)

md_parts = []
html_body_parts = []   # parallel HTML stream that handles images directly

try:
    import markdown as md_lib
    have_md = True
except Exception:
    have_md = False

def md_to_html(s):
    if have_md:
        return md_lib.markdown(s, extensions=["fenced_code", "tables", "sane_lists"])
    return "<pre>" + html.escape(s) + "</pre>"


for c in nb["cells"]:
    if c["cell_type"] == "markdown":
        s = src(c).rstrip()
        md_parts.append(s + "\n")
        html_body_parts.append(md_to_html(s))
    elif c["cell_type"] == "code":
        for kind, content in cell_outputs_for_report(c):
            if kind == "text":
                md_parts.append("```text\n" + content + "\n```\n")
                html_body_parts.append(
                    "<pre><code>" + html.escape(content) + "</code></pre>"
                )
            elif kind == "img":
                md_parts.append(f"![figure](data:image/png;base64,{content})\n")
                html_body_parts.append(
                    f'<p><img src="data:image/png;base64,{content}" '
                    f'style="max-width:100%; height:auto;"></p>'
                )

report_md = "\n".join(md_parts)
with open(REPORT_MD, "w") as f:
    f.write(report_md)
print(f"wrote {REPORT_MD}  ({len(report_md):,} bytes)")

# --- HTML doc ---------------------------------------------------------------
body_html = "\n".join(html_body_parts)
html_doc = f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>StructML_HW2_Report</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
          max-width: 950px; margin: 2em auto; padding: 0 1em; line-height: 1.5;
          color: #222; }}
  h1 {{ border-bottom: 2px solid #555; padding-bottom: .2em; }}
  h2 {{ border-bottom: 1px solid #aaa; padding-bottom: .15em; margin-top: 2em; }}
  h3 {{ margin-top: 1.5em; }}
  pre {{ background: #f4f4f4; padding: .7em 1em; border-radius: 4px;
         overflow-x: auto; font-size: 0.82em; line-height: 1.35; }}
  code {{ font-family: "SF Mono", Menlo, Consolas, monospace; }}
  table {{ border-collapse: collapse; margin: .5em 0; }}
  th, td {{ border: 1px solid #bbb; padding: 4px 8px; text-align: left; }}
  th {{ background: #eee; }}
  blockquote {{ border-left: 3px solid #ccc; margin: 0; padding-left: 1em;
                color: #555; }}
  img {{ max-width: 100%; height: auto; }}
  @media print {{
    body {{ max-width: 100%; margin: 0; }}
    pre, img  {{ break-inside: avoid; }}
  }}
</style>
</head><body>
{body_html}
</body></html>
"""
with open(REPORT_HTML, "w") as f:
    f.write(html_doc)
print(f"wrote {REPORT_HTML} ({len(html_doc):,} bytes)")

# --- code-only ipynb --------------------------------------------------------
code_cells = []
for c in nb["cells"]:
    if c["cell_type"] == "code":
        code_cells.append(c)
    elif c["cell_type"] == "markdown":
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
