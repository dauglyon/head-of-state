"""Delay against spurious switches for each decoder (HANDOFF task 6), as a self-contained HTML page.

    uv run python -I experiments/plot_tradeoff.py results/s3_dev_sweep.csv results/s3_dev_tradeoff.html \
        --title "S3 PubMed dev" --rotations 2 --stride 1 --prior-power 0
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

W, H, M = 720, 420, dict(l=64, r=150, t=24, b=52)
SERIES = [  # (decoder column value, legend name, parameter name, css slot)
    ("persist", "Persist rule", "persist", "--series-1"),
    ("filter_L×1", "Forward filter (declared L)", "η", "--series-2"),
    ("raw", "Raw argmax", "", "--series-3"),
]

CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--page:#f9f9f7;--text-primary:#0b0b0b;--text-secondary:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;
font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--page);color:var(--text-primary);padding:16px}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface-1:#1a1a19;
--page:#0d0d0d;--text-primary:#f0efec;--text-secondary:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--series-1:#3987e5;
--series-2:#d95926;--series-3:#199e70}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--page:#0d0d0d;--text-primary:#f0efec;
--text-secondary:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70}
.card{background:var(--surface-1);border-radius:8px;padding:16px;max-width:760px;position:relative}
h1{font-size:16px;margin:0 0 2px}p.sub{font-size:13px;color:var(--text-secondary);margin:0 0 8px}
svg text{font-size:12px;fill:var(--text-secondary)} .legend{display:flex;gap:16px;font-size:13px;margin:4px 0 8px}
.legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:5px;margin-right:6px;background:var(--c)}
.tip{position:absolute;pointer-events:none;background:var(--surface-1);border:1px solid var(--axis);border-radius:6px;
padding:6px 8px;font-size:12px;display:none;white-space:nowrap;color:var(--text-primary)}
table{border-collapse:collapse;font-size:12px;margin-top:12px}td,th{padding:3px 8px;text-align:right;border-bottom:1px solid var(--grid)}
th:first-child,td:first-child{text-align:left}details{margin-top:8px;font-size:13px}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--title", required=True)
    ap.add_argument("--rotations", default="2")
    ap.add_argument("--stride", default="1")
    ap.add_argument("--prior-power", default="0.0")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(a.csv.open())
            if r["rotations"] == a.rotations and r["stride"] == a.stride
            and float(r["prior_power"]) == float(a.prior_power)]
    pts = {}
    for key, *_ in SERIES:
        sel = [r for r in rows if r["decoder"] == key]
        pts[key] = sorted(({"x": float(r["mean_delay_tokens"]), "y": float(r["spurious_per_1k_tokens"]),
                            "p": r["param"], "acc": float(r["accuracy_committed"]),
                            "m": f'{r["switches_matched"]}/{r["true_switches"]}'} for r in sel),
                          key=lambda d: d["x"])
    allp = [d for v in pts.values() for d in v]
    xmax = max(d["x"] for d in allp) * 1.08
    # Log y: the raw decoder's spurious rate is an order of magnitude above the useful settings.
    pos_y = [d["y"] for d in allp if d["y"] > 0]
    ymin, ymax = min(pos_y) / 1.5, max(pos_y) * 1.3   # zero rates are drawn on the floor (tooltip shows 0)
    pw, ph = W - M["l"] - M["r"], H - M["t"] - M["b"]
    sx = lambda x: M["l"] + pw * x / xmax
    sy = lambda y: M["t"] + ph * (1 - (math.log10(max(y, ymin)) - math.log10(ymin))
                                   / (math.log10(ymax) - math.log10(ymin)))

    def ticks(vmax):
        step = next(s for s in [1, 2, 5, 10, 20, 25, 50, 100] if vmax / s <= 6)
        return [i * step for i in range(int(vmax / step) + 1)]

    svg = [f'<svg width="{W}" height="{H}" role="img" aria-label="{html.escape(a.title)}">']
    yt = [m * 10 ** e for e in range(-2, 4) for m in (1, 2, 5) if ymin <= m * 10 ** e <= ymax]
    for t in yt:
        svg.append(f'<line x1="{M["l"]}" x2="{M["l"] + pw}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke="var(--grid)"/>'
                   f'<text x="{M["l"] - 8}" y="{sy(t) + 4:.1f}" text-anchor="end">{t:g}</text>')
    for t in ticks(xmax):
        svg.append(f'<text x="{sx(t):.1f}" y="{M["t"] + ph + 18}" text-anchor="middle">{t:g}</text>')
    svg.append(f'<line x1="{M["l"]}" x2="{M["l"] + pw}" y1="{M["t"] + ph}" y2="{M["t"] + ph}" stroke="var(--axis)"/>')
    svg.append(f'<text x="{M["l"] + pw / 2}" y="{H - 10}" text-anchor="middle">Mean delay after a true switch (tokens)</text>')
    svg.append(f'<text transform="translate(16,{M["t"] + ph / 2}) rotate(-90)" text-anchor="middle">'
               'Spurious switches per 1,000 tokens (log)</text>')
    for key, name, pname, slot in SERIES:
        ps = pts[key]
        if len(ps) > 1:
            d = " ".join(f'{"M" if i == 0 else "L"}{sx(p["x"]):.1f},{sy(p["y"]):.1f}' for i, p in enumerate(ps))
            svg.append(f'<path d="{d}" fill="none" stroke="var({slot})" stroke-width="2"/>')
        for p in ps:
            tip = html.escape(json.dumps({"name": name, "pname": pname, **p}))
            svg.append(f'<circle cx="{sx(p["x"]):.1f}" cy="{sy(p["y"]):.1f}" r="4.5" fill="var({slot})" '
                       f'stroke="var(--surface-1)" stroke-width="2"/>'
                       f'<circle class="hit" cx="{sx(p["x"]):.1f}" cy="{sy(p["y"]):.1f}" r="12" fill="transparent" '
                       f'data-tip="{tip}"/>')
    # Direct labels at each series' last point (largest delay), pushed apart vertically.
    labels = sorted(([sy(pts[k][-1]["y"]) + 4, sx(pts[k][-1]["x"]) + 10, n] for k, n, *_ in SERIES if pts[k]),
                    key=lambda l: l[0])
    for i in range(1, len(labels)):
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + 15)
    for y, x, n in labels:
        svg.append(f'<text x="{x:.1f}" y="{y:.1f}" style="fill:var(--text-primary)">{html.escape(n)}</text>')
    svg.append("</svg>")

    table = ["<table><tr><th>Decoder</th><th>Parameter</th><th>Mean delay</th><th>Spurious /1k</th>"
             "<th>Matched</th><th>Committed acc.</th></tr>"]
    for key, name, pname, _ in SERIES:
        for p in pts[key]:
            table.append(f'<tr><td>{html.escape(name)}</td><td>{html.escape(pname + " " + p["p"]) if pname else ""}</td>'
                         f'<td>{p["x"]:.1f}</td><td>{p["y"]:.2f}</td><td>{p["m"]}</td><td>{p["acc"]:.3f}</td></tr>')
    table.append("</table>")
    legend = "".join(f'<span style="--c:var({slot})">{html.escape(name)}</span>' for _, name, _, slot in SERIES)
    sub = (f"rotations {a.rotations}, stride {a.stride}, prior-power {a.prior_power}. "
           "Lower left is better. Hover a point for its parameter.")
    page = f"""<!doctype html><meta charset="utf-8"><title>{html.escape(a.title)}</title><style>{CSS}</style>
<div class="viz-root"><div class="card"><h1>{html.escape(a.title)}: delay against spurious switches</h1>
<p class="sub">{html.escape(sub)}</p><div class="legend">{legend}</div>{"".join(svg)}<div class="tip"></div>
<details><summary>Table</summary>{"".join(table)}</details></div></div>
<script>
const card=document.querySelector('.card'),tip=document.querySelector('.tip');
document.querySelectorAll('circle.hit').forEach(c=>{{
 c.addEventListener('mouseenter',()=>{{const d=JSON.parse(c.dataset.tip);
  tip.innerHTML=`<b>${{d.name}}</b>${{d.pname?' — '+d.pname+' '+d.p:''}}<br>delay ${{d.x.toFixed(1)}} tokens · spurious ${{d.y.toFixed(2)}}/1k<br>matched ${{d.m}} · committed acc. ${{d.acc.toFixed(3)}}`;
  tip.style.display='block';const r=card.getBoundingClientRect(),b=c.getBoundingClientRect();
  tip.style.left=(b.left-r.left+16)+'px';tip.style.top=(b.top-r.top-8)+'px';}});
 c.addEventListener('mouseleave',()=>tip.style.display='none');}});
</script>"""
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(page, encoding="utf-8")
    print(a.out)


if __name__ == "__main__":
    main()
