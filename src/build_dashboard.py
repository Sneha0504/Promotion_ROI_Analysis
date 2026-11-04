"""Unit 5, step 12: precomputed interactive dashboard (PLAN sec. 7).

Reads the committed per-event result tables and summary files from ``data/processed`` and
writes a single **self-contained static page** to ``docs_site/index.html``. Every number and
chart series is **precomputed and embedded** as inline JSON -- the page makes no network calls
and runs no backend at view time, so it works both on GitHub Pages and from a local ``file://``
path. Plotly is vendored locally under ``docs_site/assets`` and referenced by a relative path.

The page has two views:

* **Overview** -- the overall net-profit classification split (counts + %), the Key-Question-3
  headline (naive wins that flip to net negative), the discount x margin sensitivity grid, the
  early-warning backtest precision/recall with its base-rate caveat, and the clearly-labeled
  modeled economic assumptions.
* **Explore** -- a searchable selector over all analyzed promotions; for the chosen promotion a
  single chart overlays actual sales, the ITS counterfactual and the synthetic-control
  counterfactual with the promotion window shaded, alongside the net-profit figure, its
  components, the classification badge, the chosen method, the fit quality and any caution flags.

Nothing is hard-coded: all statistics are read from the precomputed files; paths come from
``src.config``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import Config, load_config
from src.logging_utils import get_logger

logger = get_logger(__name__)

# Vendored Plotly (relative path from docs_site/index.html) and the pinned CDN fallback.
PLOTLY_ASSET_REL = "assets/plotly.min.js"
PLOTLY_CDN_FALLBACK = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly-basic.min.js"


def _r(value: Any, ndigits: int = 1) -> float | None:
    """Round a scalar for embedding, mapping NaN/inf/None to JSON ``null``."""
    if value is None:
        return None
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fval):
        return None
    return round(fval, ndigits)


def _series_payload(group: pd.DataFrame) -> dict[str, Any]:
    """Compact daily-series payload for one event (dates reconstructed in JS from the start)."""
    group = group.sort_values("date")
    dates = pd.to_datetime(group["date"])

    def col(name: str, ndigits: int = 1) -> list[float | None]:
        return [_r(v, ndigits) for v in group[name].to_numpy()]

    return {
        "start": dates.iloc[0].strftime("%Y-%m-%d"),
        "n": int(len(group)),
        "actual": col("actual", 1),
        "its": col("its_cf", 1),
        "sc": col("sc_cf", 1),
        "promo_start": pd.to_datetime(group["promo_start"].iloc[0]).strftime("%Y-%m-%d"),
        "promo_end": pd.to_datetime(group["promo_end"].iloc[0]).strftime("%Y-%m-%d"),
    }


def _build_payload(cfg: Config) -> dict[str, Any]:
    """Assemble the full embedded-data object from the precomputed result files."""
    proc = cfg.paths.data_processed
    display = pd.read_parquet(proc / "display_series.parquet")
    net = pd.read_parquet(proc / "net_impact.parquet")
    events = pd.read_parquet(proc / "analyzable_events.parquet")
    its = pd.read_parquet(proc / "its_results.parquet")
    synth = pd.read_parquet(proc / "synth_results.parquet")
    comparison = pd.read_parquet(proc / "effect_comparison.parquet")
    net_summary = json.loads((proc / "net_impact_summary.json").read_text(encoding="utf-8"))
    ew_summary = json.loads((proc / "early_warning_summary.json").read_text(encoding="utf-8"))

    net_i = net.set_index("event_id")
    ev_i = events.set_index("event_id")
    its_i = its.set_index("event_id")
    syn_i = synth.set_index("event_id")
    cmp_i = comparison.set_index("event_id")
    series_by_event = {eid: g for eid, g in display.groupby("event_id", sort=False)}

    event_rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    for event_id in ev_i.index:
        ev = ev_i.loc[event_id]
        ni = net_i.loc[event_id]
        it = its_i.loc[event_id]
        sc = syn_i.loc[event_id]
        cm = cmp_i.loc[event_id]
        chosen = str(ni["chosen_method"])

        # Chosen method's promo/post percentile-bootstrap effect bounds.
        if chosen == "SC":
            promo_lo, promo_hi = _r(sc["promo_effect_lo"]), _r(sc["promo_effect_hi"])
            post_lo, post_hi = _r(sc["post_effect_lo"]), _r(sc["post_effect_hi"])
        else:
            promo_lo, promo_hi = _r(it["promo_effect_lo"]), _r(it["promo_effect_hi"])
            post_lo, post_hi = _r(it["post_effect_lo"]), _r(it["post_effect_hi"])

        label = f"Store {int(ev['store_nbr'])} · Item {int(ev['item_nbr'])} · {ev['family']}"
        event_rows.append({
            "id": event_id,
            "label": label,
            "store": int(ev["store_nbr"]),
            "item": int(ev["item_nbr"]),
            "family": str(ev["family"]),
            "cls": str(ni["classification"]),
            "method": chosen,
            "flag": bool(ni["low_confidence"] or ni["post_near_zero"]
                         or bool(it["cf_implausible"])),
        })
        detail[event_id] = {
            "series": _series_payload(series_by_event[event_id]),
            "label": label,
            "family": str(ev["family"]),
            "store": int(ev["store_nbr"]),
            "item": int(ev["item_nbr"]),
            "class_num": int(ev["class"]),
            "n_donors": int(ev["n_donors"]),
            "donor_source": str(ev["donor_source"]),
            "classification": str(ni["classification"]),
            "chosen_method": chosen,
            "net_profit": _r(ni["net_profit_impact"]),
            "net_profit_lo": _r(ni["net_profit_lo"]),
            "net_profit_hi": _r(ni["net_profit_hi"]),
            "lift": _r(ni["lift"]),
            "post_effect": _r(ni["post_effect"]),
            "cannibalization": _r(ni["cannibalization"]),
            "cannibalization_share": _r(ni["cannibalization_share"], 3),
            "promo_units_sold": _r(ni["promo_units_sold"]),
            "net_incremental_units": _r(ni["net_incremental_units"]),
            "discount_cost": _r(ni["discount_cost"]),
            "incremental_margin": _r(ni["incremental_margin"]),
            "promo_effect_lo": promo_lo,
            "promo_effect_hi": promo_hi,
            "post_effect_lo": post_lo,
            "post_effect_hi": post_hi,
            "low_confidence": bool(ni["low_confidence"]),
            "post_near_zero": bool(ni["post_near_zero"]),
            "cf_implausible": bool(it["cf_implausible"]),
            "its_r2": _r(it["pre_r2"], 3),
            "its_poor_fit": bool(it["poor_fit"]),
            "sc_rmspe": _r(sc["norm_smooth_rmspe"], 3),
            "sc_poor_fit": bool(sc["poor_fit"]),
            "sc_low_credibility": bool(ni["sc_low_credibility"]),
            "estimate_credible": bool(ni["estimate_credible"]),
            "placebo_p_promo": _r(sc["placebo_pvalue_promo"], 3),
            "placebo_p_post": _r(sc["placebo_pvalue_post"], 3),
            "break_even_discount": _r(ni["break_even_discount"], 4),
            "break_even_margin": _r(ni["break_even_margin"], 4),
            "never_breaks_even": bool(ni["never_breaks_even"]),
            "promo_sign_agree": bool(cm["promo_sign_agree"]),
            "both_trustworthy": bool(cm["both_trustworthy"]),
        }

    event_rows.sort(key=lambda r: (r["store"], r["item"]))

    return {
        "events": event_rows,
        "detail": detail,
        "net_summary": net_summary,
        "early_warning": ew_summary,
    }


def _render_html(payload: dict[str, Any], plotly_src: str) -> str:
    """Inject the embedded data and the Plotly source path into the page template."""
    data_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return (_PAGE_TEMPLATE
            .replace("__PLOTLY_SRC__", plotly_src)
            .replace("__DATA_JSON__", data_json))


def build_dashboard(cfg: Config) -> Path:
    """Build the dashboard and return the path to the written ``index.html``."""
    payload = _build_payload(cfg)

    out_dir = cfg.paths.docs_site
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    vendored = assets_dir / "plotly.min.js"
    if vendored.exists() and vendored.stat().st_size > 0:
        plotly_src = PLOTLY_ASSET_REL
        logger.info("Using vendored Plotly at %s (%d bytes)",
                    vendored, vendored.stat().st_size)
    else:
        plotly_src = PLOTLY_CDN_FALLBACK
        logger.warning("Vendored Plotly not found at %s; falling back to pinned CDN URL %s",
                       vendored, PLOTLY_CDN_FALLBACK)

    html = _render_html(payload, plotly_src)
    out_path = out_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")

    logger.info("Dashboard written: %s (%d events, %.2f MB)", out_path,
                len(payload["events"]), out_path.stat().st_size / 1e6)
    return out_path


def main() -> None:
    """CLI entry point for the ``dashboard`` Makefile target."""
    parser = argparse.ArgumentParser(description="Build the precomputed Plotly dashboard.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    build_dashboard(cfg)


# ---------------------------------------------------------------------------
# Static page template. ``__PLOTLY_SRC__`` and ``__DATA_JSON__`` are substituted at build time.
# All styling and logic is inline so the page is fully self-contained.
# ---------------------------------------------------------------------------
_PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Promotion ROI Analysis</title>
<script src="__PLOTLY_SRC__" charset="utf-8"></script>
<style>
:root{
  --bg:#f7f8fa; --panel:#ffffff; --ink:#1a1d23; --muted:#5b6472; --line:#e3e7ee;
  --accent:#2563eb; --pos:#15803d; --neg:#b91c1c; --inc:#b45309;
  --actual:#2563eb; --its:#d97706; --sc:#0d9488;
  --pos-bg:#e7f6ec; --neg-bg:#fbeaea; --inc-bg:#fdf3e3;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#121519; --panel:#1b1f27; --ink:#eef1f5; --muted:#9aa5b4; --line:#2b313c;
    --accent:#60a5fa; --pos:#4ade80; --neg:#f87171; --inc:#fbbf24;
    --actual:#60a5fa; --its:#fbbf24; --sc:#2dd4bf;
    --pos-bg:#14321f; --neg-bg:#3a1a1a; --inc-bg:#332411;
  }
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:1040px;margin:0 auto;padding:16px}
header h1{font-size:1.6rem;margin:4px 0 2px}
header p{color:var(--muted);margin:0 0 12px;font-size:.95rem}
.assume{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:10px 14px;font-size:.84rem;color:var(--muted);margin-bottom:14px}
.assume b{color:var(--ink)}
.tabs{display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap}
.tab{appearance:none;cursor:pointer;border:1px solid var(--line);background:var(--panel);
  color:var(--ink);padding:8px 16px;border-radius:999px;font-size:.92rem}
.tab[aria-selected="true"]{background:var(--accent);color:#fff;border-color:var(--accent)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:16px;margin-bottom:16px}
h2{font-size:1.15rem;margin:0 0 10px}
h3{font-size:1rem;margin:16px 0 8px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.card{border:1px solid var(--line);border-radius:10px;padding:12px}
.card .big{font-size:1.5rem;font-weight:700}
.card .lbl{color:var(--muted);font-size:.8rem;margin-top:2px}
.split{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;font-weight:600;font-size:.82rem}
.b-pos{background:var(--pos-bg);color:var(--pos)}
.b-neg{background:var(--neg-bg);color:var(--neg)}
.b-inc{background:var(--inc-bg);color:var(--inc)}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{border:1px solid var(--line);padding:7px 9px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{background:rgba(127,127,127,.08);font-weight:600}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
input[type=search],select{font-size:.95rem;padding:8px 10px;border:1px solid var(--line);
  border-radius:8px;background:var(--bg);color:var(--ink)}
input[type=search]{flex:1 1 200px;min-width:0}
select{flex:2 1 320px;min-width:0}
#chart{width:100%;height:440px}
.metrics{display:grid;grid-template-columns:1.1fr .9fr;gap:16px;margin-top:8px}
@media (max-width:720px){.metrics{grid-template-columns:1fr}.split{grid-template-columns:1fr}}
.kv{display:flex;justify-content:space-between;gap:12px;padding:5px 0;
  border-bottom:1px dashed var(--line);font-size:.9rem}
.kv:last-child{border-bottom:none}
.kv .k{color:var(--muted)}
.flags{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap}
.flag{background:var(--inc-bg);color:var(--inc);border-radius:8px;padding:4px 9px;
  font-size:.78rem;font-weight:600}
.note{color:var(--muted);font-size:.82rem;margin-top:8px}
.hidden{display:none}
a{color:var(--accent)}
.legend-pill{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:4px;
  vertical-align:middle}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>Promotion ROI Analysis</h1>
  <p>Did real grocery promotions actually make money? Each effect is isolated with two
  independent counterfactuals (interrupted time series and synthetic control), then netted of
  cannibalization and discount cost. 300 promotions, Corporaci&oacute;n Favorita data.</p>
</header>

<div class="assume" id="assume"></div>

<div class="tabs" role="tablist">
  <button class="tab" id="tab-overview" role="tab" aria-selected="true"
    onclick="showView('overview')">Overview</button>
  <button class="tab" id="tab-explore" role="tab" aria-selected="false"
    onclick="showView('explore')">Explore promotions</button>
</div>

<section id="view-overview">
  <div class="panel">
    <h2>Overall verdict</h2>
    <p class="note" id="headline-lead"></p>
    <div class="split" id="split-cards"></div>
    <p class="note" id="split-note"></p>
  </div>

  <div class="panel">
    <h2>How credible is the net-negative verdict?</h2>
    <p class="note" id="cred-lead"></p>
    <div class="cards" id="cred-cards"></div>
    <p class="note" id="cred-note"></p>
  </div>

  <div class="panel">
    <h2>Break-even &mdash; how much the verdict leans on the assumptions</h2>
    <div class="cards" id="be-cards"></div>
    <p class="note" id="be-note"></p>
  </div>

  <div class="panel">
    <h2>Key question 3 &mdash; naive wins that lose money</h2>
    <div class="cards" id="kq3-cards"></div>
    <p class="note">A "naive win" is any promotion whose promotion-window sales rose versus the
    counterfactual. The flip count is how many of those are net negative once cannibalization
    (pulled-forward demand) and the discount cost are subtracted.</p>
  </div>

  <div class="panel">
    <h2>Sensitivity to the modeled assumptions</h2>
    <p class="note">Discount depth &times; gross margin. Each cell shows median net profit
    (margin-units) and the % net negative. The verdict direction is robust across the grid.</p>
    <div style="overflow-x:auto"><table id="sens-table"></table></div>
  </div>

  <div class="panel">
    <h2>Early-warning rule &mdash; 5-fold cross-validation</h2>
    <p class="note" id="ew-rule"></p>
    <div class="cards" id="ew-cards"></div>
    <p class="note" id="ew-metrics"></p>
    <p class="note" id="ew-base"></p>
    <h3>Pooled out-of-fold confusion matrix (n=300)</h3>
    <div style="overflow-x:auto"><table id="ew-confusion"></table></div>
  </div>
</section>

<section id="view-explore" class="hidden">
  <div class="panel">
    <h2>Explore a promotion</h2>
    <div class="controls">
      <input type="search" id="search" placeholder="Filter by store, item or family&hellip;"
        aria-label="Filter promotions" oninput="filterOptions()">
      <select id="eventSelect" aria-label="Select a promotion" onchange="renderEvent()"></select>
    </div>
    <div>
      <span class="note">
        <span class="legend-pill" style="background:var(--actual)"></span>Actual sales
        &nbsp;<span class="legend-pill" style="background:var(--its)"></span>ITS counterfactual
        &nbsp;<span class="legend-pill" style="background:var(--sc)"></span>Synthetic control
        &nbsp;&mdash; shaded band = promotion window
      </span>
    </div>
    <div id="chart"></div>
    <p class="note">Counterfactual = modeled sales had there been no promotion. Per-event effect
    confidence intervals below are <b>percentile bootstrap</b> intervals.</p>

    <div class="metrics">
      <div>
        <h3>Net impact <span id="m-badge"></span></h3>
        <div class="kv"><span class="k">Net profit impact (margin-units)</span>
          <span class="v" id="m-net"></span></div>
        <div class="kv"><span class="k">Net profit 95% interval</span>
          <span class="v" id="m-netci"></span></div>
        <div class="kv"><span class="k">Promotion-window lift (units)</span>
          <span class="v" id="m-lift"></span></div>
        <div class="kv"><span class="k">Cannibalization / post dip (units)</span>
          <span class="v" id="m-cannib"></span></div>
        <div class="kv"><span class="k">Cannibalization share of lift</span>
          <span class="v" id="m-cshare"></span></div>
        <div class="kv"><span class="k">Net incremental units</span>
          <span class="v" id="m-netunits"></span></div>
        <div class="kv"><span class="k">Discount cost (on all promo units sold)</span>
          <span class="v" id="m-discount"></span></div>
        <div class="kv"><span class="k">Incremental margin</span>
          <span class="v" id="m-margin"></span></div>
        <p class="note" id="cshare-note">Cannibalization share &gt;100% means the post-period
        dip exceeded the promotion-window gain (a small lift followed by a larger later dip);
        it is not capped.</p>
      </div>
      <div>
        <h3>Method &amp; fit</h3>
        <div class="kv"><span class="k">Chosen counterfactual</span>
          <span class="v" id="m-method"></span></div>
        <div class="kv"><span class="k">Fit quality</span>
          <span class="v" id="m-fit"></span></div>
        <div class="kv"><span class="k">Chosen promo effect (95% CI)</span>
          <span class="v" id="m-promoci"></span></div>
        <div class="kv"><span class="k">Chosen post effect (95% CI)</span>
          <span class="v" id="m-postci"></span></div>
        <div class="kv"><span class="k">ITS vs SC promo-sign agreement</span>
          <span class="v" id="m-agree"></span></div>
        <div class="kv"><span class="k">Donors used</span>
          <span class="v" id="m-donors"></span></div>
        <div class="kv"><span class="k">SC placebo p-value (promo / post)</span>
          <span class="v" id="m-placebo"></span></div>
        <div class="kv"><span class="k">Estimate credibility</span>
          <span class="v" id="m-cred"></span></div>
        <div class="kv"><span class="k">Break-even discount / margin</span>
          <span class="v" id="m-be"></span></div>
        <div class="flags" id="m-flags"></div>
      </div>
    </div>
  </div>
</section>

<footer class="note">Results are precomputed; the page loads no external data. Economics are
modeled assumptions (the source data has no price or margin). Confidence intervals are
percentile bootstrap intervals.</footer>
</div>

<script>
"use strict";
const DATA = __DATA_JSON__;

function fmt(x, d){
  if(x===null||x===undefined) return "—";
  return Number(x).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d});
}
function badgeClass(cls){
  if(cls==="net positive") return "b-pos";
  if(cls==="net negative") return "b-neg";
  return "b-inc";
}
function badgeLabel(cls){
  if(cls==="net positive") return "NET POSITIVE";
  if(cls==="net negative") return "NET NEGATIVE";
  return "INCONCLUSIVE";
}

// ---- Overview ----
function renderOverview(){
  const ns = DATA.net_summary, a = ns.assumptions;
  document.getElementById("assume").innerHTML =
    "<b>Modeled economic assumptions</b> (the source data has no price or margin): "+
    "price normalized to <b>1.0</b> so results read in <b>margin-units</b>; discount depth <b>"+
    (a.discount_depth*100)+"%</b>; gross margin <b>"+(a.gross_margin*100)+"%</b>. "+ a.note;

  const med = ns.median_net_profit_impact;
  const exc = ns.overall_split_excl_low_confidence;
  document.getElementById("headline-lead").innerHTML =
    "Across 300 promotions the <b>median</b> net profit impact is <b>"+fmt(med,0)+
    "</b> margin-units &mdash; net negative. Among the <b>"+exc.total+
    "</b> confidently-estimated promotions (excluding "+ns.n_low_confidence+
    " low-confidence), <b>"+exc.pct["net negative"]+"%</b> are net negative and only <b>"+
    exc.pct["net positive"]+"%</b> net positive. (The mean is skewed by a few capped "+
    "low-confidence outliers, so the median and this confident subset lead here.)";

  const sp = ns.overall_split, order=["net positive","net negative","inconclusive"];
  document.getElementById("split-cards").innerHTML = order.map(c=>
    '<div class="card"><div class="badge '+badgeClass(c)+'">'+badgeLabel(c)+'</div>'+
    '<div class="big" style="margin-top:8px">'+sp.counts[c]+'</div>'+
    '<div class="lbl">'+sp.pct[c]+'% of 300</div></div>').join("");
  document.getElementById("split-note").innerHTML =
    "Classification uses the chosen method's promotion + post percentile-bootstrap CIs; an "+
    "event whose net-profit interval straddles zero (or whose both counterfactuals are "+
    "poor-fit) is inconclusive. Overall median "+fmt(med,0)+", mean "+
    fmt(ns.mean_net_profit_impact,0)+" (all) / "+
    fmt(ns.mean_net_profit_impact_excl_low_confidence,0)+" (excluding low-confidence).";

  // Credibility tiers + conservative bound.
  const tn = ns.credibility_tiers["net negative"], tp = ns.credibility_tiers["net positive"];
  const cons = ns.sc_low_credibility_downgrade.overall_split.counts;
  document.getElementById("cred-lead").innerHTML =
    "The "+tn.total+" net-negative labels are stratified by whether the chosen estimate is "+
    "itself credible &mdash; a synthetic control not flagged by placebo inference, or an ITS "+
    "with a good pre-fit.";
  document.getElementById("cred-cards").innerHTML =
    '<div class="card"><div class="big" style="color:var(--pos)">'+tn.high_confidence+
    '</div><div class="lbl">high-confidence net negative</div></div>'+
    '<div class="card"><div class="big" style="color:var(--inc)">'+tn.lower_confidence+
    '</div><div class="lbl">lower-confidence net negative (SC lift not separable from placebo noise)</div></div>'+
    '<div class="card"><div class="big">'+tp.high_confidence+' / '+tp.lower_confidence+
    '</div><div class="lbl">net positive: high / lower confidence</div></div>'+
    '<div class="card"><div class="big">'+cons["net positive"]+' / '+cons["net negative"]+' / '+
    cons["inconclusive"]+'</div><div class="lbl">conservative bound (all low-credibility SC &rarr; inconclusive)</div></div>';
  const rob = ns.placebo_robustness;
  document.getElementById("cred-note").innerHTML =
    "Across all 300 events, "+ns.n_sc_low_credibility_all+" individual SC promotion-window lifts "+
    "are not separable from donor placebo noise (in-space placebo p &gt; 0.10), which is why the "+
    "net-negatives are stratified rather than taken at face value. Even so, <b>"+rob.survive_p90+
    " of the "+rob.n_lower_conf_net_negative+"</b> lower-confidence net-negatives stay net-negative "+
    "under 90th-percentile placebo noise ("+rob.survive_max+" under the maximum placebo). All "+
    tp.total+" net-positive promotions are high-confidence.";

  // Break-even.
  const be = ns.break_even;
  document.getElementById("be-cards").innerHTML =
    '<div class="card"><div class="big">'+be.n_never_break_even+
    '</div><div class="lbl">never break even (net incremental units &le; 0)</div></div>'+
    '<div class="card"><div class="big">'+(be.median_break_even_discount_positive_lift*100).toFixed(1)+
    '%</div><div class="lbl">median break-even discount (positive-lift events)</div></div>'+
    '<div class="card"><div class="big">'+(be.median_break_even_margin_positive_lift*100).toFixed(0)+
    '%</div><div class="lbl">median break-even margin (at the current discount)</div></div>';
  document.getElementById("be-note").innerHTML =
    "At the assumed "+(a.discount_depth*100)+"% discount / "+(a.gross_margin*100)+"% margin, the "+
    "median positive-lift promotion would need a discount below ~"+
    (be.median_break_even_discount_positive_lift*100).toFixed(1)+"% or a margin near ~"+
    (be.median_break_even_margin_positive_lift*100).toFixed(0)+"% to break even. "+be.n_flip_positive_at_alt_discount+
    " currently-non-positive promotions would flip net positive at a "+(be.alt_discount_depth*100)+
    "% discount and "+be.n_flip_positive_at_alt_margin+" at a "+(be.alt_gross_margin*100)+"% margin "+
    "&mdash; point-estimate flips that hold the lift fixed; a shallower discount would likely yield "+
    "a smaller lift, so these are upper bounds.";

  document.getElementById("kq3-cards").innerHTML =
    '<div class="card"><div class="big">'+ns.naive_wins+'</div>'+
    '<div class="lbl">naive wins (sales rose vs counterfactual)</div></div>'+
    '<div class="card"><div class="big" style="color:var(--neg)">'+ns.naive_win_but_net_negative+
    '</div><div class="lbl">of those flip to net negative after cannibalization + discount</div></div>'+
    '<div class="card"><div class="big">'+
    Math.round(100*ns.naive_win_but_net_negative/ns.naive_wins)+
    '%</div><div class="lbl">of naive wins actually lose money</div></div>';

  // Sensitivity grid.
  const cells = ns.sensitivity.cells;
  const margins = ns.sensitivity.gross_margins, depths = ns.sensitivity.discount_depths;
  const byKey = {};
  cells.forEach(c=>byKey[c.discount_depth+"|"+c.gross_margin]=c);
  let html = "<thead><tr><th>discount &darr; / margin &rarr;</th>"+
    margins.map(m=>"<th>"+(m*100)+"%</th>").join("")+"</tr></thead><tbody>";
  depths.forEach(d=>{
    html += "<tr><th>"+(d*100)+"%</th>"+margins.map(m=>{
      const c=byKey[d+"|"+m];
      return "<td>"+fmt(c.median_net_profit_impact,0)+"<br><span class='lbl'>"+
        c.pct_negative+"% neg</span></td>";
    }).join("")+"</tr>";
  });
  html += "</tbody>";
  document.getElementById("sens-table").innerHTML = html;

  // Early warning: 5-fold stratified CV, pooled out-of-fold over n=300.
  const ew = DATA.early_warning;
  const pooled = ew.test;                       // pooled OOF under the deployable metric (Youden)
  const yo = ew.cv_by_metric.youden, f1 = ew.cv_by_metric.f1;
  const yp = yo.pooled_out_of_fold, fp = f1.pooled_out_of_fold;
  document.getElementById("ew-rule").innerHTML =
    ew.rule+" Deployable metric: <b>"+ew.deployable_metric+"</b>'s threshold "+
    "<b>"+ew.deployable_threshold+"</b> (chosen on the CV, not nested).";
  document.getElementById("ew-cards").innerHTML =
    '<div class="card"><div class="big">'+pooled.precision+'</div>'+
    '<div class="lbl">pooled out-of-fold precision (Youden)</div></div>'+
    '<div class="card"><div class="big">'+pooled.recall+'</div>'+
    '<div class="lbl">pooled out-of-fold recall</div></div>'+
    '<div class="card"><div class="big">'+Math.round(100*pooled.base_rate_net_negative)+
    '%</div><div class="lbl">base rate (net negative)</div></div>'+
    '<div class="card"><div class="big">'+fp.precision+' / '+fp.recall+'</div>'+
    '<div class="lbl">F1 alternative (near-trivial: flags almost everything)</div></div>';
  document.getElementById("ew-metrics").innerHTML =
    "5-fold stratified CV (seeded), pooled over every event's out-of-fold prediction. "+
    "Youden across folds: precision "+yo.precision_mean+" &plusmn; "+yo.precision_std+
    ", recall "+yo.recall_mean+" &plusmn; "+yo.recall_std+", F1 "+yo.f1_mean+" &plusmn; "+
    yo.f1_std+". F1 tuning instead drives the threshold to flag almost everything "+
    "(precision "+fp.precision+" &asymp; base rate), so Youden is the deployable choice.";
  document.getElementById("ew-base").innerHTML =
    "<b>Honest caveat.</b> "+ew.verdict+" The ~"+
    (pooled.precision - pooled.base_rate_net_negative).toFixed(3)+
    " precision lift is modest, and the metric was chosen on the CV (not in a nested loop), a "+
    "mild optimism. The earlier single 60/40 split is not reported (one favourable draw).";

  const cm = pooled.confusion_matrix;
  document.getElementById("ew-confusion").innerHTML =
    "<thead><tr><th></th><th>predicted net neg</th><th>predicted not</th></tr></thead><tbody>"+
    "<tr><th>actual net neg</th><td>"+cm.tp+" (TP)</td><td>"+cm.fn+" (FN)</td></tr>"+
    "<tr><th>actual not</th><td>"+cm.fp+" (FP)</td><td>"+cm.tn+" (TN)</td></tr></tbody>";
}

// ---- Explore ----
function populateSelect(filter){
  const sel = document.getElementById("eventSelect");
  const prev = sel.value;
  const f = (filter||"").trim().toLowerCase();
  const opts = DATA.events.filter(e=> !f || e.label.toLowerCase().includes(f) ||
    e.cls.toLowerCase().includes(f));
  sel.innerHTML = opts.map(e=>{
    const mark = e.flag ? " ⚠" : "";
    return '<option value="'+e.id+'">'+e.label+" — "+badgeLabel(e.cls)+mark+"</option>";
  }).join("");
  if(opts.some(e=>e.id===prev)) sel.value = prev;
  return opts.length;
}
function filterOptions(){
  const n = populateSelect(document.getElementById("search").value);
  if(n>0) renderEvent();
}

function themeColors(){
  const dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  const css = getComputedStyle(document.documentElement);
  return {
    ink: css.getPropertyValue("--ink").trim(),
    muted: css.getPropertyValue("--muted").trim(),
    line: css.getPropertyValue("--line").trim(),
    actual: css.getPropertyValue("--actual").trim(),
    its: css.getPropertyValue("--its").trim(),
    sc: css.getPropertyValue("--sc").trim(),
    shade: dark ? "rgba(148,163,184,0.16)" : "rgba(37,99,235,0.08)"
  };
}
function datesFrom(start, n){
  const out=[], d=new Date(start+"T00:00:00");
  for(let i=0;i<n;i++){ out.push(new Date(d.getTime()+i*86400000).toISOString().slice(0,10)); }
  return out;
}
function renderChart(det){
  const c = themeColors(), s = det.series, x = datesFrom(s.start, s.n);
  const traces = [
    {x:x, y:s.actual, name:"Actual", mode:"lines", line:{color:c.actual,width:2.4},
      hovertemplate:"%{x}<br>Actual: %{y:.0f}<extra></extra>"},
    {x:x, y:s.its, name:"ITS counterfactual", mode:"lines",
      line:{color:c.its,width:1.8,dash:"dot"},
      hovertemplate:"%{x}<br>ITS cf: %{y:.0f}<extra></extra>"},
    {x:x, y:s.sc, name:"Synthetic control", mode:"lines",
      line:{color:c.sc,width:1.8,dash:"dash"},
      hovertemplate:"%{x}<br>SC cf: %{y:.0f}<extra></extra>"}
  ];
  const layout = {
    margin:{l:52,r:16,t:10,b:40}, height:440, paper_bgcolor:"rgba(0,0,0,0)",
    plot_bgcolor:"rgba(0,0,0,0)", font:{color:c.ink,size:12},
    xaxis:{gridcolor:c.line, zeroline:false},
    yaxis:{title:"Daily unit sales", gridcolor:c.line, zeroline:false, rangemode:"tozero"},
    legend:{orientation:"h", y:1.12, x:0, font:{color:c.ink}},
    hovermode:"x unified",
    shapes:[{type:"rect", xref:"x", yref:"paper", x0:s.promo_start, x1:s.promo_end,
      y0:0, y1:1, fillcolor:c.shade, line:{width:0}, layer:"below"}],
    annotations:[{x:s.promo_start, xref:"x", y:1, yref:"paper", yanchor:"bottom",
      text:"Promotion", showarrow:false, font:{color:c.muted,size:11}, xanchor:"left"}]
  };
  Plotly.react("chart", traces, layout, {responsive:true, displayModeBar:false});
}
function renderMetrics(det){
  const badge = document.getElementById("m-badge");
  badge.className = "badge "+badgeClass(det.classification);
  badge.textContent = badgeLabel(det.classification);
  document.getElementById("m-net").textContent = fmt(det.net_profit,1);
  document.getElementById("m-netci").textContent =
    "["+fmt(det.net_profit_lo,0)+", "+fmt(det.net_profit_hi,0)+"]";
  document.getElementById("m-lift").textContent = fmt(det.lift,1);
  document.getElementById("m-cannib").textContent = fmt(det.cannibalization,1);
  const cs = det.cannibalization_share;
  document.getElementById("m-cshare").textContent =
    cs===null ? "— (no positive lift)"
      : (cs*100).toFixed(1)+"%"+(cs>1 ? " (dip > gain)" : "");
  document.getElementById("cshare-note").style.display = (cs!==null && cs>1) ? "" : "none";
  document.getElementById("m-netunits").textContent = fmt(det.net_incremental_units,1);
  document.getElementById("m-discount").textContent = fmt(det.discount_cost,1);
  document.getElementById("m-margin").textContent = fmt(det.incremental_margin,1);

  document.getElementById("m-method").textContent =
    det.chosen_method==="SC" ? "Synthetic control" : "Interrupted time series";
  document.getElementById("m-fit").textContent = det.chosen_method==="SC"
    ? "SC pre-RMSPE "+fmt(det.sc_rmspe,3)+(det.sc_poor_fit?" (poor)":" (ok)")
    : "ITS pre-period R² "+fmt(det.its_r2,3)+(det.its_poor_fit?" (poor)":" (ok)");
  document.getElementById("m-promoci").textContent =
    fmt(det.lift,0)+"  ["+fmt(det.promo_effect_lo,0)+", "+fmt(det.promo_effect_hi,0)+"]";
  document.getElementById("m-postci").textContent =
    fmt(det.post_effect,0)+"  ["+fmt(det.post_effect_lo,0)+", "+fmt(det.post_effect_hi,0)+"]";
  document.getElementById("m-agree").textContent = det.promo_sign_agree ? "agree" : "disagree";
  document.getElementById("m-donors").textContent = det.n_donors+" ("+det.donor_source+")";
  document.getElementById("m-placebo").textContent =
    (det.chosen_method==="SC")
      ? fmt(det.placebo_p_promo,3)+" / "+fmt(det.placebo_p_post,3)
        +(det.sc_low_credibility?" (low credibility)":" (credible)")
      : fmt(det.placebo_p_promo,3)+" / "+fmt(det.placebo_p_post,3)+" (ITS chosen)";
  document.getElementById("m-cred").textContent =
    det.estimate_credible ? "high (chosen estimate is credible)"
                          : "lower (chosen estimate not credible)";
  document.getElementById("m-be").textContent =
    (det.never_breaks_even
      ? "never (net incremental units ≤ 0)"
      : (det.break_even_discount===null ? "—" : (det.break_even_discount*100).toFixed(1)+"%")
        +" / "+(det.break_even_margin===null ? "—" : (det.break_even_margin*100).toFixed(0)+"%"));

  const flags=[];
  if(det.low_confidence) flags.push("Low confidence (both methods poor-fit)");
  if(det.cf_implausible) flags.push("ITS counterfactual capped (implausible)");
  if(det.post_near_zero) flags.push("Post window near zero (possible delisting/stockout)");
  if(det.chosen_method==="SC" && det.sc_low_credibility)
    flags.push("SC lift not separable from placebo noise");
  const fe = document.getElementById("m-flags");
  fe.innerHTML = flags.map(f=>'<span class="flag">⚠ '+f+'</span>').join("");
}
function renderEvent(){
  const id = document.getElementById("eventSelect").value;
  const det = DATA.detail[id];
  if(!det) return;
  renderChart(det);
  renderMetrics(det);
}

function showView(v){
  const ov = v==="overview";
  document.getElementById("view-overview").classList.toggle("hidden", !ov);
  document.getElementById("view-explore").classList.toggle("hidden", ov);
  document.getElementById("tab-overview").setAttribute("aria-selected", ov);
  document.getElementById("tab-explore").setAttribute("aria-selected", !ov);
  if(!ov){
    // The chart is first drawn while the explore view is display:none, where Plotly falls back
    // to its 700px default. Redraw and resize now that the container has real width.
    renderEvent();
    Plotly.Plots.resize("chart");
  }
}

renderOverview();
populateSelect("");
renderEvent();
if(window.matchMedia){
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", ()=>{
    renderOverview();
    if(!document.getElementById("view-explore").classList.contains("hidden")) renderEvent();
  });
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
