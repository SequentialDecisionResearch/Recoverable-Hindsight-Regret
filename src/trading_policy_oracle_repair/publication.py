from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .hashing import sha256_file


def support_flag(low: float, high: float) -> str:
    if np.isfinite(low) and float(low) > 0.0:
        return "positive_95"
    if np.isfinite(high) and float(high) < 0.0:
        return "negative_95"
    return "inconclusive"


def add_support_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    pairs = [
        ("log_utility_improvement", "path_support"),
        ("mean_actual_advantage", "decision_advantage_support"),
        ("continuous_log_utility_improvement", "continuous_support"),
    ]
    for prefix, flag in pairs:
        lo, hi = f"{prefix}_ci_low", f"{prefix}_ci_high"
        if lo in out.columns and hi in out.columns:
            out[flag] = [support_flag(a, b) for a, b in zip(out[lo], out[hi])]
    return out


def compact_selective_table(pub: pd.DataFrame) -> pd.DataFrame:
    p = add_support_flags(pub)
    cols = [
        "split", "model_name", "lcb_method", "baseline_policy_id",
        "log_utility_improvement_observed", "log_utility_improvement_ci_low", "log_utility_improvement_ci_high",
        "log_utility_improvement_prob_gt_zero", "path_support",
        "mean_actual_advantage_observed", "mean_actual_advantage_ci_low", "mean_actual_advantage_ci_high",
        "mean_actual_advantage_prob_gt_zero", "decision_advantage_support",
        "harm_rate_observed", "harm_rate_ci_low", "harm_rate_ci_high",
        "coverage_observed", "coverage_ci_low", "coverage_ci_high",
        "positive_year_fraction", "positive_asset_fraction",
        "continuous_log_utility_improvement_observed", "continuous_log_utility_improvement_ci_low",
        "continuous_log_utility_improvement_ci_high", "continuous_log_utility_improvement_prob_gt_zero",
        "continuous_support",
    ]
    return p[[c for c in cols if c in p.columns]].copy()


def compact_meta_table(pub: pd.DataFrame) -> pd.DataFrame:
    p = add_support_flags(pub)
    cols = [
        "split", "strategy_type", "model_name", "lcb_method",
        "log_utility_improvement_observed", "log_utility_improvement_ci_low", "log_utility_improvement_ci_high",
        "log_utility_improvement_prob_gt_zero", "path_support",
        "coverage_observed", "harm_rate_observed", "mean_actual_advantage_observed",
        "positive_year_fraction",
        "continuous_log_utility_improvement_observed", "continuous_log_utility_improvement_ci_low",
        "continuous_log_utility_improvement_ci_high", "continuous_log_utility_improvement_prob_gt_zero",
        "continuous_support",
    ]
    return p[[c for c in cols if c in p.columns]].copy()



def _fmt_num(x: Any, digits: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return ""
    return f"{v:.{digits}f}" if np.isfinite(v) else ""


def _fmt_ci(row: pd.Series, prefix: str) -> str:
    return f"{_fmt_num(row.get(prefix + '_observed'))} [{_fmt_num(row.get(prefix + '_ci_low'))}, {_fmt_num(row.get(prefix + '_ci_high'))}]"


def paper_selective_table(pub: pd.DataFrame) -> pd.DataFrame:
    p = add_support_flags(pub)
    rows = []
    for _, r in p.iterrows():
        rows.append({
            "split": r.get("split", ""),
            "model": r.get("model_name", ""),
            "baseline": r.get("baseline_policy_id", ""),
            "path dlogU [95% CI]": _fmt_ci(r, "log_utility_improvement"),
            "Pboot(dlogU>0)": _fmt_num(r.get("log_utility_improvement_prob_gt_zero"), 3),
            "local advantage [95% CI]": _fmt_ci(r, "mean_actual_advantage"),
            "harm": _fmt_num(r.get("harm_rate_observed"), 3),
            "coverage": _fmt_num(r.get("coverage_observed"), 3),
            "positive years": _fmt_num(r.get("positive_year_fraction"), 2),
            "continuous dlogU [95% CI]": _fmt_ci(r, "continuous_log_utility_improvement"),
            "path support": r.get("path_support", ""),
            "local support": r.get("decision_advantage_support", ""),
        })
    return pd.DataFrame(rows)


def paper_meta_table(pub: pd.DataFrame) -> pd.DataFrame:
    p = add_support_flags(pub)
    rows = []
    for _, r in p.iterrows():
        rows.append({
            "split": r.get("split", ""),
            "strategy": r.get("strategy_type", ""),
            "model": r.get("model_name", ""),
            "path dlogU [95% CI]": _fmt_ci(r, "log_utility_improvement"),
            "Pboot(dlogU>0)": _fmt_num(r.get("log_utility_improvement_prob_gt_zero"), 3),
            "coverage": _fmt_num(r.get("coverage_observed"), 3),
            "harm": _fmt_num(r.get("harm_rate_observed"), 3),
            "continuous dlogU [95% CI]": _fmt_ci(r, "continuous_log_utility_improvement"),
            "path support": r.get("path_support", ""),
            "continuous support": r.get("continuous_support", ""),
        })
    return pd.DataFrame(rows)


def paper_synthetic_table(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "world", "signal_strength", "prediction_spearman_mean", "coverage_mean",
        "harm_rate_mean", "deployed_gain_per_decision_mean",
        "oracle_positive_gain_per_decision_mean", "recoverable_fraction_mean",
    ]
    return summary[[c for c in cols if c in summary.columns]].copy()

def claim_evidence_table(selective: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    s = add_support_flags(selective)
    m = add_support_flags(meta)
    rows: list[dict[str, Any]] = []
    if not s.empty:
        decision_positive = int((s.get("decision_advantage_support") == "positive_95").sum())
        path_positive = int((s.get("path_support") == "positive_95").sum())
        path_negative = int((s.get("path_support") == "negative_95").sum())
        cont_positive = int((s.get("continuous_support") == "positive_95").sum()) if "continuous_support" in s else 0
        rows.extend([
            {
                "claim_id": "C1_local_recoverability",
                "claim": "Calibrated selective intervention identifies local states with positive realized continuation advantage.",
                "status": "supported" if decision_positive > 0 else "not_supported",
                "evidence": f"{decision_positive}/{len(s)} selective-repair groups have a 95% decision-advantage CI entirely above zero.",
            },
            {
                "claim_id": "C2_end_to_end_wealth",
                "claim": "Selective repair produces statistically supported post-cost path-level utility improvement in at least some registered settings.",
                "status": "supported" if path_positive > 0 else "not_supported",
                "evidence": f"{path_positive}/{len(s)} groups have positive 95% path-level utility CI; {path_negative} are significantly negative.",
            },
            {
                "claim_id": "C3_continuous_robustness",
                "claim": "Continuous pseudo-live wealth improvement is statistically supported in at least some registered settings.",
                "status": "supported" if cont_positive > 0 else "underpowered_or_not_supported",
                "evidence": f"{cont_positive}/{len(s)} groups have positive 95% continuous utility CI; continuous inference uses only the available asset paths.",
            },
        ])
    if not m.empty:
        meta_pos = int(((m["strategy_type"] == "meta_selector") & (m.get("path_support") == "positive_95")).sum())
        meta_neg_cont = int(((m["strategy_type"] == "meta_selector") & (m.get("continuous_support") == "negative_95")).sum()) if "continuous_support" in m else 0
        rows.append({
            "claim_id": "C4_global_meta_selector",
            "claim": "Global meta-policy selection is a robust primary contribution.",
            "status": "not_supported" if meta_pos == 0 else "mixed",
            "evidence": f"{meta_pos} fold-local meta-selector groups have positive 95% utility CI; {meta_neg_cont} continuous meta groups have negative 95% CI.",
        })
    return pd.DataFrame(rows)


def _write_latex(frame: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    # Escape all text; floats are deliberately compact for paper-ready review.
    latex = frame.to_latex(
        index=False,
        escape=True,
        float_format=lambda x: f"{x:.4f}" if np.isfinite(x) else "",
        caption=caption,
        label=label,
        longtable=False,
    )
    path.write_text(latex, encoding="utf-8")


def write_table_bundle(
    frame: pd.DataFrame,
    *,
    stem: str,
    table_dir: Path,
    caption: str,
    label: str,
    write_latex: bool,
) -> list[Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / f"{stem}.csv"
    frame.to_csv(csv_path, index=False)
    out = [csv_path]
    if write_latex:
        tex_path = table_dir / f"{stem}.tex"
        _write_latex(frame, tex_path, caption=caption, label=label)
        out.append(tex_path)
    return out


def _save_figure(fig: plt.Figure, stem: str, figure_dir: Path, *, write_pdf: bool) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    png = figure_dir / f"{stem}.png"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    out = [png]
    if write_pdf:
        pdf = figure_dir / f"{stem}.pdf"
        fig.savefig(pdf, bbox_inches="tight")
        out.append(pdf)
    plt.close(fig)
    return out


def _nice_label(row: pd.Series) -> str:
    model = str(row.get("model_name", ""))
    baseline = str(row.get("baseline_policy_id", ""))
    return f"{model} | {baseline}"


def plot_forest(
    frame: pd.DataFrame,
    *,
    metric_prefix: str,
    title: str,
    stem_prefix: str,
    figure_dir: Path,
    write_pdf: bool,
) -> list[Path]:
    paths: list[Path] = []
    for split, g in frame.groupby("split", observed=True, dropna=False):
        obs = f"{metric_prefix}_observed"
        lo = f"{metric_prefix}_ci_low"
        hi = f"{metric_prefix}_ci_high"
        if not {obs, lo, hi}.issubset(g.columns):
            continue
        gg = g.loc[np.isfinite(pd.to_numeric(g[obs], errors="coerce"))].copy()
        if gg.empty:
            continue
        gg["label"] = gg.apply(_nice_label, axis=1)
        gg = gg.sort_values(obs).reset_index(drop=True)
        y = np.arange(len(gg))
        x = gg[obs].to_numpy(dtype=float)
        xlo = gg[lo].to_numpy(dtype=float)
        xhi = gg[hi].to_numpy(dtype=float)
        err = np.vstack([x - xlo, xhi - x])
        fig, ax = plt.subplots(figsize=(8.2, max(4.6, 0.42 * len(gg) + 1.8)))
        ax.errorbar(x, y, xerr=err, fmt="o", capsize=3)
        ax.axvline(0.0, linewidth=1.0)
        ax.set_yticks(y, gg["label"].tolist())
        ax.set_xlabel(metric_prefix.replace("_", " "))
        ax.set_title(f"{title} - {split}")
        paths.extend(_save_figure(fig, f"{stem_prefix}__{split}", figure_dir, write_pdf=write_pdf))
    return paths


def plot_annual_heatmaps(annual: pd.DataFrame, figure_dir: Path, *, write_pdf: bool) -> list[Path]:
    paths: list[Path] = []
    if annual.empty:
        return paths
    a = annual.copy()
    a["series"] = a["model_name"].astype(str) + " | " + a["baseline_policy_id"].astype(str)
    for split, g in a.groupby("split", observed=True, dropna=False):
        pivot = g.pivot_table(index="series", columns="fold_id", values="mean_log_utility_improvement", aggfunc="mean")
        if pivot.empty:
            continue
        arr = pivot.to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(11.5, max(4.8, 0.42 * len(pivot.index) + 1.6)))
        im = ax.imshow(arr, aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)), [str(x).replace("test_", "") for x in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)), pivot.index.tolist())
        ax.set_title(f"Annual mean log-utility improvement - {split}")
        ax.set_xlabel("OOS test year")
        fig.colorbar(im, ax=ax, shrink=0.85, label="mean log-utility improvement")
        paths.extend(_save_figure(fig, f"annual_repair_gain_heatmap__{split}", figure_dir, write_pdf=write_pdf))
    return paths


def plot_synthetic_recoverability(summary: pd.DataFrame, figure_dir: Path, *, write_pdf: bool) -> list[Path]:
    paths: list[Path] = []
    if summary.empty:
        return paths
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for world, g in summary.groupby("world", observed=True):
        g = g.sort_values("signal_strength")
        ax.errorbar(
            g["signal_strength"], g["recoverable_fraction_mean"],
            yerr=g["recoverable_fraction_std"].fillna(0.0), marker="o", capsize=3, label=str(world),
        )
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("Synthetic feature signal strength")
    ax.set_ylabel("Recovered fraction of hindsight-positive opportunity")
    ax.set_title("Synthetic recoverability vs signal strength")
    ax.legend()
    paths.extend(_save_figure(fig, "synthetic_recoverability_curve", figure_dir, write_pdf=write_pdf))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for world, g in summary.groupby("world", observed=True):
        g = g.sort_values("signal_strength")
        ax.plot(g["signal_strength"], g["coverage_mean"], marker="o", label=f"{world} coverage")
    ax.set_xlabel("Synthetic feature signal strength")
    ax.set_ylabel("Deployment coverage")
    ax.set_title("Synthetic abstention behavior")
    ax.legend()
    paths.extend(_save_figure(fig, "synthetic_deployment_coverage", figure_dir, write_pdf=write_pdf))
    return paths


def plot_coverage_harm(frontier: pd.DataFrame, figure_dir: Path, *, write_pdf: bool) -> list[Path]:
    paths: list[Path] = []
    if frontier.empty:
        return paths
    needed = {"split", "model_name", "baseline_policy_id", "lcb_method", "deploy_margin", "coverage_mean", "harm_rate_mean"}
    if not needed.issubset(frontier.columns):
        return paths
    primary = frontier.loc[frontier["lcb_method"].astype(str) == "empirical"].copy()
    for (split, model), g in primary.groupby(["split", "model_name"], observed=True):
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        for baseline, h in g.groupby("baseline_policy_id", observed=True):
            h = h.sort_values("deploy_margin")
            ax.plot(h["coverage_mean"], h["harm_rate_mean"], marker="o", label=str(baseline))
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Harm rate among deployed decisions")
        ax.set_title(f"Coverage-harm frontier - {split} - {model}")
        ax.legend(fontsize=8)
        paths.extend(_save_figure(fig, f"coverage_harm__{split}__{model}", figure_dir, write_pdf=write_pdf))
    return paths


def plot_prediction_deciles(deciles: pd.DataFrame, figure_dir: Path, *, write_pdf: bool) -> list[Path]:
    paths: list[Path] = []
    needed = {"split", "policy_id", "task", "model_name", "decile", "mean_actual"}
    if deciles.empty or not needed.issubset(deciles.columns):
        return paths
    d = deciles.loc[(deciles["task"].astype(str) == "regret") & deciles["split"].astype(str).isin(["same_asset_test", "asset_oos_test"])].copy()
    for (split, model), g in d.groupby(["split", "model_name"], observed=True):
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
        agg = g.groupby(["policy_id", "decile"], observed=True)["mean_actual"].mean().reset_index()
        for policy, h in agg.groupby("policy_id", observed=True):
            h = h.sort_values("decile")
            ax.plot(h["decile"], h["mean_actual"], marker="o", label=str(policy))
        ax.set_xlabel("Predicted-regret decile")
        ax.set_ylabel("Mean realized regret")
        ax.set_title(f"Predicted vs realized regret by decile - {split} - {model}")
        ax.legend(fontsize=8)
        paths.extend(_save_figure(fig, f"regret_deciles__{split}__{model}", figure_dir, write_pdf=write_pdf))
    return paths


def plot_switching_frontier(repair_oos: pd.DataFrame, figure_dir: Path, *, write_pdf: bool) -> list[Path]:
    paths: list[Path] = []
    needed = {"split", "model_name", "baseline_policy_id", "switching_burden", "log_utility_improvement"}
    if repair_oos.empty or not needed.issubset(repair_oos.columns):
        return paths
    for split, g in repair_oos.groupby("split", observed=True):
        agg = g.groupby(["model_name", "baseline_policy_id"], observed=True).agg(
            switching_burden=("switching_burden", "mean"),
            utility_gain=("log_utility_improvement", "mean"),
        ).reset_index()
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        for model, h in agg.groupby("model_name", observed=True):
            ax.scatter(h["switching_burden"], h["utility_gain"], label=str(model))
            for r in h.itertuples(index=False):
                ax.annotate(str(r.baseline_policy_id), (r.switching_burden, r.utility_gain), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xlabel("Mean deployment / switching burden")
        ax.set_ylabel("Mean post-cost log-utility improvement")
        ax.set_title(f"Switching burden vs utility - {split}")
        ax.legend()
        paths.extend(_save_figure(fig, f"switching_burden_utility__{split}", figure_dir, write_pdf=write_pdf))
    return paths


def _strategy_id_for_repair(baseline: str, model: str, method: str) -> str:
    return f"continuous_repair__{baseline}__{model}__{method}"


def plot_continuous_relative_wealth(
    ledger: pd.DataFrame,
    *,
    baselines: Iterable[str],
    models: Iterable[str],
    lcb_method: str,
    figure_dir: Path,
    write_pdf: bool,
) -> list[Path]:
    paths: list[Path] = []
    needed = {"date", "asset_id", "split", "strategy_id", "wealth_close"}
    if ledger.empty or not needed.issubset(ledger.columns):
        return paths
    x = ledger.copy()
    x["date"] = pd.to_datetime(x["date"]).dt.normalize()
    for baseline in baselines:
        ref_sid = f"continuous_fixed__{baseline}"
        ref = x.loc[x["strategy_id"].astype(str) == ref_sid, ["date", "asset_id", "split", "wealth_close"]].rename(columns={"wealth_close": "reference"})
        if ref.empty:
            continue
        for model in models:
            sid = _strategy_id_for_repair(str(baseline), str(model), str(lcb_method))
            adp = x.loc[x["strategy_id"].astype(str) == sid, ["date", "asset_id", "split", "wealth_close"]].rename(columns={"wealth_close": "adaptive"})
            if adp.empty:
                continue
            m = ref.merge(adp, on=["date", "asset_id", "split"], how="inner")
            if m.empty:
                continue
            m = m.loc[(m["reference"] > 0) & (m["adaptive"] > 0)].copy()
            m["log_relative"] = np.log(m["adaptive"].to_numpy(dtype=float) / m["reference"].to_numpy(dtype=float))
            pooled = m.groupby(["split", "date"], observed=True)["log_relative"].mean().reset_index()
            pooled["relative_wealth"] = np.exp(pooled["log_relative"])
            fig, ax = plt.subplots(figsize=(8.0, 4.8))
            for split, g in pooled.groupby("split", observed=True):
                ax.plot(g["date"], g["relative_wealth"], label=str(split))
            ax.axhline(1.0, linewidth=1.0)
            ax.set_ylabel("Geometric-mean repair / baseline wealth")
            ax.set_title(f"Continuous relative wealth - {baseline} - {model}")
            ax.legend()
            paths.extend(_save_figure(fig, f"continuous_relative_wealth__{baseline}__{model}", figure_dir, write_pdf=write_pdf))
    return paths


def build_results_readme(
    *,
    selective: pd.DataFrame,
    meta: pd.DataFrame,
    synthetic_report: dict[str, Any],
    optional_inputs_present: dict[str, bool],
) -> str:
    s = add_support_flags(selective)
    m = add_support_flags(meta)
    n_dec_pos = int((s.get("decision_advantage_support") == "positive_95").sum()) if not s.empty else 0
    n_path_pos = int((s.get("path_support") == "positive_95").sum()) if not s.empty else 0
    n_path_neg = int((s.get("path_support") == "negative_95").sum()) if not s.empty else 0
    n_cont_pos = int((s.get("continuous_support") == "positive_95").sum()) if (not s.empty and "continuous_support" in s) else 0
    meta_cont_neg = int(((m["strategy_type"] == "meta_selector") & (m.get("continuous_support") == "negative_95")).sum()) if (not m.empty and "continuous_support" in m) else 0
    opt = "\n".join(f"- {k}: {'present' if v else 'not present; corresponding extended figure skipped'}" for k, v in optional_inputs_present.items())
    return f"""# Phase 12 Publication Bundle\n\nThis bundle is generated from frozen Phase-10/11 outcomes plus a preregistered synthetic sanity experiment. Phase 12 fits no new real-data trading model and does not tune any threshold.\n\n## Automatic evidence summary\n\n- Selective-repair decision-advantage groups with 95% CI entirely above zero: **{n_dec_pos}/{len(s)}**.\n- Selective-repair path-level utility groups with 95% CI entirely above zero: **{n_path_pos}/{len(s)}**.\n- Selective-repair path-level utility groups with 95% CI entirely below zero: **{n_path_neg}/{len(s)}**.\n- Continuous selective-repair groups with 95% utility CI entirely above zero: **{n_cont_pos}/{len(s)}**.\n- Continuous meta-selector groups with 95% utility CI entirely below zero: **{meta_cont_neg}**.\n\n## Synthetic sanity result\n\n- Recoverable-world high-signal deployed gain per decision: `{synthetic_report.get('recoverable_high_signal_gain', float('nan')):.6f}`.\n- Recoverable-world high-signal recovered fraction of hindsight-positive opportunity: `{synthetic_report.get('recoverable_high_signal_fraction', float('nan')):.4f}`.\n- Clairvoyance-only average hindsight-positive opportunity: `{synthetic_report.get('nonrecoverable_oracle_gain', float('nan')):.6f}`.\n- Clairvoyance-only maximum deployment coverage: `{synthetic_report.get('nonrecoverable_max_coverage', float('nan')):.4f}`.\n\nThese synthetic checks are leakage/identifiability sanity tests, not evidence about real-market profitability.\n\n## Extended frozen inputs\n\n{opt}\n\n## Interpretation guardrails\n\n1. Decision-level 60-day continuation advantages are overlapping observations; the block bootstrap is intended to preserve temporal dependence approximately.\n2. `bootstrap_prob_gt_zero` is a bootstrap sign probability, not a classical p-value.\n3. Continuous asset-OOS inference has few asset paths and can be economically large while statistically underpowered.\n4. The publication bundle reports negative controls and failures; it does not delete unfavorable policy/model combinations.\n5. The synthetic clairvoyance-only world must contain hindsight opportunities while the deployable gate largely abstains. If that condition fails, inspect leakage before interpreting real-data results.\n"""


def write_publication_manifest(bundle_dir: Path, *, input_files: Iterable[Path], metadata: dict[str, Any]) -> Path:
    files = sorted(p for p in bundle_dir.rglob("*") if p.is_file() and p.name != "publication_manifest.json")
    payload = {
        "schema_version": 1,
        **metadata,
        "inputs": [
            {"path": str(p), "sha256": sha256_file(p), "size_bytes": int(p.stat().st_size)}
            for p in input_files if p.exists()
        ],
        "artifacts": [
            {"path": str(p.relative_to(bundle_dir)), "sha256": sha256_file(p), "size_bytes": int(p.stat().st_size)}
            for p in files
        ],
    }
    path = bundle_dir / "publication_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def zip_publication_bundle(bundle_dir: Path, destination_without_suffix: Path) -> Path:
    if destination_without_suffix.with_suffix(".zip").exists():
        destination_without_suffix.with_suffix(".zip").unlink()
    made = shutil.make_archive(str(destination_without_suffix), "zip", root_dir=bundle_dir.parent, base_dir=bundle_dir.name)
    return Path(made)
