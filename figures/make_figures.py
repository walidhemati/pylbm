"""Make every figure and ``paper/results.tex`` from the JSON in ``results/``.

    python figures/make_figures.py

Benchmark data: ``results/rtx3070-laptop.json`` (the publishable ``--full`` run) if it
exists, otherwise the newest provisional run (``dev.json`` / ``quick.json``). Validation
data: ``results/validation.json`` (``pylbm validate``) and ``results/karman_vorticity.npz``.
Numbers reach the paper only through the macros written here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG, PAPER = ROOT / "results", ROOT / "figures", ROOT / "paper"
sys.path.insert(0, str(ROOT / "src"))

from pylbm.backends import COLORS  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})
COL, FULL = 3.4, 7.0
LABEL = {
    ("numpy", "roll"): "NumPy (fp64)",
    ("torch-eager", "roll"): "torch eager, roll",
    ("torch-eager", "gather"): "torch eager, gather",
    ("torch-compile", "original"): "original code (torch.compile)",
    ("torch-compile", "roll"): "torch.compile, roll",
    ("torch-compile", "gather"): "torch.compile, gather",
    ("numba", "fused"): "Numba-CUDA fused",
    ("warp", "fused"): "Warp fused",
    ("triton", "fused"): "Triton fused",
    ("cupy-cuda-c", "fused"): "CUDA C fused",
    ("triton", "fused-aa"): "Triton fused AA",
    ("cupy-cuda-c", "fused-aa"): "CUDA C fused AA",
    ("triton", "fused-fp16"): "Triton fused fp16",
    ("triton", "fused-bf16"): "Triton fused bf16",
    ("cupy-cuda-c", "fused-fp16"): "CUDA C fused fp16",
    ("triton", "fused-aa-fp16"): "Triton AA fp16",
    ("cupy-cuda-c", "fused-aa-fp16"): "CUDA C AA fp16",
}


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  figures/{name}.pdf/png")


def load_bench() -> tuple[dict, str]:
    for name in ("rtx3070-laptop.json", "dev.json", "quick.json"):
        p = RES / name
        if p.exists():
            return json.loads(p.read_text()), name
    raise SystemExit("no benchmark JSON in results/ (run bench/run.py)")


def runs_of(bench: dict, lattice: str, collision: str = "bgk") -> list[dict]:
    return [r for r in bench["runs"] if r["lattice"] == lattice and r["collision"] == collision]


def best_at(runs: list[dict], nodes: int | None = None) -> dict[tuple[str, str], dict]:
    """Per (backend, variant): the run at ``nodes`` (or the largest size measured)."""
    out: dict[tuple[str, str], dict] = {}
    for r in runs:
        key = (r["backend"], r["variant"])
        n = int(np.prod(r["size"]))
        if nodes is not None:
            if n == nodes:
                out[key] = r
        elif key not in out or n > np.prod(out[key]["size"]):
            out[key] = r
    return out


# ----------------------------------------------------------------------------- figures
def fig_ladder(bench: dict, copy_bw: float) -> dict:
    """Optimization ladder: MLUPS per rung at the reference size, with roofline ceilings."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    info = {}
    for ax, lattice, q in ((axes[0], "D2Q9", 9), (axes[1], "D3Q19", 19)):
        runs = runs_of(bench, lattice)
        if not runs:
            ax.set_visible(False)
            continue
        # reference size: largest size measured by the Triton fused kernel
        tri = [r for r in runs if r["backend"] == "triton" and r["variant"] == "fused"]
        ref_nodes = int(max(np.prod(r["size"]) for r in tri)) if tri else None
        sel = best_at(runs, ref_nodes)
        for key, r in best_at(runs).items():  # slow rungs: their largest size
            sel.setdefault(key, r)
        order = [k for k in LABEL if k in sel]
        vals = [sel[k]["median"] for k in order]
        y = np.arange(len(order))[::-1]
        ax.barh(y, vals, color=[COLORS[k[0]] for k in order],
                xerr=[[sel[k]["median"] - sel[k]["min"] for k in order],
                      [sel[k]["max"] - sel[k]["median"] for k in order]],
                error_kw={"lw": 0.6, "capsize": 1.5})
        labels = []
        for k in order:
            r = sel[k]
            sz = "x".join(str(s) for s in r["size"][:1]) + (f"$^{len(r['size'])}$")
            labels.append(f"{LABEL[k]} ({sz})")
        ax.set_yticks(y, labels, fontsize=7)
        ax.set_xscale("log")
        for bpu, ls in ((8 * q, "--"), (4 * q, ":")):
            ceil = copy_bw * 1e9 / bpu / 1e6
            ax.axvline(ceil, color="k", ls=ls, lw=0.9, zorder=0)
            ax.text(ceil, 1.0, "fp32" if bpu == 8 * q else "16-bit",
                    transform=ax.get_xaxis_transform(), fontsize=6.5,
                    ha="right" if bpu == 8 * q else "left", va="bottom")
        for yi, v in zip(y, vals, strict=True):
            ax.text(v * 1.12, yi, f"{v:,.0f}", va="center", fontsize=6,
                    bbox={"boxstyle": "square,pad=0.1", "fc": "white", "ec": "none",
                          "alpha": 0.85})
        ax.set_xlabel("MLUPS (log scale)")
        n_ref = ref_nodes or 0
        side = round(n_ref ** (1 / (2 if lattice == "D2Q9" else 3)))
        ax.set_title(f"{lattice} cavity (fused kernels at {side}$^{2 if q == 9 else 3}$)",
                     fontsize=8, pad=10)
        ax.set_xlim(left=max(1.0, 0.5 * min(vals)), right=4 * copy_bw * 1e9 / (4 * q) / 1e6)
        info[lattice] = sel
    fig.tight_layout()
    save(fig, "ladder")
    return info


def fig_scaling(bench: dict, copy_bw: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.8))
    for ax, lattice, q in ((axes[0], "D2Q9", 9), (axes[1], "D3Q19", 19)):
        runs = runs_of(bench, lattice)
        groups: dict[tuple[str, str], list[dict]] = {}
        for r in runs:
            groups.setdefault((r["backend"], r["variant"]), []).append(r)
        for key, rs in groups.items():
            if len(rs) < 2 or key[0] == "numpy":
                continue
            rs.sort(key=lambda r: np.prod(r["size"]))
            n = [np.prod(r["size"]) for r in rs]
            ls = ":" if "fp16" in key[1] or "bf16" in key[1] else (
                "--" if "aa" in key[1] else "-")
            ax.plot(n, [r["median"] for r in rs], ls, marker="o", ms=2.5, lw=1,
                    color=COLORS[key[0]], label=LABEL.get(key, f"{key[0]}/{key[1]}"))
        for bpu, ls in ((8 * q, "--"), (4 * q, ":")):
            ax.axhline(copy_bw * 1e9 / bpu / 1e6, color="k", ls=ls, lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("lattice nodes")
        ax.set_ylabel("MLUPS")
        ax.set_title(f"{lattice}", fontsize=8)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=6, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False)
    fig.tight_layout()
    save(fig, "scaling")


def fig_ghia(val: dict) -> None:
    from pylbm.validation.ghia import reference

    runs = [r for r in val.get("ghia", []) if r["collision"] == "bgk"
            and r["backend"] in ("triton/fused", "numpy/roll")]
    by_re: dict[int, dict] = {}
    for r in runs:  # finest grid per Re
        if r["re"] not in by_re or r["n"] > by_re[r["re"]]["n"]:
            by_re[r["re"]] = r
    if not by_re:
        return
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 3.0))
    cols = {100: "#1f77b4", 400: "#2ca02c", 1000: "#d62728"}
    for re, r in sorted(by_re.items()):
        ref = reference(re)
        pr = r["profile"]
        c = cols.get(re, "k")
        axes[0].plot(pr["u"], pr["y"], color=c, lw=1, label=f"Re={re} ({r['n']}$^2$)")
        axes[0].plot(ref["u"], ref["y"], "o", mfc="none", color=c, ms=3.5)
        axes[1].plot(pr["x"], pr["v"], color=c, lw=1)
        axes[1].plot(ref["x"], ref["v"], "o", mfc="none", color=c, ms=3.5)
    axes[0].set_xlabel("$u/U$ on $x=1/2$")
    axes[0].set_ylabel("$y$")
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$v/U$ on $y=1/2$")
    axes[0].plot([], [], "o", mfc="none", color="k", ms=3.5, label="Ghia et al. (1982)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "ghia")


def fig_viscosity(val: dict) -> None:
    sw = val.get("shear_wave", [])
    if not sw:
        return
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6))
    tau = np.array([r["tau"] for r in sw])
    axes[0].plot(tau, [r["nu_theory"] for r in sw], "k--", lw=0.8,
                 label=r"theory $(\tau-1/2)/3$")
    axes[0].plot(tau, [r["nu_measured"] for r in sw], "o", color=COLORS["triton"], ms=3.5,
                 label=f"measured ({sw[0]['backend']}, $n={sw[0]['n']}$)")
    axes[0].set_xlabel(r"$\tau$")
    axes[0].set_ylabel(r"$\nu$ (lattice units)")
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].semilogy(tau, [max(r["rel_error"], 1e-8) for r in sw], "o-", ms=3.5,
                     color=COLORS["triton"], lw=0.8)
    axes[1].set_xlabel(r"$\tau$")
    axes[1].set_ylabel(r"$|\nu_{\rm meas}-\nu|/\nu$")
    fig.tight_layout()
    save(fig, "viscosity")


def fig_poiseuille(val: dict) -> None:
    po = val.get("poiseuille")
    if not po:
        return
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6))
    for coll, mk in (("bgk", "o"), ("trt", "s")):
        if coll not in po:
            continue
        rs = po[coll]["runs"]
        h = np.array([r["h"] for r in rs])
        e = np.array([max(r["L2"], 1e-16) for r in rs])
        lab = coll.upper() + (f" (order {po[coll]['order']:.2f})" if coll == "bgk" else
                              ", magic $\\Lambda=3/16$")
        axes[0].loglog(h, e, mk + "-", ms=3.5, lw=0.8, label=lab,
                       color=COLORS["numpy"] if coll == "bgk" else COLORS["cupy-cuda-c"])
    h = np.array([8, 64])
    e0 = po["bgk"]["runs"][0]["L2"]
    axes[0].loglog(h, e0 * (h / 8.0) ** -2, "k:", lw=0.8, label="slope $-2$")
    axes[0].set_xlabel("channel width $H$ (nodes)")
    axes[0].set_ylabel("relative $L_2$ error")
    axes[0].legend(fontsize=7, frameon=False)
    r = po["bgk"]["runs"][1]
    y = np.linspace(0, 1, 200)
    axes[1].plot(4 * y * (1 - y), y, "k-", lw=0.8, label="analytic")
    axes[1].plot(r["u"], r["y"], "o", ms=3, color=COLORS["numpy"], label=f"BGK, $H={r['h']}$")
    axes[1].set_xlabel("$u/u_{\\max}$")
    axes[1].set_ylabel("$y/H$")
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "poiseuille")


def fig_karman(val: dict) -> None:
    k = val.get("karman")
    npz = RES / "karman_vorticity.npz"
    if not k or not npz.exists():
        return
    z = np.load(npz)
    w = z["vorticity"].astype(float)
    w[z["solid"]] = np.nan
    fig = plt.figure(figsize=(FULL, 3.2))
    ax = fig.add_axes([0.0, 0.42, 1.0, 0.58])
    ax.imshow(w, origin="lower", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="bilinear")
    ax.set_axis_off()
    ax2 = fig.add_axes([0.08, 0.08, 0.86, 0.28])
    t = np.array(k["t_conv"])
    ax2.plot(t, k["cl"], color=COLORS["torch-compile"], lw=0.8, label="$c_L$")
    ax2.plot(t, np.array(k["cd"]) - 3.0, color="k", lw=0.8, label="$c_D-3$")
    ax2.set_xlabel("$t\\,U/D$")
    ax2.legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
    save(fig, "karman")


# ----------------------------------------------------------------------------- results.tex
def _fmt(v: float, nd: int = 0) -> str:
    return f"{v:,.{nd}f}".replace(",", "{,}")


def write_results_tex(bench: dict, src: str, ladder: dict, val: dict) -> None:
    copy_bw = bench["ceiling"]["copy_bandwidth_GBs"]
    spec_bw = bench["ceiling"]["spec_bandwidth_GBs"]
    m: dict[str, str] = {}
    prov = bool(bench.get("provisional", True))
    m["benchSource"] = src.replace("_", r"\_")
    m["copyBW"] = _fmt(copy_bw, 0)
    m["specBW"] = _fmt(spec_bw, 0)
    m["gpuName"] = bench["machine"].get("gpu", "GPU")
    m["benchDate"] = bench["machine"].get("date", "")[:10]
    m["ceilFpThirtyTwo"] = _fmt(copy_bw * 1e9 / 72 / 1e6, 0)
    m["ceilSixteen"] = _fmt(copy_bw * 1e9 / 36 / 1e6, 0)
    m["ceilThreeD"] = _fmt(copy_bw * 1e9 / 152 / 1e6, 0)
    rows = []
    for lattice, sel in ladder.items():
        for key in LABEL:
            if key not in sel:
                continue
            r = sel[key]
            size = "$" + r"\times".join(str(s) for s in r["size"]) + "$"
            rows.append(f"{lattice} & {LABEL[key]} & {size} & {_fmt(r['median'], 0)} & "
                        f"{r['bytes_per_unit']} & {_fmt(r['effective_GBs'], 0)} & "
                        f"{r['pct_copy_bw']:.0f} & {r['pct_spec_bw']:.0f} \\\\")
        rows.append(r"\midrule")
    m["ladderRows"] = "\n".join(rows[:-1])
    d2 = ladder.get("D2Q9", {})

    def best(pred: object) -> dict | None:
        c = [r for k, r in d2.items() if pred(k)]  # type: ignore[operator]
        return max(c, key=lambda r: r["median"]) if c else None

    b32 = best(lambda k: "16" not in k[1] and k[0] not in ("numpy",))
    b16 = best(lambda k: "16" in k[1])
    if b32:
        m["bestFpThirtyTwoMLUPS"] = _fmt(b32["median"])
        m["bestFpThirtyTwoName"] = LABEL[(b32["backend"], b32["variant"])]
        m["bestFpThirtyTwoPct"] = f"{b32['pct_copy_bw']:.0f}"
        m["bestFpThirtyTwoGBs"] = _fmt(b32["effective_GBs"])
    if b16:
        m["bestSixteenMLUPS"] = _fmt(b16["median"])
        m["bestSixteenName"] = LABEL[(b16["backend"], b16["variant"])]
        m["bestSixteenPct"] = f"{b16['pct_copy_bw']:.0f}"
    allbest = max(d2.values(), key=lambda r: r["median"]) if d2 else None
    if allbest:
        m["bestMLUPS"] = _fmt(allbest["median"])
        m["bestName"] = LABEL[(allbest["backend"], allbest["variant"])]
        m["bestPctCopy"] = f"{allbest['pct_copy_bw']:.0f}"
        m["bestPctSpec"] = f"{allbest['pct_spec_bw']:.0f}"
        m["bestGBs"] = _fmt(allbest["effective_GBs"])
    orig = d2.get(("torch-compile", "original"))
    if orig and b32:
        m["origMLUPS"] = _fmt(orig["median"])
        m["origPct"] = f"{orig['pct_copy_bw']:.1f}"
        m["speedupFpThirtyTwo"] = f"{b32['median'] / orig['median']:.1f}"
        if allbest:
            m["speedupBest"] = f"{allbest['median'] / orig['median']:.1f}"
    for key, name in ((("torch-compile", "roll"), "compileRoll"),
                      (("torch-eager", "roll"), "eagerRoll"), (("numpy", "roll"), "numpyRef"),
                      (("triton", "fused"), "tritonFused"), (("warp", "fused"), "warpFused"),
                      (("numba", "fused"), "numbaFused"), (("cupy-cuda-c", "fused"), "cudaFused")):
        if key in d2:
            m[name + "MLUPS"] = _fmt(d2[key]["median"])
            m[name + "Pct"] = f"{d2[key]['pct_copy_bw']:.0f}"
    d3 = ladder.get("D3Q19", {})
    if d3:
        b3 = max((r for k, r in d3.items() if "16" not in k[1] and k[0] != "numpy"),
                 key=lambda r: r["median"])
        m["bestThreeDMLUPS"] = _fmt(b3["median"])
        m["bestThreeDName"] = LABEL[(b3["backend"], b3["variant"])]
        m["bestThreeDPct"] = f"{b3['pct_copy_bw']:.0f}"
        b3h = [r for k, r in d3.items() if "16" in k[1]]
        if b3h:
            b3h = max(b3h, key=lambda r: r["median"])
            m["bestThreeDSixteenMLUPS"] = _fmt(b3h["median"])
            m["bestThreeDSixteenPct"] = f"{b3h['pct_copy_bw']:.0f}"
    colls = [r for r in bench["runs"] if r["collision"] != "bgk" and r["backend"] == "triton"
             and r["variant"] == "fused"]
    for r in colls:
        m[f"triton{r['collision'].upper()}Pct"] = f"{r['pct_copy_bw']:.0f}"
        m[f"triton{r['collision'].upper()}MLUPS"] = _fmt(r["median"])
    # ---------------------------------------------------------------- validation
    for r in val.get("ghia", []):
        tag = {"triton/fused": "", "triton/fused-fp16": "Hal", "triton/fused-bf16": "Bf",
               "torch-compile/original": "Orig", "triton/fused-aa": "AA",
               "cupy-cuda-c/fused-aa-fp16": "CudaAAHal"}.get(r["backend"])
        if tag is None:
            continue
        coll = "" if r["collision"] == "bgk" else r["collision"].upper()
        rn = {100: "One", 400: "Four", 1000: "Thou"}[r["re"]]
        ntag = {64: "Sixtyfour", 128: "", 256: "", 512: "Fivetwelve"}.get(r["n"], "")
        if r["re"] == 100 and r["n"] == 256 and tag == "":
            ntag = "Twofiftysix"
        name = f"ghia{rn}{coll}{tag}{ntag}"
        m[name + "Ltwo"] = f"{r['errors']['L2']:.4f}"
        m[name + "Linf"] = f"{r['errors']['Linf']:.4f}"
        m[name + "N"] = str(r["n"])
    sw = val.get("shear_wave", [])
    if sw:
        m["shearMaxErr"] = f"{100 * max(r['rel_error'] for r in sw):.2f}"
        m["shearN"] = str(sw[0]["n"])
        m["shearTauMin"] = f"{min(r['tau'] for r in sw):.2f}"
        m["shearTauMax"] = f"{max(r['tau'] for r in sw):.2f}"
        mid = [r for r in sw if 0.6 <= r["tau"] <= 1.6]
        m["shearMidErr"] = f"{100 * max(r['rel_error'] for r in mid):.2f}" if mid else "--"
    po = val.get("poiseuille", {})
    if "bgk" in po:
        m["poisOrder"] = f"{po['bgk']['order']:.3f}"
        m["poisBgkFinest"] = f"{po['bgk']['runs'][-1]['L2']:.1e}"
    if "trt" in po:
        m["poisTrtMax"] = f"{max(r['L2'] for r in po['trt']['runs']):.0e}"
    if "bgk_fp32_h32" in po:
        m["poisFpThirtyTwo"] = f"{po['bgk_fp32_h32']['L2']:.1e}"
        h32 = [r for r in po.get("bgk", {}).get("runs", []) if r["h"] == 32]
        if h32:
            m["poisFpSixtyFourHThirtyTwo"] = f"{h32[0]['L2']:.1e}"
    k = val.get("karman")
    if k:
        m["karmanSt"] = f"{k['St']:.4f}"
        m["karmanCd"] = f"{k['cd_max']:.3f}"
        m["karmanCl"] = f"{k['cl_max']:.3f}"
        m["karmanD"] = str(k["d"])
        m["karmanTau"] = f"{k['tau']:.3f}"
        m["karmanNx"], m["karmanNy"] = (str(v) for v in k["shape"])
    tg = val.get("taylor_green_3d", [])
    if tg:
        gpu = [r for r in tg if r["backend"] != "numpy/roll"] or tg
        m["tgvMaxErr"] = f"{100 * max(r['rel_error'] for r in gpu):.2f}"
        m["tgvN"] = str(gpu[0]["n"])
    eq = val.get("equivalence", [])
    if eq:
        f32 = [r["max_abs_diff"] for r in eq if r["storage"] == "f32"]
        f16 = [r["max_abs_diff"] for r in eq if r["storage"] == "f16"]
        bf = [r["max_abs_diff"] for r in eq if r["storage"] == "bf16"]
        m["eqMaxFpThirtyTwo"] = f"{max(f32):.1e}" if f32 else "--"
        m["eqMaxFpSixteen"] = f"{max(f16):.1e}" if f16 else "--"
        m["eqMaxBf"] = f"{max(bf):.1e}" if bf else "--"
        m["eqCount"] = str(len({(r['backend'], r['variant']) for r in eq}))
    lines = ["% GENERATED by figures/make_figures.py -- do not edit by hand.",
             f"% benchmark source: results/{src}; validation: results/validation.json",
             r"\newif\ifprovisional",
             r"\provisionaltrue" if prov else r"\provisionalfalse"]
    for key, val_ in sorted(m.items()):
        lines.append(f"\\newcommand{{\\{key}}}{{{val_}}}")
    (PAPER / "results.tex").write_text("\n".join(lines) + "\n")
    print(f"  paper/results.tex ({len(m)} macros, provisional={prov})")


def _replace_block(text: str, tag: str, body: str) -> str:
    start = f"<!-- BEGIN:{tag} (generated by figures/make_figures.py) -->"
    end = f"<!-- END:{tag} -->"
    i, j = text.index(start), text.index(end)
    return text[: i + len(start)] + "\n" + body.strip() + "\n" + text[j:]


def write_readme(bench: dict, src: str, ladder: dict, val: dict) -> None:
    """Fill the generated README blocks from the JSON (headline, validation, performance)."""
    readme = ROOT / "README.md"
    text = readme.read_text()
    copy_bw = bench["ceiling"]["copy_bandwidth_GBs"]
    gpu = bench["machine"].get("gpu", "GPU")
    prov = bench.get("provisional", True)
    d2 = ladder.get("D2Q9", {})
    d3 = ladder.get("D3Q19", {})
    head = []
    if d2:
        f32 = max((r for k, r in d2.items() if "16" not in k[1] and k[0] != "numpy"),
                  key=lambda r: r["median"])
        f16 = [r for k, r in d2.items() if "16" in k[1]]
        orig = d2.get(("torch-compile", "original"))
        lines = [f"> **Headline ({gpu}{', PROVISIONAL dev run' if prov else ''}).** "
                 f"D2Q9 fp32: **{f32['median']:,.0f} MLUPS** "
                 f"({LABEL[(f32['backend'], f32['variant'])]}, "
                 f"{'x'.join(map(str, f32['size']))}) = "
                 f"**{f32['pct_copy_bw']:.0f}% of the measured {copy_bw:.0f} GB/s copy "
                 f"bandwidth**, {f32['pct_spec_bw']:.0f}% of the 448 GB/s spec."]
        if f16:
            b = max(f16, key=lambda r: r["median"])
            lines.append(f"> 16-bit storage: **{b['median']:,.0f} MLUPS** "
                         f"({LABEL[(b['backend'], b['variant'])]}, "
                         f"{b['pct_copy_bw']:.0f}% of copy bandwidth at 36 B/update).")
        if d3:
            b3 = max((r for k, r in d3.items() if "16" not in k[1] and k[0] != "numpy"),
                     key=lambda r: r["median"])
            lines.append(f"> D3Q19 fp32: **{b3['median']:,.0f} MLUPS** "
                         f"({b3['pct_copy_bw']:.0f}% of copy bandwidth).")
        if orig:
            lines.append(f"> The original script of this repository: {orig['median']:,.0f} "
                         f"MLUPS ({f32['median'] / orig['median']:.1f}x slower than fp32 "
                         f"fused).")
        lines.append(f"> Source: `results/{src}`.")
        head = lines
    text = _replace_block(text, "headline", "\n>\n".join(head))
    rows = ["| lattice | rung | size | MLUPS | B/update | GB/s | % copy | % spec |",
            "|---|---|---|---:|---:|---:|---:|---:|"]
    for lattice, sel in ladder.items():
        for key in LABEL:
            if key in sel:
                r = sel[key]
                rows.append(f"| {lattice} | {LABEL[key]} | {'x'.join(map(str, r['size']))} | "
                            f"{r['median']:,.0f} | {r['bytes_per_unit']} | "
                            f"{r['effective_GBs']:.0f} | {r['pct_copy_bw']:.0f} | "
                            f"{r['pct_spec_bw']:.0f} |")
    perf = (f"Median of 5 timed runs on the {gpu}; ceiling = measured device copy bandwidth "
            f"({copy_bw:.0f} GB/s). " + ("**Provisional numbers from a development run on a "
                                          "shared GPU.**" if prov else "") + "\n\n" +
            "\n".join(rows))
    text = _replace_block(text, "performance", perf)
    vrows = ["| check | reference | result |", "|---|---|---|"]
    for r in val.get("ghia", []):
        if r["backend"] == "triton/fused" and r["collision"] == "bgk" and (
                (r["re"], r["n"]) in ((100, 128), (400, 256), (1000, 256))):
            vrows.append(f"| cavity Re={r['re']}, {r['n']}² | Ghia et al. 1982 centrelines | "
                         f"L2 = {r['errors']['L2']:.4f}, Linf = {r['errors']['Linf']:.4f} "
                         f"(units of U) |")
    for r in val.get("ghia", []):
        if r["backend"] in ("triton/fused-fp16", "triton/fused-bf16",
                            "torch-compile/original"):
            vrows.append(f"| cavity Re={r['re']}, {r['n']}², {r['backend']} | Ghia et al. "
                         f"1982 | Linf = {r['errors']['Linf']:.4f} |")
    sw = val.get("shear_wave", [])
    if sw:
        vrows.append(f"| shear-wave decay, tau {min(r['tau'] for r in sw)}-"
                     f"{max(r['tau'] for r in sw)} | nu = (tau - 1/2)/3 | max rel. error "
                     f"{100 * max(r['rel_error'] for r in sw):.2f}% |")
    po = val.get("poiseuille", {})
    if "bgk" in po:
        vrows.append(f"| Poiseuille, H = 8..64 (float64) | analytic parabola | BGK order "
                     f"{po['bgk']['order']:.3f}; TRT max L2 "
                     f"{max(r['L2'] for r in po['trt']['runs']):.0e} |")
    k = val.get("karman")
    if k:
        vrows.append(f"| cylinder 2D-2, D = {k['d']} | Schäfer–Turek 1996: St 0.295-0.305, "
                     f"cD 3.22-3.24, cL 0.99-1.01 | St = {k['St']:.4f}, cD,max = "
                     f"{k['cd_max']:.3f}, cL,max = {k['cl_max']:.3f} |")
    tg = [r for r in val.get("taylor_green_3d", []) if r["backend"] != "numpy/roll"]
    if tg:
        vrows.append(f"| D3Q19 Taylor–Green, {tg[0]['n']}³, 3 planes | nu = (tau - 1/2)/3 | "
                     f"max rel. error {100 * max(r['rel_error'] for r in tg):.2f}% |")
    text = _replace_block(text, "validation", "\n".join(vrows))
    readme.write_text(text)
    print("  README.md generated blocks")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    bench, src = load_bench()
    val_path = RES / "validation.json"
    val = json.loads(val_path.read_text()) if val_path.exists() else {}
    copy_bw = bench["ceiling"]["copy_bandwidth_GBs"]
    print(f"[make_figures] bench={src} (provisional={bench.get('provisional')})")
    ladder = fig_ladder(bench, copy_bw)
    fig_scaling(bench, copy_bw)
    fig_ghia(val)
    fig_viscosity(val)
    fig_poiseuille(val)
    fig_karman(val)
    write_results_tex(bench, src, ladder, val)
    write_readme(bench, src, ladder, val)


if __name__ == "__main__":
    main()
