#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
ALL-TO-ALL SWEEP -- 6, 8, 10 qubits  x  L = 6, 8
================================================
    python -m iontraps_qrc.sweep                    # all six cells
    python -m iontraps_qrc.sweep --dry-run          # budgets only
    python -m iontraps_qrc.sweep --quick            # smoke test
    python -m iontraps_qrc.sweep --only 8q_a2a_L8   # one cell

    # THE EXPERIMENT TO RUN FIRST (set BACKEND = "exact" below): the
    # shot-noise-free ceiling. Everything measured so far is an exact
    # statevector WITH sampled shots, i.e.
    #     real hardware <= numpy backend <= exact backend
    # Improving hardware moves the left term toward the middle one and can
    # never pass it, so if the exact ceiling also loses to the classical
    # baseline then no shot count and no hardware can win on this dataset.
    # Costs 4^N memory: fine at 6-10 qubits, seconds to ~20 minutes.

WHAT THIS SWEEP IS FOR
----------------------
Two axes, six cells, everything else held fixed:

                 L = 6                L = 8
        6q    3 in + 3 mem         3 in + 3 mem
        8q    4 in + 4 mem         4 in + 4 mem
       10q    5 in + 5 mem         5 in + 5 mem

All-to-all coupling throughout, which is the one thing a trapped ion gives
us for free: every pair is a direct interaction, no SWAP network. The
question this sweep answers is whether spending that freedom on MORE QUBITS
or on MORE TIMESTEPS buys more, and whether either one buys anything at all.

The qubit split is held at half input / half memory so that N is the only
thing changing down a column. Do not "improve" one cell's split without
rerunning the whole column -- the comparison is the point.

THE COUPLING NORMALISATION -- read this before changing it
----------------------------------------------------------
All-to-all means qubit i feels a local field  sum_j J_ij Z_j  built from
N-1 random terms. For zero-mean J that sum has standard deviation growing
like sqrt(N-1). So if j_scale were held at 1.0, going from 6 to 10 qubits
would increase the total coupling each qubit feels by sqrt(9/5) = 1.34x,
and the sweep would be confounding "more qubits" with "more mixing" --
two different experiments reported as one.

COUPLING_NORM = "sk" (default) therefore sets

        j_scale = sqrt(7 / (N - 1))

which holds the local-field variance constant (Sherrington-Kirkpatrick
scaling, the standard choice for random all-to-all couplings). It is
anchored at N = 8 so that the 8-qubit cells keep j_scale = 1.0 exactly and
stay directly comparable to the completed run of 2026-10-10.

        6q  -> j_scale 1.183        8q -> 1.000        10q -> 0.882

COUPLING_NORM = "none" keeps j_scale = 1.0 everywhere. That is a legitimate
experiment -- "what does a bigger, more strongly mixed reservoir do" -- but
it is NOT a clean qubit-count sweep, and the write-up has to say which one
was run. Both are worth having; run "sk" first.

WHAT CHANGED SINCE THE 2026-10-10 RUNS, AND WHY
-----------------------------------------------
Every default here exists because a measurement on those runs demanded it.

  1. BALANCED SAMPLING.  Proportional sampling on a 95.5%-benign dataset
     gave 3 test positives at n_test=100 and 1 at n_test=20. An AUC with
     one positive row is one pairwise comparison; supervised_classical=1.0
     on the 20-qubit run was a single row that happened to rank first.
     ROC-AUC is a rank statistic and is not biased by the evaluation ratio,
     so balancing costs nothing and removes the variance.

  2. NO CONSTANT PADDING.  NF-UNSW-NB15 has ~24 features. The 8q run asked
     for 64 encoding slots and filled 40 of them (62%) with the constant
     0.5; the 20q run filled 237 of 256 (93%). Those gates evolved a fixed
     input. pad_mode="reupload" tiles the real features instead, with a
     frozen per-repetition permutation.

  3. CORRELATORS OFF.  Same-step <Z_i Z_j> needs ~10,000 shots to rise
     above its own sampling error; at 100 shots all 48 of them were pure
     noise. Dropping them moved the 8q supervised AUC 0.9519 -> 0.9897,
     past the 0.9725 classical baseline.

  4. 512 SHOTS, not 100.  Only 8 of 88 features cleared the shot-noise
     floor at 100 shots. The mid-circuit channels want ~756 for a 3x
     margin; 512 is the knee of the cost curve.

  5. PCA FITTED BEFORE SUBSAMPLING.  The old code capped PCA at
     len(train_subsample) - 1, so the 20-qubit run at n_train=20 got 19
     components. Still train-only, so nothing leaks.

  6. final_mix IS NOW FIELD -> ZZ -> FIELD.  A freshly reset input qubit is
     in a Z eigenstate, which a diagonal ZZ layer cannot move, so the old
     ZZ -> field ordering returned a constant. Measured: 0 of 8 and 0 of 20
     final-snapshot columns cleared the noise floor.

RUNTIME, MEASURED ON THIS MACHINE (512 shots, 850 samples per cell)
-------------------------------------------------------------------
        6q  L=6   0.8 min        6q  L=8   0.7 min
        8q  L=6   5.3 min        8q  L=8   6.8 min
       10q  L=6  28.7 min       10q  L=8  36.5 min
                                    total ~79 min
10 qubits is 1024 amplitudes and still exact -- there is no approximation
anywhere in the numpy backend. Use --quick (about 2 min total) to check the
wiring before committing to the full run.

HARDWARE COST, FOR REFERENCE
----------------------------
At 512 shots the 10q L=8 cell is roughly 440 HQC per sample, so 850 samples
would be ~374,000 HQC. Nothing in this sweep is affordable on real hardware
at a shot count that makes the features meaningful. That is a genuine
finding and belongs in the write-up: the honest plan is simulator/emulator
for the headline number, plus a small reduced-shot hardware run as a
confirmation that the circuit executes as compiled.
"""

import json
import sys
from pathlib import Path


# ---- point this at the parquet if it is not beside the core module -------
DATA_PATH = None          # e.g. r"D:\claude things\QRC\data\NF-UNSW-NB15.parquet"

from iontraps_qrc import core
from iontraps_qrc.core import ReservoirSpec, run_experiment, budget_report

if DATA_PATH:
    core.NFUNSW_PARQUET_CANDIDATES.insert(0, str(DATA_PATH))

# ----------------------------------------------------------------------
# The grid
# ----------------------------------------------------------------------
QUBITS = [(6, 3, 3), (8, 4, 4), (10, 5, 5)]      # (N, n_in, n_mem)
L_VALUES = [6, 8]
COUPLING_NORM = "sk"      # "sk" (default, see docstring) | "none"

# dt is now a real axis. It had been frozen at 0.3 since the start of the
# project without ever being scanned, which made it the biggest untested
# assumption in the physics: dt multiplies BOTH J and h in to_halfturns(),
# so it is a uniform time dilation of the whole reservoir -- the single
# knob that sets how far the state moves per step.
#
#   dt -> 0      exp(-i H dt) ~ I. The encoding writes a phase, the field
#                layer has no time to turn it into population, every <Z>
#                stays pinned near its initial value and the signal
#                collapses BELOW THE SHOT FLOOR. Looks like a shot problem.
#   dt optimal   enough rotation to be measurable, enough ZZ phase to
#                correlate qubits, but each observable still depends on the
#                inputs in a recoverable LOW-ORDER way. Edge of chaos.
#   dt -> large  every observable becomes a high-order near-random function
#                of all inputs. Plenty of variance, so it sails through the
#                noise filter, but a LINEAR readout cannot decode it. Looks
#                like a model-capacity problem.
#
# Both failure modes masquerade as something else, which is exactly why it
# has to be measured rather than reasoned about.
#
# At the current operating point (dt=0.3, j_scale/h_scale=1, annulus):
#     ZZ conditional phase  dt*j_scale*sqrt((N-1)/3) = 1.53*dt = 0.46 rad/step
#     field rotation        2*|h|*dt                           = 0.18-0.60 rad/step
# The two channels are matched in magnitude and accumulate of order one
# full turn over the circuit -- a plausible regime, never verified.
#
# EXPECT A PEAK, NOT A LINE. A peak is the better result: it means the
# operating point is located and both failure modes are explainable from
# the Hamiltonian. It also predicts that the optimum dt should shift with L
# so as to hold L*dt roughly constant, which is a testable scaling claim.
DT_VALUES = [0.3]         # e.g. [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]

# The input gain, formerly a hardcoded 3.0 inside angles_from_features.
# SMALLER = bigger angle excursions = more signal per shot, until tanh
# saturates. Roughly 4x cheaper than shots per unit of SNR, and free.
IN_GAIN_VALUES = [3.0]    # e.g. [1.0, 1.5, 2.0, 3.0, 4.5]

# Final-snapshot readout basis. "mixed" measures a frozen per-qubit choice
# of {Z,X,Y} and costs one 1q gate per rotated qubit -- no 2q gates, no
# extra shots. Measured on a 4-qubit test reservoir, std of the final
# snapshot of the two INPUT qubits (the columns that have been dead in
# every run of this project): 0.0011 -> 0.0516 and 0.0003 -> 0.0116, i.e.
# 47x and 39x. The reason is first-order sensitivity: a qubit reset to |0>
# and nudged by a small angle theta has <Z> ~ 1 - theta^2/2 (SECOND order)
# but <X> ~ theta (FIRST order). Z is the worst possible basis for reading
# a small excursion from |0>, which is precisely what the final snapshot of
# a just-reset input qubit is.
FINAL_BASIS = "z"         # "z" (historical) | "x" | "y" | "mixed"

# ----------------------------------------------------------------------
# Shared settings -- identical for every cell, or the sweep means nothing
# ----------------------------------------------------------------------
# "numpy" = exact statevector + shot sampling (what every result so far
# used). "exact" = density matrix, shot noise removed entirely: the ceiling
# that perfect hardware at infinite shots would reach. Run "exact" at 6q/8q
# before spending time on anything else -- if the ceiling also loses to the
# classical baseline then no shot count and no hardware can win on this
# dataset, and if it is far above then the architecture is shot-limited and
# the shot-scaling curve is a real claim.
BACKEND = "numpy"         # "numpy" | "exact" | "syntax" | "nexus"
SAVE_BITS = False         # keep the raw shot table (enables a shot-scaling
                          # curve by subsampling shots from ONE run)
DEVICE = "H1-1E"
CONFIRM_HARDWARE = False

N_SHOTS = 512
N_TRAIN = 400
N_TEST = 300              # balanced -> 150 benign / 150 attack
N_ZERODAY = 150

DT = DT_VALUES[0]         # default cell; the axis itself is DT_VALUES
SEED = 1234               # THE reservoir seed. Same for every cell.

# THE DATA SPLIT SEED, which is a different seed from the reservoir one and
# had never been exposed. run_experiment(seed=...) controls train_test_split,
# the stratified subsample, the zero-day draw and the readout's CV folds;
# SEED above controls only the J/h/mem_init draw. The sweep never passed
# seed, so every cell and every run so far used the SAME 150/150 split --
# which is why supervised_classical is the identical 0.9531 in all six
# cells and zeroday_classical the identical 0.8009. Six cells against one
# baseline number is ONE comparison repeated six times.
#
# That matters because every difference this project has measured is
# smaller than the Hanley-McNeil marginal standard error (+-0.013
# supervised, +-0.026 zero-day). A single split cannot tell +0.002 from 0.
# Listing several seeds here re-draws the split and gives the SPREAD of the
# difference, which is the honest error bar and the one a reviewer will
# ask for. Cost is linear: the 6q_a2a_L6 cell is ~0.8 min, so five seeds of
# that one cell is ~4 min.
#
# Keep the DEFAULT at [42] -- that is the split every existing number was
# measured on, so a single-seed run still reproduces them exactly.
SPLIT_SEEDS = [42]        # e.g. [42, 43, 44, 45, 46]
SEQUENCE_MODE = "features"
SLICE_ORDER = "interleave"
CORRELATORS = False       # see note 3
BALANCE = "balanced"      # see note 1
PAD_MODE = "reupload"     # see note 2
DROP_NOISE_FLOOR = True
ZERO_DAY = "Exploits"
LOG1P_FIX = True
DATASET = "nfunsw"
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "sweep_alltoall"


def j_scale_for(n: int) -> float:
    if COUPLING_NORM == "sk":
        return (7.0 / (n - 1)) ** 0.5          # anchored at N = 8 -> 1.0
    if COUPLING_NORM == "none":
        return 1.0
    raise ValueError(f"unknown COUPLING_NORM {COUPLING_NORM!r}")


def make_spec(n, n_in, n_mem, L, dt=None, in_gain=None,
              split_seed=None) -> ReservoirSpec:
    dt = DT if dt is None else dt
    in_gain = IN_GAIN_VALUES[0] if in_gain is None else in_gain
    # The tag only grows extra fields when the corresponding axis actually
    # has more than one value, so the single-value defaults keep the tags
    # (and therefore the frozen reservoir files) of the 2026-10-11 sweep.
    tag = f"{n}q_a2a_L{L}"
    if len(DT_VALUES) > 1:
        tag += f"_dt{dt:g}"
    if len(IN_GAIN_VALUES) > 1:
        tag += f"_g{in_gain:g}"
    if FINAL_BASIS != "z":
        tag += f"_{FINAL_BASIS}"
    # The split seed has to reach the tag or two seeds would overwrite each
    # other's features_/results_ files. It does NOT change the reservoir
    # draw (that is SEED), so the extra reservoir_*.npz files all hold the
    # same J/h -- harmless, and freeze()'s drift check confirms it.
    if split_seed is not None and len(SPLIT_SEEDS) > 1:
        tag += f"_s{split_seed}"
    return ReservoirSpec(
        tag=tag,
        n_in=n_in,
        n_mem=n_mem,
        n_steps=L,
        dt=dt,
        topology="all_to_all",
        seed=SEED,
        j_scale=j_scale_for(n),
        h_scale=1.0,
        in_gain=in_gain,
        final_basis=FINAL_BASIS,
        in_sites=None,          # all-to-all has no geometry; placement is moot
    )


def grid():
    """Every cell, as (spec, split_seed).

    dt and in_gain vary INSIDE one reservoir draw, because neither touches
    J/h/mem_init -- that is what isolates them. split_seed varies the DATA
    and leaves the reservoir alone, which is the orthogonal axis.
    """
    for (n, n_in, n_mem) in QUBITS:
        for L in L_VALUES:
            for dt in DT_VALUES:
                for g in IN_GAIN_VALUES:
                    for s in SPLIT_SEEDS:
                        yield make_spec(n, n_in, n_mem, L, dt, g, s), s


# ----------------------------------------------------------------------
def dry_run():
    """Budgets and gate counts for all six cells. Touches no data."""
    print(__doc__.split("WHAT THIS SWEEP IS FOR")[0].strip())
    print(f"\nCOUPLING_NORM = {COUPLING_NORM!r}, DT = {DT}, "
          f"N_SHOTS = {N_SHOTS}\n")
    rows = []
    seen = set()
    for spec, _ in grid():
        if spec.tag in seen:          # same circuit, different split
            continue
        seen.add(spec.tag)
        # deliberately NOT frozen: circuit_budget needs only the shape of
        # the register, and a dry run must not write reservoir files
        b = core.circuit_budget(spec)
        rows.append((spec, b))
        print(f"{spec.tag:<12} j_scale={spec.j_scale:.3f}  "
              f"edges={len(spec.edges):>2}  2q={b.n_2q:>4}  1q={b.n_1q:>4}  "
              f"meas={b.n_meas:>3}  depth2q={b.depth_2q:>3}  "
              f"survival={b.survival():6.1%}  slots={spec.n_features_ingested:>3}  "
              f"readout={core.n_readout_features(spec, CORRELATORS):>3}  "
              f"HQC/sample@{N_SHOTS}={b.hqc(N_SHOTS):>7.1f}")
    print(f"\n{'':<12} total HQC for {N_TRAIN + N_TEST + N_ZERODAY} samples "
          f"per cell, all six cells: "
          f"{sum(b.hqc(N_SHOTS) for _, b in rows) * (N_TRAIN + N_TEST + N_ZERODAY):,.0f}")
    print("\nNothing in that column is affordable on hardware. Run the sweep "
          "on BACKEND='numpy', then use BACKEND='syntax' to prove the "
          "circuits compile for the real target at zero HQC.")


def run_all(quick=False, only=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_shots = 64 if quick else N_SHOTS
    n_train = 60 if quick else N_TRAIN
    n_test = 40 if quick else N_TEST
    n_zd = 20 if quick else N_ZERODAY
    if quick:
        print("!! --quick: tiny sizes and 64 shots. This checks the WIRING.\n"
              "!! Every metric it prints is below the noise floor and below\n"
              "!! the minimum test-positive count. Do not quote any of it.\n")

    summary = []
    for spec, split_seed in grid():
        if only and spec.tag != only:
            continue
        print("\n" + "#" * 72)
        print(f"#  {spec.tag}   {spec.n_qubits} qubits "
              f"({spec.n_in} in + {spec.n_mem} mem), all-to-all, "
              f"L={spec.n_steps}, j_scale={spec.j_scale:.3f}")
        print("#" * 72)
        res = run_experiment(
            spec,
            backend=BACKEND,
            seed=split_seed,
            n_shots=n_shots,
            save_bits=SAVE_BITS,
            n_train=n_train,
            n_test=n_test,
            n_zeroday=n_zd,
            device=DEVICE,
            confirm_hardware=CONFIRM_HARDWARE,
            sequence_mode=SEQUENCE_MODE,
            slice_order=SLICE_ORDER,
            correlators=CORRELATORS,
            balance=BALANCE,
            pad_mode=PAD_MODE,
            drop_noise_floor=DROP_NOISE_FLOOR,
            zero_day=ZERO_DAY,
            log1p_fix=LOG1P_FIX,
            dataset=DATASET,
            quiet_sim=False,
            out_dir=OUT_DIR,
        )
        if res is None:                      # syntax-check backend
            continue
        b = core.circuit_budget(spec)
        summary.append({
            "tag": spec.tag, "n_qubits": spec.n_qubits, "L": spec.n_steps,
            "n_in": spec.n_in, "j_scale": spec.j_scale,
            "dt": spec.dt, "in_gain": spec.in_gain,
            "final_basis": spec.final_basis, "split_seed": split_seed,
            "slots": spec.n_features_ingested,
            "n_2q": b.n_2q, "depth_2q": b.depth_2q,
            "survival": b.survival(), "hqc_per_sample": b.hqc(n_shots),
            **{k: v for k, v in res.items() if k != "diagnostics"},
            **{f"diag_{k}": v for k, v in res["diagnostics"].items()},
        })
        _write_summary(summary, n_shots, quick)
    _print_table(summary)
    return summary


def _write_summary(summary, n_shots, quick):
    meta = {"coupling_norm": COUPLING_NORM, "seed": SEED,
            "split_seeds": SPLIT_SEEDS,
            "dt_values": DT_VALUES, "in_gain_values": IN_GAIN_VALUES,
            "final_basis": FINAL_BASIS,
            "n_shots": (None if BACKEND == "exact" else n_shots),
            "correlators": CORRELATORS,
            "balance": BALANCE, "pad_mode": PAD_MODE,
            "drop_noise_floor": DROP_NOISE_FLOOR, "zero_day": ZERO_DAY,
            "quick": quick, "backend": BACKEND}
    (OUT_DIR / "sweep_summary.json").write_text(
        json.dumps({"meta": meta, "cells": summary}, indent=2, default=float))
    if summary:
        keys = list(summary[0].keys())
        lines = [",".join(keys)]
        for r in summary:
            lines.append(",".join(str(r.get(k, "")) for k in keys))
        (OUT_DIR / "sweep_summary.csv").write_text("\n".join(lines) + "\n")


def _print_table(summary):
    if not summary:
        print("\nNothing ran.")
        return

    def g(r, *path, default=float("nan")):
        v = r
        for p in path:
            if not isinstance(v, dict) or p not in v:
                return default
            v = v[p]
        return v

    print("\n" + "=" * 128)
    print("SWEEP SUMMARY   all-to-all,  COUPLING_NORM=%s,  backend=%s,  "
          "correlators=%s" % (COUPLING_NORM, BACKEND, CORRELATORS))
    print("=" * 128)
    # EVERY quantum column is now printed next to its classical counterpart.
    # The 2026-10-11 table printed three quantum zero-day columns and NO
    # classical one, although run_experiment had computed and saved it all
    # along -- so the report built from this output shipped without the one
    # comparison the project's headline claim rests on. Never again.
    hdr = (f"{'cell':<16}{'N':>3}{'L':>3}{'dt':>6}{'feat':>5}{'live':>5}"
           f"{'2q':>5}{'surv':>7}"
           f"{'sup_q*':>8}{'sup_c':>8}{'d_sup':>8}"
           f"{'zd*AUC':>8}{'zd_c':>8}{'d_zd':>8}"
           f"{'zd*@1%':>8}{'zdc@1%':>8}{'zd*@5%':>8}{'zdc@5%':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in summary:
        sq = r.get('supervised_quantum_denoised', float('nan'))
        sc = r['supervised_classical']
        zq = g(r, 'zeroday_quantum_denoised', 'AUC')
        zc = g(r, 'zeroday_classical', 'AUC')
        print(f"{r['tag']:<16}{r['n_qubits']:>3}{r['L']:>3}"
              f"{r.get('dt', float('nan')):>6.2f}"
              f"{r['diag_n_features']:>5}"
              f"{r['diag_n_features_above_shot_floor']:>5}"
              f"{r['n_2q']:>5}{r['survival']:>6.1%}"
              f"{sq:>8.4f}{sc:>8.4f}{sq - sc:>+8.4f}"
              f"{zq:>8.4f}{zc:>8.4f}{zq - zc:>+8.4f}"
              f"{g(r, 'zeroday_quantum_denoised', '@1%FA'):>8.3f}"
              f"{g(r, 'zeroday_classical', '@1%FA'):>8.3f}"
              f"{g(r, 'zeroday_quantum_denoised', '@5%FA'):>8.3f}"
              f"{g(r, 'zeroday_classical', '@5%FA'):>8.3f}")
    print("-" * len(hdr))
    print("  feat = readout features;  live = features above the shot "
          "floor (all of them on the exact backend)")
    print("  sup_q*/zd*  = quantum arm on the live features, selected on "
          "TRAIN ONLY;  _c = classical PCA baseline")
    print("  d_sup, d_zd = quantum MINUS classical. Negative means the "
          "reservoir lost. THESE ARE THE TWO NUMBERS THAT MATTER.")
    print("  surv = product-of-fidelities estimate at Helios numbers, NOT "
          "a measured figure, and NOT applied to any feature above.")

    # An AUC difference has to be read against its own error bar.
    npos = int(summary[0].get("diag_test_attack", 0) or 0)
    nneg = int(summary[0].get("diag_test_benign", 0) or 0)
    if npos > 1 and nneg > 1 and len(SPLIT_SEEDS) == 1:
        def hm_se(a, n1, n2):
            q1 = a / (2 - a)
            q2 = 2 * a * a / (1 + a)
            return ((a * (1 - a) + (n1 - 1) * (q1 - a * a)
                     + (n2 - 1) * (q2 - a * a)) / (n1 * n2)) ** 0.5
        se_s = hm_se(max(0.5, min(0.999, summary[0]['supervised_classical'])),
                     npos, nneg)
        zc0 = g(summary[0], 'zeroday_classical', 'AUC')
        se_z = (hm_se(max(0.5, min(0.999, zc0)), npos, nneg)
                if zc0 == zc0 else float('nan'))
        print(f"\n  Hanley-McNeil standard error at {npos}+/{nneg}-: "
              f"supervised +-{se_s:.4f}, zero-day +-{se_z:.4f}. A paired "
              f"difference on one test set is tighter than that, but treat "
              f"any |d| below ~{se_s:.3f} as a tie.")

    # ---- aggregate over split seeds ----------------------------------
    # With one split seed this block prints nothing. With several it is the
    # only part of the table worth reading: the spread of d across splits
    # is the error bar on the claim, measured rather than estimated.
    if len(SPLIT_SEEDS) > 1:
        groups = {}
        for r in summary:
            key = (r["n_qubits"], r["L"], r.get("dt"), r.get("in_gain"))
            groups.setdefault(key, []).append(r)
        print("\n" + "=" * 128)
        print(f"ACROSS {len(SPLIT_SEEDS)} DATA SPLITS -- mean +- sd of the "
              f"quantum MINUS classical difference")
        print("=" * 128)
        ahdr = (f"{'cell':<16}{'n':>3}{'d_sup':>9}{'sd':>8}{'t':>7}"
                f"{'d_zd':>9}{'sd':>8}{'t':>7}{'zd wins':>9}")
        print(ahdr)
        print("-" * len(ahdr))
        for key, rs in sorted(groups.items()):
            ds = [r.get("supervised_quantum_denoised", float("nan"))
                  - r["supervised_classical"] for r in rs]
            dz = [g(r, "zeroday_quantum_denoised", "AUC")
                  - g(r, "zeroday_classical", "AUC") for r in rs]
            n = len(rs)

            def ms(v):
                v = [x for x in v if x == x]
                if not v:
                    return float("nan"), float("nan"), float("nan")
                m = sum(v) / len(v)
                sd = ((sum((x - m) ** 2 for x in v)
                       / max(1, len(v) - 1)) ** 0.5)
                # paired t on the differences: is the mean d distinguishable
                # from zero at all, given how much it moves between splits?
                t = (m / (sd / len(v) ** 0.5)) if sd > 0 else float("nan")
                return m, sd, t

            ms_s, sd_s, t_s = ms(ds)
            ms_z, sd_z, t_z = ms(dz)
            wins = sum(1 for x in dz if x == x and x > 0)
            name = rs[0]["tag"].rsplit("_s", 1)[0]
            print(f"{name:<16}{n:>3}{ms_s:>+9.4f}{sd_s:>8.4f}{t_s:>7.2f}"
                  f"{ms_z:>+9.4f}{sd_z:>8.4f}{t_z:>7.2f}"
                  f"{wins:>6}/{n}")
        print("-" * len(ahdr))
        print("  sd = spread of the DIFFERENCE across data splits. This is "
              "the error bar to quote.")
        print("  t  = mean / (sd/sqrt(n)), a paired t on the per-split "
              "differences. |t| < ~2.5 at n=5 means")
        print("       the difference is not distinguishable from zero no "
              "matter which way its sign points.")
        print("  The reservoir is IDENTICAL across these rows (SEED is "
              "fixed); only the data split moves.")

    first = summary[0]
    pos = first.get("diag_test_attack", 0)
    print(f"\n  Test positives per cell: {pos}. "
          + ("Enough to report." if pos >= 20 else
             "!! TOO FEW -- raise N_TEST before quoting anything."))
    if BACKEND == "exact":
        print("\n  BACKEND='exact': these are SHOT-NOISE-FREE ceilings. If "
              "d_sup and d_zd are still\n  negative here, no shot count and "
              "no hardware improvement can win on this dataset\n  and the "
              "encoding or the readout has to change. If they are strongly "
              "positive,\n  the architecture is shot-limited and the "
              "shot-scaling curve is the result to show.")
    print("\nWHAT TO READ OFF THIS TABLE")
    print("  1. d_sup and d_zd FIRST. The 2026-10-11 sweep was +0.005 on "
          "d_sup (inside the error\n     bar, and inflated by a readout "
          "bug since fixed) and -0.004 to -0.050 on d_zd --\n     i.e. a "
          "tie and a loss. Anything better than that is new information.")
    print("  2. Compare down a column (N at fixed L) against across a row "
          "(L at fixed N).\n     Whichever axis moves the number is the one "
          "worth spending the gate budget on.")
    print("  3. Watch 'live'. If it does not grow with 'feat', the extra "
          "qubits are adding\n     noise columns rather than signal, and "
          "that is a shot-count problem, not a\n     physics result.")
    print("  4. zd@1%FA is the pitch number. The supervised columns mostly "
          "say whether the\n     pipeline is sane -- tree ensembles already "
          "beat everything here on\n     supervised detection, so that is "
          "not the claim to make.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--dry-run" in args:
        dry_run()
    else:
        only = None
        if "--only" in args:
            only = args[args.index("--only") + 1]
        run_all(quick="--quick" in args, only=only)
