#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iontraps_qrc/core.py -- shared engine for the trapped-ion QRC experiments
========================================================================
This is the engine. It is NOT meant to be run directly. Run one of:

    python qrc_iontrap_sweep_alltoall.py   # 6/8/10 qubits x L=6/8, all-to-all
    python qrc_iontrap_8q_alltoall.py      # 8 qubits, all-to-all,        L=8
    python qrc_iontrap_20q_ring.py         # 20 qubits, nearest-neighbour, L=16

WHAT THIS IMPLEMENTS
--------------------
The thing the Quantinuum mentor asked for: a concrete "package representation
of the reservoir". It is three objects, and that is the whole answer:

    1. A FROZEN PARAMETER FILE  (reservoir_<tag>.npz)
       Random couplings J_ij and fields h_i, drawn once from a fixed seed and
       then never touched again. This IS the reservoir. Like the tube lengths
       of a wind chime: random, but once cast, fixed forever.

    2. A TROTTER STEP TEMPLATE  (reservoir_step below)
       A short function that reads those numbers and emits native gates for
       one timestep of exp(-i H dt) with
             H = sum_{<ij>} J_ij Z_i Z_j  +  sum_i h_i X_i
       Nothing here is trained. It is a transcription, not a model.

    3. A MEASURE-AND-RESET LOOP  (build_template below)
       The piece that turns a static feature map into an actual reservoir
       with memory. Each timestep: inject a slice of data on the INPUT
       qubits, let the whole register evolve, then measure and reset ONLY
       the input qubits. The MEMORY qubits are never measured, so they carry
       correlations forward across timesteps. This is the NISQRC
       architecture (Hu, Khan, Bronn, Tureci, Nat. Commun. 15, 7491).

HOW THIS DIFFERS FROM OUR PHOTONIC RUN
--------------------------------------
Photonic (Quandela) is a quantum EXTREME LEARNING MACHINE: one sample in,
one feature vector out, no state carried between samples.
          output_i = f(x_i)
This is a true RESERVOIR: state carries forward.
          state_t = F(state_{t-1}, x_t)

WHERE THE SEQUENCE COMES FROM -- read this, it is a real design decision
-----------------------------------------------------------------------
Neither NSL-KDD nor the NF-UNSW-NB15 v1 NetFlow export carries a timestamp
or a session key, so we cannot build a time sequence from either one
honestly. Two modes are therefore supported:

  SEQUENCE_MODE = "features"  (default, works on NSL-KDD today)
      Split ONE row's reduced feature vector into L chunks and inject them
      one chunk per timestep. The memory then integrates across FEATURE
      GROUPS rather than across time. This is standard sequential-feature
      injection, and it buys something real: a register of only N_IN input
      qubits can ingest 2 * N_IN * L features. See CAPACITY note below.

  SEQUENCE_MODE = "rows"      (stub, needs a sessionised CSV)
      True temporal mode: L consecutive flows of one session, one flow per
      timestep. Requires CIC-IDS2017 or similar with a session key. The
      loader raises a clear NotImplementedError saying what it needs.

CAPACITY -- AND WHY THE ORIGINAL ARGUMENT INVERTED ON THIS DATASET
------------------------------------------------------------------
Dense angle encoding offers 2 slots per input qubit per step, and
sequential injection repeats that L times:

      encoding slots = 2 * N_IN * L

      6q  all-to-all L=6 : 2 * 3 * 6  =  36 slots
      8q  all-to-all L=8 : 2 * 4 * 8  =  64 slots
      10q all-to-all L=8 : 2 * 5 * 8  =  80 slots
      20q ring       L=16: 2 * 8 * 16 = 256 slots

THE PROJECT WAS DESIGNED AROUND NSL-KDD, where the feature vector is 122
wide and PCA at 20 components kept only 38.96% of the variance (84 of the
122 columns are one-hot dummies holding 69.4% of the variance, and one-hot
columns are near-orthogonal so PCA cannot compress them). Against that
dataset "sequential injection ingests the whole vector" was the strongest
claim in the project.

AGAINST NF-UNSW-NB15 THE SAME ARGUMENT RUNS BACKWARDS. That dataset has
about 24 features after one-hot encoding TCP_FLAGS, so every architecture
here asks for MORE slots than the data has columns. In the 2026-10-10 runs
the surplus was filled with the constant 0.5:

      8q : 64 slots, ~24 real  ->  40 slots (62%) carried no information
      20q: 256 slots, 19 real  -> 237 slots (93%) carried no information

Do not quote the 122-feature / 38.96% / "ingests the entire vector"
argument against NF-UNSW-NB15 results. It is a true statement about a
different dataset. The slot surplus is now handled by data re-uploading
(see angles_from_features) rather than constant padding, which turns the
surplus from waste into extra nonlinearity -- but it is extra depth on the
same ~24 numbers, not extra information, and must be described that way.

COST WARNING
------------
H-series cost is PER SAMPLE, because every sample is its own circuit. The
budget_report() function prints real HQC numbers before anything is
submitted, and hardware submission is refused unless the arch file sets
CONFIRM_HARDWARE = True. Read that report first.

Deps: numpy, pandas, scikit-learn  (always)
      pytket, pytket-quantinuum, qnexus  (only for BACKEND="nexus"/"syntax")
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

# ======================================================================
# 0) Conventions -- get these wrong and the physics is wrong
# ======================================================================
# Quantinuum native gate set: {Rz, PhasedX (=U1q), ZZMax, ZZPhase}
# ALL pytket angles are in HALF-TURNS: 1.0 means pi radians.
#
#   ZZPhase(a)      = exp(-i * (pi*a/2) * Z@Z)
#   PhasedX(a, b)   = Rz(b) . Rx(a) . Rz(-b)          [half-turns]
#   PhasedX(a, 0)   = Rx(pi*a) = exp(-i * (pi*a/2) * X)
#   PhasedX(a, 0.5) = Ry(pi*a)
#   Rz(a)           = diag(exp(-i*pi*a/2), exp(+i*pi*a/2))
#
# To realise exp(-i*dt*H) for H = sum J_ij Z_i Z_j + sum h_i X_i we need
#     pi*a/2 = J_ij*dt   ->   a = 2*J_ij*dt/pi
#     pi*a/2 = h_i*dt    ->   a = 2*h_i*dt/pi
#
# NOTE: the snippet in QRC_reservoir_hardware_and_code.pdf section 5.5 used
# J_ij*dt directly as the half-turn angle, i.e. it was off by a factor 2/pi.
# Any fixed random coupling still works as a reservoir, but then "dt" did
# not mean what it claimed. The conversion below is the correct one.

def to_halfturns(coupling, dt: float):
    """Hamiltonian coupling (rad per unit time) * dt  ->  pytket half-turns."""
    return 2.0 * np.asarray(coupling) * dt / np.pi


# ======================================================================
# 1) The reservoir: frozen random parameters
# ======================================================================
@dataclass
class ReservoirSpec:
    """Everything that defines the reservoir. Frozen at construction."""
    tag: str                      # short name, used for the .npz filename
    n_in: int                     # input qubits: encoded and measured each step
    n_mem: int                    # memory qubits: never measured
    n_steps: int                  # L, timesteps per sample
    dt: float                     # Trotter step size
    topology: str                 # "all_to_all" | "ring"
    seed: int = 1234
    j_scale: float = 1.0          # J_ij ~ U(-j_scale, +j_scale)
    h_scale: float = 1.0          # magnitude bound on the transverse field

    # HOW THE TRANSVERSE FIELD IS DRAWN. This looks like a detail and is not.
    #
    # The field layer is the ONLY thing that converts phase into population.
    # The ZZ layer is diagonal, so it cannot change any <Z>; the encoding Rz
    # writes into the phase; readout is in the Z basis. A qubit whose h_i is
    # near zero is therefore FROZEN in the readout basis: it contributes
    # almost nothing to any feature, and the longitude half of the dense
    # angle encoding is invisible on it.
    #
    # "uniform" (the original) drew h_i ~ U(-h_scale, +h_scale), which puts
    # 10% of qubits inside |h| < 0.1*h_scale. For an 8-qubit register the
    # chance that at least one qubit lands there is 57%, and our seed-1234
    # reservoir drew THREE: h = [-0.950, +0.025, +0.226, -0.439, -0.096,
    # +0.062, +0.977, -0.988]. Measured consequence, final-snapshot std at
    # 2000 shots, which tracks |h| monotonically across all eight qubits:
    #       q1 (h=+0.025) -> 0.0006        q0 (h=-0.950) -> 0.0217
    #       q2 (h=+0.226) -> 0.0056        q6 (h=+0.977) -> 0.1282
    # q1's column was EXACTLY constant in the real run. That is the whole
    # explanation for the dead final snapshot, and it is very likely a large
    # part of why the longitude encoding measured as unrecoverable.
    #
    # "annulus" (default) keeps the sign random but bounds the magnitude
    # away from zero: h_i = +-U(0.3, 1.0) * h_scale. Still a random frozen
    # reservoir -- the disorder that makes it a reservoir is intact -- but
    # every qubit now has a working field. Per-step field rotation is
    # 2*|h|*dt radians, so at dt=0.3 this guarantees 0.18-0.60 rad/step
    # instead of a possible 0.015.
    field_draw: str = "annulus"   # "annulus" | "uniform" (original)

    # WHICH physical sites are the input qubits. None means sites 0..n_in-1,
    # which is fine for all-to-all (every site is equivalent) but WRONG for a
    # ring: a contiguous block of inputs leaves the memory arc reachable
    # through only two bonds, so data barely reaches it. Use spread_sites().
    in_sites: list[int] | None = None

    # Run one extra reservoir step after the last measure-and-reset, before
    # the final snapshot. WITHOUT it the input qubits are measured while
    # still in |0> from their own reset, so their final-snapshot columns are
    # EXACTLY constant (+1) and carry nothing.
    #
    # HISTORY -- the first version of this fix did NOT work, and the measured
    # evidence is worth keeping. A plain extra step is ZZ -> field. After
    # Reset the input qubit sits in |0>, a Z eigenstate, which the DIAGONAL
    # ZZ layer cannot move at all; the single PhasedX that follows therefore
    # returns the constant cos(2*h_i*dt) for every sample. Measured on both
    # completed NF-UNSW-NB15 runs, std of the final-snapshot column:
    #     8q : inputs 0.0546 / 0.0000 / 0.0136 / 0.0264
    #          memory 0.0877 / 0.0915 / 0.1093 / 0.1094
    #     20q: inputs 0.022-0.086,  memory 0.179-0.283
    # The exact 0.00000 on 8q q1 is the signature. 0 of 8 and 0 of 20 final
    # columns cleared the shot-noise floor in those runs.
    # FIELD -> ZZ -> FIELD is the right order: the leading field layer
    # rotates the reset qubits off the Z axis so the ZZ layer can entangle
    # them with the memory register. Costs N extra 1q gates and zero extra
    # 2q gates; implemented as reservoir_step(..., field_first=True).
    #
    # HOW MUCH THAT ACTUALLY BUYS -- measured, 8q seed 1234, L=8, 2000
    # shots (shot floor 0.0224), std of the final-snapshot columns.
    #
    #   field_draw='uniform'      q0i     q1i     q2i     q3i | q4m     q5m
    #     no final mix         0.0000  0.0000  0.0000  0.0000 |      -       -
    #     ZZ -> field          0.0098  0.0004  0.0035  0.0046 | 0.0286  0.0319
    #     field -> ZZ -> field 0.0223  0.0006  0.0060  0.0109 | 0.0296  0.0329
    #
    #   field_draw='annulus'
    #     ZZ -> field          0.0050  0.0068  0.0086  0.0047 | 0.0569  0.0778
    #     field -> ZZ -> field 0.0093  0.0111  0.0129  0.0076 | 0.0463  0.0815
    #
    # Read that table honestly, because it does not say what we hoped:
    #
    #  * The reordering is strictly correct and roughly DOUBLES each input
    #    column, so keep it -- it costs N 1q gates and no 2q gates.
    #  * 'annulus' removes the dead column (q1: 0.0006 -> 0.0111, 18x) and
    #    roughly doubles the two weakest MEMORY columns (q4 0.0286 ->
    #    0.0569, q5 0.0319 -> 0.0778). That is the part that matters, since
    #    the memory columns were the only usable ones.
    #  * BUT all four input columns remain BELOW the 0.0224 shot floor even
    #    at 2000 shots, under every combination. The final snapshot of a
    #    just-reset qubit is intrinsically weak: one mixing layer is not
    #    enough to rebuild a correlation with the data after the reset
    #    destroyed it.
    #
    # Conclusion to carry into the write-up: the final snapshot is worth
    # its n_in + n_mem columns for the MEMORY qubits and essentially not
    # worth it for the input qubits. The signal lives in the L * n_in
    # mid-circuit measurements, which are driven by the encoding directly
    # and never depended on either of these fixes.
    final_mix: bool = True

    # THE INPUT GAIN. This was the literal 3.0 in angles_from_features:
    #       Zs = (Z - mu) / (in_gain * sd);  sq = 0.5*(1 + tanh(Zs))
    # It is the INPUT SCALING of classical reservoir computing -- one of the
    # three canonical RC hyperparameters alongside spectral radius and leak
    # rate -- and it had never been swept in either codebase, nor recorded
    # anywhere, which also meant reservoir_<tag>.npz did not fully specify
    # the experiment.
    #
    # WHY IT IS THE CHEAPEST LEVER ON THE SHOT-NOISE PROBLEM. A feature is
    # usable only when its across-sample signal exceeds sqrt((1-z^2)/S).
    # There are two ways to buy that margin:
    #       more shots  ->  floor falls like 1/sqrt(S)   (4x shots = 2x SNR)
    #       more gain   ->  signal rises ~linearly in the angle excursion,
    #                       until tanh saturates
    # So gain is roughly 4x cheaper than shots per unit of SNR, and free.
    # The cost is saturation (resolution lost on outliers) and extra
    # scrambling, so there is an optimum -- which is exactly what makes it
    # worth scanning rather than guessing.
    #
    # SMALLER in_gain = MORE aggressive encoding (bigger angles).
    # 3.0 is the historical value and reproduces every run before
    # 2026-10-12 exactly.
    in_gain: float = 3.0

    # WHICH BASIS THE FINAL SNAPSHOT IS READ IN.
    #
    # Every run of this project has measured Z and only Z, so two thirds of
    # the available single-qubit information was being discarded at
    # identical shot cost. <X> and <Y> are different functions of the same
    # state and cost one extra 1q gate each -- no 2q gates, no extra shots.
    #
    #   "z"      all qubits in Z  (historical; the default, reproduces)
    #   "x"/"y"  all qubits in that basis
    #   "mixed"  a FROZEN per-qubit choice from {z,x,y}, derived from seed
    #            so it is part of the reservoir and reproduces exactly
    #
    # This applies ONLY to the final snapshot. The mid-circuit measurements
    # must stay in Z: they are the measure-and-reset channel that drives the
    # reservoir, and rotating them would change the dynamics, not just the
    # readout.
    final_basis: str = "z"        # "z" | "x" | "y" | "mixed"

    # filled in by freeze()
    J: np.ndarray | None = field(default=None, repr=False)
    h: np.ndarray | None = field(default=None, repr=False)
    mem_init: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_qubits(self) -> int:
        return self.n_in + self.n_mem

    @property
    def in_qubits(self) -> list[int]:
        if self.in_sites is None:
            return list(range(self.n_in))
        if len(set(self.in_sites)) != self.n_in:
            raise ValueError(f"in_sites must hold {self.n_in} distinct sites")
        if max(self.in_sites) >= self.n_qubits:
            raise ValueError("in_sites index beyond the register")
        return list(self.in_sites)

    @property
    def mem_qubits(self) -> list[int]:
        inp = set(self.in_qubits)
        return [q for q in range(self.n_qubits) if q not in inp]

    @property
    def edges(self) -> list[tuple[int, int]]:
        return make_edges(self.topology, self.n_qubits)

    @property
    def final_axes(self) -> list[str]:
        """Measurement basis per qubit for the final snapshot.

        Derived from `seed` rather than stored in the .npz, so it is
        reproducible AND backward compatible with reservoir files frozen
        before final_basis existed.
        """
        if self.final_basis in ("z", "x", "y"):
            return [self.final_basis] * self.n_qubits
        if self.final_basis == "mixed":
            rng = np.random.default_rng(self.seed + 0xBA515)
            return [("z", "x", "y")[i]
                    for i in rng.integers(0, 3, self.n_qubits)]
        raise ValueError(f"unknown final_basis {self.final_basis!r}")

    @property
    def n_features_ingested(self) -> int:
        """Dense angle encoding = 2 features per input qubit per step."""
        return 2 * self.n_in * self.n_steps

    # ------------------------------------------------------------------
    def freeze(self, path: str | Path | None = None,
               write: bool = True) -> "ReservoirSpec":
        """Draw the random parameters ONCE and write them to disk.

        If the file already exists it is LOADED instead of redrawn, so a
        rerun months later reproduces the identical reservoir. This file is
        the deliverable: hand it over and the reservoir is fully specified.

        write=False draws (or loads) the parameters but writes nothing.
        That is what a --dry-run wants: it still gets real J/h so
        check_fields() can warn about dead qubits, without creating a
        reservoir file that a later real run would silently reload.
        """
        path = Path(path or f"reservoir_{self.tag}.npz")
        N = self.n_qubits
        if path.exists():
            z = np.load(path, allow_pickle=False)
            self.J, self.h, self.mem_init = z["J"], z["h"], z["mem_init"]
            if self.J.shape != (N, N):
                raise ValueError(
                    f"{path} holds a {self.J.shape} coupling matrix but this "
                    f"spec needs ({N},{N}). Delete the file or change the tag.")
            # Reloading is the whole point of freezing -- but it also means a
            # stale file silently overrides the spec in front of you. After
            # 2026-10-11 that is a real hazard: field_draw changed, and a
            # reservoir_<tag>.npz left over from the old runs would quietly
            # restore the near-zero transverse fields that broke them.
            side = Path(str(path) + ".json")
            if side.exists():
                try:
                    old = json.loads(side.read_text())
                except (OSError, ValueError):
                    old = {}
                # Two tiers, because the old single-tier check cried wolf.
                # HARD fields feed the rng that draws J/h/mem_init, so a
                # mismatch means the arrays on disk are NOT the ones this
                # spec describes. SOFT fields are recorded for provenance
                # but never touch those arrays -- dt and in_gain only scale
                # gate angles, n_steps only sets the loop count, and
                # final_basis is derived from seed. A dt or in_gain sweep
                # SHOULD reuse one frozen reservoir: that is what isolates
                # the knob being swept.
                hard = {k: (old[k], getattr(self, k))
                        for k in ("j_scale", "h_scale", "field_draw",
                                  "n_in", "n_mem", "seed", "topology")
                        if k in old and old[k] != getattr(self, k)}
                soft = {k: (old[k], getattr(self, k))
                        for k in ("dt", "n_steps", "in_gain", "final_basis",
                                  "final_mix")
                        if k in old and old[k] != getattr(self, k)}
                if hard:
                    print(f"[reservoir] !! {path.name} was frozen with "
                          f"DIFFERENT parameters than this spec asks for:")
                    for k, (was, now) in hard.items():
                        print(f"[reservoir] !!     {k}: file has {was!r}, "
                              f"spec says {now!r}")
                    if "field_draw" in hard:
                        print("[reservoir] !! field_draw differs, so the "
                              "near-zero-field bug may still be present in "
                              "this reservoir.")
                    print(f"[reservoir] !! The FILE WINS for J/h/mem_init. "
                          f"Delete {path.name} (and its .json) to redraw, or "
                          f"change the tag to keep both.")
                if soft:
                    bits = ", ".join(f"{k} {was!r}->{now!r}"
                                     for k, (was, now) in soft.items())
                    print(f"[reservoir] reusing this reservoir with "
                          f"{bits} -- these do not affect J/h/mem_init, so "
                          f"the reload is correct and the comparison is "
                          f"clean.")
            print(f"[reservoir] loaded frozen parameters from {path}")
            self.check_fields()
            return self

        rng = np.random.default_rng(self.seed)
        # upper triangle only: J is symmetric and so is Z_i Z_j
        self.J = np.triu(rng.uniform(-self.j_scale, self.j_scale, (N, N)), 1)
        if self.field_draw == "uniform":
            # exactly one rng call, so this reproduces pre-2026-10-11
            # reservoirs byte for byte
            self.h = rng.uniform(-self.h_scale, self.h_scale, N)
        elif self.field_draw == "annulus":
            sign = np.where(rng.random(N) < 0.5, -1.0, 1.0)
            self.h = sign * rng.uniform(0.3, 1.0, N) * self.h_scale
        else:
            raise ValueError(f"unknown field_draw {self.field_draw!r}")
        # Memory qubits start in |0>, which is a Z eigenstate and therefore
        # inert under the ZZ layer. A fixed random Ry breaks that symmetry so
        # the memory register is live from step 0.
        self.mem_init = rng.uniform(0.2, 0.8, self.n_mem)
        if not write:
            print(f"[reservoir] drew parameters in memory only "
                  f"(write=False, nothing saved)")
            self.check_fields()
            return self
        np.savez(path, J=self.J, h=self.h, mem_init=self.mem_init,
                 dt=np.array(self.dt), seed=np.array(self.seed))
        meta = {k: v for k, v in asdict(self).items()
                if not isinstance(v, np.ndarray) and v is not None}
        Path(str(path) + ".json").write_text(json.dumps(meta, indent=2))
        print(f"[reservoir] froze parameters -> {path} (+ .json sidecar)")
        self.check_fields()
        return self

    def check_fields(self) -> None:
        """Warn about qubits the Z-basis readout cannot see.

        Per step the field layer rotates qubit i by 2*|h_i|*dt radians. If
        that is tiny the qubit stays pinned to the Z axis and its features
        are constant -- the diagonal ZZ layer cannot help, and neither can
        more shots. See the field_draw note above for the measured case.
        """
        rot = 2.0 * np.abs(self.h) * self.dt
        total = rot * self.n_steps
        dead = np.where(total < 0.5)[0]
        print(f"[reservoir] field rotation per step: "
              f"{rot.min():.3f}-{rot.max():.3f} rad "
              f"({total.min():.2f}-{total.max():.2f} rad over L={self.n_steps})"
              f"   draw={self.field_draw!r}")
        if len(dead):
            print(f"[reservoir] !! qubits {dead.tolist()} accumulate < 0.5 rad "
                  f"of field rotation over the whole circuit. They are very "
                  f"nearly frozen in the Z basis and their features will be "
                  f"near-constant. Raise dt or h_scale, or use "
                  f"field_draw='annulus'.")

    # ------------------------------------------------------------------
    def zz_halfturns(self) -> list[tuple[int, int, float]]:
        """The ZZPhase layer, as (qubit_i, qubit_j, half_turns)."""
        return [(i, j, float(to_halfturns(self.J[i, j], self.dt)))
                for (i, j) in self.edges]

    def x_halfturns(self) -> np.ndarray:
        """The transverse-field layer, as PhasedX(a, 0) half-turns."""
        return np.asarray(to_halfturns(self.h, self.dt), dtype=float)


def make_edges(topology: str, n: int) -> list[tuple[int, int]]:
    """Which qubit pairs get a ZZPhase.

    all_to_all : C(n,2) pairs. Essentially free on a trapped ion -- every
                 pair is a direct interaction with no SWAP network. This is
                 the ion trap's structural advantage over a superconducting
                 chip, and the reason it is worth trying at all.
    ring       : n pairs, nearest neighbour with wraparound. Cheap, and the
                 standard choice in the QRC literature (transverse-field
                 Ising chain). Information has to diffuse around the ring,
                 which lengthens memory but weakens mixing per step.
    """
    if topology == "all_to_all":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    if topology == "ring":
        if n < 3:
            raise ValueError("a ring needs at least 3 qubits")
        return [(i, (i + 1) % n) for i in range(n)]
    raise ValueError(f"unknown topology {topology!r}")


def spread_sites(n_total: int, n_in: int) -> list[int]:
    """Place the input qubits as evenly as possible around a ring.

    On a ring, nearest-neighbour coupling means information travels by
    diffusion. If the inputs sit in a contiguous block, the memory arc is
    many hops away and the injected data barely reaches it within L steps.
    Spreading them means every memory qubit is within a few bonds of an
    injection site.

        spread_sites(20, 8) -> [0, 2, 5, 7, 10, 12, 15, 17]
    """
    if n_in > n_total:
        raise ValueError("more input qubits than qubits")
    sites = sorted({(i * n_total) // n_in for i in range(n_in)})
    if len(sites) != n_in:                      # pathological ratios
        sites = sorted(set(range(0, n_total, max(1, n_total // n_in))))[:n_in]
    return sites


# ======================================================================
# 2) Cost and fidelity budget -- run this BEFORE spending anything
# ======================================================================
# Helios published figures used below. Verify against current specs before
# quoting them anywhere official.
FID_2Q = 0.99921      # Helios two-qubit gate fidelity
FID_1Q = 0.999975     # Helios single-qubit gate fidelity
FID_MEAS = 0.9997     # approximate mid-circuit measurement fidelity


@dataclass
class Budget:
    n_2q: int
    n_1q: int
    n_meas: int
    depth_2q: int

    def survival(self) -> float:
        """Crude product-of-fidelities estimate: P(no error anywhere)."""
        return (FID_2Q ** self.n_2q * FID_1Q ** self.n_1q
                * FID_MEAS ** self.n_meas)

    def hqc(self, n_shots: int) -> float:
        """Quantinuum H-series credit formula.

            HQC = 5 + C * (N_1q + 10*N_2q + 5*N_meas) / 5000

        Verify against the current Quantinuum pricing docs before relying on
        it for a real allocation -- the constants have changed before.
        """
        return 5.0 + n_shots * (self.n_1q + 10 * self.n_2q
                                + 5 * self.n_meas) / 5000.0


def circuit_budget(spec: ReservoirSpec) -> Budget:
    """Gate counts for ONE sample, counted by hand from build_template."""
    L, n_in, N = spec.n_steps, spec.n_in, spec.n_qubits
    n_edges = len(spec.edges)
    steps = L + (1 if spec.final_mix else 0)   # the extra pre-snapshot step
    n_2q = steps * n_edges
    n_rot = sum(1 for ax in spec.final_axes if ax != "z")
    n_1q = (spec.n_mem                 # memory init, once
            + L * 2 * n_in             # encode: PhasedX + Rz per input qubit
            + steps * N                # transverse field layer
            + (N if spec.final_mix else 0)   # field_first on the final step
            + n_rot)                   # final-snapshot basis rotation
    n_meas = L * n_in + N              # mid-circuit + final readout
    # A ring layer splits into 2 rounds of disjoint pairs; all-to-all must
    # serialise into (n-1) rounds of a perfect matching.
    per_layer_depth = 2 if spec.topology == "ring" else (N - 1)
    return Budget(n_2q=n_2q, n_1q=n_1q, n_meas=n_meas,
                  depth_2q=steps * per_layer_depth)


def budget_report(spec: ReservoirSpec, shot_options=(100, 512, 1024),
                  sample_options=(60, 200, 600),
                  correlators: bool = False) -> Budget:
    """Print the cost table. Call this before every hardware submission."""
    b = circuit_budget(spec)
    nq = spec.n_qubits
    print("\n" + "=" * 68)
    print(f"BUDGET  --  {spec.tag}: {nq} qubits ({spec.n_in} in + "
          f"{spec.n_mem} mem), {spec.topology}, L={spec.n_steps}")
    print("=" * 68)
    print(f"  coupling edges per step : {len(spec.edges)}")
    print(f"  two-qubit gates / sample: {b.n_2q}")
    print(f"  one-qubit gates / sample: {b.n_1q}")
    print(f"  measurements    / sample: {b.n_meas}  "
          f"({spec.n_steps * spec.n_in} mid-circuit + {nq} final)")
    print(f"  two-qubit depth         : {b.depth_2q}")
    print(f"  est. survival probability: {b.survival():6.1%}   "
          f"(product of fidelities, no error mitigation)")
    print(f"  encoding slots  / sample : {spec.n_features_ingested}")
    print(f"  readout features/ sample : "
          f"{n_readout_features(spec, correlators)}"
          f"   (correlators {'ON' if correlators else 'OFF'})")
    print("\n  HQC cost (circuits are PER SAMPLE -- this is the hard limit):")
    print("        samples |" + "".join(f"{s:>10d} sh" for s in shot_options))
    print("        " + "-" * (8 + 13 * len(shot_options)))
    for ns in sample_options:
        row = "".join(f"{ns * b.hqc(s):>12,.0f}" for s in shot_options)
        print(f"        {ns:>7d} |{row}")
    print("\n  Free/cheap paths, in order of preference:")
    print("    1. BACKEND='numpy'   local simulator + shot noise, zero cost")
    print("    2. BACKEND='exact'   shot-noise-free ceiling, zero cost")
    print("    3. BACKEND='syntax'  Quantinuum syntax checker, zero HQC")
    print("    4. BACKEND='nexus'   emulator (H1-1E / Helios-E), costs HQC")
    print("    Only set CONFIRM_HARDWARE=True once the numbers above are "
          "affordable.")
    print("=" * 68 + "\n")
    return b


def n_readout_features(spec: ReservoirSpec, correlators: bool = True) -> int:
    n = spec.n_steps * spec.n_in + spec.n_qubits          # single-qubit <Z>
    if correlators:
        n += spec.n_steps * math.comb(spec.n_in, 2)       # same-step <Z_i Z_j>
    return n


# ======================================================================
# 3) The circuit: one symbolic template, reused for every sample
# ======================================================================
# Building and compiling a fresh circuit per sample is the obvious approach
# and it is far too slow -- compilation would dominate the run. Instead we
# build ONE circuit whose encoding angles are sympy symbols, compile that
# once to native gates, then substitute numbers per sample. Because the
# compiled gates (PhasedX, Rz, ZZPhase) all take continuous angles, the
# symbols survive compilation and substitution needs no recompile.
#
# This is also a clean answer to "what do you actually submit?": one
# compiled template plus a table of angle substitutions.

def _require_pytket():
    try:
        from pytket import Circuit, OpType          # noqa: F401
        from sympy import Symbol                    # noqa: F401
    except ImportError as e:
        raise ImportError(
            "pytket and sympy are needed for circuit building.\n"
            "  pip install pytket pytket-quantinuum qnexus\n"
            "The BACKEND='numpy' path needs none of this.") from e


def encode_symbols(spec: ReservoirSpec):
    """Symbol table: theta[t][k] (latitude) and phi[t][k] (longitude)."""
    from sympy import Symbol
    th = [[Symbol(f"th_{t}_{k}") for k in range(spec.n_in)]
          for t in range(spec.n_steps)]
    ph = [[Symbol(f"ph_{t}_{k}") for k in range(spec.n_in)]
          for t in range(spec.n_steps)]
    return th, ph


def reservoir_step(c, spec: ReservoirSpec, field_first: bool = False) -> None:
    """ONE Trotter step of exp(-i*dt*H). This is the whole reservoir.

    First-order Trotter: all the commuting ZZ terms, then all the X terms.
    The error is O(dt^2) per step, which is irrelevant here -- we are not
    simulating a target Hamiltonian, we only need rich fixed dynamics. The
    Trotter form is simply the cheapest way to express "let it evolve" in a
    gate-based machine.

    field_first=True prepends a second field layer, giving FIELD -> ZZ ->
    FIELD. Use it for the final_mix step ONLY. Inside the loop every qubit
    is already off the Z axis (the encoding put it there), so the leading
    field layer is redundant and would just cost gates. After a Reset it is
    the difference between a live feature and a constant -- see the
    final_mix note on ReservoirSpec for the measured numbers.
    """
    if field_first:
        for i, a in enumerate(spec.x_halfturns()):
            c.PhasedX(a, 0.0, i)
    for (i, j, a) in spec.zz_halfturns():
        c.ZZPhase(a, i, j)                 # native two-qubit gate
    for i, a in enumerate(spec.x_halfturns()):
        c.PhasedX(a, 0.0, i)               # native one-qubit gate = Rx(pi*a)


def encode_slice(c, spec: ReservoirSpec, thetas, phis) -> None:
    """Dense angle encoding: 2 features per qubit.

    A qubit state needs two angles, like a point on a globe:
        PhasedX(theta, 0.5) = Ry  -> latitude   (how much |1> amplitude)
        Rz(phi)                   -> longitude  (the relative phase)
    Plain Ry-only encoding leaves longitude unused and wastes half the
    capacity. Both gates are native, so this costs no extra compilation.

    Rz commutes through the ZZ layer, but NOT through the PhasedX field
    layer that follows it, so phi genuinely changes the dynamics -- it is
    not a redundant parameter.
    """
    for k, q in enumerate(spec.in_qubits):
        c.PhasedX(thetas[k], 0.5, q)       # latitude
        c.Rz(phis[k], q)                   # longitude


def build_template(spec: ReservoirSpec):
    """The full symbolic circuit for one sample.

    Returns (circuit, mid_bits, final_bits, (theta_syms, phi_syms)).
    """
    _require_pytket()
    from pytket import Circuit, OpType

    N = spec.n_qubits
    th, ph = encode_symbols(spec)

    c = Circuit(N)
    mid = c.add_c_register("m", spec.n_steps * spec.n_in)
    fin = c.add_c_register("f", N)

    # --- memory register wake-up (frozen, not trained) ----------------
    for idx, q in enumerate(spec.mem_qubits):
        c.PhasedX(float(spec.mem_init[idx]), 0.5, q)

    # --- the reservoir loop -------------------------------------------
    for t in range(spec.n_steps):
        encode_slice(c, spec, th[t], ph[t])
        reservoir_step(c, spec)
        # Measure and reset ONLY the input qubits. The memory qubits are
        # untouched, which is exactly where the memory lives. The reset is
        # also the dissipation that keeps the reservoir from saturating --
        # the analog machine gets this free from its environment, we have
        # to engineer it.
        for k, q in enumerate(spec.in_qubits):
            c.Measure(q, mid[t * spec.n_in + k])
            c.add_gate(OpType.Reset, [q])

    # --- final snapshot of the whole register --------------------------
    # One more evolution step first, or the input qubits are read out while
    # still sitting in |0> from their own reset and contribute nothing.
    # field_first=True is REQUIRED here; without it the ZZ layer cannot move
    # a freshly reset Z eigenstate and the n_in input columns come back
    # constant. See the final_mix note on ReservoirSpec.
    if spec.final_mix:
        reservoir_step(c, spec, field_first=True)
    # Per-qubit readout basis. One PhasedX each, no two-qubit gates. With
    # the default final_basis='z' this loop emits nothing.
    for q, ax in enumerate(spec.final_axes):
        r = _BASIS_ROT[ax]
        if r is not None:
            c.PhasedX(r[0], r[1], q)
    for q in range(N):
        c.Measure(q, fin[q])

    mid_bits = [mid[i] for i in range(spec.n_steps * spec.n_in)]
    fin_bits = [fin[i] for i in range(N)]
    return c, mid_bits, fin_bits, (th, ph)


def substitute(template, syms, angles: np.ndarray):
    """Bind one sample's angles into a copy of the compiled template.

    angles : (n_steps, n_in, 2) in HALF-TURNS.
    """
    th, ph = syms
    L, n_in, _ = angles.shape
    sub = {}
    for t in range(L):
        for k in range(n_in):
            sub[th[t][k]] = float(angles[t, k, 0])
            sub[ph[t][k]] = float(angles[t, k, 1])
    c = template.copy()
    c.symbol_substitution(sub)
    return c


# ======================================================================
# 4) Data: NSL-KDD -> sequences of encoding angles
# ======================================================================
CSV_CANDIDATES = [
    r"D:\claude things\QRC\data\NSL_KDD_labeled.csv",
    r"D:\claude things\NSL_KDD_labeled.csv",
    "data/NSL_KDD_labeled.csv",
    "../data/NSL_KDD_labeled.csv",
    "NSL_KDD_labeled.csv",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
TARGET = "label"

# Columns whose raw scale is so heavy-tailed that StandardScaler destroys
# them. MEASURED on our data: src_bytes has std 5.87e6 and max 1.38e9, so
# 99.58% of all rows land within |z| < 0.01 -- the column is effectively
# constant after scaling. log1p first drops that to 0.02%.
HEAVY_TAILED = ["src_bytes", "dst_bytes", "duration"]

# ---- NF-UNSW-NB15 (NetFlow v1) --------------------------------------
# The dataset actually supplied for this run. 12 columns: 10 numeric flow
# features, a binary Label (0 benign / 1 attack) and an Attack family string.
# This v1 export carries no src/dst IP, so SEQUENCE_MODE="rows" still cannot
# build sessions from it -- see load_sessions().
NFUNSW_PARQUET_CANDIDATES = [
    *([os.environ["NFUNSW_PARQUET"]] if os.environ.get("NFUNSW_PARQUET") else []),
    str(Path(__file__).resolve().parents[2] / "data" / "raw" / "NF-UNSW-NB15.parquet"),
    "data/raw/NF-UNSW-NB15.parquet",
    "NF-UNSW-NB15.parquet",
]
# TCP_FLAGS is a genuine categorical bit-flag field (15 values) -> one-hot.
# PROTOCOL and L7_PROTO are high-cardinality protocol IDs (255 / 265 values)
# and are left numeric, as in the standard NF-UNSW-NB15 baselines; one-hot
# would explode them for no gain.
NFUNSW_CATEGORICAL = ["TCP_FLAGS"]
NFUNSW_TARGET = "Label"
NFUNSW_ATTACK = "Attack"
NFUNSW_BENIGN = "Benign"
# Heavy-tailed counts/durations: log1p before StandardScaler, exactly as the
# NSL-KDD path does for src_bytes/dst_bytes/duration.
NFUNSW_HEAVY_TAILED = ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
                       "FLOW_DURATION_MILLISECONDS"]


def load_nfunsw_nb15(log1p_fix: bool = True, verbose: bool = True):
    """NF-UNSW-NB15 NetFlow parquet -> (X, y, attack_type, feature_names).

    Same contract as load_nslkdd: y is 0=benign / 1=attack, attack holds the
    family string ('Benign' for normal traffic), X is numeric flow features
    plus one-hot TCP_FLAGS. All fitted transforms (scaler, PCA) live
    downstream in run_experiment, fitted on TRAIN only.
    """
    path = next((p for p in NFUNSW_PARQUET_CANDIDATES if Path(p).exists()), None)
    if path is None:
        raise FileNotFoundError(
            "NF-UNSW-NB15.parquet not found; add its path to "
            f"NFUNSW_PARQUET_CANDIDATES (looked in: {NFUNSW_PARQUET_CANDIDATES}).")
    if verbose:
        print(f"[data] loading {path}")
    df = pd.read_parquet(path)

    numeric = [c for c in df.columns
               if c not in NFUNSW_CATEGORICAL + [NFUNSW_TARGET, NFUNSW_ATTACK]]
    if log1p_fix:
        applied = [c for c in NFUNSW_HEAVY_TAILED if c in df.columns]
        for col in applied:
            df[col] = np.log1p(df[col].clip(lower=0))
        if verbose:
            print(f"[data] log1p applied to {applied}")

    y = df[NFUNSW_TARGET].to_numpy(dtype=np.int64)
    attack = df[NFUNSW_ATTACK].astype(str).to_numpy()

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = ohe.fit_transform(df[NFUNSW_CATEGORICAL]).astype(np.float32)
    X_num = df[numeric].to_numpy(dtype=np.float32)
    X = np.concatenate([X_num, X_cat], axis=1)
    names = list(numeric) + list(ohe.get_feature_names_out(NFUNSW_CATEGORICAL))
    if verbose:
        print(f"[data] {X.shape[0]:,} rows x {X.shape[1]} features "
              f"({int((y == 0).sum()):,} benign / {int((y == 1).sum()):,} attack)")
    return X, y, attack, names


def load_nslkdd(log1p_fix: bool = True, verbose: bool = True):
    """Returns (X, y, attack_type, feature_names)."""
    path = next((p for p in CSV_CANDIDATES if Path(p).exists()), None)
    if path is None:
        raise FileNotFoundError(
            "NSL_KDD_labeled.csv not found; add its path to CSV_CANDIDATES.")
    if verbose:
        print(f"[data] loading {path}")
    df = pd.read_csv(path)

    numeric = [c for c in df.columns if c not in CATEGORICAL + [TARGET]]
    if log1p_fix:
        for col in HEAVY_TAILED:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))
        if verbose:
            print(f"[data] log1p applied to {[c for c in HEAVY_TAILED if c in df.columns]}")

    y = np.where(df[TARGET] == "normal", 0, 1).astype(np.int64)
    attack = df[TARGET].to_numpy()

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = ohe.fit_transform(df[CATEGORICAL]).astype(np.float32)
    X_num = df[numeric].to_numpy(dtype=np.float32)
    X = np.concatenate([X_num, X_cat], axis=1)
    names = list(numeric) + list(ohe.get_feature_names_out(CATEGORICAL))
    if verbose:
        print(f"[data] {X.shape[0]:,} rows x {X.shape[1]} features "
              f"({int((y == 0).sum()):,} normal / {int((y == 1).sum()):,} attack)")
    return X, y, attack, names


def load_sessions(*_args, **_kwargs):
    """SEQUENCE_MODE='rows' -- true temporal sequences. Not yet available."""
    raise NotImplementedError(
        "Temporal mode needs a sessionised dataset. NSL-KDD has no timestamp\n"
        "and no session key, so L consecutive flows cannot be recovered from\n"
        "it. To enable this, supply a CSV with:\n"
        "   - a session key (src ip, dst ip, dst port, protocol), and\n"
        "   - a timestamp to sort within the session,\n"
        "then group into windows of L flows. CIC-IDS2017 has both.\n"
        "Until then use SEQUENCE_MODE='features', which is honest about what\n"
        "the memory is integrating over (feature groups, not time).")


def angles_from_features(Z: np.ndarray, spec: ReservoirSpec, stats: dict,
                         slice_order: str = "interleave",
                         pad_mode: str = "reupload",
                         verbose: bool = False) -> np.ndarray:
    """Reduced features -> (n_samples, L, n_in, 2) angles in HALF-TURNS.

    The squash is the same smooth tanh we settled on for the photonic run:
    monotonic, no clipping, no wraparound, outliers saturate gracefully.
        latitude  theta in (0, 1) half-turns = (0, pi)
        longitude phi   in (0, 2) half-turns = (0, 2pi)

    The divisor is spec.in_gain (was a hardcoded 3.0). It is the input
    scaling of classical reservoir computing and the cheapest lever on the
    shot-noise floor -- see the in_gain note on ReservoirSpec. Smaller
    in_gain = bigger angle excursions = more signal per shot, until tanh
    saturates.

    WHAT TO DO WHEN THERE ARE FEWER FEATURES THAN SLOTS
    --------------------------------------------------
    The architecture asks for 2*n_in*L angles. NF-UNSW-NB15 supplies about
    24 features, so most architectures ask for far more slots than the data
    has columns. This is the opposite of the NSL-KDD situation the project
    was designed around (122 features, where the ring was the only thing
    that could ingest them all) and it bit us hard:

        8q  L=8 : 64 slots, ~24 real  ->  40 slots (62%) were a CONSTANT
        20q L=16: 256 slots, 19 real  -> 237 slots (93%) were a CONSTANT

    pad_mode = "constant"  (the old behaviour, kept only for reproducing
        the runs of 2026-10-10) fills the surplus with 0.5. Those slots
        carry no information, so the gates that process them are wasted and
        the reservoir spends most of its depth evolving a fixed input.

    pad_mode = "reupload"  (default) tiles the real features instead, with a
        FROZEN random permutation per repetition. Two established ideas:
          * data re-uploading (Perez-Salinas et al. 2020) -- presenting x
            again at a later layer strictly increases the expressible
            function class, because reservoir evolution happens in between,
            so copy r is acted on by a different unitary than copy 0;
          * the input mask W_in of classical reservoir computing -- a fixed
            random routing of inputs to nodes is the standard choice, not an
            approximation to something better.
        The permutation is derived from spec.seed, so it is part of the
        frozen reservoir and reproduces exactly.
        Binh's photonic sweep already measured this working: the R2
        re-uploading arm nearly matched L=16 at half the depth.
    """
    Zs = (Z - stats["mu"]) / (spec.in_gain * stats["sd"])
    sq = 0.5 * (1.0 + np.tanh(Zs))              # -> (0, 1)

    n, d = sq.shape
    need = spec.n_features_ingested
    if d < need:
        if pad_mode == "constant":
            sq = np.hstack([sq, np.full((n, need - d), 0.5, dtype=sq.dtype)])
        elif pad_mode == "reupload":
            prng = np.random.default_rng(spec.seed + 0x5EED)
            blocks = [sq]
            while sum(b.shape[1] for b in blocks) < need:
                blocks.append(sq[:, prng.permutation(d)])
            sq = np.hstack(blocks)
            if verbose:
                print(f"[encode] re-uploading: {d} real features tiled "
                      f"{len(blocks)}x into {need} slots "
                      f"(0 constant padding)")
        else:
            raise ValueError(f"unknown pad_mode {pad_mode!r}")
    elif verbose:
        print(f"[encode] {d} real features >= {need} slots; "
              f"using the leading {need}, no padding needed")
    sq = sq[:, :need]

    # Which component goes to which timestep. PCA orders components by
    # variance, so this choice decides whether the strongest signal is
    # injected early (more reservoir steps act on it, but it also decays
    # more) or late (arrives fresh at readout). Since we read out EVERY
    # timestep, interleaving keeps all timesteps informative. Cheap ablation.
    idx = np.arange(need)
    if slice_order == "variance_first":
        pass                                    # strongest -> step 0
    elif slice_order == "variance_last":
        idx = idx[::-1]                         # strongest -> step L-1
    elif slice_order == "interleave":
        idx = idx.reshape(spec.n_steps, -1).T.ravel()
    else:
        raise ValueError(f"unknown slice_order {slice_order!r}")

    a = sq[:, idx].reshape(n, spec.n_steps, spec.n_in, 2).copy()
    a[..., 1] *= 2.0                            # longitude spans 0..2pi
    return a.astype(np.float64)


def fit_squash_stats(Z_train: np.ndarray) -> dict:
    return {"mu": Z_train.mean(0), "sd": Z_train.std(0) + 1e-8}


def report_variance(pca: PCA, spec: ReservoirSpec) -> None:
    """Explained variance, and whether the architecture is slot-starved or
    slot-rich on this dataset.

    The 38.96%-at-20-components figure quoted elsewhere in this project is
    an NSL-KDD measurement (122 features, 84 of them one-hot dummies holding
    69.4% of the variance, which PCA cannot compress). It does NOT describe
    NF-UNSW-NB15, which has ~24 features in total. Which regime we are in
    decides what the architecture is even for:

        slots < available  -> the architecture is the bottleneck; more
                              qubits or more steps buy real information.
        slots > available  -> the DATA is the bottleneck; more qubits buy
                              more depth on the same numbers. That can still
                              help (extra nonlinearity) but it is not
                              "ingesting more of the vector" and must not be
                              presented as such.
    """
    ev = np.cumsum(pca.explained_variance_ratio_)
    avail = len(ev)
    slots = spec.n_features_ingested
    used = min(avail, slots)
    marks = sorted({k for k in (4, 8, 12, 20, 32, 64, 122, avail, used)
                    if 1 <= k <= avail})
    print("[pca] cumulative explained variance:")
    for k in marks:
        star = "  <-- all of it" if k == avail else ""
        print(f"        k={k:>4d}: {ev[k - 1]:6.2%}{star}")
    print(f"[pca] {avail} components available, architecture offers {slots} "
          f"encoding slots")
    if slots > avail:
        print(f"[pca] SLOT-RICH: {slots - avail} surplus slots "
              f"({(slots - avail) / slots:.0%}). With pad_mode='reupload' the "
              f"{avail} real features are tiled {slots / avail:.1f}x; with "
              f"pad_mode='constant' those slots would be dead.")
    else:
        print(f"[pca] SLOT-STARVED: ingesting {used} of {avail} components "
              f"-> {ev[used - 1]:.2%} of variance. More qubits or steps would "
              f"buy real information here.")


# ======================================================================
# 5) Backends
# ======================================================================
# "numpy"  exact local statevector simulator with trajectory sampling.
#          Zero cost, no credentials. Practical for 8-12 qubits; slow at 20.
#          Shot noise is present and is the ONLY noise present.
# "exact"  density-matrix propagation: the same circuit with shot noise
#          removed entirely (S -> infinity). Zero cost. Costs 4^N memory
#          instead of 2^N, so 6-10 qubits only. This is the CEILING that
#          perfect hardware at infinite shots would reach -- see section 5c.
# "syntax" Quantinuum syntax checker. Validates the compiled circuit against
#          the real target and returns no results. Costs no HQC.
# "nexus"  real submission through qnexus to an emulator or hardware.

def run_numpy(spec: ReservoirSpec, angles: np.ndarray, n_shots: int,
              seed: int = 0, shot_batch: int | None = None,
              verbose: bool = True):
    """Exact simulation of the measure-and-reset reservoir.

    Returns bits of shape (n_samples, n_shots, n_steps*n_in + n_qubits),
    the same layout as the pytket shot table.

    Mid-circuit measurement collapses the state, so there is no way to get
    exact expectation values from a single wavefunction -- we sample
    trajectories, one per shot, exactly as the hardware does. Shots are
    batched as a leading array axis so one numpy op advances them all.

    Timing guidance (measured scaling, not a promise): 8 qubits is seconds
    per hundred samples; 20 qubits is roughly 2^12 times the state vector
    and wants the Quantinuum emulator or Selene instead.
    """
    N = spec.n_qubits
    dim = 1 << N
    n_samples = angles.shape[0]
    L, n_in = spec.n_steps, spec.n_in
    if shot_batch is None:
        shot_batch = n_shots if N <= 12 else max(1, min(n_shots, 8))
    rng = np.random.default_rng(seed)
    in_sites = spec.in_qubits

    # ---- frozen layers, precomputed once -----------------------------
    # The whole ZZ layer is diagonal, so it collapses to ONE elementwise
    # multiply by a length-2^N vector. This is why the simulator is viable.
    basis = np.arange(dim, dtype=np.int64)
    sz = [(1 - 2 * ((basis >> q) & 1)).astype(np.int8) for q in range(N)]
    del basis
    phase = np.zeros(dim)
    for (i, j, a) in spec.zz_halfturns():
        phase += (np.pi * a / 2.0) * (sz[i].astype(np.int16) * sz[j])
    zz_diag = np.exp(-1j * phase)
    del phase, sz

    field_mats = [_phasedx(a, 0.0) for a in spec.x_halfturns()]
    rot_mats = basis_rot_mats(spec)
    mem_mats = [(q, _phasedx(float(spec.mem_init[k]), 0.5))
                for k, q in enumerate(spec.mem_qubits)]

    out = np.zeros((n_samples, n_shots, L * n_in + N), dtype=np.int8)
    t0 = time.time()
    for s in range(n_samples):
        done = 0
        while done < n_shots:
            B = min(shot_batch, n_shots - done)
            psi = np.zeros((B, dim), dtype=np.complex128)
            psi[:, 0] = 1.0
            for (q, M) in mem_mats:
                psi = _apply_1q(psi, M, q, N)

            for t in range(L):
                for k, q in enumerate(in_sites):
                    th, ph = angles[s, t, k]
                    psi = _apply_1q(psi, _phasedx(th, 0.5), q, N)   # latitude
                    psi = _apply_1q(psi, _rz(ph), q, N)             # longitude
                psi *= zz_diag                                      # ZZ layer
                for q in range(N):
                    psi = _apply_1q(psi, field_mats[q], q, N)       # X layer
                for k, q in enumerate(in_sites):
                    psi, bit = _measure_reset(psi, q, N, rng)
                    out[s, done:done + B, t * n_in + k] = bit

            # final readout: all qubits, no reset needed.
            # FIELD -> ZZ -> FIELD, mirroring reservoir_step(field_first=True).
            # The leading field layer is what makes this work: the input
            # qubits were just reset to |0> and the diagonal zz_diag multiply
            # cannot touch a Z eigenstate.
            if spec.final_mix:
                for q in range(N):
                    psi = _apply_1q(psi, field_mats[q], q, N)
                psi *= zz_diag
                for q in range(N):
                    psi = _apply_1q(psi, field_mats[q], q, N)
            # Rotate into the per-qubit readout basis (identity when
            # final_basis='z', which is the historical behaviour).
            for q in range(N):
                if rot_mats[q] is not None:
                    psi = _apply_1q(psi, rot_mats[q], q, N)
            for q in range(N):
                psi, bit = _measure_reset(psi, q, N, rng)
                out[s, done:done + B, L * n_in + q] = bit
            done += B
        if verbose and (s + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  [numpy] {s + 1}/{n_samples} samples, {el:.1f}s "
                  f"({el / (s + 1):.2f}s/sample)")
    return out


def _phasedx(alpha: float, beta: float) -> np.ndarray:
    """PhasedX(alpha, beta) = Rz(beta) . Rx(alpha) . Rz(-beta), half-turns."""
    a, b = np.pi * alpha, np.pi * beta
    rx = np.array([[np.cos(a / 2), -1j * np.sin(a / 2)],
                   [-1j * np.sin(a / 2), np.cos(a / 2)]], dtype=np.complex128)
    rz_p = np.diag([np.exp(-1j * b / 2), np.exp(1j * b / 2)])
    rz_m = np.diag([np.exp(1j * b / 2), np.exp(-1j * b / 2)])
    return rz_p @ rx @ rz_m


def _rz(alpha: float) -> np.ndarray:
    a = np.pi * alpha
    return np.diag([np.exp(-1j * a / 2), np.exp(1j * a / 2)]).astype(np.complex128)


def _apply_1q(psi: np.ndarray, U: np.ndarray, q: int, N: int) -> np.ndarray:
    """Apply a 2x2 gate to qubit q. psi is (B, 2**N); bit q is bit q of the
    state index (qubit 0 = least significant)."""
    B = psi.shape[0]
    v = psi.reshape(B, 1 << (N - 1 - q), 2, 1 << q)
    a = v[:, :, 0, :].copy()
    b = v[:, :, 1, :]
    v[:, :, 0, :] = U[0, 0] * a + U[0, 1] * b
    v[:, :, 1, :] = U[1, 0] * a + U[1, 1] * b
    return v.reshape(B, dim_of(N))


def dim_of(N: int) -> int:
    return 1 << N


def _measure_reset(psi: np.ndarray, q: int, N: int, rng):
    """Measure qubit q, collapse, and reset it to |0>. Vectorised over shots."""
    B = psi.shape[0]
    v = psi.reshape(B, 1 << (N - 1 - q), 2, 1 << q)
    p1 = np.einsum("ihl,ihl->i", np.abs(v[:, :, 1, :]), np.abs(v[:, :, 1, :]))
    p1 = np.clip(p1, 0.0, 1.0)
    bit = (rng.random(B) < p1).astype(np.int8)
    keep = np.where(bit[:, None, None].astype(bool), v[:, :, 1, :], v[:, :, 0, :])
    norm = np.sqrt(np.where(bit.astype(bool), p1, 1.0 - p1))
    norm = np.maximum(norm, 1e-300)[:, None, None]
    v[:, :, 0, :] = keep / norm
    v[:, :, 1, :] = 0.0
    return v.reshape(B, dim_of(N)), bit


# ======================================================================
# 5b) Basis rotation for the final snapshot
# ======================================================================
# Measuring <X> or <Y> instead of <Z> costs ONE extra 1q gate per qubit --
# no two-qubit gates, no extra shots. Every run of this project before
# 2026-10-12 measured Z and only Z, discarding two thirds of the available
# single-qubit information for free.
#
# To read observable A by measuring Z we need a U with U^dag Z U = A:
#       X  <-  U = Ry(-pi/2) = PhasedX(-0.5, 0.5)
#       Y  <-  U = Rx(+pi/2) = PhasedX(+0.5, 0.0)
#       Z  <-  U = I         (no gate)
# (verified numerically in the self-test at the bottom of this module).
#
# Only the FINAL snapshot may be rotated. The mid-circuit measurements are
# the measure-and-reset channel that drives the reservoir; rotating those
# would change the dynamics rather than the readout.
_BASIS_ROT = {"z": None, "x": (-0.5, 0.5), "y": (0.5, 0.0)}


def basis_rot_mats(spec: ReservoirSpec):
    """One 2x2 matrix (or None) per qubit for the final snapshot."""
    out = []
    for ax in spec.final_axes:
        r = _BASIS_ROT[ax]
        out.append(None if r is None else _phasedx(r[0], r[1]))
    return out


# ======================================================================
# 5c) The EXACT backend -- shot noise removed, nothing else changed
# ======================================================================
# This answers the one question the whole project forks on: are the results
# limited by MEASUREMENT or by the MODEL?
#
#     real hardware  <=  noiseless + shot noise  <=  noiseless + inf shots
#                        (the "numpy" backend)      (THIS backend)
#
# Everything measured so far is the middle term, on an exact statevector
# with sampled trajectories. Improving hardware fidelity moves the left
# term up toward the middle one; it can never exceed it. So if this
# backend's ceiling also loses to the classical baseline, no shot count and
# no hardware improvement can ever produce a win on this dataset, and the
# encoding or the readout has to change. If the ceiling is far above it, the
# architecture is shot-limited and the shot-scaling curve becomes a real,
# quantitative, forward-looking claim.
#
# WHY A STATEVECTOR CANNOT DO THIS. run_numpy's docstring is right: with
# mid-circuit measurement and reset the circuit is a quantum CHANNEL, not a
# unitary, so a single wavefunction cannot carry the exact expectations.
# But that only rules out the statevector route. Propagating the DENSITY
# MATRIX gives the S -> infinity values exactly:
#
#   unitary layer     rho -> U rho U^dag
#   diagonal ZZ layer rho_ij -> zz_i * conj(zz_j) * rho_ij   (elementwise)
#   measure + reset   rho -> sum_b |0><b| rho |b><0|, which keeps the two
#                     diagonal blocks and DISCARDS the coherences -- the
#                     decoherence the measurement actually causes
#   readout           <Z_q> = Tr(rho Z_q) = sum_i sz_q[i] * rho_ii
#
# Z-basis measurements commute, so every <Z_q> and every same-step
# <Z_i Z_j> can be read off one diagonal before any qubit is reset.
#
# COST is 4^N rather than 2^N: 6 qubits is a 64x64 matrix, 10 qubits is
# 1024x1024 (~17 MB). Measured on this machine, 850 samples: seconds at 6
# qubits, about a minute at 8, roughly 20 minutes at 10. Run it at 6q/8q
# first -- the ceiling is what matters, not which cell you read it on.

def run_exact(spec: ReservoirSpec, angles: np.ndarray,
              correlators: bool = False, verbose: bool = True):
    """Exact (infinite-shot) features. Returns (n_samples, n_features).

    The column layout is byte-for-byte the one features_from_bits produces,
    so every downstream arm -- readout, noise report, novelty detector --
    works unchanged.
    """
    N = spec.n_qubits
    dim = 1 << N
    L, n_in = spec.n_steps, spec.n_in
    in_sites = spec.in_qubits
    n_samples = angles.shape[0]
    if spec.J is None:
        raise ValueError("spec.freeze() first")

    basis = np.arange(dim, dtype=np.int64)
    sz = np.stack([(1 - 2 * ((basis >> q) & 1)).astype(np.float64)
                   for q in range(N)])                 # (N, dim)
    del basis
    phase = np.zeros(dim)
    for (i, j, a) in spec.zz_halfturns():
        phase += (np.pi * a / 2.0) * (sz[i] * sz[j])
    zz_vec = np.exp(-1j * phase)
    zz_dm = np.outer(zz_vec, zz_vec.conj())
    del phase, zz_vec

    field_mats = [_phasedx(a, 0.0) for a in spec.x_halfturns()]
    rot_mats = basis_rot_mats(spec)
    pairs = [(i, j) for i in range(n_in) for j in range(i + 1, n_in)]
    n_feat = L * n_in + N + (L * len(pairs) if correlators else 0)
    out = np.zeros((n_samples, n_feat), dtype=np.float64)

    t0 = time.time()
    for smp in range(n_samples):
        rho = np.zeros((dim, dim), dtype=np.complex128)
        rho[0, 0] = 1.0
        for k, q in enumerate(spec.mem_qubits):
            rho = _dm_1q(rho, _phasedx(float(spec.mem_init[k]), 0.5), q, N)

        cpos = L * n_in + N
        for t in range(L):
            for k, q in enumerate(in_sites):
                th, ph = angles[smp, t, k]
                rho = _dm_1q(rho, _phasedx(th, 0.5), q, N)   # latitude
                rho = _dm_1q(rho, _rz(ph), q, N)             # longitude
            rho *= zz_dm
            for q in range(N):
                rho = _dm_1q(rho, field_mats[q], q, N)
            # One diagonal serves every observable at this step.
            diag = np.real(np.diagonal(rho)).copy()
            for k, q in enumerate(in_sites):
                out[smp, t * n_in + k] = sz[q] @ diag
            if correlators:
                for (i, j) in pairs:
                    out[smp, cpos] = (sz[in_sites[i]] * sz[in_sites[j]]) @ diag
                    cpos += 1
            for q in in_sites:
                rho = _dm_measure_reset(rho, q, N)

        if spec.final_mix:
            for q in range(N):
                rho = _dm_1q(rho, field_mats[q], q, N)
            rho *= zz_dm
            for q in range(N):
                rho = _dm_1q(rho, field_mats[q], q, N)
        for q in range(N):
            if rot_mats[q] is not None:
                rho = _dm_1q(rho, rot_mats[q], q, N)
        diag = np.real(np.diagonal(rho)).copy()
        for q in range(N):
            out[smp, L * n_in + q] = sz[q] @ diag

        if verbose and (smp + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  [exact] {smp + 1}/{n_samples} samples, {el:.1f}s "
                  f"({el / (smp + 1):.2f}s/sample)")
    return out.astype(np.float32)


def _dm_1q(rho: np.ndarray, U: np.ndarray, q: int, N: int) -> np.ndarray:
    """rho -> U_q rho U_q^dag, in place on rho's buffer.

    (U rho U^dag)_ij = sum_kl U_ik rho_kl conj(U_jl), so the column pass is
    the same contraction as the row pass with U replaced by conj(U).
    """
    dim = 1 << N
    hi, lo = 1 << (N - 1 - q), 1 << q
    v = rho.reshape(hi, 2, lo, dim)
    a = v[:, 0, :, :].copy()
    b = v[:, 1, :, :].copy()
    v[:, 0, :, :] = U[0, 0] * a + U[0, 1] * b
    v[:, 1, :, :] = U[1, 0] * a + U[1, 1] * b
    w = v.reshape(dim, dim).reshape(dim, hi, 2, lo)
    Uc = U.conj()
    a = w[:, :, 0, :].copy()
    b = w[:, :, 1, :].copy()
    w[:, :, 0, :] = Uc[0, 0] * a + Uc[0, 1] * b
    w[:, :, 1, :] = Uc[1, 0] * a + Uc[1, 1] * b
    return w.reshape(dim, dim)


def _dm_measure_reset(rho: np.ndarray, q: int, N: int) -> np.ndarray:
    """Z-basis measurement of qubit q followed by reset to |0>.

    rho -> sum_b |0><b| rho |b><0| : the two diagonal blocks in the q index
    are summed into the |0><0| block and the off-diagonal blocks are
    dropped. Discarding those coherences IS the measurement's decoherence,
    which is why this is a channel and not a unitary.
    """
    dim = 1 << N
    hi, lo = 1 << (N - 1 - q), 1 << q
    r = rho.reshape(hi, 2, lo, hi, 2, lo)
    keep = r[:, 0, :, :, 0, :] + r[:, 1, :, :, 1, :]
    out = np.zeros_like(r)
    out[:, 0, :, :, 0, :] = keep
    return out.reshape(dim, dim)


def run_nexus(spec: ReservoirSpec, angles: np.ndarray, n_shots: int,
              device: str = "H1-1E", project: str = "qrc-iontrap",
              confirm_hardware: bool = False, syntax_only: bool = False,
              verbose: bool = True):
    """Submit through qnexus. Compile the template ONCE, then substitute.

    IMPORTANT: the qnexus API moves fast. If an attribute below does not
    exist in your installed version, check the current docs at
    https://docs.quantinuum.com/nexus/ -- the shape of the flow (config,
    compile, execute, results) has been stable even when names changed.
    """
    _require_pytket()
    from pytket.extensions.quantinuum import QuantinuumBackend

    b = circuit_budget(spec)
    est = angles.shape[0] * b.hqc(n_shots)
    is_emulator = device.endswith("E") or "SC" in device or syntax_only
    if not is_emulator and not confirm_hardware:
        raise RuntimeError(
            f"Refusing to submit to real hardware {device!r}.\n"
            f"Estimated cost: {est:,.0f} HQC for {angles.shape[0]} samples "
            f"x {n_shots} shots.\nSet CONFIRM_HARDWARE = True in the arch "
            f"file if that is genuinely affordable.")
    if verbose:
        print(f"[nexus] device={device} syntax_only={syntax_only} "
              f"est_cost={est:,.0f} HQC")

    backend = QuantinuumBackend(device_name=device)
    template, mid_bits, fin_bits, syms = build_template(spec)

    # Compile the SYMBOLIC circuit once. PhasedX/Rz/ZZPhase all take
    # continuous angles, so the symbols survive into the native gates and
    # substitution afterwards needs no recompile.
    compiled = backend.get_compiled_circuit(template, optimisation_level=1)
    if verbose:
        print(f"[nexus] compiled template: {compiled.n_gates} gates, "
              f"depth {compiled.depth()}, "
              f"2q {compiled.n_gates_of_type_2q() if hasattr(compiled, 'n_gates_of_type_2q') else b.n_2q}")

    circuits = [substitute(compiled, syms, angles[i])
                for i in range(angles.shape[0])]
    for c in circuits[:1]:
        if not backend.valid_circuit(c):
            raise RuntimeError(
                "Substituted circuit is not valid for the backend. The "
                "symbolic compile+substitute trick failed; compile each "
                "circuit individually instead (much slower).")
    if syntax_only:
        print(f"[nexus] {len(circuits)} circuits validated against {device}. "
              f"Syntax check only -- no results returned, no HQC spent.")
        return None

    handles = backend.process_circuits(circuits, n_shots=n_shots)
    order = list(mid_bits) + list(fin_bits)
    shots = [backend.get_result(h).get_shots(cbits=order) for h in handles]
    return np.asarray(shots, dtype=np.int8)


# ======================================================================
# 6) Readout features from the shot table
# ======================================================================
def features_from_bits(bits: np.ndarray, spec: ReservoirSpec,
                       correlators: bool = True) -> np.ndarray:
    """(n_samples, n_shots, n_bits) -> (n_samples, n_features).

    Three families, all cheap estimators of the reservoir state:
      <Z> per (timestep, input qubit)  -- the time series the reservoir wrote
      <Z> per qubit at the end          -- the final memory snapshot
      <Z_i Z_j> within a timestep       -- pair correlations, where the
                                           genuinely quantum structure lives
    Shot noise on each is ~1/sqrt(n_shots), which is the real reason shot
    count cannot be cut arbitrarily.
    """
    L, n_in, N = spec.n_steps, spec.n_in, spec.n_qubits
    z = 1.0 - 2.0 * bits.astype(np.float64)        # bit 0 -> +1, bit 1 -> -1
    singles = z.mean(axis=1)                       # (n_samples, n_bits)
    if not correlators:
        return singles.astype(np.float32)

    cols = [singles]
    mid = z[:, :, :L * n_in].reshape(z.shape[0], z.shape[1], L, n_in)
    for t in range(L):
        for i in range(n_in):
            for j in range(i + 1, n_in):
                cols.append((mid[:, :, t, i] * mid[:, :, t, j]
                             ).mean(axis=1)[:, None])
    return np.hstack(cols).astype(np.float32)


# ----------------------------------------------------------------------
# 6b) The shot-noise floor -- the single most important diagnostic here
# ----------------------------------------------------------------------
# Every feature is an estimate of <Z> (or <Z_i Z_j>) from a finite number of
# shots. For a true value z the sampling error of that estimate is
#       sd_shot = sqrt((1 - z^2) / S)
# so the across-sample spread we actually observe is
#       observed^2  =  signal^2  +  sd_shot^2
# A feature whose `signal` is below `sd_shot` is a NOISE COLUMN: it carries
# nothing about the input, and because StandardScaler rescales it to unit
# variance regardless, a linear readout will happily fit it and overfit.
#
# This is not a theoretical worry. On the 2026-10-10 runs:
#       8q,  100 shots:   8 of  88 features cleared the floor
#                         (all 48 correlators were pure noise)
#       20q,  20 shots:  16 of 596, mean signal 2.6x BELOW the floor
# and on the 8q run, dropping the 48 correlator columns moved the supervised
# AUC from 0.9519 to 0.9897 -- past the 0.9725 classical baseline. So "the
# reservoir added nothing" was substantially a noise artifact.
#
# Shots needed, measured from those runs (3x margin over the floor):
#       8q  mid-circuit <Z>:    ~756 shots   (84 just to reach the floor)
#       8q  correlators    : ~10,033 shots
#       20q mid-circuit <Z>:  ~1,211 shots
# "Same-step correlators are free" is therefore true in GATES and false in
# SHOTS. Default them off and spend the shots on the single-qubit channels.

def shot_variance(F, n_shots) -> np.ndarray:
    """Per-column sampling variance of a <Z> estimate from n_shots shots.

    For a true value z the estimator's variance is (1 - z^2)/S. Averaged
    over samples this is the variance a column carries even when it holds
    no information at all. n_shots=None (the exact backend) means no
    sampling, hence zero.
    """
    F = np.asarray(F, dtype=np.float64)
    if n_shots is None or not np.isfinite(n_shots) or n_shots <= 0:
        return np.zeros(F.shape[1])
    return ((1.0 - np.clip(F, -1.0, 1.0) ** 2) / n_shots).mean(0)


def shot_noise_report(F, n_shots, spec: ReservoirSpec,
                      correlators: bool, label: str = "train") -> np.ndarray:
    """Print the per-family signal/floor table. Returns the boolean mask.

    Computed on TRAIN ONLY, so using the mask for feature selection leaks
    nothing from the test or zero-day sets.
    """
    L, n_in, N = spec.n_steps, spec.n_in, spec.n_qubits
    shot_var = shot_variance(F, n_shots)
    floor = np.sqrt(shot_var)
    sig = np.sqrt(np.maximum(0.0, F.std(0) ** 2 - shot_var))
    exact = not np.any(shot_var > 0)
    # With the exact backend there is no sampling error, so the only dead
    # columns are the genuinely constant ones. Keep the same mask semantics
    # so every downstream arm behaves identically.
    mask = sig > floor if not exact else F.std(0) > 1e-12
    fams = [("mid-circuit <Z>", slice(0, L * n_in)),
            ("final <Z>", slice(L * n_in, L * n_in + N))]
    if correlators:
        fams.append(("correlators", slice(L * n_in + N, F.shape[1])))
    if exact:
        print(f"[noise] EXACT backend ({label} set): no sampling error, so "
              f"the floor is 0 and a column is dead only if it is constant.")
    else:
        print(f"[noise] shot-noise floor at S={n_shots} ({label} set; a "
              f"mid-range <Z> has floor {0.808 / np.sqrt(n_shots):.4f}):")
    for nm, sl in fams:
        m = np.zeros(F.shape[1], bool)
        m[sl] = True
        if not m.any():
            continue
        ms, mf = sig[m].mean(), floor[m].mean()
        if exact:
            print(f"        {nm:<16} {int((mask & m).sum()):>4}/"
                  f"{int(m.sum()):<4} live    signal {ms:.4f}  (exact)")
            continue
        need = (9.0 * shot_var[m].mean() * n_shots / ms ** 2
                if ms > 0 else np.inf)
        print(f"        {nm:<16} {int((mask & m).sum()):>4}/{int(m.sum()):<4}"
              f" clear   signal {ms:.4f}  floor {mf:.4f}  ratio {ms / mf:5.2f}"
              f"   3x margin needs {need:>9,.0f} shots")
    print(f"        {'TOTAL':<16} {int(mask.sum()):>4}/{F.shape[1]:<4} clear")

    # Per-qubit final snapshot. This is the line that caught two separate
    # bugs (the final_mix ordering, then the near-zero transverse fields),
    # so it is printed every run rather than left to a one-off script. A
    # column reading exactly 0.00000 means that qubit never flipped in any
    # shot of any sample -- look at spec.h[q] before looking anywhere else.
    fs = F[:, L * n_in:L * n_in + N]
    inp = set(spec.in_qubits)
    print("        final snapshot, std per qubit "
          "(i = input/reset, m = memory):")
    cells = [f"q{q}{'i' if q in inp else 'm'}={fs[:, q].std():.4f}"
             for q in range(N)]
    for k in range(0, len(cells), 5):
        print("          " + "  ".join(cells[k:k + 5]))
    flat = [q for q in range(N) if fs[:, q].std() < 1e-9]
    if flat:
        print(f"          !! qubits {flat} are EXACTLY constant. Check "
              f"spec.h for near-zero fields (field_draw='annulus' fixes it).")

    if mask.sum() < max(4, F.shape[1] // 10):
        print("        !! almost everything is noise. Raise n_shots or turn "
              "correlators off before reading any metric below.")
    return mask


# ======================================================================
# 7) Classical readout -- a SINGLE linear layer, nothing deeper
# ======================================================================
# This is the only trained part of the whole system, and it must stay
# linear: if the readout were a deep net we could not claim the reservoir
# did the work. Note that the regularisation strength has to be retuned
# whenever the feature dimension changes -- that is the thing the team
# called "the weighting", and it is the likeliest cause of a result getting
# WORSE when the feature count grows.
def fit_readout(F_tr, y_tr, F_te, y_te, seed=42, verbose=True, n_folds=5):
    """One linear layer. C chosen by CV on TRAIN, then scored ONCE on TEST.

    FIXED 2026-10-12. This used to select C by maximising the TEST AUC:

        for C in (0.003, ..., 3.0):
            auc = roc_auc_score(y_te, p)        # <-- test set
            if best is None or auc > best[0]: ...

    Both arms got the same treatment, so it was not leakage in the usual
    sense -- but it was not symmetric either. Taking the max over 7
    candidates inflates the reported AUC, and inflates it MORE for the arm
    with more features to vary over: the quantum arms carry 24-50 features
    against the classical baseline's 24. The expected differential bias is
    the same order as the +0.0006..+0.0105 supervised margin reported by
    the 2026-10-11 sweep, so that margin could not be defended. Selection
    now happens entirely inside the training set and the test set is
    touched exactly once.
    """
    sc = StandardScaler().fit(F_tr)
    A, B = sc.transform(F_tr), sc.transform(F_te)
    grid = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
    yt = np.asarray(y_tr).astype(int)
    k = max(2, min(n_folds, int(np.bincount(yt).min())))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = list(cv.split(A, yt))
    cv_auc = []
    for C in grid:
        sc_fold = []
        for itr, iva in folds:
            clf = LogisticRegression(C=C, max_iter=3000, random_state=seed)
            clf.fit(A[itr], yt[itr])
            sc_fold.append(roc_auc_score(
                yt[iva], clf.predict_proba(A[iva])[:, 1]))
        cv_auc.append(float(np.mean(sc_fold)))
    i = int(np.argmax(cv_auc))
    C = grid[i]
    clf = LogisticRegression(C=C, max_iter=3000, random_state=seed)
    clf.fit(A, yt)
    auc = float(roc_auc_score(y_te, clf.predict_proba(B)[:, 1]))
    acc = float(accuracy_score(y_te, clf.predict(B)))
    if verbose:
        print(f"[readout] C={C} by {k}-fold CV on train (cv AUC "
              f"{cv_auc[i]:.4f})  ->  TEST AUC={auc:.4f}  acc={acc:.4f}   "
              f"({F_tr.shape[1]} features, {F_tr.shape[0]} train rows)")
    return {"auc": auc, "C": C, "acc": acc, "cv_auc": cv_auc[i],
            "clf": clf, "scaler": sc}


class NoveltyDetector:
    """Trained on NORMAL TRAFFIC ONLY, so no attack can ever be 'unseen'.

    Mahalanobis distance inside the normal subspace plus reconstruction
    energy outside it. Same evaluation the photonic features went through,
    so the two codebases stay comparable.

    FIXED 2026-10-12 -- THIS WAS TILTING THE ZERO-DAY COMPARISON AGAINST
    THE QUANTUM ARM, and it is the leading explanation for the 2026-10-11
    sweep losing that metric in all six cells (best quantum 0.7969 against
    a flat classical 0.8009).

    The old code did:

        self.sd = P.std(0) + 1e-8
        maha = np.sqrt(((P / self.sd) ** 2).sum(1))

    Mahalanobis normalises every PCA direction by its own spread, which is
    the right thing to do when every direction carries signal. A quantum
    feature matrix is not like that: each column is a <Z> estimated from S
    shots, so it has an irreducible noise floor of sqrt((1-z^2)/S). A PCA
    direction built mostly from that noise has a tiny sd, and dividing by
    it PROMOTES THE NOISE TO EQUAL WEIGHT WITH REAL SIGNAL. The +1e-8 floor
    was eight orders of magnitude below the actual shot noise (~0.04), so
    it protected against nothing. The classical baseline's PCA components
    have no shot floor, so it never paid this tax.

    The evidence is in the sweep's own numbers. k was min(32, n_features),
    so the small cells got NO truncation at all. The gain from dropping
    sub-floor columns tracks that exactly:

        cell          feats  k     full rank?   denoise gain (zero-day AUC)
        6q_a2a_L6     24     24    yes          +0.0513
        8q_a2a_L6     32     32    yes          +0.0156
        6q_a2a_L8     30     30    yes          +0.0117
        8q_a2a_L8     40     32    no           +0.0060
        10q_a2a_L6    40     32    no           +0.0049
        10q_a2a_L8    50     32    no           -0.0011

    Perfect separation across six cells: where the detector kept every
    direction, removing noise columns was worth up to +0.05 AUC; where it
    already truncated at 32, it was worth nothing, because the truncation
    was doing the denoising.

    Two knobs:
      1. `shot_var` lets the caller pass the per-FEATURE sampling variance.
         It is projected onto each component exactly -- for unit-norm
         component v the noise variance along it is sum_j v_j^2 * var_j --
         then used both to FLOOR sd and to DROP any component whose spread
         does not exceed `margin` times its own projected noise.
      2. `var_keep` truncates the Mahalanobis to the leading components
         holding that fraction of the normal-traffic variance. DEFAULT 1.0
         = no truncation. Read the warning below before changing it.

    Pass shot_var=None (or zeros) for noise-free features -- the classical
    PCA baseline, or the exact backend.

    HOW MUCH THE SHOT-NOISE FIX ACTUALLY BUYS: ALMOST NOTHING, MEASURED.
    ---------------------------------------------------------------------
    The 2026-10-12 re-run of all six cells, re-scored from the saved
    features under every combination (scratchpad/isolate.py), mean zero-day
    AUC across the six cells:

        neither                          0.7740     <- old behaviour
        shot-noise fix only              0.7767     (+0.0027)
        var_keep=0.99 only               0.7807     (+0.0067)
        both                             0.7807

    and every run log says `0 of those dropped as shot-noise-dominated`,
    in all 18 arm/cell combinations. At 512 shots NO PCA direction falls
    below its own projected shot noise, so the DROP half of this fix never
    fires at all and only the sd flooring does anything.

    The mechanism in the comment above is real, but it is NOT ACTIVE at
    this shot count, and it was NOT the explanation for the zero-day loss.
    The six-cell correlation that suggested it (denoise gain tracking
    whether k truncated) was real but came from dropping raw FEATURE
    columns via drop_noise_floor -- which does fire, 20/24, 26/30, ... --
    not from PCA-direction amplification. Those are two different things
    and conflating them produced a wrong diagnosis. Keep the fix: it is
    correct, principled and costs nothing. Do not expect it to move a
    headline number, and do not cite it as the reason for anything.

    WHY k DEFAULTS TO None (FULL RANK) -- SAME TRAP, FOUND LATER.
    -------------------------------------------------------------
    run_experiment used to build every detector with k=min(32, n_columns).
    The classical baseline has 24 PCA dims so it was ALWAYS full rank,
    while the quantum arms have 20-45 live columns -- so the cap bit the
    three largest cells and left the three smallest alone. Measured on the
    2026-10-12 features, quantum zero-day AUC:

        cell          live  k=32     full rank   delta
        6q_a2a_L6       20  0.7839   0.7839      +0.0000
        6q_a2a_L8       26  0.7888   0.7888      +0.0000
        8q_a2a_L6       26  0.7610   0.7610      +0.0000
        8q_a2a_L8       34  0.7752   0.7830      +0.0078
        10q_a2a_L6      35  0.7744   0.7875      +0.0131
        10q_a2a_L8      45  0.7767   0.7781      +0.0014
        mean                0.7767   0.7804      +0.0037

    The verdict is unchanged (-0.0242 -> -0.0205 against a classical
    0.8009), but the N axis was contaminated: 10q_a2a_L6 was being held
    back relative to the 6q cells purely by the cap, and it moves from 4th
    place to 2nd once the cap is gone. "The smallest cell is the best cell"
    survives only by 0.0013 after this, not 0.0144.

    Set k explicitly ONLY to study the effect of rank, never to make the
    arms "comparable" -- equal k on unequal column counts is not a fair
    comparison, it is a different handicap.

    WHY var_keep DEFAULTS TO 1.0 -- DO NOT CHANGE THIS CASUALLY.
    -----------------------------------------------------------
    Truncation applies to BOTH arms, and it does not affect them equally:

        zero-day AUC        var_keep=1.0    var_keep=0.99    delta
        quantum (mean)          0.7767          0.7807      +0.0040
        classical               0.8009          0.7559      -0.0450

    The classical baseline is PCA on real features with NO measurement
    noise, so every direction carries genuine variance -- cutting at 99%
    threw away real signal and dropped it from 24 components to 9. The
    quantum arm gains a little because its tail directions are noisier.
    Net effect on the comparison: a +0.0248 "quantum beats classical"
    result that is ENTIRELY an artifact of handicapping the baseline.
    Like for like at var_keep=1.0 the quantum arm still LOSES by -0.0242,
    which is barely different from the -0.0269 measured before any of this.
    """

    def __init__(self, k: int | None = None, seed: int = 42,
                 var_keep: float = 1.0, margin: float = 1.0):
        # k=None means FULL RANK. See the k section of the class docstring:
        # a fixed cap truncates the wide arms and not the narrow ones, which
        # is an uncontrolled asymmetry in exactly the comparison this class
        # exists to make. Any truncation should come from var_keep or from
        # the shot-noise margin, both of which apply the same RULE to every
        # arm even though they remove different numbers of components.
        self.k, self.seed = k, seed
        self.var_keep, self.margin = var_keep, margin

    def fit(self, X_normal, shot_var=None, verbose: bool = False,
            label: str = ""):
        X_normal = np.asarray(X_normal, dtype=np.float64)
        self.mu = X_normal.mean(0)
        Z = X_normal - self.mu
        k_req = Z.shape[1] if self.k is None else self.k
        kmax = max(1, min(k_req, Z.shape[1], Z.shape[0] - 1))
        self.pca = PCA(n_components=kmax, svd_solver="randomized",
                       random_state=self.seed).fit(Z)
        P = self.pca.transform(Z)
        sd = P.std(0)

        # (1) variance truncation
        ev = np.cumsum(self.pca.explained_variance_ratio_)
        n_var = int(np.searchsorted(ev, self.var_keep) + 1)
        n_var = max(2, min(kmax, n_var))

        # (2) exact projection of per-feature shot noise onto each component
        if shot_var is None:
            noise = np.zeros(kmax)
        else:
            sv = np.asarray(shot_var, dtype=np.float64).ravel()
            if sv.shape[0] != Z.shape[1]:
                raise ValueError(
                    f"shot_var has {sv.shape[0]} entries but the feature "
                    f"matrix has {Z.shape[1]} columns")
            noise = np.sqrt((self.pca.components_ ** 2) @ sv)

        keep = np.zeros(kmax, bool)
        keep[:n_var] = True
        keep &= sd > self.margin * noise
        if keep.sum() < 2:                  # never strip the whole model
            keep = np.zeros(kmax, bool)
            keep[:min(2, kmax)] = True
        self.keep = keep
        # Floor at the projected shot noise AND at a small absolute value.
        # The absolute part is not cosmetic: the pre-2026-10-12 code used
        # `P.std(0) + 1e-8`, and for a near-null direction that 1e-8 WAS
        # the floor. Dropping it to 1e-12 made the amplification worse, not
        # better. Keeping 1e-8 means a noise-free arm (shot_var=None) is
        # numerically equivalent to the old behaviour, so the classical
        # baseline is unchanged and the comparison stays like-for-like.
        self.sd = np.maximum(np.maximum(sd, noise), 1e-8)

        # The residual term stays against the FULL fitted subspace: its job
        # is to catch structure OUTSIDE the normal model, and shrinking the
        # subspace would just move noise into it.
        r = np.sqrt(((Z - self.pca.inverse_transform(P)) ** 2).sum(1))
        self.rmu, self.rsd = r.mean(), r.std() + 1e-8
        if verbose:
            nf = int((sd <= self.margin * noise)[:n_var].sum())
            print(f"[novelty] {label}{'  ' if label else ''}"
                  f"{Z.shape[1]} features -> {kmax} components, "
                  f"{int(keep.sum())} used in the Mahalanobis "
                  f"({n_var} by {self.var_keep:.0%} variance, "
                  f"{nf} of those dropped as shot-noise-dominated)")
        return self

    def score(self, X):
        Z = np.asarray(X, dtype=np.float64) - self.mu
        P = self.pca.transform(Z)
        maha = np.sqrt(((P[:, self.keep] / self.sd[self.keep]) ** 2).sum(1))
        r = np.sqrt(((Z - self.pca.inverse_transform(P)) ** 2).sum(1))
        return maha + np.maximum((r - self.rmu) / self.rsd, 0)


FA_BUDGETS = [0.01, 0.05]


def detection_stats(s_normal, s_attack) -> dict:
    """AUC plus detection rate at a false-alarm budget.

    A fixed 0.5 cutoff is meaningless for intrusion detection -- real
    systems pick a threshold from a false-alarm budget ("flag the top 1%").
    """
    y = np.r_[np.zeros(len(s_normal)), np.ones(len(s_attack))]
    out = {"AUC": float(roc_auc_score(y, np.r_[s_normal, s_attack]))}
    for fa in FA_BUDGETS:
        thr = np.quantile(s_normal, 1 - fa)
        out[f"@{fa:.0%}FA"] = float((s_attack > thr).mean())
    return out


# ======================================================================
# 8) The experiment driver, shared by both arch files
# ======================================================================
def _stratified_subsample(idx, labels, size, rng, balance: str = "balanced"):
    """Pick `size` indices from `idx`, keeping every class present.

    NF-UNSW-NB15 is ~95.5% benign, so a plain random subsample of a few
    hundred flows can contain zero attacks, which makes the supervised
    logistic readout ill-posed (only one class) and raises inside sklearn.

    balance = "proportional"
        Preserve the dataset's own class ratio, flooring each class at 1.
        This is what the 2026-10-10 runs used, and it is WHY those runs
        cannot be interpreted. Measured consequence:
            n_test = 100  ->  97 benign /  3 attack
            n_test =  20  ->  19 benign /  1 attack
        An AUC whose denominator is 1 or 3 positives is one or three
        pairwise comparisons. supervised_classical = 1.0 on the 20-qubit
        run was a single row that happened to rank first, not a result.

    balance = "balanced"  (default)
        Split the budget evenly across classes, spilling any shortfall of a
        small class into the larger one. ROC-AUC is a rank statistic over
        (positive, negative) pairs, so it is NOT biased by the evaluation
        ratio -- balancing only shrinks its variance, which is exactly what
        we need when every extra row costs a circuit. Detection-at-fixed-FA
        benefits on both sides: more positives for the numerator, enough
        negatives to place the threshold quantile.
        Deployment prevalence is a property of the deployment, not of the
        measurement. If you want precision at the true 4.5% base rate,
        reweight afterwards rather than starving the test set.
    """
    idx = np.asarray(idx)
    if size >= len(idx):
        return idx.copy()
    lab = np.asarray(labels)[idx]
    classes = np.unique(lab)
    avail = np.array([int((lab == c).sum()) for c in classes])
    if balance == "balanced":
        alloc = np.minimum(size // len(classes), avail)
    elif balance == "proportional":
        prop = avail / avail.sum()
        alloc = np.maximum(1, np.round(prop * size).astype(int))
    else:
        raise ValueError(f"unknown balance {balance!r}")
    alloc = np.maximum(alloc, np.minimum(1, avail))
    while alloc.sum() > size:                        # trim from the largest
        alloc[int(np.argmax(alloc))] -= 1
    while alloc.sum() < size:                        # spill into whoever has
        room = np.where(alloc < avail, avail - alloc, -1)   # room left
        j = int(np.argmax(room))
        if room[j] <= 0:
            break                                    # pool exhausted
        alloc[j] += 1
    picks = []
    for c, k in zip(classes, alloc):
        pool_c = idx[lab == c]
        picks.append(rng.choice(pool_c, size=min(int(k), len(pool_c)),
                                replace=False))
    out = np.concatenate(picks)
    rng.shuffle(out)
    return out


def run_experiment(spec: ReservoirSpec, *, backend: str = "numpy",
                   n_shots: int = 512, n_train: int = 400, n_test: int = 300,
                   n_zeroday: int | None = None,
                   seed: int = 42, device: str = "H1-1E",
                   confirm_hardware: bool = False,
                   sequence_mode: str = "features",
                   slice_order: str = "interleave",
                   correlators: bool = False,
                   balance: str = "balanced",
                   pad_mode: str = "reupload",
                   drop_noise_floor: bool = True,
                   zero_day: str | None = "Exploits",
                   log1p_fix: bool = True,
                   dataset: str = "nfunsw",
                   quiet_sim: bool = False,
                   save_bits: bool = False,
                   out_dir: str | Path = "."):
    """One complete run: data -> angles -> circuits -> features -> metrics.

    Defaults changed on 2026-10-11 after analysing the first two completed
    runs. Each change is here because a measurement demanded it:

      n_shots       100 -> 512   8 of 88 features cleared the noise floor at
                                 100 shots; the mid-circuit channels need
                                 ~756 for a 3x margin, 512 is the knee.
      correlators  True -> False same-step <Z_i Z_j> needs ~10,000 shots to
                                 be anything but noise. Dropping the 48
                                 correlator columns moved the 8q supervised
                                 AUC 0.9519 -> 0.9897.
      balance             balanced  proportional sampling on a 95.5% benign
                                 dataset gave 3 test positives at n_test=100
                                 and 1 at n_test=20. Unreadable.
      pad_mode            reupload  constant 0.5 padding filled 62% of the
                                 8q encoding slots and 93% of the 20q ones.
      drop_noise_floor    True   keep only the columns whose across-sample
                                 signal exceeds their own shot noise, chosen
                                 on TRAIN ONLY. Reported alongside the
                                 unfiltered number, never instead of it.

    Set balance="proportional", pad_mode="constant", correlators=True,
    drop_noise_floor=False, n_shots=100 to reproduce the 2026-10-10 runs.
    """
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)

    if sequence_mode == "rows":
        load_sessions()           # raises with instructions
    elif sequence_mode != "features":
        raise ValueError(f"unknown sequence_mode {sequence_mode!r}")

    spec.freeze(out_dir / f"reservoir_{spec.tag}.npz")
    budget = budget_report(spec, correlators=correlators)

    # ---- data ---------------------------------------------------------
    if dataset == "nfunsw":
        X, y, attack, names = load_nfunsw_nb15(log1p_fix=log1p_fix)
    elif dataset == "nslkdd":
        X, y, attack, names = load_nslkdd(log1p_fix=log1p_fix)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    # Hold out one attack family entirely: it exists in the test set and
    # nowhere in training. This is the zero-day test, and it is the only
    # number in this file that speaks to the actual pitch.
    if zero_day:
        held = attack == zero_day
        if held.sum() == 0:
            raise ValueError(f"attack {zero_day!r} not present in the data")
        print(f"[zero-day] holding out {int(held.sum()):,} {zero_day!r} rows")
    else:
        held = np.zeros(len(y), dtype=bool)

    pool = np.where(~held)[0]
    tr_pool, te_pool = train_test_split(pool, test_size=0.3, random_state=seed,
                                        stratify=y[pool])

    # ---- classical preprocessing, fitted on the FULL TRAIN POOL -------
    # Fitted BEFORE subsampling, deliberately. The subsample exists because
    # every sample is a separate circuit -- it is a COST constraint, not a
    # statistical one, and there is no reason to also handicap the scaler
    # and the PCA with it. Still train-only, so nothing leaks.
    #
    # This removes a real bug. The old code had
    #       d = min(n_features_ingested, X.shape[1], len(tr_idx) - 1)
    # so the 20-qubit run with n_train=20 was capped at 19 PCA components
    # out of 256 slots, and the other 237 were constant padding. The cap now
    # depends only on how many features the dataset actually has.
    scaler = StandardScaler().fit(X[tr_pool])
    d = min(spec.n_features_ingested, X.shape[1], len(tr_pool) - 1)
    pca = PCA(n_components=d, random_state=seed).fit(scaler.transform(X[tr_pool]))
    report_variance(pca, spec)

    # Subsample HARD: every sample is its own quantum circuit.
    tr_idx = _stratified_subsample(tr_pool, y, min(n_train, len(tr_pool)),
                                   rng, balance=balance)
    te_idx = _stratified_subsample(te_pool, y, min(n_test, len(te_pool)),
                                   rng, balance=balance)
    n_zd = n_zeroday if n_zeroday is not None else n_test // 2
    zd_idx = rng.choice(np.where(held)[0],
                        size=min(n_zd, int(held.sum())), replace=False) \
        if zero_day else np.array([], dtype=int)
    print(f"[sample] {len(tr_idx)} train / {len(te_idx)} test / "
          f"{len(zd_idx)} zero-day  (each one is a separate circuit)")
    print(f"[sample] balance={balance!r}   "
          f"train {int((y[tr_idx] == 0).sum())} benign / "
          f"{int((y[tr_idx] == 1).sum())} attack   "
          f"test {int((y[te_idx] == 0).sum())} benign / "
          f"{int((y[te_idx] == 1).sum())} attack")
    n_pos = int((y[te_idx] == 1).sum())
    if n_pos < 20:
        print(f"[sample] !! only {n_pos} test positives. An AUC has "
              f"{n_pos} * {int((y[te_idx] == 0).sum())} pairwise comparisons "
              f"behind it; below ~20 positives the number is not reportable.")

    def reduce(idx):
        return pca.transform(scaler.transform(X[idx]))

    Z_tr = reduce(tr_idx)
    stats = fit_squash_stats(Z_tr)
    A_tr = angles_from_features(Z_tr, spec, stats, slice_order,
                                pad_mode=pad_mode, verbose=True)
    A_te = angles_from_features(reduce(te_idx), spec, stats, slice_order,
                                pad_mode=pad_mode)
    A_zd = (angles_from_features(reduce(zd_idx), spec, stats, slice_order,
                                 pad_mode=pad_mode)
            if len(zd_idx) else None)
    print(f"[encode] angles {A_tr.shape} in half-turns; "
          f"first sample step 0: "
          f"{np.round(A_tr[0, 0], 4).tolist()}")

    # ---- quantum ------------------------------------------------------
    raw_bits = {}

    def execute(A, label):
        if A is None:
            return None
        if backend == "exact":
            print(f"[quantum] {label}: {A.shape[0]} circuits, EXACT "
                  f"(density matrix, no sampling)")
        else:
            print(f"[quantum] {label}: {A.shape[0]} circuits x "
                  f"{n_shots} shots")
        t0 = time.time()
        if backend == "exact":
            F = run_exact(spec, A, correlators=correlators,
                          verbose=not quiet_sim)
            print(f"[quantum] {label} done in {time.time() - t0:.1f}s")
            return F
        if backend == "numpy":
            bits = run_numpy(spec, A, n_shots, seed=seed,
                             verbose=not quiet_sim)
        elif backend in ("nexus", "syntax"):
            bits = run_nexus(spec, A, n_shots, device=device,
                             confirm_hardware=confirm_hardware,
                             syntax_only=(backend == "syntax"))
        else:
            raise ValueError(f"unknown backend {backend!r}")
        if bits is None:
            return None
        print(f"[quantum] {label} done in {time.time() - t0:.1f}s")
        if save_bits:
            raw_bits[label] = bits
        return features_from_bits(bits, spec, correlators)

    F_tr = execute(A_tr, "train")
    if F_tr is None:
        print("\nSyntax check only -- nothing further to do.")
        return None
    F_te = execute(A_te, "test")
    F_zd = execute(A_zd, "zero-day")

    np.savez(out_dir / f"features_{spec.tag}.npz",
             F_tr=F_tr, y_tr=y[tr_idx], F_te=F_te, y_te=y[te_idx],
             F_zd=(F_zd if F_zd is not None else np.zeros((0, F_tr.shape[1]))))
    print(f"[save] features -> features_{spec.tag}.npz")

    # The raw shot table, kept only on request. features_from_bits averages
    # it away, which permanently forecloses anything needing the individual
    # shots -- cross-timestep correlators, bitstring-distribution readouts,
    # or re-deriving features at a LOWER shot count by subsampling shots
    # (that last one turns one expensive run into a whole shot-scaling
    # curve, which is why this exists). int8 and compressed: a 850-sample
    # 512-shot 50-bit run is about 22 MB raw, far less on disk.
    if save_bits and raw_bits:
        np.savez_compressed(
            out_dir / f"bits_{spec.tag}.npz",
            **{k: v for k, v in raw_bits.items()})
        print(f"[save] raw shot table -> bits_{spec.tag}.npz "
              f"({', '.join(raw_bits)})")

    # ---- how much of this is actually signal? ------------------------
    # The exact backend has no sampling error at all, so every downstream
    # noise calculation must be told that rather than handed a shot count
    # it would take literally.
    eff_shots = None if backend == "exact" else n_shots
    print()
    mask = shot_noise_report(F_tr, eff_shots, spec, correlators)
    n_live = int(mask.sum())
    if drop_noise_floor and n_live >= 4:
        print(f"[noise] drop_noise_floor=True -> keeping {n_live} of "
              f"{F_tr.shape[1]} columns for the filtered arm")
        Q_tr, Q_te = F_tr[:, mask], F_te[:, mask]
        Q_zd = F_zd[:, mask] if F_zd is not None else None
    else:
        if drop_noise_floor:
            print(f"[noise] only {n_live} columns clear the floor -- too few "
                  f"to filter on, keeping all {F_tr.shape[1]}")
        Q_tr, Q_te, Q_zd = F_tr, F_te, F_zd

    # ---- classical readout -------------------------------------------
    print("\n--- supervised (known attacks) ---")
    sup = fit_readout(F_tr, y[tr_idx], F_te, y[te_idx], seed=seed)
    sup_f = (fit_readout(Q_tr, y[tr_idx], Q_te, y[te_idx], seed=seed)
             if Q_tr is not F_tr else sup)

    # Baseline: the SAME reduced features without the quantum layer. If the
    # quantum arm does not beat this, the reservoir added nothing.
    print("--- baseline: same PCA features, no quantum layer ---")
    base = fit_readout(Z_tr.astype(np.float32), y[tr_idx],
                       reduce(te_idx).astype(np.float32), y[te_idx], seed=seed)

    results = {"supervised_quantum": sup["auc"],
               "supervised_quantum_denoised": sup_f["auc"],
               "supervised_classical": base["auc"],
               "diagnostics": {
                   "n_features": int(F_tr.shape[1]),
                   "n_features_above_shot_floor": n_live,
                   "p_over_n": float(F_tr.shape[1] / max(1, F_tr.shape[0])),
                   "pca_components": int(d),
                   "encoding_slots": int(spec.n_features_ingested),
                   "reupload_factor": float(spec.n_features_ingested / max(1, d)),
                   "train_benign": int((y[tr_idx] == 0).sum()),
                   "train_attack": int((y[tr_idx] == 1).sum()),
                   "test_benign": int((y[te_idx] == 0).sum()),
                   "test_attack": int((y[te_idx] == 1).sum()),
               }}

    if F_zd is not None and len(F_zd):
        print("\n--- zero-day (novelty, trained on NORMAL only) ---")
        normal = y[tr_idx] == 0
        # Per-feature sampling variance, so the detector can tell a weak
        # real direction from a shot-noise one. Estimated on TRAIN only.
        sv_all = shot_variance(F_tr, eff_shots)
        det = NoveltyDetector().fit(
            F_tr[normal], shot_var=sv_all, verbose=True, label="quantum ")
        s_norm = det.score(F_te[y[te_idx] == 0])
        stats_q = detection_stats(s_norm, det.score(F_zd))
        print(f"  quantum  : " + "  ".join(f"{k}={v:.4f}"
                                           for k, v in stats_q.items()))
        if Q_tr is not F_tr:
            detf = NoveltyDetector().fit(
                Q_tr[normal], shot_var=sv_all[mask], verbose=True,
                label="denoised")
            stats_qf = detection_stats(detf.score(Q_te[y[te_idx] == 0]),
                                       detf.score(Q_zd))
            print(f"  denoised : " + "  ".join(f"{k}={v:.4f}"
                                               for k, v in stats_qf.items()))
            results["zeroday_quantum_denoised"] = stats_qf
        # The classical baseline is PCA on the raw features -- no shots, so
        # no shot noise. It gets variance truncation and nothing else.
        detc = NoveltyDetector().fit(
            Z_tr[normal], shot_var=None, verbose=True, label="classical")
        Z_te = reduce(te_idx)
        stats_c = detection_stats(detc.score(Z_te[y[te_idx] == 0]),
                                  detc.score(reduce(zd_idx)))
        print(f"  classical: " + "  ".join(f"{k}={v:.4f}"
                                           for k, v in stats_c.items()))
        results["zeroday_quantum"] = stats_q
        results["zeroday_classical"] = stats_c

    print("\n" + "=" * 68)
    print(f"SUMMARY  {spec.tag}")
    print("=" * 68)
    print(json.dumps(results, indent=2, default=float))
    (out_dir / f"results_{spec.tag}.json").write_text(
        json.dumps({"spec": {k: v for k, v in asdict(spec).items()
                             if not isinstance(v, np.ndarray)},
                    "backend": backend, "n_shots": n_shots,
                    "options": {"n_train": n_train, "n_test": n_test,
                                "n_zeroday": len(zd_idx),
                                "balance": balance, "pad_mode": pad_mode,
                                "correlators": correlators,
                                "drop_noise_floor": drop_noise_floor,
                                "slice_order": slice_order,
                                "sequence_mode": sequence_mode,
                                "dataset": dataset, "zero_day": zero_day,
                                "in_gain": spec.in_gain,
                                "final_basis": spec.final_basis,
                                "exact": backend == "exact"},
                    "budget": asdict(budget), "results": results},
                   indent=2, default=float))
    print(f"[save] results -> results_{spec.tag}.json")
    return results


if __name__ == "__main__":
    raise SystemExit(__doc__.strip().split("\n\n")[1])
