#!/usr/bin/env python3
"""Run iontraps_qrc.core.run_numpy across several processes, BIT-IDENTICAL to
the single-process run.

Why this exists: run_numpy simulates one circuit at a time on one core. At 16
and 20 qubits a sweep cell would take days. Circuits are independent, so they
can be simulated in parallel -- the only thing that couples them is the random
number stream used to sample measurement outcomes.

How the result stays identical: run_numpy makes ONE generator from `seed` and
consumes it circuit by circuit. Each circuit draws exactly

        n_shots * (n_steps * n_in + n_qubits)

uniform numbers (one per shot per measurement), regardless of the data. So the
stream position at the start of circuit k is known in advance. Each worker
builds the same generator, jumps it forward to that position
(PCG64.advance), and then calls the ORIGINAL, unmodified run_numpy on its
slice. Concatenated, the output equals the serial output exactly. This is
checked by selftest() below, not assumed.

Nothing in iontraps_qrc/core.py is edited. install(core) swaps the module's
run_numpy for the parallel front end at run time only.
"""
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from iontraps_qrc import core

_SERIAL = core.run_numpy          # the original, captured before any patching
_POOL = None
_WORKERS = None


def draws_per_circuit(spec, n_shots: int) -> int:
    return n_shots * (spec.n_steps * spec.n_in + spec.n_qubits)


def _run_slice(job):
    """Worker: simulate circuits [start, start+len(A)) of the serial run."""
    spec, A, n_shots, seed, start = job
    offset = start * draws_per_circuit(spec, n_shots)
    orig = np.random.default_rng

    def positioned(s=None):
        g = orig(s)
        g.bit_generator.advance(offset)
        return g

    np.random.default_rng = positioned
    try:
        return start, _SERIAL(spec, A, n_shots, seed=seed, verbose=False)
    finally:
        np.random.default_rng = orig


def _pool(workers):
    global _POOL, _WORKERS
    if _POOL is None or _WORKERS != workers:
        if _POOL is not None:
            _POOL.close()
        _POOL = mp.get_context("spawn").Pool(workers)
        _WORKERS = workers
    return _POOL


def slice_size(n_qubits: int) -> int:
    """Circuits per job. Small at high qubit counts so the fast and slow cores
    of this machine stay balanced; larger at low counts to limit overhead."""
    if n_qubits >= 18:
        return 1
    if n_qubits >= 16:
        return 2
    if n_qubits >= 14:
        return 4
    return 8


def run_numpy_parallel(spec, angles, n_shots, seed=0, shot_batch=None,
                       verbose=True, workers=24):
    """Drop-in replacement for run_numpy. Same arguments, same return value."""
    if shot_batch is not None:
        # a non-default batch does not change the draw order, but keep the
        # serial path for anything this wrapper was not tested with
        return _SERIAL(spec, angles, n_shots, seed=seed, shot_batch=shot_batch,
                       verbose=verbose)
    n = angles.shape[0]
    step = slice_size(spec.n_qubits)
    jobs = [(spec, angles[a:a + step], n_shots, seed, a) for a in range(0, n, step)]
    out = None
    done, t0, next_report = 0, time.time(), 25
    for start, bits in _pool(workers).imap_unordered(_run_slice, jobs):
        if out is None:
            out = np.zeros((n,) + bits.shape[1:], dtype=bits.dtype)
        out[start:start + bits.shape[0]] = bits
        done += bits.shape[0]
        if verbose and (done >= next_report or done == n):
            el = time.time() - t0
            print(f"  [numpy x{workers}] {done}/{n} samples, {el:.1f}s "
                  f"({el / done:.2f}s/sample wall)", flush=True)
            next_report = (done // 25 + 1) * 25
    return out


def install(core_module=core, workers=24):
    """Make run_experiment use the parallel front end. Call in the parent only."""
    def front(spec, angles, n_shots, seed=0, shot_batch=None, verbose=True):
        return run_numpy_parallel(spec, angles, n_shots, seed=seed,
                                  shot_batch=shot_batch, verbose=verbose,
                                  workers=workers)
    core_module.run_numpy = front


def selftest(workers=6):
    """Parallel output must equal serial output exactly, for both shot-batch
    regimes of run_numpy (N <= 12: one batch; N > 12: batches of 8)."""
    import contextlib
    import io
    ok = True
    for (n_in, n_mem, L, n_samples, n_shots) in ((4, 4, 6, 21, 64), (7, 7, 3, 9, 24)):
        spec = core.ReservoirSpec(tag="selftest", n_in=n_in, n_mem=n_mem, n_steps=L,
                                  dt=0.3, topology="all_to_all", seed=1234)
        with contextlib.redirect_stdout(io.StringIO()):
            spec.freeze(write=False)
        A = np.random.default_rng(7).uniform(0.1, 0.9, (n_samples, L, n_in, 2))
        A[..., 1] *= 2
        serial = _SERIAL(spec, A, n_shots, seed=42, verbose=False)
        par = run_numpy_parallel(spec, A, n_shots, seed=42, verbose=False,
                                 workers=workers)
        same = bool(np.array_equal(serial, par)) and serial.dtype == par.dtype
        ok &= same
        print(f"  {n_in + n_mem} qubits, L={L}, {n_samples} circuits x {n_shots} shots "
              f"(slices of {slice_size(n_in + n_mem)}): parallel == serial: {same}")
    return ok


if __name__ == "__main__":
    print("selftest:", "PASS" if selftest() else "FAIL")
