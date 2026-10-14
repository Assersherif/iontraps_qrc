"""Load the committed results and build the Results table / compare it with the report."""
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

# Table "3. Results" as shown in the report (values copied from the figure):
# (qubits, L): (sup_quantum, sup_classical, zd_quantum, zd_classical)
REPORT = {
    (6, 6): (0.965, 0.984, 0.867, 0.888), (6, 8): (0.975, 0.985, 0.859, 0.888),
    (6, 10): (0.980, 0.986, 0.878, 0.888), (8, 6): (0.977, 0.985, 0.864, 0.888),
    (8, 8): (0.979, 0.986, 0.871, 0.888), (8, 10): (0.984, 0.986, 0.883, 0.888),
    (10, 6): (0.984, 0.986, 0.882, 0.888), (10, 8): (0.976, 0.986, 0.875, 0.888),
    (10, 10): (0.984, 0.986, 0.882, 0.888), (12, 6): (0.962, 0.986, 0.861, 0.888),
    (12, 8): (0.984, 0.986, 0.882, 0.888), (12, 10): (0.986, 0.986, 0.882, 0.888),
    (14, 6): (0.963, 0.986, 0.889, 0.888), (14, 8): (0.970, 0.986, 0.871, 0.888),
    (14, 10): (0.973, 0.986, 0.889, 0.888), (16, 6): (0.985, 0.986, 0.888, 0.888),
    (16, 8): (0.978, 0.986, 0.891, 0.888), (16, 10): (0.976, 0.986, 0.882, 0.888),
}


def load_grid(results_dir: Path = RESULTS) -> pd.DataFrame:
    """One row per (qubits, L): AUCs (denoised quantum = the report's columns),
    paired difference and bootstrap 95% interval for both tasks."""
    summ = json.loads((results_dir / "sweep_summary_full_grid.json").read_text())
    eff = json.loads((results_dir / "grid_effects_check.json").read_text())
    rows = []
    for c in summ["cells"]:
        rows.append(dict(
            tag=c["tag"], qubits=c["n_qubits"], L=c["L"],
            sup_q=c["supervised_quantum_denoised"], sup_q_raw=c["supervised_quantum"],
            sup_c=c["supervised_classical"],
            zd_q=c["zeroday_quantum_denoised"]["AUC"], zd_q_raw=c["zeroday_quantum"]["AUC"],
            zd_c=c["zeroday_classical"]["AUC"],
            zd_q_1fa=c["zeroday_quantum_denoised"]["@1%FA"], zd_c_1fa=c["zeroday_classical"]["@1%FA"],
            feats=c["diag_n_features"], feats_live=c["diag_n_features_above_shot_floor"],
            source=c["source"]))
    df = pd.DataFrame(rows).set_index("tag")
    for key, name in (("supervised", "sup"), ("zeroday", "zd")):
        for cell in eff[key]["cells"]:
            t = cell["tag"]
            df.loc[t, f"{name}_diff"] = cell["gap"]
            df.loc[t, f"{name}_lo"], df.loc[t, f"{name}_hi"] = cell["ci95"]
    return df.sort_values(["qubits", "L"]).reset_index()


def compare_with_report(df: pd.DataFrame, tol: float = 0.0015) -> pd.DataFrame:
    out = []
    for _, r in df.iterrows():
        rep = REPORT[(r.qubits, r.L)]
        mine = (r.sup_q, r.sup_c, r.zd_q, r.zd_c)
        d = [abs(round(m, 3) - x) for m, x in zip(mine, rep)]
        out.append(dict(qubits=r.qubits, L=r.L, max_abs_diff=max(d), match=max(d) <= tol,
                        **{n: (round(m, 3), x) for n, m, x in zip(("sup_q", "sup_c", "zd_q", "zd_c"), mine, rep)}))
    return pd.DataFrame(out)
