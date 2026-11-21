"""Tabulate the synchronized whole-window memory and latency measurements."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def main():
    rows=json.loads((ROOT/"results/lm/inference_profile.json").read_text())
    assert len(rows)==16
    lines=[r"\begin{center}\small\begin{tabular}{llrrrr}\toprule",
           r"Model & Policy & Batch & Median ms & P90 ms & Peak MiB\\\midrule"]
    for row in rows:
        # The full machine-readable set contains all 16 profiles. Compare each hard
        # adaptive policy with its observed same-depth control for this table.
        k=row['mode']
        if k not in ("adaptive","1" if row["model"]=="118K" else "4"):continue
        label="Learned" if k=="adaptive" else f"Fixed {k}"
        lines.append(f"{row['model']} & {label} & {row['batch']} & {row['median_ms_per_batch']:.2f} & {row['p90_ms_per_batch']:.2f} & {row['max_peak_mlx_bytes']/2**20:.1f} "+r"\\")
    lines += [r"\bottomrule\end{tabular}\end{center}",r"\captext{Milliseconds per batch. Each learned policy is paired here with a forced exit at its observed depth, one for the small checkpoint and four for the larger one. Full results include forced depths one, two and four for both checkpoints.}"]
    (HERE/"profile-rows.tex").write_text("\n".join(lines)+"\n")
if __name__=="__main__": main()
