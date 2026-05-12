"""
compare.py — 실험 결과 수집 → 순위표 출력 → experiments_summary.yaml 저장

사용법:
  python src/compare.py
  python src/compare.py --runs-dir /workspace/runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import save_yaml


def _load_results(runs_dir: Path) -> list[dict]:
    results = []
    for result_yaml in sorted(runs_dir.glob("exp_*/result.yaml")):
        with open(result_yaml) as f:
            data = yaml.safe_load(f) or {}
        best = data.get("best", {})
        if "test_angular_err" not in best:
            continue  # 아직 eval 미완료
        exp_info = data.get("experiment", {})
        results.append({
            "exp_dir":       result_yaml.parent.name,
            "det_filter":    exp_info.get("det_filter",  "?"),
            "pose_filter":   exp_info.get("pose_filter", "?"),
            "crop_filter":   exp_info.get("crop_filter", "?"),
            "backbone":      exp_info.get("backbone",    "?"),
            "val_angular_err":  float(best.get("val_angular_err",  999.0)),
            "test_angular_err": float(best.get("test_angular_err", 999.0)),
            "best_epoch":    int(best.get("epoch", 0)),
        })
    return sorted(results, key=lambda r: r["test_angular_err"])


def _phase_of(row: dict) -> str:
    if row["crop_filter"] == "adaptive":
        return "phase2"
    return "phase1"


def _print_table(results: list[dict]) -> None:
    if not results:
        print("결과 없음.")
        return

    header = (
        f"{'순위':>4}  {'det':10} {'pose':10} {'crop':10} {'backbone':14}"
        f"{'val_err°':>9} {'test_err°':>10} {'epoch':>6}  exp_dir"
    )
    print(header)
    print("-" * len(header))

    for rank, row in enumerate(results, start=1):
        print(
            f"  {rank:2d}  {row['det_filter']:10} {row['pose_filter']:10} "
            f"{row['crop_filter']:10} {row['backbone']:14}"
            f"  {row['val_angular_err']:7.3f}°  {row['test_angular_err']:8.3f}°"
            f"  {row['best_epoch']:5d}  {row['exp_dir']}"
        )


def compare(runs_dir: Path) -> None:
    results = _load_results(runs_dir)
    if not results:
        print(f"[compare] {runs_dir} 내 완료된 실험이 없습니다.")
        return

    phase1 = [r for r in results if _phase_of(r) == "phase1"]
    phase2 = [r for r in results if _phase_of(r) == "phase2"]

    print(f"\n{'='*60}")
    print(f"  Phase 1 결과 ({len(phase1)}개)")
    print(f"{'='*60}")
    _print_table(phase1)

    if phase2:
        print(f"\n{'='*60}")
        print(f"  Phase 2 결과 ({len(phase2)}개)")
        print(f"{'='*60}")
        _print_table(phase2)

    # 요약 저장
    summary = {"experiments": results}

    if phase1:
        best_p1 = phase1[0]
        summary["phase1_best"] = {
            k: best_p1[k]
            for k in ("det_filter", "pose_filter", "crop_filter",
                      "backbone", "test_angular_err")
        }

    if phase2:
        best_p2 = phase2[0]
        summary["phase2_best"] = {
            k: best_p2[k]
            for k in ("det_filter", "pose_filter", "crop_filter",
                      "backbone", "test_angular_err")
        }

    if phase1 and phase2:
        improvement = phase1[0]["test_angular_err"] - phase2[0]["test_angular_err"]
        summary["improvement_deg"] = round(improvement, 4)

    out_path = runs_dir / "experiments_summary.yaml"
    save_yaml(summary, out_path)
    print(f"\n[compare] 요약 저장: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="/workspace/runs")
    args = parser.parse_args()
    compare(Path(args.runs_dir))


if __name__ == "__main__":
    main()
