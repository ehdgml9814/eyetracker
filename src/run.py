"""
run.py — YAML 실험 스크립트 단일 진입점

실험 YAML 형식:
  name: phase1_step1_det
  description: "..."
  sweep:
    fixed:
      category.pose.selected: none
      category.crop.selected: none
    grid:
      category.det.selected: [none, clahe, gamma, bilateral]
      model.backbone: [resnet18, mobilenet_v2]

사용법:
  python src/run.py experiments/phase1_step1_det.yaml          # 실행
  python src/run.py experiments/phase1_step1_det.yaml --list   # 목록만
  python src/run.py experiments/phase1_step1_det.yaml --dry-run
  python src/run.py experiments/phase1_step1_det.yaml --force  # 강제 재실행
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, save_yaml


# ── 실험 목록 생성 ────────────────────────────────────────────────────────────

def _expand_sweep(sweep: dict) -> list[dict[str, str]]:
    """
    sweep.fixed + sweep.grid의 Cartesian product → override 목록.

    각 원소: {"key": "value", ...}
    """
    fixed: dict = sweep.get("fixed", {}) or {}
    grid:  dict = sweep.get("grid",  {}) or {}

    # 고정값 문자열화
    fixed_items = {k: str(v) for k, v in fixed.items()}

    if not grid:
        return [fixed_items]

    keys   = list(grid.keys())
    values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]

    experiments = []
    for combo in itertools.product(*values):
        overrides = dict(fixed_items)
        for k, v in zip(keys, combo):
            overrides[k] = str(v)
        experiments.append(overrides)

    return experiments


def _exp_name(base_name: str, overrides: dict[str, str]) -> str:
    """
    실험 디렉토리 이름 생성.
    예) exp_clahe_none_gabor_resnet18
    """
    det  = overrides.get("category.det.selected",  "none")
    pose = overrides.get("category.pose.selected", "none")
    crop = overrides.get("category.crop.selected", "none")
    bb   = overrides.get("model.backbone",         "resnet18")
    return f"exp_{det}_{pose}_{crop}_{bb}"


def _overrides_to_set_args(overrides: dict[str, str]) -> list[str]:
    return [f"{k}={v}" for k, v in overrides.items()]


# ── 완료 여부 확인 ─────────────────────────────────────────────────────────────

def _is_done(runs_dir: Path, exp_name: str) -> bool:
    """train + eval 모두 완료 여부 (result.yaml에 test_angular_err 존재)."""
    result_path = runs_dir / exp_name / "result.yaml"
    if not result_path.exists():
        return False
    try:
        with open(result_path) as f:
            data = yaml.safe_load(f) or {}
        return "test_angular_err" in data.get("best", {})
    except Exception:
        return False


def _is_trained(runs_dir: Path, exp_name: str) -> bool:
    """train만 완료 여부 (best.pt 존재)."""
    return (runs_dir / exp_name / "best.pt").exists()


# ── 서브프로세스 실행 ─────────────────────────────────────────────────────────

def _run_subprocess(cmd: list[str], label: str) -> bool:
    print(f"\n[run] {label}")
    print(f"  $ {' '.join(cmd)}")
    ret = subprocess.call(cmd)
    if ret != 0:
        print(f"[run] 오류: {label} 실패 (코드 {ret})")
        return False
    return True


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="실험 스크립트 단일 진입점")
    parser.add_argument("experiment_yaml", help="experiments/*.yaml 경로")
    parser.add_argument("--config-dir", default="/workspace/configs")
    parser.add_argument("--runs-dir",   default="/workspace/runs")
    parser.add_argument("--list",       action="store_true", help="실험 목록만 출력")
    parser.add_argument("--dry-run",    action="store_true", help="실제 실행 없이 계획 출력")
    parser.add_argument("--force",      action="store_true", help="완료된 실험도 재실행")
    args = parser.parse_args()

    exp_yaml_path = Path(args.experiment_yaml)
    if not exp_yaml_path.exists():
        print(f"오류: {exp_yaml_path} 없음")
        sys.exit(1)

    with open(exp_yaml_path) as f:
        exp_spec = yaml.safe_load(f) or {}

    base_name = exp_spec.get("name", exp_yaml_path.stem)
    desc      = exp_spec.get("description", "")
    sweep     = exp_spec.get("sweep", {})

    experiments = _expand_sweep(sweep)
    runs_dir    = Path(args.runs_dir)

    print(f"\n실험: {base_name}")
    if desc:
        print(f"설명: {desc}")
    print(f"총 {len(experiments)}개 실험\n")

    # ── 목록 출력 ─────────────────────────────────────────────────────────────
    done_count = 0
    for i, overrides in enumerate(experiments, start=1):
        name   = _exp_name(base_name, overrides)
        done   = _is_done(runs_dir, name)
        status = "✓ 완료" if done else "○ 대기"
        if done:
            done_count += 1
        print(f"  {i:2d}. [{status}] {name}")
        for k, v in overrides.items():
            print(f"        {k}={v}")

    print(f"\n완료: {done_count}/{len(experiments)}")

    if args.list:
        return

    # ── 실행 ──────────────────────────────────────────────────────────────────
    success_count = 0
    skip_count    = 0
    fail_count    = 0

    for i, overrides in enumerate(experiments, start=1):
        name    = _exp_name(base_name, overrides)
        exp_dir = runs_dir / name

        if _is_done(runs_dir, name) and not args.force:
            print(f"\n[run] 스킵 ({i}/{len(experiments)}): {name}")
            skip_count += 1
            continue

        print(f"\n{'='*60}")
        print(f"[run] 실험 {i}/{len(experiments)}: {name}")
        print(f"{'='*60}")

        set_args = _overrides_to_set_args(overrides)
        set_flags = []
        for s in set_args:
            set_flags += ["--set", s]

        if args.dry_run:
            print(f"  [dry-run] train: --exp-dir {exp_dir} {set_flags}")
            print(f"  [dry-run] eval:  --exp-dir {exp_dir}")
            success_count += 1
            continue

        # train — best.pt 없을 때만 실행 (eval 실패 후 재시도 시 train 스킵)
        if args.force or not _is_trained(runs_dir, name):
            train_cmd = [
                sys.executable, "src/train.py",
                "--config-dir", args.config_dir,
                "--exp-dir", str(exp_dir),
            ] + set_flags

            if not _run_subprocess(train_cmd, f"train: {name}"):
                fail_count += 1
                continue
        else:
            print(f"\n[run] train 스킵 (best.pt 존재): {name}")

        # eval — test_angular_err 없을 때만 실행
        if args.force or not _is_done(runs_dir, name):
            eval_cmd = [
                sys.executable, "src/eval.py",
                "--exp-dir", str(exp_dir),
                "--config-dir", args.config_dir,
            ]

            if not _run_subprocess(eval_cmd, f"eval: {name}"):
                fail_count += 1
                continue

        success_count += 1

    print(f"\n{'='*60}")
    print(f"[run] 완료: 성공={success_count} 스킵={skip_count} 실패={fail_count}")

    if success_count > 0 and not args.dry_run:
        print("[run] 결과 비교:")
        compare_cmd = [
            sys.executable, "src/compare.py",
            "--runs-dir", args.runs_dir,
        ]
        subprocess.call(compare_cmd)


if __name__ == "__main__":
    main()
