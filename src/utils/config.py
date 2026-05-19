"""
config.py — YAML 설정 로드 / 저장 / 병합 / 로그 추가
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """override 값을 base 위에 재귀 병합. base를 복사하여 반환."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """'a.b.c' 형태의 키로 중첩 dict에 값을 설정."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    # 값 타입 추론 (문자열 → bool/int/float)
    if isinstance(value, str):
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
    d[keys[-1]] = value


# ── 공개 API ─────────────────────────────────────────────────────────────────

def load_config(
    config_dir: str | Path = "/workspace/configs",
    overrides: list[str] | None = None,
) -> dict:
    """
    static.yaml + dynamic.yaml + filters.yaml을 병합한 뒤
    overrides(["key.path=value", ...])를 적용하여 반환.
    """
    config_dir = Path(config_dir)

    static  = yaml.safe_load((config_dir / "static.yaml").read_text())  or {}
    dynamic = yaml.safe_load((config_dir / "dynamic.yaml").read_text()) or {}
    # filters = yaml.safe_load((config_dir / "filters.yaml").read_text()) or {}
    filters_path = config_dir / "filters.yaml"
    filters = yaml.safe_load(filters_path.read_text()) if filters_path.exists() else {}

    cfg = _deep_merge(static, dynamic)
    cfg = _deep_merge(cfg, filters)

    if overrides:
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"override 형식 오류: '{item}' (key=value 형식 필요)")
            key, val = item.split("=", 1)
            _set_nested(cfg, key.strip(), val.strip())

    return cfg


def save_yaml(data: Any, path: str | Path) -> None:
    """데이터를 YAML 파일로 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def append_yaml_log(record: dict, path: str | Path) -> None:
    """
    YAML 리스트 파일에 record를 추가.
    파일이 없으면 생성, 있으면 기존 리스트에 append.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: list = []
    if path.exists():
        data = yaml.safe_load(path.read_text()) or []
        if isinstance(data, list):
            existing = data

    existing.append(record)
    save_yaml(existing, path)


def get_filter_params(cfg: dict, filter_name: str, point: str) -> dict:
    """
    필터 파라미터를 두 단계로 조회하여 반환.

    1. filters.<filter_name>.params  (전역 기본값)
    2. category.<point>.params       (포인트별 오버라이드, 비어있으면 전역 유지)

    반환: 병합된 파라미터 dict (빈 dict 가능)
    """
    global_params = (
        cfg.get("filters", {})
           .get(filter_name, {})
           .get("params", {})
    ) or {}

    point_params = (
        cfg.get("category", {})
           .get(point, {})
           .get("params", {})
    ) or {}

    return _deep_merge(global_params, point_params)
