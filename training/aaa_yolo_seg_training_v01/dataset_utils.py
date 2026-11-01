from __future__ import annotations

from pathlib import Path

import yaml


def materialize_absolute_dataset_yaml(source: str | Path) -> str:
    """Write a runtime YAML whose dataset root is absolute and portable."""
    source_path = Path(source).resolve()
    config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    configured_root = Path(config.get("path", "dataset"))
    if not configured_root.is_absolute():
        configured_root = (Path.cwd() / configured_root).resolve()
    config["path"] = str(configured_root)
    runtime_path = source_path.with_name(f"{source_path.stem}.runtime.yaml")
    runtime_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return str(runtime_path)

