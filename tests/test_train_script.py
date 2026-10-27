"""Config handling of scripts/train_ppo.py (no game, no torch)."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("yaml")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_ppo.py"
spec = importlib.util.spec_from_file_location("train_ppo", SCRIPT)
assert spec and spec.loader
train_ppo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_ppo)


def test_config_file_and_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("name: a\ntasks: [reach]\ntotal_turns: 10\n")
    cfg = train_ppo.load_config(
        str(path), ["total_turns=20", "tasks=[reach, escape]", "ppo.seed=3"]
    )
    assert cfg.name == "a" and cfg.total_turns == 20
    assert cfg.tasks == ["reach", "escape"] and cfg.ppo == {"seed": 3}


def test_unknown_key_is_rejected():
    with pytest.raises(SystemExit):
        train_ppo.load_config(None, ["nope=1"])


def test_shipped_configs_parse():
    for name in ("ppo_reach", "ppo_smoke"):
        path = SCRIPT.parents[1] / "configs" / f"{name}.yaml"
        assert train_ppo.load_config(str(path), []).name == name
