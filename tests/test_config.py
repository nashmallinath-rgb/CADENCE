from cadence.config import load_cascade_config, load_run_config


def test_cascade_defaults_load_from_repo_config() -> None:
    cfg = load_cascade_config()
    assert 0 < cfg.tau <= 1 and cfg.delta > 0
    assert cfg.cheap_trials >= 1 and cfg.strong_trials >= 1


def test_run_config_loads_with_llm_section() -> None:
    cfg = load_run_config()
    assert cfg.max_gen >= 1 and cfg.max_archive >= 1
    assert cfg.llm.cache_dir == ".cache/llm"
    assert set(cfg.llm.rpm) >= {"gemini", "xai"}
