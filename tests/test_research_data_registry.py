import pandas as pd


def test_clean_dataset_registry_roundtrip(tmp_path, monkeypatch):
    import research.data_registry as registry

    monkeypatch.setattr(registry, "ROOT", tmp_path)
    monkeypatch.setattr(registry, "STORE", tmp_path / "data" / "research_lab")
    monkeypatch.setattr(registry, "DATA_STORE", tmp_path / "data" / "research_lab" / "clean_datasets")
    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "data" / "research_lab" / "clean_dataset_registry.json")
    registry.STORE.mkdir(parents=True)
    registry.DATA_STORE.mkdir(parents=True)

    idx = pd.date_range("2025-01-01", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame({
        "open": range(1, 11), "high": range(2, 12), "low": range(1, 11),
        "close": range(2, 12), "volume": [1.0] * 10,
    }, index=idx)
    meta = registry.register_clean_dataset(df, "TESTUSD", "1h", cleaning_info={"huecos_eliminados": 1})
    assert registry.available_clean_assets("1h") == ["TESTUSD"]
    out = registry.load_clean_dataset("TESTUSD", "1h")
    assert len(out) == len(df)
    assert out.index.equals(df.index)
    assert meta["status"] == "clean"


def test_research_orchestrator_does_not_fallback_to_mt5(monkeypatch):
    import research.orchestrator.research_orchestrator as module
    from research.config import ResearchConfig

    cfg = ResearchConfig(assets=["TESTUSD"], timeframe="1h", objectives=["sharpe"], population_size=20, generations=3)
    orch = module.ResearchOrchestrator(cfg, preloaded_data={})
    monkeypatch.setattr(module, "load_clean_dataset", lambda *args, **kwargs: None)
    try:
        orch._load_asset("TESTUSD")
    except ValueError as exc:
        assert "dataset limpio" in str(exc).lower()
    else:
        raise AssertionError("El orquestador no debe aceptar datos sin registro limpio.")
