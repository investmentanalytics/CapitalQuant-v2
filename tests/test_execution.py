"""
tests/test_execution.py
Cobertura para Fase 3 del roadmap (bloque D): risk overlay y el guardia
de cuenta DEMO de execution/mt5_execution.py.

No requiere MetaTrader5 instalado: se reemplaza market_data.mt5_provider
por un doble de prueba (fake) que simula cuentas demo/real, posiciones,
ticks y symbol_info, para poder verificar TODA la lógica de seguridad
sin depender de una terminal MT5 real.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from execution.risk_overlay import RiskOverlay, RiskOverlayConfig
import execution.mt5_execution as mt5_execution_module
from execution.mt5_execution import MT5ExecutionClient, DemoAccountRequiredError


# ---------------------------------------------------------------------------
# RiskOverlay — lógica pura, sin MT5
# ---------------------------------------------------------------------------

class TestRiskOverlay:
    def setup_method(self):
        self.overlay = RiskOverlay(RiskOverlayConfig(
            risk_per_trade_pct=0.02, max_daily_loss_pct=0.05, max_open_positions=1,
        ))

    def test_rejects_real_account_without_explicit_opt_in(self):
        result = self.overlay.check_order(
            account_equity=10_000, account_is_demo=False, n_open_positions=0,
            entry_price=100.0, stop_loss_price=95.0, contract_size=1,
            volume_min=0.01, volume_max=100, volume_step=0.01,
        )
        assert result.allowed is False
        assert "REAL" in result.reason

    def test_allows_real_account_with_explicit_opt_in(self):
        result = self.overlay.check_order(
            account_equity=10_000, account_is_demo=False, n_open_positions=0,
            entry_price=100.0, stop_loss_price=95.0, contract_size=1,
            volume_min=0.01, volume_max=100, volume_step=0.01,
            allow_real_account=True,
        )
        assert result.allowed is True
        assert result.suggested_volume is not None

    def test_allows_valid_demo_order(self):
        result = self.overlay.check_order(
            account_equity=10_000, account_is_demo=True, n_open_positions=0,
            entry_price=100.0, stop_loss_price=95.0, contract_size=1,
            volume_min=0.01, volume_max=100, volume_step=0.01,
        )
        assert result.allowed is True
        assert result.suggested_volume is not None
        assert result.suggested_volume > 0

    def test_position_sizing_matches_risk_formula(self):
        # risk_amount = 10_000 * 0.02 = 200; risk_per_unit = |100-95|*1 = 5
        # size = 200 / 5 = 40 (antes de tope de notional / redondeo a step)
        result = self.overlay.check_order(
            account_equity=10_000, account_is_demo=True, n_open_positions=0,
            entry_price=100.0, stop_loss_price=95.0, contract_size=1,
            volume_min=0.01, volume_max=1000, volume_step=1.0,
        )
        assert result.suggested_volume == pytest.approx(40.0, abs=1.0)

    def test_rejects_missing_stop_loss(self):
        result = self.overlay.check_order(
            account_equity=10_000, account_is_demo=True, n_open_positions=0,
            entry_price=100.0, stop_loss_price=None, contract_size=1,
            volume_min=0.01, volume_max=100, volume_step=0.01,
        )
        assert result.allowed is False

    def test_rejects_when_max_open_positions_reached(self):
        result = self.overlay.check_order(
            account_equity=10_000, account_is_demo=True, n_open_positions=1,
            entry_price=100.0, stop_loss_price=95.0, contract_size=1,
            volume_min=0.01, volume_max=100, volume_step=0.01,
        )
        assert result.allowed is False
        assert "posici" in result.reason.lower()

    def test_rejects_when_size_below_minimum_lot(self):
        # riesgo muy chico + SL muy ancho -> tamaño calculado menor al mínimo
        overlay = RiskOverlay(RiskOverlayConfig(risk_per_trade_pct=0.0001))
        result = overlay.check_order(
            account_equity=1_000, account_is_demo=True, n_open_positions=0,
            entry_price=100.0, stop_loss_price=50.0, contract_size=1,
            volume_min=1.0, volume_max=100, volume_step=1.0,
        )
        assert result.allowed is False

    def test_kill_switch_triggers_after_daily_loss_limit(self):
        overlay = RiskOverlay(RiskOverlayConfig(max_daily_loss_pct=0.05))
        overlay.check_daily_loss(current_equity=10_000)   # ancla el día en 10_000
        result = overlay.check_daily_loss(current_equity=9_400)  # -6%, supera el -5%
        assert result.allowed is False
        assert overlay.kill_switch_triggered is True

    def test_kill_switch_blocks_subsequent_orders(self):
        overlay = RiskOverlay(RiskOverlayConfig(max_daily_loss_pct=0.05))
        overlay.check_daily_loss(current_equity=10_000)
        overlay.check_daily_loss(current_equity=9_000)  # -10%, dispara kill switch
        result = overlay.check_order(
            account_equity=9_000, account_is_demo=True, n_open_positions=0,
            entry_price=100.0, stop_loss_price=95.0, contract_size=1,
            volume_min=0.01, volume_max=100, volume_step=0.01,
        )
        assert result.allowed is False

    def test_no_kill_switch_within_daily_loss_tolerance(self):
        overlay = RiskOverlay(RiskOverlayConfig(max_daily_loss_pct=0.05))
        overlay.check_daily_loss(current_equity=10_000)
        result = overlay.check_daily_loss(current_equity=9_800)  # -2%, dentro de tolerancia
        assert result.allowed is True
        assert overlay.kill_switch_triggered is False

    def test_volume_respects_step_rounding(self):
        result = self.overlay.check_order(
            account_equity=10_000, account_is_demo=True, n_open_positions=0,
            entry_price=100.0, stop_loss_price=99.7, contract_size=1,
            volume_min=0.01, volume_max=1000, volume_step=0.1,
        )
        if result.allowed:
            # El volumen debe ser múltiplo del step (dentro de tolerancia flotante)
            remainder = round(result.suggested_volume / 0.1) * 0.1 - result.suggested_volume
            assert abs(remainder) < 1e-6


# ---------------------------------------------------------------------------
# MT5ExecutionClient — guardia de cuenta DEMO (con proveedor MT5 falso)
# ---------------------------------------------------------------------------

class _FakeProvider:
    """Doble de prueba de MT5Provider — sin ninguna dependencia de
    MetaTrader5 real, para poder testear el guardia DEMO/REAL en
    cualquier sistema operativo."""

    def __init__(self, is_demo: bool, equity: float = 10_000.0):
        self._is_demo = is_demo
        self._equity = equity

    def account_info(self):
        return {
            "login": 12345, "server": "Broker-Demo", "name": "Test",
            "balance": self._equity, "equity": self._equity, "margin": 0,
            "margin_free": self._equity, "currency": "USD",
            "trade_mode": 0 if self._is_demo else 2,
            "is_demo": self._is_demo, "leverage": 100,
        }

    def get_raw_mt5(self):
        return None  # no se usa en estos tests (no llegamos a order_send)

    def get_symbol_info(self, symbol):
        return {"contract_size": 100_000.0, "min_lot": 0.01, "max_lot": 100.0, "lot_step": 0.01}


def test_send_market_order_requires_explicit_confirm(monkeypatch):
    monkeypatch.setattr(mt5_execution_module, "get_mt5_provider", lambda: _FakeProvider(is_demo=True))
    client = MT5ExecutionClient()
    result = client.send_market_order("EURUSD", "buy", 0.1, stop_loss_price=1.05, confirm=False)
    assert result.success is False
    assert "confirmaci" in result.message.lower()


def test_send_market_order_blocked_on_real_account_without_opt_in(monkeypatch):
    monkeypatch.setattr(mt5_execution_module, "get_mt5_provider", lambda: _FakeProvider(is_demo=False))
    client = MT5ExecutionClient()  # allow_real_account=False por defecto
    result = client.send_market_order("EURUSD", "buy", 0.1, stop_loss_price=1.05, confirm=True)
    assert result.success is False
    assert "REAL" in result.message


def test_close_all_positions_requires_explicit_confirm(monkeypatch):
    monkeypatch.setattr(mt5_execution_module, "get_mt5_provider", lambda: _FakeProvider(is_demo=True))
    client = MT5ExecutionClient()
    results = client.close_all_positions(confirm=False)
    assert len(results) == 1
    assert results[0].success is False


def test_is_demo_reflects_provider(monkeypatch):
    monkeypatch.setattr(mt5_execution_module, "get_mt5_provider", lambda: _FakeProvider(is_demo=True))
    client = MT5ExecutionClient()
    assert client.is_demo() is True

    monkeypatch.setattr(mt5_execution_module, "get_mt5_provider", lambda: _FakeProvider(is_demo=False))
    client2 = MT5ExecutionClient()
    assert client2.is_demo() is False


def test_verify_account_allowed_raises_on_real_account_without_opt_in(monkeypatch):
    monkeypatch.setattr(mt5_execution_module, "get_mt5_provider", lambda: _FakeProvider(is_demo=False))
    client = MT5ExecutionClient()
    with pytest.raises(DemoAccountRequiredError):
        client._verify_account_allowed()


def test_verify_account_allowed_passes_on_real_account_with_opt_in(monkeypatch):
    monkeypatch.setattr(mt5_execution_module, "get_mt5_provider", lambda: _FakeProvider(is_demo=False))
    client = MT5ExecutionClient(allow_real_account=True)
    info = client._verify_account_allowed()
    assert info["is_demo"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
