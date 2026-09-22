"""
ui/state.py
Gestión centralizada del estado de sesión de Streamlit.

Elimina la dispersión de st.session_state["clave"] por todo el código.
Todas las claves del estado se definen y acceden desde aquí.

REGLA: ninguna página accede directamente a st.session_state.
       Toda lectura/escritura pasa por AppState.
"""
from __future__ import annotations
from typing import Optional, Any
import streamlit as st


class AppState:
    """Acceso tipado y centralizado al session_state de Streamlit."""

    # ------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------

    @staticmethod
    def init() -> None:
        """
        Inicializa todos los valores del estado con sus defaults.
        Llamar UNA vez al inicio de app.py antes de cualquier renderizado.
        """
        defaults = {
            # Backtest
            "backtest_results":    {},       # {key: BacktestResults}
            "last_backtest_key":   None,
            "last_live_configs":   {},       # {(strategy,asset,timeframe): {"params":..,"config":..}}
                                              # usado por Mercado en Vivo para reproducir EXACTAMENTE
                                              # los mismos parámetros/entradas del último backtest.

            # Portfolio
            "portfolio_results":   None,     # PortfolioResults
            "portfolio_config":    {},       # {asset: {strategy: weight}}
            "asset_weights":       {},       # {asset: weight}

            # Optimización
            "opt_history":         None,     # pd.DataFrame
            "opt_best_params":     {},
            "opt_strategy":        None,
            "opt_asset":           None,
            "opt_timeframe":       None,

            # Walk-Forward
            "wfo_results":         None,     # WFOResults

            # CPCV (Combinatorial Purged Cross-Validation) + PBO
            "cpcv_results":        None,     # CPCVResults

            # Monte Carlo
            "mc_results":          None,     # MonteCarloResults

            # Sensibilidad
            "sensitivity_result":  None,     # SensitivityResult

            # Selección actual
            "selected_asset":      "BTCUSD",
            "selected_timeframe":  "1d",
            "selected_strategy":   None,

            # Datos cargados en caché
            "loaded_data":         {},       # {(asset, tf): df}
            "research_clean_assets": {},    # {(asset, tf): metadata} registrados desde Gráficos
            "historical_data_source": "MT5 / datos originales",  # fuente histórica para módulos de análisis

            # Constructor de Estrategias (editor de código)
            "builder_compiled":       None,  # Type[BaseStrategy] validada, lista para usar
            "builder_compiled_name":  None,  # nombre de la clase compilada
            "builder_backtest":       None,  # BacktestResults del backtest rápido en el Constructor
            "builder_opt_history":    None,  # historial de optimización lanzada desde el Constructor
            "builder_opt_best":       {},

            # Descubridor Genético (Aletheia)
            "discovery_result":       None,  # dict con leaderboard de la última corrida
            "discovery_running":      False,
            "discovery_sent_ok":      None,  # último mensaje de envío a Backtesting

            # Research Lab — el batch queda además persistido en data/research
            # para poder recuperar el último trabajo al volver a la página.
            "last_research_batch":    None,
            "research_assets":        [],
            "research_progress":      0.0,
            "research_progress_text": "Esperando inicio de investigación..."
        }
        for key, default in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default

    # ------------------------------------------------------------------
    # Constructor de Estrategias
    # ------------------------------------------------------------------

    @staticmethod
    def save_builder_compiled(strategy_class: Any) -> None:
        st.session_state.builder_compiled = strategy_class
        st.session_state.builder_compiled_name = strategy_class.__name__ if strategy_class else None

    @staticmethod
    def get_builder_compiled() -> Optional[Any]:
        return st.session_state.get("builder_compiled")

    @staticmethod
    def save_builder_backtest(results: Any) -> None:
        st.session_state.builder_backtest = results

    @staticmethod
    def get_builder_backtest() -> Optional[Any]:
        return st.session_state.get("builder_backtest")

    @staticmethod
    def save_builder_optimization(history_df: Any, best_params: dict) -> None:
        st.session_state.builder_opt_history = history_df
        st.session_state.builder_opt_best = best_params

    @staticmethod
    def get_builder_optimization() -> dict:
        return {
            "history": st.session_state.get("builder_opt_history"),
            "best_params": st.session_state.get("builder_opt_best", {}),
        }

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    @staticmethod
    def save_backtest(key: str, results: Any) -> None:
        st.session_state.backtest_results[key] = results
        st.session_state.last_backtest_key = key

    @staticmethod
    def get_backtest(key: str) -> Optional[Any]:
        return st.session_state.backtest_results.get(key)

    @staticmethod
    def has_backtest(key: str) -> bool:
        return key in st.session_state.backtest_results

    @staticmethod
    def save_live_config(strategy_name: str, asset: str, timeframe: str,
                          params: dict, config: dict) -> None:
        """Guarda los parámetros de estrategia y la configuración de riesgo/capital
        del último backtest ejecutado para (estrategia, activo, temporalidad), de
        forma que Mercado en Vivo pueda reproducir exactamente esas mismas entradas."""
        st.session_state.last_live_configs[(strategy_name, asset, timeframe)] = {
            "params": params, "config": config,
        }

    @staticmethod
    def get_live_config(strategy_name: str, asset: str, timeframe: str) -> Optional[dict]:
        return st.session_state.last_live_configs.get((strategy_name, asset, timeframe))

    @staticmethod
    def get_last_backtest() -> Optional[Any]:
        key = st.session_state.get("last_backtest_key")
        if key:
            return st.session_state.backtest_results.get(key)
        return None

    @staticmethod
    def list_backtests() -> list:
        return list(st.session_state.backtest_results.keys())

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------

    @staticmethod
    def save_portfolio(results: Any) -> None:
        st.session_state.portfolio_results = results

    @staticmethod
    def get_portfolio() -> Optional[Any]:
        return st.session_state.portfolio_results

    @staticmethod
    def save_portfolio_config(config: dict, weights: dict) -> None:
        st.session_state.portfolio_config = config
        st.session_state.asset_weights = weights

    @staticmethod
    def get_portfolio_config() -> tuple[dict, dict]:
        return (
            st.session_state.portfolio_config,
            st.session_state.asset_weights,
        )

    # ------------------------------------------------------------------
    # Optimización
    # ------------------------------------------------------------------

    @staticmethod
    def save_optimization(history_df: Any, best_params: dict,
                           strategy: str, asset: str, timeframe: str,
                           allowed_regimes: Optional[list] = None) -> None:
        st.session_state.opt_history = history_df
        st.session_state.opt_best_params = best_params
        st.session_state.opt_strategy = strategy
        st.session_state.opt_asset = asset
        st.session_state.opt_timeframe = timeframe
        st.session_state.opt_allowed_regimes = allowed_regimes or []

    @staticmethod
    def get_optimization() -> dict:
        return {
            "history":    st.session_state.opt_history,
            "best_params": st.session_state.opt_best_params,
            "strategy":   st.session_state.opt_strategy,
            "asset":      st.session_state.opt_asset,
            "timeframe":  st.session_state.opt_timeframe,
            "allowed_regimes": st.session_state.get("opt_allowed_regimes", []),
        }

    @staticmethod
    def save_wfo(results: Any) -> None:
        st.session_state.wfo_results = results

    @staticmethod
    def get_wfo() -> Optional[Any]:
        return st.session_state.wfo_results

    @staticmethod
    def save_cpcv(results: Any) -> None:
        st.session_state.cpcv_results = results

    @staticmethod
    def get_cpcv() -> Optional[Any]:
        return st.session_state.cpcv_results

    @staticmethod
    def save_mc(results: Any) -> None:
        st.session_state.mc_results = results

    @staticmethod
    def get_mc() -> Optional[Any]:
        return st.session_state.mc_results

    @staticmethod
    def save_sensitivity(result: Any) -> None:
        st.session_state.sensitivity_result = result

    @staticmethod
    def get_sensitivity() -> Optional[Any]:
        return st.session_state.sensitivity_result

    # ------------------------------------------------------------------
    # Datos
    # ------------------------------------------------------------------

    @staticmethod
    def cache_data(asset: str, timeframe: str, df: Any) -> None:
        st.session_state.loaded_data[(asset, timeframe)] = df

    @staticmethod
    def get_cached_data(asset: str, timeframe: str) -> Optional[Any]:
        return st.session_state.loaded_data.get((asset, timeframe))

    @staticmethod
    def set_historical_data_source(source: str) -> None:
        if source not in ("MT5 / datos originales", "Datos limpios / verificados"):
            source = "MT5 / datos originales"
        st.session_state.historical_data_source = source

    @staticmethod
    def get_historical_data_source() -> str:
        return st.session_state.get("historical_data_source", "MT5 / datos originales")

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    @staticmethod
    def clear_all() -> None:
        """Reinicia toda la sesión."""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        AppState.init()

    @staticmethod
    def clear_results() -> None:
        """Limpia solo resultados (mantiene configuración)."""
        for key in ["backtest_results", "portfolio_results", "opt_history",
                    "wfo_results", "cpcv_results", "mc_results", "sensitivity_result",
                    "builder_backtest", "builder_opt_history"]:
            if key == "backtest_results":
                st.session_state[key] = {}
            else:
                st.session_state[key] = None
