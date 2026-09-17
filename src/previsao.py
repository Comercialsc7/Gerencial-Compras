from __future__ import annotations

from functools import lru_cache

import pandas as pd

MOTORES = ("padrao_semanal", "chronos2")


def prever(
    serie: pd.Series,
    horizonte: int = 14,
    motor: str = "chronos2",
    sku: str = "sku",
) -> pd.DataFrame:
    """
    Descrição: Gera previsão diária no horizonte pedido.
    Parâmetros: serie (qtde diária em caixas), horizonte (int), motor (str), sku (str)
    Retorno: DataFrame com data, sku, previsao, p10, p90, motor
    """
    if horizonte < 1:
        raise ValueError("Horizonte deve ser pelo menos 1 dia.")
    if serie.empty:
        raise ValueError("Série vazia: não há o que prever.")

    serie = serie.sort_index().astype(float)
    if motor == "chronos2":
        return _prever_chronos(serie, horizonte=horizonte, sku=sku)
    if motor == "padrao_semanal":
        return _prever_padrao_semanal(serie, horizonte=horizonte, sku=sku)
    raise ValueError(f"Motor desconhecido: {motor}. Use um de {MOTORES}.")


def _prever_padrao_semanal(serie: pd.Series, horizonte: int, sku: str) -> pd.DataFrame:
    """Repete a última semana (mesmo weekday). Piso em zero."""
    ultima_semana = serie.tail(7)
    media_weekday = ultima_semana.groupby(ultima_semana.index.weekday).mean()
    media_7 = float(serie.tail(7).mean())
    desvio = float(ultima_semana.std(ddof=0) or 0.0)

    datas = pd.date_range(serie.index.max() + pd.Timedelta(days=1), periods=horizonte, freq="D")
    pontos = []
    for data in datas:
        if data.weekday() in media_weekday.index:
            ponto = float(media_weekday.loc[data.weekday()])
        else:
            ponto = media_7
        pontos.append(max(0.0, ponto))

    previsao = pd.Series(pontos, index=datas)
    faixa = max(desvio, media_7 * 0.15, 1.0)
    return pd.DataFrame(
        {
            "data": previsao.index,
            "sku": sku,
            "previsao": previsao.to_numpy(),
            "p10": (previsao - faixa).clip(lower=0).to_numpy(),
            "p90": (previsao + faixa).to_numpy(),
            "motor": "padrao_semanal",
        }
    )


@lru_cache(maxsize=1)
def _pipeline_chronos():
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")


def chronos_disponivel() -> bool:
    try:
        import chronos  # noqa: F401
    except ImportError:
        return False
    return True


def _prever_chronos(serie: pd.Series, horizonte: int, sku: str) -> pd.DataFrame:
    if not chronos_disponivel():
        raise RuntimeError(
            "Chronos-2 não está instalado. Rode: pip install 'chronos-forecasting>=2.2'"
        )

    pipeline = _pipeline_chronos()
    contexto = pd.DataFrame(
        {
            "id": sku,
            "timestamp": serie.index,
            "target": serie.to_numpy(dtype=float),
        }
    )
    pred = pipeline.predict_df(
        contexto,
        prediction_length=horizonte,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
        freq="D",
    )
    ponto = pred["predictions"] if "predictions" in pred.columns else pred["0.5"]
    return pd.DataFrame(
        {
            "data": pd.to_datetime(pred["timestamp"]),
            "sku": pred.get("id", sku),
            "previsao": ponto.clip(lower=0).to_numpy(),
            "p10": pred["0.1"].clip(lower=0).to_numpy() if "0.1" in pred.columns else ponto.to_numpy(),
            "p90": pred["0.9"].clip(lower=0).to_numpy() if "0.9" in pred.columns else ponto.to_numpy(),
            "motor": "chronos2",
        }
    )
