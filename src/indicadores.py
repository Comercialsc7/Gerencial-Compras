from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.cadastro import COLS_CADASTRO
from src.ingestao import FORNECEDOR_PADRAO

# Prioridade de atenção com base no índice de consumo (MA7/MA30).
SCORE_FAIXA = {
    "aumento_forte": 90,
    "queda": 75,
    "aumento_moderado": 55,
    "indefinido": 40,
    "normal": 10,
}

ROTULO_FAIXA = {
    "indefinido": "Histórico insuficiente",
    "queda": "Consumo em queda",
    "normal": "Consumo normal",
    "aumento_moderado": "Aumento moderado",
    "aumento_forte": "Aumento forte",
}


@dataclass(frozen=True)
class IndicadoresConsumo:
    media_7: float
    media_14: float
    media_30: float
    indice: float | None
    faixa: str
    rotulo: str
    queda_percentual: float | None
    score_risco: int
    nivel_risco: str


def _media_janela(serie: pd.Series, janela: int) -> float:
    if serie.empty:
        return 0.0
    return float(serie.tail(janela).mean())


def calcular_medias(serie: pd.Series) -> dict[str, float]:
    """
    Descrição: Calcula médias móveis de qtde (caixas) em 7, 14 e 30 dias.
    Parâmetros: serie (pd.Series diária de qtde)
    Retorno: dict com media_7, media_14 e media_30
    """
    return {
        "media_7": _media_janela(serie, 7),
        "media_14": _media_janela(serie, 14),
        "media_30": _media_janela(serie, 30),
    }


def classificar_indice(indice: float | None) -> tuple[str, str]:
    """
    Descrição: Classifica o índice MA7/MA30 nas faixas de consumo.
    Parâmetros: indice (float ou None)
    Retorno: (faixa, rotulo)
    """
    if indice is None or (isinstance(indice, float) and np.isnan(indice)):
        return "indefinido", ROTULO_FAIXA["indefinido"]
    if indice < 0.80:
        return "queda", ROTULO_FAIXA["queda"]
    if indice < 1.20:
        return "normal", ROTULO_FAIXA["normal"]
    if indice <= 1.50:
        return "aumento_moderado", ROTULO_FAIXA["aumento_moderado"]
    return "aumento_forte", ROTULO_FAIXA["aumento_forte"]


def calcular_score_risco(faixa: str, indice: float | None) -> tuple[int, str]:
    """
    Descrição: Converte faixa/índice em score 0–100 e nível de risco.
    Parâmetros: faixa (str), indice (float ou None)
    Retorno: (score_risco, nivel_risco)
    """
    base = SCORE_FAIXA.get(faixa, 40)
    if indice is not None and not (isinstance(indice, float) and np.isnan(indice)):
        base = min(100, base + int(min(9, abs(indice - 1.0) * 20)))
    if base >= 80:
        nivel = "Alto"
    elif base >= 50:
        nivel = "Médio"
    else:
        nivel = "Baixo"
    return int(base), nivel


def calcular_indicadores(serie: pd.Series) -> IndicadoresConsumo:
    """
    Descrição: Consolida médias, índice de consumo e classificação.
    Parâmetros: serie (pd.Series diária de qtde)
    Retorno: IndicadoresConsumo
    """
    medias = calcular_medias(serie)
    media_7 = medias["media_7"]
    media_30 = medias["media_30"]

    if media_30 <= 0:
        indice = None
        queda = None
    else:
        indice = media_7 / media_30
        queda = max(0.0, (1 - indice) * 100) if indice < 1 else 0.0

    faixa, rotulo = classificar_indice(indice)
    score, nivel = calcular_score_risco(faixa, indice)
    return IndicadoresConsumo(
        media_7=media_7,
        media_14=medias["media_14"],
        media_30=media_30,
        indice=indice,
        faixa=faixa,
        rotulo=rotulo,
        queda_percentual=queda,
        score_risco=score,
        nivel_risco=nivel,
    )


def _medias_por_sku(movimento: pd.DataFrame) -> pd.DataFrame:
    """Médias das últimas 7/14/30 linhas por SKU (calendário já contínuo)."""
    ordenado = movimento.sort_values(["sku", "data"], kind="mergesort").copy()
    ordenado["_from_end"] = ordenado.groupby("sku", sort=False).cumcount(ascending=False)
    return pd.DataFrame(
        {
            "media_7": ordenado.loc[ordenado["_from_end"] < 7].groupby("sku", sort=False)["qtde"].mean(),
            "media_14": ordenado.loc[ordenado["_from_end"] < 14].groupby("sku", sort=False)["qtde"].mean(),
            "media_30": ordenado.loc[ordenado["_from_end"] < 30].groupby("sku", sort=False)["qtde"].mean(),
        }
    )


def _classificar_indice_vetor(indice: pd.Series) -> pd.Series:
    faixa = pd.Series("indefinido", index=indice.index, dtype="object")
    valido = indice.notna()
    faixa = faixa.mask(valido & (indice < 0.80), "queda")
    faixa = faixa.mask(valido & (indice >= 0.80) & (indice < 1.20), "normal")
    faixa = faixa.mask(valido & (indice >= 1.20) & (indice <= 1.50), "aumento_moderado")
    faixa = faixa.mask(valido & (indice > 1.50), "aumento_forte")
    return faixa


def _score_risco_vetor(faixa: pd.Series, indice: pd.Series) -> tuple[pd.Series, pd.Series]:
    base = faixa.map(SCORE_FAIXA).fillna(40).astype(float)
    ajuste = (indice - 1.0).abs().clip(upper=0.45) * 20
    ajuste = ajuste.clip(upper=9).fillna(0)
    score = (base + ajuste).clip(upper=100).astype(int)
    nivel = pd.Series(
        np.where(score >= 80, "Alto", np.where(score >= 50, "Médio", "Baixo")),
        index=score.index,
    )
    return score, nivel


def _meta_do_cadastro(skus: pd.Index, cadastro: pd.DataFrame | None) -> pd.DataFrame:
    """Uma linha de metadados por SKU — sem repetir no calendário diário."""
    sku_list = [str(s) for s in skus]
    if cadastro is None or cadastro.empty:
        meta = pd.DataFrame({"sku": sku_list}).set_index("sku")
        for col in COLS_CADASTRO:
            if col == "sku":
                continue
            if col in ("nome_produto", "desc_reduzida"):
                meta[col] = meta.index.astype(str)
            else:
                meta[col] = FORNECEDOR_PADRAO
        meta["fornecedor"] = meta["nome_fornecedor"]
        return meta

    meta = cadastro.copy()
    meta["sku"] = meta["sku"].astype(str)
    meta = meta.drop_duplicates(subset=["sku"], keep="first").set_index("sku")
    meta = meta.reindex(sku_list)
    meta.index.name = "sku"

    if "nome_produto" not in meta.columns:
        meta["nome_produto"] = pd.NA
    meta["nome_produto"] = meta["nome_produto"].fillna(pd.Series(sku_list, index=meta.index))
    if "desc_reduzida" not in meta.columns:
        meta["desc_reduzida"] = meta["nome_produto"]
    else:
        meta["desc_reduzida"] = meta["desc_reduzida"].fillna(meta["nome_produto"])

    for col in COLS_CADASTRO:
        if col == "sku":
            continue
        if col not in meta.columns:
            meta[col] = FORNECEDOR_PADRAO
        else:
            meta[col] = meta[col].fillna(FORNECEDOR_PADRAO)

    meta["fornecedor"] = meta["nome_fornecedor"]
    return meta


def ranking_risco(
    movimento: pd.DataFrame,
    cadastro: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Descrição: Ranking vetorizado; metadados vêm do cadastro (1x por SKU).
    Parâmetros:
      movimento — série diária lean (data, sku, qtde)
      cadastro — opcional, 1 linha por SKU com fornecedor/ABC/nome
    Retorno: DataFrame ordenado por score_risco desc
    """
    if movimento.empty:
        return pd.DataFrame()

    medias = _medias_por_sku(movimento)
    media_7 = medias["media_7"]
    media_30 = medias["media_30"]

    indice = media_7 / media_30
    indice = indice.where(media_30 > 0)
    queda = ((1 - indice).clip(lower=0) * 100).where(indice < 1, 0.0)
    queda = queda.where(indice.notna())

    faixa = _classificar_indice_vetor(indice)
    rotulo = faixa.map(ROTULO_FAIXA)
    score, nivel = _score_risco_vetor(faixa, indice)

    ranking = _meta_do_cadastro(medias.index, cadastro)
    ranking["media_7"] = media_7
    ranking["media_14"] = medias["media_14"]
    ranking["media_30"] = media_30
    ranking["indice"] = indice
    ranking["queda_percentual"] = queda
    ranking["faixa"] = faixa
    ranking["rotulo"] = rotulo
    ranking["score_risco"] = score
    ranking["nivel_risco"] = nivel

    ranking = ranking.reset_index()
    if "sku" not in ranking.columns and "index" in ranking.columns:
        ranking = ranking.rename(columns={"index": "sku"})
    return ranking.sort_values(
        ["score_risco", "sku"], ascending=[False, True]
    ).reset_index(drop=True)
