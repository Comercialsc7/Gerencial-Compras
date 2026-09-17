from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingestao import ler_csv_colunas

COLS_ESTOQUE = (
    "sku",
    "media_dia_erp",
    "estoque",
    "reservada",
    "em_transito",
    "ultima_compra_qtd",
)

# Contrato CSV estoque (SELECT / export):
# SEQPRODUTO,MD_C5,ESTOQUE,RESERVADA,EMTRANSITO,ULT_COMPRA
COLS_ESTOQUE_OBRIG = {
    "sku": ("seqproduto",),
    "media_dia_erp": ("md_c5",),
    "estoque": ("estoque",),
    "reservada": ("reservada",),
    "em_transito": ("emtransito",),
    "ultima_compra_qtd": ("ult_compra",),
}


def carregar_estoque(origem) -> pd.DataFrame:
    """
    Descrição: Lê estoque.csv só com colunas do contrato.
    Parâmetros: origem (caminho, buffer ou DataFrame)
    Retorno: DataFrame 1 linha/SKU com saldos padronizados
    """
    bruto = ler_csv_colunas(origem, COLS_ESTOQUE_OBRIG)
    colunas = {str(c).strip().upper(): c for c in bruto.columns}

    def _col(nome: str) -> str:
        if nome in colunas:
            return colunas[nome]
        raise ValueError(f"Coluna de estoque não encontrada: {nome}")

    saida = pd.DataFrame(
        {
            "sku": bruto[_col("SEQPRODUTO")].astype(str).str.strip(),
            "media_dia_erp": pd.to_numeric(bruto[_col("MD_C5")], errors="coerce").fillna(0.0),
            "estoque": pd.to_numeric(bruto[_col("ESTOQUE")], errors="coerce").fillna(0.0),
            "reservada": pd.to_numeric(bruto[_col("RESERVADA")], errors="coerce").fillna(0.0),
            "em_transito": pd.to_numeric(bruto[_col("EMTRANSITO")], errors="coerce").fillna(0.0),
            "ultima_compra_qtd": pd.to_numeric(bruto[_col("ULT_COMPRA")], errors="coerce"),
        }
    )
    saida = saida.dropna(subset=["sku"])
    saida = saida.loc[saida["sku"] != ""].copy()
    saida = saida.drop_duplicates(subset=["sku"], keep="first").reset_index(drop=True)
    return saida


def _status_cobertura(dias: float | None) -> str:
    if dias is None or pd.isna(dias):
        return "Sem ritmo"
    if dias <= 0:
        return "Sem estoque"
    if dias < 7:
        return "Crítico"
    if dias < 14:
        return "Baixo"
    if dias <= 45:
        return "Adequado"
    return "Alto"


def anexar_estoque_ao_ranking(
    ranking: pd.DataFrame,
    estoque: pd.DataFrame | None,
    dias_planejamento: int = 30,
) -> pd.DataFrame:
    """
    Descrição: Cruza ranking com estoque e calcula cobertura + sugestão de compra.
    Parâmetros:
      ranking — saída de ranking_risco
      estoque — saída de carregar_estoque (ou None)
      dias_planejamento — horizonte para sugestão (ex.: 30 = mês)
    Retorno: ranking enriquecido
    """
    out = ranking.copy()
    if estoque is None or estoque.empty:
        for col in (
            "estoque",
            "reservada",
            "em_transito",
            "media_dia_erp",
            "ultima_compra_qtd",
            "demanda_dia",
            "cobertura_dias",
            "demanda_periodo",
            "sugestao_compra",
            "status_cobertura",
        ):
            out[col] = pd.NA if col != "status_cobertura" else "Sem estoque"
        out["status_cobertura"] = "Sem cadastro de estoque"
        return out

    meta = estoque.set_index("sku")
    out["estoque"] = out["sku"].map(meta["estoque"])
    out["reservada"] = out["sku"].map(meta["reservada"])
    out["em_transito"] = out["sku"].map(meta["em_transito"]).fillna(0.0)
    out["media_dia_erp"] = out["sku"].map(meta["media_dia_erp"])
    out["ultima_compra_qtd"] = out["sku"].map(meta["ultima_compra_qtd"])

    out = atualizar_cobertura_sugestao(out, dias_planejamento=dias_planejamento)
    return out


def atualizar_cobertura_sugestao(
    ranking: pd.DataFrame,
    dias_planejamento: int = 30,
) -> pd.DataFrame:
    """
    Descrição: Recalcula cobertura e sugestão com o horizonte informado.
    Parâmetros: ranking já com colunas de estoque; dias_planejamento (int)
    Retorno: DataFrame com cobertura_dias, demanda_periodo, sugestao_compra, status_cobertura
    """
    out = ranking.copy()
    if "estoque" not in out.columns:
        return out

    media_30 = pd.to_numeric(out.get("media_30"), errors="coerce").fillna(0.0)
    media_erp = pd.to_numeric(out.get("media_dia_erp"), errors="coerce").fillna(0.0)
    out["demanda_dia"] = media_30.where(media_30 > 0, media_erp)

    # ESTOQUE e RESERVADA entram crus do extrato (sem recalcular disponível).
    estoque = pd.to_numeric(out["estoque"], errors="coerce").fillna(0.0)
    em_transito = pd.to_numeric(out.get("em_transito"), errors="coerce").fillna(0.0)
    demanda_dia = pd.to_numeric(out["demanda_dia"], errors="coerce").fillna(0.0)

    out["cobertura_dias"] = (estoque / demanda_dia).where(demanda_dia > 0, pd.NA)
    out["demanda_periodo"] = demanda_dia * float(max(1, int(dias_planejamento)))
    necessidade = out["demanda_periodo"] - estoque - em_transito
    out["sugestao_compra"] = necessidade.clip(lower=0.0)
    out["status_cobertura"] = [
        _status_cobertura(None if pd.isna(v) else float(v)) for v in out["cobertura_dias"]
    ]
    sem_cadastro = out["estoque"].isna()
    out.loc[sem_cadastro, "status_cobertura"] = "Sem cadastro de estoque"
    out.loc[sem_cadastro, "sugestao_compra"] = pd.NA
    out.loc[sem_cadastro, "cobertura_dias"] = pd.NA
    return out


def mtime_arquivo(caminho: Path) -> float:
    if not caminho.exists():
        return 0.0
    return caminho.stat().st_mtime
