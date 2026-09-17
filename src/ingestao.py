from __future__ import annotations

import pandas as pd

COLUNAS_DATA = ("data",)
COLUNAS_SKU = ("sku", "produto")
COLUNAS_QTDE = ("target", "qtde")
COLUNAS_FORNECEDOR = ("fornecedor",)
FORNECEDOR_PADRAO = "—"
ABC_PADRAO = "—"

# Janela diária suficiente p/ MA30 + gráfico (60d) + previsão.
JANELA_CALENDARIO_DIAS = 90

# Contrato CSV volume (SELECT / export): DATA,SKU,TARGET,FORNECEDOR
# TARGET = quantidade em caixas (volume). Alias interno: qtde.
COLS_QTDE_OBRIG = {
    "data": COLUNAS_DATA,
    "sku": COLUNAS_SKU,
    "qtde": COLUNAS_QTDE,
}
COLS_QTDE_OPC = {
    "fornecedor": COLUNAS_FORNECEDOR,
}


def _normalizar_coluna(nome: str) -> str:
    return str(nome).strip().lower().replace(" ", "_")


def _resolver_coluna(colunas: list[str], candidatos: tuple[str, ...]) -> str:
    mapa = {_normalizar_coluna(c): c for c in colunas}
    for candidato in candidatos:
        if candidato in mapa:
            return mapa[candidato]
    raise ValueError(
        f"Coluna obrigatória não encontrada. Esperado uma de: {', '.join(candidatos)}. "
        f"Recebido: {', '.join(colunas)}"
    )


def _resolver_coluna_opcional(colunas: list[str], candidatos: tuple[str, ...]) -> str | None:
    mapa = {_normalizar_coluna(c): c for c in colunas}
    for candidato in candidatos:
        if candidato in mapa:
            return mapa[candidato]
    return None


def _listar_headers(origem) -> list[str]:
    if isinstance(origem, pd.DataFrame):
        return list(origem.columns)
    headers = list(pd.read_csv(origem, nrows=0, encoding="utf-8-sig").columns)
    if hasattr(origem, "seek"):
        origem.seek(0)
    return headers


def _usecols_resolvidos(
    headers: list[str],
    obrigatorias: dict[str, tuple[str, ...]],
    opcionais: dict[str, tuple[str, ...]] | None = None,
) -> list[str]:
    """Resolve nomes reais no arquivo e devolve lista única para usecols."""
    escolhidas: list[str] = []
    for candidatos in obrigatorias.values():
        escolhidas.append(_resolver_coluna(headers, candidatos))
    if opcionais:
        for candidatos in opcionais.values():
            col = _resolver_coluna_opcional(headers, candidatos)
            if col is not None:
                escolhidas.append(col)
    # preserva ordem, sem duplicata
    vistos: set[str] = set()
    saida: list[str] = []
    for col in escolhidas:
        if col not in vistos:
            vistos.add(col)
            saida.append(col)
    return saida


def ler_csv_colunas(
    origem,
    obrigatorias: dict[str, tuple[str, ...]],
    opcionais: dict[str, tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    """
    Descrição: Lê CSV/DataFrame só com as colunas do contrato (usecols).
    Parâmetros: origem; obrigatorias/opcionais = chave lógica → candidatos de nome
    Retorno: DataFrame com as colunas físicas selecionadas
    """
    headers = _listar_headers(origem)
    usecols = _usecols_resolvidos(headers, obrigatorias, opcionais)
    if isinstance(origem, pd.DataFrame):
        return origem.loc[:, usecols].copy()
    bruto = pd.read_csv(origem, encoding="utf-8-sig", usecols=usecols)
    if hasattr(origem, "seek"):
        origem.seek(0)
    return bruto


def _fornecedor_predominante(serie: pd.Series) -> str:
    limpa = serie.dropna().astype(str).str.strip()
    limpa = limpa[limpa != ""]
    if limpa.empty:
        return FORNECEDOR_PADRAO
    return str(limpa.mode().iloc[0])


def carregar_qtde(origem) -> pd.DataFrame:
    """
    Descrição: Lê CSV de volume e padroniza data, sku, qtde e fornecedor.
    Parâmetros: origem (caminho, buffer ou DataFrame)
    Retorno: DataFrame com data, sku, qtde (caixas) e fornecedor (opcional)
    Contrato: DATA,SKU,TARGET,FORNECEDOR (TARGET = volume em caixas)
    """
    bruto = ler_csv_colunas(origem, COLS_QTDE_OBRIG, COLS_QTDE_OPC)

    col_data = _resolver_coluna(list(bruto.columns), COLUNAS_DATA)
    col_sku = _resolver_coluna(list(bruto.columns), COLUNAS_SKU)
    col_qtde = _resolver_coluna(list(bruto.columns), COLUNAS_QTDE)
    col_fornecedor = _resolver_coluna_opcional(list(bruto.columns), COLUNAS_FORNECEDOR)

    # Aceita ISO (2026-03-20 ...) e BR (20/03/2026).
    datas = pd.to_datetime(bruto[col_data], errors="coerce", format="mixed", dayfirst=True)
    saida = pd.DataFrame(
        {
            "data": datas,
            "sku": bruto[col_sku].astype(str).str.strip(),
            "qtde": pd.to_numeric(bruto[col_qtde], errors="coerce"),
        }
    )
    if col_fornecedor is not None:
        saida["fornecedor"] = (
            bruto[col_fornecedor]
            .astype(str)
            .str.strip()
            .replace({"": FORNECEDOR_PADRAO, "nan": FORNECEDOR_PADRAO})
        )
    else:
        saida["fornecedor"] = FORNECEDOR_PADRAO

    saida = saida.dropna(subset=["data", "sku"])
    saida["qtde"] = saida["qtde"].fillna(0).clip(lower=0)

    fornecedor_map = saida.groupby("sku", as_index=True)["fornecedor"].agg(_fornecedor_predominante)
    saida = (
        saida.groupby(["sku", "data"], as_index=False)["qtde"]
        .sum()
        .sort_values(["sku", "data"])
        .reset_index(drop=True)
    )
    # Mantém fornecedor só para extrair mapa sku→fornecedor antes do calendário lean.
    saida["fornecedor"] = saida["sku"].map(fornecedor_map).fillna(FORNECEDOR_PADRAO)
    if saida.empty:
        raise ValueError("Nenhuma linha válida após a leitura do arquivo de volume (QTDE).")
    return saida


def mapa_fornecedor_por_sku(movimento: pd.DataFrame) -> pd.Series:
    """Extrai Series sku → fornecedor (1 valor por SKU)."""
    if "fornecedor" not in movimento.columns:
        return pd.Series(dtype=str)
    return movimento.groupby("sku", sort=False)["fornecedor"].agg(_fornecedor_predominante)


def completar_calendario(
    movimento: pd.DataFrame,
    janela_max_dias: int = JANELA_CALENDARIO_DIAS,
) -> pd.DataFrame:
    """
    Descrição: Preenche dias ausentes por SKU com qtde 0 (apenas data/sku/qtde).
    Parâmetros:
      movimento — DataFrame padronizado
      janela_max_dias — mantém só o fim da série (default 90); 0 = série inteira
    Retorno: DataFrame diário contínuo lean por SKU
    """
    if movimento.empty:
        return pd.DataFrame(columns=["data", "sku", "qtde"])

    base = movimento[["sku", "data", "qtde"]].copy()
    base["data"] = pd.to_datetime(base["data"])
    base["sku"] = base["sku"].astype(str)
    base = base.groupby(["sku", "data"], as_index=False)["qtde"].sum()

    if janela_max_dias and janela_max_dias > 0:
        fim = base.groupby("sku")["data"].transform("max")
        corte = fim - pd.Timedelta(days=int(janela_max_dias) - 1)
        base = base.loc[base["data"] >= corte]

    blocos = []
    for sku, grupo in base.groupby("sku", sort=False):
        serie = grupo.set_index("data")["qtde"].sort_index()
        calendario = pd.date_range(serie.index.min(), serie.index.max(), freq="D")
        serie = serie.reindex(calendario, fill_value=0.0)
        blocos.append(
            pd.DataFrame(
                {
                    "data": serie.index,
                    "sku": sku,
                    "qtde": serie.to_numpy(dtype=float),
                }
            )
        )
    if not blocos:
        return pd.DataFrame(columns=["data", "sku", "qtde"])
    return pd.concat(blocos, ignore_index=True)


def serie_do_sku(movimento: pd.DataFrame, sku: str) -> pd.Series:
    recorte = movimento.loc[movimento["sku"] == str(sku), ["data", "qtde"]]
    if recorte.empty:
        raise ValueError(f"SKU {sku} não encontrado.")
    return recorte.set_index("data")["qtde"].sort_index()


def filtrar_por_fornecedor(df: pd.DataFrame, fornecedor: str | None) -> pd.DataFrame:
    if not fornecedor or fornecedor == "Todos":
        return df
    if "fornecedor_rotulo" in df.columns:
        return df.loc[df["fornecedor_rotulo"] == fornecedor].copy()
    if "fornecedor" in df.columns:
        return df.loc[df["fornecedor"] == fornecedor].copy()
    return df


def filtrar_por_abc(df: pd.DataFrame, abc: str | None) -> pd.DataFrame:
    if not abc or abc == "Todos":
        return df
    if "abc" not in df.columns:
        return df
    return df.loc[df["abc"].astype(str).str.upper() == str(abc).upper()].copy()


def listar_fornecedores(df: pd.DataFrame) -> list[str]:
    """Lista fornecedores com nome/rótulo preenchido (espera ranking ou cadastro)."""
    from src.cadastro import nome_fornecedor_valido

    if "nome_fornecedor" in df.columns and "fornecedor_rotulo" in df.columns:
        com_nome = df.loc[
            df["nome_fornecedor"].map(nome_fornecedor_valido),
            "fornecedor_rotulo",
        ]
        return sorted(
            {
                str(v)
                for v in com_nome.dropna().unique()
                if str(v).strip() and str(v) != FORNECEDOR_PADRAO
            }
        )

    chave = "fornecedor_rotulo" if "fornecedor_rotulo" in df.columns else "fornecedor"
    if chave not in df.columns:
        return []
    valores = sorted({str(v) for v in df[chave].dropna().unique() if str(v).strip()})
    return [v for v in valores if nome_fornecedor_valido(v)]


def listar_abc(df: pd.DataFrame) -> list[str]:
    if "abc" not in df.columns:
        return []
    return sorted(
        {
            str(v).upper()
            for v in df["abc"].dropna().unique()
            if str(v).strip() and str(v).strip() != ABC_PADRAO
        }
    )
