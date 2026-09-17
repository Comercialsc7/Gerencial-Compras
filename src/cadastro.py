from __future__ import annotations

import pandas as pd

from src.ingestao import (
    ABC_PADRAO,
    FORNECEDOR_PADRAO,
    _resolver_coluna,
    _resolver_coluna_opcional,
    ler_csv_colunas,
)

COLS_CADASTRO = (
    "sku",
    "cod_fornecedor",
    "nome_fornecedor",
    "fornecedor_rotulo",
    "nome_produto",
    "desc_reduzida",
    "abc",
    "abc_rentabilidade",
    "categoria",
    "familia",
)

# Contrato CSV produtos (SELECT / export):
# CODFORNEC,NOMEFORN,PRODUTO,NOMEPRODUTO,DESCREDUZIDA,CLASSIFICACAOABC,
# ABCRENTPRODUTO,NOMECATEGORIA,NOMEFAMILIA,SEGMENTO,DATAATUALIZACAO
COLS_PRODUTOS_OBRIG = {
    "cod_fornecedor": ("codfornec",),
    "nome_fornecedor": ("nomeforn",),
    "sku": ("produto",),
    "nome_produto": ("nomeproduto",),
}
COLS_PRODUTOS_OPC = {
    "abc": ("classificacaoabc",),
    "abc_rentabilidade": ("abcrentproduto",),
    "desc_reduzida": ("descreduzida",),
    "categoria": ("nomecategoria",),
    "familia": ("nomefamilia",),
    "segmento": ("segmento",),
    "data_atualizacao": ("dataatualizacao",),
}


def nome_fornecedor_valido(nome) -> bool:
    if nome is None or (isinstance(nome, float) and pd.isna(nome)):
        return False
    texto = str(nome).strip()
    return bool(texto) and texto.lower() not in {"nan", "none", "null", FORNECEDOR_PADRAO, "-"}


def rotulo_fornecedor(cod: str, nome: str) -> str:
    cod = str(cod).strip()
    nome = str(nome).strip() if nome else FORNECEDOR_PADRAO
    if not cod or cod.lower() in {"nan", "none", FORNECEDOR_PADRAO}:
        return nome or FORNECEDOR_PADRAO
    try:
        cod = str(int(float(cod)))
    except (TypeError, ValueError):
        pass
    return f"{cod} | {nome}"


def carregar_produtos(origem) -> pd.DataFrame:
    """
    Descrição: Lê cadastro de produtos e deduplica por SKU (só colunas do contrato).
    Parâmetros: origem (caminho, buffer ou DataFrame)
    Retorno: DataFrame com um registro por sku (preferência SEGMENTO 1)
    """
    bruto = ler_csv_colunas(origem, COLS_PRODUTOS_OBRIG, COLS_PRODUTOS_OPC)
    colunas = list(bruto.columns)

    col_cod_forn = _resolver_coluna(colunas, COLS_PRODUTOS_OBRIG["cod_fornecedor"])
    col_nome_forn = _resolver_coluna(colunas, COLS_PRODUTOS_OBRIG["nome_fornecedor"])
    col_sku = _resolver_coluna(colunas, COLS_PRODUTOS_OBRIG["sku"])
    col_nome_prod = _resolver_coluna(colunas, COLS_PRODUTOS_OBRIG["nome_produto"])
    col_abc = _resolver_coluna_opcional(colunas, COLS_PRODUTOS_OPC["abc"])
    col_abc_rent = _resolver_coluna_opcional(colunas, COLS_PRODUTOS_OPC["abc_rentabilidade"])
    col_desc = _resolver_coluna_opcional(colunas, COLS_PRODUTOS_OPC["desc_reduzida"])
    col_categ = _resolver_coluna_opcional(colunas, COLS_PRODUTOS_OPC["categoria"])
    col_fam = _resolver_coluna_opcional(colunas, COLS_PRODUTOS_OPC["familia"])
    col_seg = _resolver_coluna_opcional(colunas, COLS_PRODUTOS_OPC["segmento"])
    col_atualizacao = _resolver_coluna_opcional(colunas, COLS_PRODUTOS_OPC["data_atualizacao"])

    saida = pd.DataFrame(
        {
            "cod_fornecedor": bruto[col_cod_forn].astype(str).str.strip(),
            "nome_fornecedor": bruto[col_nome_forn].astype(str).str.strip(),
            "sku": bruto[col_sku].astype(str).str.strip(),
            "nome_produto": bruto[col_nome_prod].astype(str).str.strip(),
        }
    )
    saida["abc"] = (
        bruto[col_abc].astype(str).str.strip().str.upper() if col_abc else ABC_PADRAO
    )
    saida["abc_rentabilidade"] = (
        bruto[col_abc_rent].astype(str).str.strip().str.upper() if col_abc_rent else ABC_PADRAO
    )
    saida["desc_reduzida"] = (
        bruto[col_desc].astype(str).str.strip() if col_desc else saida["nome_produto"]
    )
    saida["categoria"] = bruto[col_categ].astype(str).str.strip() if col_categ else ABC_PADRAO
    saida["familia"] = bruto[col_fam].astype(str).str.strip() if col_fam else ABC_PADRAO
    saida["segmento"] = (
        pd.to_numeric(bruto[col_seg], errors="coerce").fillna(999).astype(int)
        if col_seg
        else 999
    )
    saida["data_atualizacao"] = (
        pd.to_datetime(bruto[col_atualizacao], errors="coerce")
        if col_atualizacao
        else pd.NaT
    )

    saida = saida.dropna(subset=["sku"])
    saida = saida.loc[saida["sku"] != ""].copy()
    saida = saida.sort_values(
        ["sku", "segmento", "data_atualizacao"],
        ascending=[True, True, False],
        na_position="last",
    )
    saida = saida.drop_duplicates(subset=["sku"], keep="first").reset_index(drop=True)
    saida["fornecedor_rotulo"] = [
        rotulo_fornecedor(c, n)
        for c, n in zip(saida["cod_fornecedor"], saida["nome_fornecedor"], strict=False)
    ]
    if saida.empty:
        raise ValueError("Nenhum produto válido no cadastro.")
    return saida[list(COLS_CADASTRO)]


def _linha_fallback(sku: str, nome_csv: str) -> dict:
    nome = nome_csv if nome_fornecedor_valido(nome_csv) else FORNECEDOR_PADRAO
    return {
        "sku": sku,
        "cod_fornecedor": FORNECEDOR_PADRAO,
        "nome_fornecedor": nome,
        "fornecedor_rotulo": nome,
        "nome_produto": sku,
        "desc_reduzida": sku,
        "abc": ABC_PADRAO,
        "abc_rentabilidade": ABC_PADRAO,
        "categoria": ABC_PADRAO,
        "familia": ABC_PADRAO,
    }


def montar_cadastro_efetivo(
    skus_movimento: list[str] | set[str] | pd.Index,
    produtos: pd.DataFrame | None,
    fornecedor_por_sku: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Descrição: Cadastro 1 linha/SKU para o ranking (produtos + fallback do CSV).
    Mantém só fornecedores com nome válido — mesmo critério da tela atual.
    """
    skus = sorted({str(s) for s in skus_movimento})
    if not skus:
        return pd.DataFrame(columns=list(COLS_CADASTRO))

    forn = pd.Series(dtype=str)
    if fornecedor_por_sku is not None and not fornecedor_por_sku.empty:
        forn = fornecedor_por_sku.copy()
        forn.index = forn.index.astype(str)

    if produtos is not None and not produtos.empty:
        prod = produtos.copy()
        prod["sku"] = prod["sku"].astype(str)
        no_cadastro = prod.set_index("sku")
        encontrados = no_cadastro.reindex([s for s in skus if s in no_cadastro.index]).reset_index()
        faltantes = [s for s in skus if s not in no_cadastro.index]
        extras = pd.DataFrame(
            [_linha_fallback(s, str(forn.get(s, FORNECEDOR_PADRAO))) for s in faltantes]
        )
        cadastro = pd.concat([encontrados, extras], ignore_index=True) if len(extras) else encontrados
    else:
        cadastro = pd.DataFrame(
            [_linha_fallback(s, str(forn.get(s, FORNECEDOR_PADRAO))) for s in skus]
        )

    for col in COLS_CADASTRO:
        if col not in cadastro.columns:
            cadastro[col] = ABC_PADRAO if "abc" in col else FORNECEDOR_PADRAO

    cadastro = cadastro.loc[cadastro["nome_fornecedor"].map(nome_fornecedor_valido)].copy()
    return cadastro[list(COLS_CADASTRO)].reset_index(drop=True)


def filtrar_qtde_por_skus(movimento: pd.DataFrame, skus) -> pd.DataFrame:
    if movimento.empty:
        return movimento
    permitidos = {str(s) for s in skus}
    return movimento.loc[movimento["sku"].astype(str).isin(permitidos)].copy()
