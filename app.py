from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.indicadores import ranking_risco
from src.cadastro import carregar_produtos, filtrar_qtde_por_skus, montar_cadastro_efetivo
from src.estoque import (
    anexar_estoque_ao_ranking,
    atualizar_cobertura_sugestao,
    carregar_estoque,
)
from src.ingestao import (
    carregar_qtde,
    completar_calendario,
    filtrar_por_abc,
    filtrar_por_fornecedor,
    listar_abc,
    listar_fornecedores,
    mapa_fornecedor_por_sku,
    serie_do_sku,
)
from src.previsao import chronos_disponivel, prever

ROTULO_MOTOR = {
    "padrao_semanal": "Padrão semanal (rápido)",
    "chronos2": "Chronos-2 (mais preciso)",
}

CONTRATO_QTDE = pd.DataFrame(
    {
        "DATA": ["15/09/2026", "15/09/2026"],
        "SKU": ["16459", "105112"],
        "TARGET": [12.5, 8],
        "FORNECEDOR": ["(opcional)", "(opcional)"],
    }
)

CONTRATO_PRODUTOS = pd.DataFrame(
    {
        "CODFORNEC": [3, 12],
        "NOMEFORN": ["FERRERO", "NESTLE"],
        "PRODUTO": [5903, 16459],
        "NOMEPRODUTO": ["BRACADEIRA TIC TAC", "PRODUTO EXEMPLO"],
        "DESCREDUZIDA": ["BRACADEIRA TIC TAC", "PRODUTO EXEMPLO"],
        "CLASSIFICACAOABC": ["A", "B"],
        "ABCRENTPRODUTO": ["A", "B"],
        "NOMECATEGORIA": ["MULLER", "DOCES"],
        "NOMEFAMILIA": ["MATERIAL MERCHADISING", "CHOCOLATE"],
        "SEGMENTO": [1, 1],
        "DATAATUALIZACAO": ["2014-06-20 15:52:06.000", "2026-01-10 08:00:00.000"],
    }
)

CONTRATO_ESTOQUE = pd.DataFrame(
    {
        "SEQPRODUTO": [14, 16459],
        "MD_C5": [4.38, 2.10],
        "ESTOQUE": [201.40, 48.00],
        "RESERVADA": [22.40, 3.00],
        "EMTRANSITO": [144, 0],
        "ULT_COMPRA": [336, 50],
    }
)


def _fmt_data(valor) -> str:
    return pd.Timestamp(valor).strftime("%d/%m/%Y")


def _fmt_num(valor: float, casas: int = 1) -> str:
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_data(show_spinner="Processando volume, cadastro, estoque e ranking...")
def _pipeline(
    qtde_bytes: bytes,
    produtos_bytes: bytes,
    estoque_bytes: bytes,
) -> tuple[pd.DataFrame, pd.DataFrame, int, bool]:
    """
    Pipeline cacheado: volume (qtde) → cadastro → ranking → estoque.
    Cobertura/sugestão dependem do horizonte e são calculadas fora do cache.
    """
    produtos = carregar_produtos(BytesIO(produtos_bytes))
    estoque = carregar_estoque(BytesIO(estoque_bytes))

    movimento_bruto = carregar_qtde(BytesIO(qtde_bytes))
    forn_por_sku = mapa_fornecedor_por_sku(movimento_bruto)
    movimento = completar_calendario(movimento_bruto)

    cadastro = montar_cadastro_efetivo(
        skus_movimento=movimento["sku"].unique(),
        produtos=produtos,
        fornecedor_por_sku=forn_por_sku,
    )
    movimento = filtrar_qtde_por_skus(movimento, cadastro["sku"])
    ranking = ranking_risco(movimento, cadastro=cadastro)
    ranking = anexar_estoque_ao_ranking(ranking, estoque, dias_planejamento=30)
    n_produtos = len(produtos)
    tem_estoque = estoque is not None and not estoque.empty
    return movimento, ranking, n_produtos, tem_estoque


def _mostrar_contratos() -> None:
    st.info(
        "Envie os **3 arquivos CSV** abaixo para rodar a análise. "
        "Depois dá para trocar esses uploads por consultas SQL direto no pipeline."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Volume (TARGET)**")
        st.caption("DATA · SKU · TARGET (caixas) · FORNECEDOR opcional")
        st.dataframe(CONTRATO_QTDE, hide_index=True, use_container_width=True)
        arquivo_qtde = st.file_uploader(
            "Arquivo de volume",
            type=["csv"],
            key="upload_qtde",
            help="Colunas: DATA, SKU (ou PRODUTO), TARGET (caixas). FORNECEDOR é opcional.",
        )
    with c2:
        st.markdown("**2. Produtos (cadastro)**")
        st.caption("CODFORNEC · NOMEFORN · PRODUTO · NOMEPRODUTO · ABC · …")
        st.dataframe(CONTRATO_PRODUTOS, hide_index=True, use_container_width=True)
        arquivo_produtos = st.file_uploader(
            "Arquivo de produtos",
            type=["csv"],
            key="upload_produtos",
            help="Colunas: CODFORNEC, NOMEFORN, PRODUTO, NOMEPRODUTO, CLASSIFICACAOABC, …",
        )
    with c3:
        st.markdown("**3. Estoque**")
        st.caption("SEQPRODUTO · MD_C5 · ESTOQUE · RESERVADA · EMTRANSITO · ULT_COMPRA")
        st.dataframe(CONTRATO_ESTOQUE, hide_index=True, use_container_width=True)
        arquivo_estoque = st.file_uploader(
            "Arquivo de estoque",
            type=["csv"],
            key="upload_estoque",
            help="Colunas: SEQPRODUTO, MD_C5, ESTOQUE, RESERVADA, EMTRANSITO, ULT_COMPRA.",
        )
    return arquivo_qtde, arquivo_produtos, arquivo_estoque


def _voltar_home() -> None:
    for chave in (
        "dados_prontos",
        "qtde_bytes",
        "qtde_nome",
        "produtos_bytes",
        "produtos_nome",
        "estoque_bytes",
        "estoque_nome",
        "previsao_chave",
        "upload_qtde",
        "upload_produtos",
        "upload_estoque",
    ):
        if chave in st.session_state:
            del st.session_state[chave]


def _grafico(historico: pd.Series, previsao: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=historico.index,
            y=historico.values,
            name="Qtde realizada (cx)",
            mode="lines",
            line={"width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=previsao["data"],
            y=previsao["p90"],
            name="Cenário alto (P90)",
            mode="lines",
            line={"width": 0},
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=previsao["data"],
            y=previsao["p10"],
            name="Faixa de incerteza (baixo ↔ alto)",
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(99, 110, 250, 0.18)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=previsao["data"],
            y=previsao["previsao"],
            name="Demanda esperada",
            mode="lines+markers",
            line={"width": 2, "dash": "dash"},
        )
    )
    ultimo = historico.tail(1)
    if not ultimo.empty:
        fig.add_trace(
            go.Scatter(
                x=[ultimo.index[0], previsao["data"].iloc[0]],
                y=[float(ultimo.iloc[0]), float(previsao["previsao"].iloc[0])],
                mode="lines",
                line={"width": 1, "dash": "dot", "color": "rgba(150,150,150,0.6)"},
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        height=420,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        legend={"orientation": "h", "y": 1.08},
        xaxis_title="Data",
        yaxis_title="Caixas / dia",
        hovermode="x unified",
    )
    return fig


def _sinal_tendencia(faixa: str) -> str:
    """Ícones Unicode/emoji legíveis no dataframe do Streamlit."""
    mapa = {
        "queda": "📉",
        "normal": "➡️",
        "aumento_moderado": "📈",
        "aumento_forte": "🔥",
        "indefinido": "—",
    }
    return mapa.get(str(faixa), "—")


def _fmt_indice_com_sinal(indice, faixa: str) -> str:
    if indice is None or (isinstance(indice, float) and pd.isna(indice)):
        return "—"
    return f"{_sinal_tendencia(faixa)} {_fmt_num(float(indice), 2)}"


def _fmt_opcional(valor, casas: int = 1) -> str:
    if valor is None or (isinstance(valor, float) and pd.isna(valor)) or pd.isna(valor):
        return "—"
    return _fmt_num(float(valor), casas)


def _tabela_ranking(ranking: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "#": range(1, len(ranking) + 1),
            "Cód. fornecedor": ranking["cod_fornecedor"],
            "Fornecedor": ranking["nome_fornecedor"],
            "Código produto": ranking["sku"],
            "Descrição": ranking["nome_produto"],
            "ABC": ranking["abc"],
            "Prioridade": ranking["nivel_risco"],
            "Pontuação": ranking["score_risco"],
            "Tendência de consumo": ranking["rotulo"],
            "Média 7 dias": ranking["media_7"].map(lambda x: _fmt_num(x, 1)),
            "MD_C5": ranking["media_dia_erp"].map(lambda x: _fmt_opcional(x, 2)),
            "Média 30 dias": ranking["media_30"].map(lambda x: _fmt_num(x, 1)),
            "Índice de consumo": [
                _fmt_indice_com_sinal(i, f)
                for i, f in zip(ranking["indice"], ranking["faixa"], strict=False)
            ],
            "Estoque": ranking["estoque"].map(lambda x: _fmt_opcional(x, 1)),
            "Reservada": ranking["reservada"].map(lambda x: _fmt_opcional(x, 1)),
            "Em trânsito": ranking["em_transito"].map(lambda x: _fmt_opcional(x, 0)),
            "Cobertura (dias)": ranking["cobertura_dias"].map(lambda x: _fmt_opcional(x, 0)),
            "Status estoque": ranking["status_cobertura"],
            "Sugestão compra": ranking["sugestao_compra"].map(lambda x: _fmt_opcional(x, 0)),
        }
    )


def _legenda_ranking() -> None:
    with st.expander("Como ler este ranking (compras)", expanded=False):
        st.markdown(
            """
**Para que serve:** mostrar primeiro os produtos que merecem atenção do comprador.

| Coluna | Significado |
|--------|-------------|
| **Prioridade** | Alto / Médio / Baixo — ordem sugerida de análise |
| **Pontuação** | Nota de 0 a 100 (quanto maior, mais urgente olhar) |
| **ABC** | Classificação comercial do cadastro (A = mais relevante) |
| **Tendência de consumo** | Se o volume recente está em alta, estável ou em queda |
| **Média 7 dias** | Volume médio dos últimos 7 dias (caixas/dia) |
| **MD_C5** | Média diária do ERP (cadastro de estoque) |
| **Média 30 dias** | Volume médio dos últimos 30 dias (caixas/dia) |
| **Índice de consumo** | Média 7 ÷ Média 30. Perto de **1,00** = estável; **acima de 1,20** = aquecendo; **abaixo de 0,80** = esfriando |
| **Estoque** | Saldo de estoque do extrato (como veio do banco) |
| **Reservada** | Quantidade reservada do extrato (como veio do banco) |
| **Em trânsito** | Quantidade já pedida / a caminho |
| **Cobertura (dias)** | Dias que o estoque cobre, no ritmo atual |
| **Status estoque** | Crítico (&lt;7d) · Baixo (&lt;14d) · Adequado · Alto (&gt;45d) |
| **Sugestão compra** | Quanto falta para cobrir o horizonte (já desconta estoque + trânsito) |

**Sinais no índice de consumo**
- 📉 esfriando (índice &lt; 0,80)
- ➡️ estável (entre 0,80 e 1,20)
- 📈 aquecendo (entre 1,20 e 1,50)
- 🔥 aquecendo forte (acima de 1,50)

**Dica:** comece pelos itens de **Prioridade Alta** e, em seguida, pelos de **Status estoque Crítico/Baixo**.
            """
        )


def _legenda_detalhe() -> None:
    with st.expander("Como ler os indicadores deste produto", expanded=False):
        st.markdown(
            """
| Indicador | Explicação |
|-----------|-------------------------|
| **ABC** | Importância do item no cadastro |
| **Média 7 dias** | Ritmo atual de saída |
| **Média 30 dias** | Ritmo “normal” do último mês |
| **Índice de consumo** | Compara o ritmo atual com o do mês |
| **Prioridade** | Se este SKU merece atenção agora |
| **Estoque / Reservada** | Saldos do extrato, sem recálculo |
| **Em trânsito** | O que já está a caminho |
| **Cobertura** | Quantos dias o estoque cobre no ritmo atual |
| **Sugestão de compra** | Gap para o horizonte escolhido (ex.: 30 dias) |

**Leitura rápida do índice**
- **0,80 a 1,20** → consumo normal  
- **Acima de 1,20** → consumo subindo (risco de faltar se o estoque estiver justo)  
- **Abaixo de 0,80** → consumo caindo (cuidado com excesso / pedido grande)
            """
        )


def _legenda_previsao() -> None:
    st.info(
        "**Como usar na compra:** use a **Demanda esperada** como base do pedido. "
        "O **Cenário alto (P90)** é só um alerta de pico — **não compre no P90**. "
        "O **Cenário baixo (P10)** mostra o piso pessimista de volume."
    )
    with st.expander("O que significam Demanda esperada, P10 e P90?", expanded=True):
        st.markdown(
            """
A previsão olha **cada dia futuro** e estima quantas caixas devem sair.

| Nome na tela | Nome técnico | O que significa |
|--------------|--------------|-----------------|
| **Demanda esperada** | Previsão (centro) | Valor mais provável para aquele dia — **use este para planejar** |
| **Cenário baixo (P10)** | P10 | Cenário fraco: só ~10% de chance de vender **menos** que isso |
| **Cenário alto (P90)** | P90 | Cenário forte: só ~10% de chance de vender **mais** que isso |

**Faixa sombreada no gráfico** = intervalo entre cenário baixo e alto (incerteza).

**Exemplo:** Demanda esperada 200 · P10 50 · P90 800  
→ Planeje perto de **200**. O 800 é um extremo possível, não a meta de compra.

**Horizonte (dias):** quantos dias à frente o sistema projeta (ex.: 14 = duas semanas).
            """
        )


def _rotulo_sku(ranking: pd.DataFrame, sku: str) -> str:
    linha = ranking.loc[ranking["sku"] == sku]
    if linha.empty:
        return sku
    nome = str(linha["nome_produto"].iloc[0])
    return f"{sku} — {nome}" if nome and nome != sku else sku


def _alerta_consumo(meta: pd.Series) -> None:
    faixa = str(meta.get("faixa", ""))
    queda = meta.get("queda_percentual")
    if faixa == "queda" and queda is not None and not pd.isna(queda) and float(queda) > 0:
        st.warning(
            f"Consumo caiu {_fmt_num(float(queda), 0)}% contra a média de 30 dias. "
            "Investigar preço, interesse do cliente, concorrente, sazonalidade, substituto, "
            "problema comercial ou ruptura recente."
        )
    elif faixa == "aumento_forte":
        st.error("Consumo atual está mais de 50% acima da média de 30 dias.")
    elif faixa == "aumento_moderado":
        st.info("Consumo atual está entre 20% e 50% acima da média de 30 dias.")
    elif faixa == "normal":
        st.success("Consumo dentro da faixa normal (índice entre 0,80 e 1,20).")


def main() -> None:
    st.set_page_config(page_title="Gerencial — Compras", layout="wide")
    st.title("Gerencial Compras")
    st.caption(
        "Painel para o time de **compras**: priorizar produtos, cruzar estoque/cobertura "
        "e projetar demanda."
    )

    with st.sidebar:
        if st.button("Home", use_container_width=True, help="Voltar para enviar ou trocar os arquivos"):
            _voltar_home()
            st.rerun()

    if not st.session_state.get("dados_prontos"):
        arquivo_qtde, arquivo_produtos, arquivo_estoque = _mostrar_contratos()

        faltando = []
        if arquivo_qtde is None:
            faltando.append("Volume (TARGET)")
        if arquivo_produtos is None:
            faltando.append("Produtos")
        if arquivo_estoque is None:
            faltando.append("Estoque")

        if faltando:
            st.warning("Faltando: **" + " · ".join(faltando) + "**. Envie os 3 CSVs para calcular.")
            return

        st.session_state["qtde_bytes"] = arquivo_qtde.getvalue()
        st.session_state["qtde_nome"] = arquivo_qtde.name
        st.session_state["produtos_bytes"] = arquivo_produtos.getvalue()
        st.session_state["produtos_nome"] = arquivo_produtos.name
        st.session_state["estoque_bytes"] = arquivo_estoque.getvalue()
        st.session_state["estoque_nome"] = arquivo_estoque.name
        st.session_state["dados_prontos"] = True
        st.rerun()

    try:
        movimento, ranking, n_produtos, tem_estoque = _pipeline(
            st.session_state["qtde_bytes"],
            st.session_state["produtos_bytes"],
            st.session_state["estoque_bytes"],
        )
    except Exception as erro:
        st.error(f"Não foi possível processar os dados: {erro}")
        return

    if movimento.empty or ranking.empty:
        st.warning("Nenhum produto com fornecedor nomeado após o cruzamento com o cadastro.")
        return

    st.sidebar.caption(f"Volume: `{st.session_state['qtde_nome']}`")
    st.sidebar.caption(f"Cadastro: {n_produtos:,} produtos · `{st.session_state['produtos_nome']}`")
    if tem_estoque:
        st.sidebar.caption(f"Estoque: `{st.session_state['estoque_nome']}`")
    else:
        st.sidebar.warning("Estoque carregado sem linhas úteis.")

    fornecedores = listar_fornecedores(ranking)
    abcs = listar_abc(ranking)
    with st.sidebar:
        st.header("Filtros")
        opcoes_fornecedor = ["Todos", *fornecedores] if fornecedores else ["Todos"]
        fornecedor = st.selectbox(
            "Fornecedor",
            options=opcoes_fornecedor,
            help="Filtra a lista pelos fornecedores com nome cadastrado.",
        )
        opcoes_abc = ["Todos", *abcs] if abcs else ["Todos"]
        abc = st.selectbox(
            "Classificação ABC",
            options=opcoes_abc,
            help="A = maior relevância comercial; C/D/E = menor.",
        )
        dias_planejamento = st.slider(
            "Horizonte de compra (dias)",
            min_value=7,
            max_value=60,
            value=30,
            step=1,
            help="Usado na sugestão: demanda do período − estoque − em trânsito.",
        )
    ranking = atualizar_cobertura_sugestao(ranking, dias_planejamento=dias_planejamento)
    ranking_filtrado = filtrar_por_abc(filtrar_por_fornecedor(ranking, fornecedor), abc)
    if ranking_filtrado.empty:
        st.warning("Nenhum produto para os filtros selecionados.")
        return

    skus = ranking_filtrado["sku"].tolist()
    sku_padrao = 0

    st.subheader("1. Onde olhar primeiro")
    st.caption(
        "Lista ordenada pela urgência de análise. "
        "Prioridade Alta = consumo fora do normal; Status estoque = cobertura do saldo."
    )
    _legenda_ranking()

    altos = int((ranking_filtrado["nivel_risco"] == "Alto").sum())
    medios = int((ranking_filtrado["nivel_risco"] == "Médio").sum())
    baixos = int((ranking_filtrado["nivel_risco"] == "Baixo").sum())
    criticos = int(ranking_filtrado["status_cobertura"].isin(["Crítico", "Sem estoque"]).sum())
    com_sugestao = ranking_filtrado["sugestao_compra"].dropna()
    total_sugestao = float(com_sugestao.sum()) if len(com_sugestao) else 0.0

    col_resumo = st.columns(6)
    col_resumo[0].metric("Produtos na lista", len(ranking_filtrado))
    col_resumo[1].metric("Prioridade alta", altos)
    col_resumo[2].metric("Prioridade média", medios)
    col_resumo[3].metric("Prioridade baixa", baixos)
    col_resumo[4].metric("Estoque crítico", criticos)
    col_resumo[5].metric(
        f"Sugestão ({dias_planejamento}d)",
        f"{_fmt_num(total_sugestao, 0)} cx",
    )

    st.dataframe(_tabela_ranking(ranking_filtrado), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("2. Detalhe do produto")
    with st.sidebar:
        sku = st.selectbox(
            "Produto (detalhe)",
            options=skus,
            index=sku_padrao,
            format_func=lambda s: _rotulo_sku(ranking_filtrado, s),
            help="Escolha o item para ver indicadores e gerar a previsão de demanda.",
        )

    meta = ranking_filtrado.loc[ranking_filtrado["sku"] == sku].iloc[0]
    indice = meta["indice"]

    st.markdown(
        f"**{meta['sku']} — {meta['nome_produto']}**  \n"
        f"Fornecedor: `{meta['cod_fornecedor']}` {meta['nome_fornecedor']} · "
        f"ABC: **{meta['abc']}** · "
        f"Categoria: {meta['categoria']} · "
        f"Família: {meta['familia']}"
    )
    _legenda_detalhe()

    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    col_a.metric("ABC", str(meta["abc"]))
    col_b.metric("Média 7 dias", f"{_fmt_num(float(meta['media_7']))} cx/dia")
    col_c.metric("Média 30 dias", f"{_fmt_num(float(meta['media_30']))} cx/dia")
    col_d.metric(
        "Índice de consumo",
        "—" if pd.isna(indice) or indice is None else _fmt_num(float(indice), 2),
    )
    col_e.metric("Prioridade", f"{meta['nivel_risco']} ({int(meta['score_risco'])})")

    col_e1, col_e2, col_e3, col_e4, col_e5, col_e6 = st.columns(6)
    col_e1.metric("Estoque", f"{_fmt_opcional(meta.get('estoque'), 1)} cx")
    col_e2.metric("Reservada", f"{_fmt_opcional(meta.get('reservada'), 1)} cx")
    col_e3.metric("Em trânsito", f"{_fmt_opcional(meta.get('em_transito'), 0)} cx")
    col_e4.metric("Cobertura", f"{_fmt_opcional(meta.get('cobertura_dias'), 0)} dias")
    col_e5.metric("Status estoque", str(meta.get("status_cobertura", "—")))
    col_e6.metric(
        f"Sugestão ({dias_planejamento}d)",
        f"{_fmt_opcional(meta.get('sugestao_compra'), 0)} cx",
    )
    _alerta_consumo(meta)

    st.markdown("##### 3. Previsão de demanda (próximos dias)")
    col_cfg1, col_cfg2, col_cfg3 = st.columns([1, 1, 1])
    with col_cfg1:
        horizonte = st.slider(
            "Quantos dias à frente?",
            min_value=7,
            max_value=30,
            value=14,
            step=1,
            help="Ex.: 14 = projeção para as próximas duas semanas.",
        )
    with col_cfg2:
        if chronos_disponivel():
            motores = ["chronos2", "padrao_semanal"]
            motor_padrao = 0
        else:
            motores = ["padrao_semanal"]
            motor_padrao = 0
        motor = st.selectbox(
            "Método de previsão",
            options=motores,
            index=motor_padrao,
            format_func=lambda chave: ROTULO_MOTOR[chave],
            help="Chronos-2 costuma ser mais preciso; o padrão semanal é mais rápido.",
        )
    with col_cfg3:
        st.write("")
        st.write("")
        gerar = st.button("Gerar previsão", type="primary", use_container_width=True)

    chave_previsao = f"{sku}|{horizonte}|{motor}"
    if gerar:
        st.session_state["previsao_chave"] = chave_previsao

    previsao_pronta = st.session_state.get("previsao_chave") == chave_previsao
    if not previsao_pronta:
        st.info("Clique em **Gerar previsão** para ver o gráfico e a tabela dia a dia.")
        return

    try:
        serie = serie_do_sku(movimento, sku)
        with st.spinner("Calculando previsão de demanda..."):
            previsao = prever(serie, horizonte=horizonte, motor=motor, sku=sku)
    except Exception as erro:
        st.error(str(erro))
        return

    _legenda_previsao()
    st.plotly_chart(_grafico(serie.tail(60), previsao), use_container_width=True)

    hist_tab, prev_tab = st.tabs(["Volume dos últimos 14 dias", "Previsão dia a dia"])
    with hist_tab:
        st.caption("Histórico recente usado como referência do ritmo de saída.")
        hist = serie.tail(14).reset_index()
        historico_tab = pd.DataFrame(
            {
                "Data": hist["data"].map(_fmt_data),
                "Qtde realizada (cx)": hist["qtde"].map(lambda x: _fmt_num(x, 1)),
            }
        )
        st.table(historico_tab.set_index("Data"))
    with prev_tab:
        st.caption(
            "Use **Demanda esperada** para montar o pedido. "
            "Cenário alto (P90) é alerta de pico — não é quantidade sugerida de compra."
        )
        prev = previsao[["data", "previsao", "p10", "p90"]].copy()
        tabela = pd.DataFrame(
            {
                "Data": prev["data"].map(_fmt_data),
                "Demanda esperada (cx)": prev["previsao"].map(lambda x: _fmt_num(x, 1)),
                "Cenário baixo P10 (cx)": prev["p10"].map(lambda x: _fmt_num(x, 1)),
                "Cenário alto P90 (cx)": prev["p90"].map(lambda x: _fmt_num(x, 1)),
            }
        )
        st.table(tabela.set_index("Data"))
        total_esperado = float(prev["previsao"].sum())
        total_alto = float(prev["p90"].sum())
        st.markdown(
            f"**Totais no período ({horizonte} dias):** "
            f"demanda esperada **{_fmt_num(total_esperado, 0)}** cx · "
            f"cenário alto **{_fmt_num(total_alto, 0)}** cx (só referência de risco)."
        )
        csv = tabela.reset_index().to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Baixar previsão em CSV",
            data=csv,
            file_name=f"previsao_{sku}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
