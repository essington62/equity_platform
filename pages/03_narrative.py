"""
03_narrative.py — Narrative Engine Dashboard
MongoDB: sources · scores · signals · macro_timeseries
"""
import datetime

import pandas as pd
import streamlit as st
from pymongo import MongoClient

# ── Config ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Narrative Engine", layout="wide")

MONGODB_URI = "mongodb+srv://edmundobrown_db_user:XktKCOIu54y7pRGN@equity-platform.snfkict.mongodb.net/?appName=equity-platform"
MONGODB_DB  = "equity_platform"

CHANNELS  = ["Genial Investimentos", "XP Investimentos", "BTG Pactual"]
VERTICAIS = ["AI_Compute", "China_Coreia", "Brasil_EM", "Commodities", "Crypto", "Agro", "Macro"]

SENTIMENT_COLOR = {
    "BULLISH": "background-color: #bbf7d0; color: #166534",
    "BEARISH": "background-color: #fecaca; color: #991b1b",
    "NEUTRO":  "background-color: #e5e7eb; color: #374151",
    "—":       "background-color: #f9fafb; color: #9ca3af",
}

JUROS_INDICATORS = {
    "treasury_30y_eua", "treasury_10y_eua", "treasury_2y_eua",
    "juros_30y_japao", "juros_10y_japao", "juros_40y_japao",
    "juros_30y_uk", "juros_10y_uk", "juros_10y_br",
    "selic_projecao", "ntnb_longo", "fed_funds",
    "ipca_projecao", "inflacao_servicos_br", "cpi_eua", "ppi_eua",
    "juros_brasileiros_10_anos", "juros_10_anos_americanos",
}

# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_sources(date_str: str) -> list:
    client = MongoClient(MONGODB_URI)
    docs   = list(client[MONGODB_DB]["sources"].find({"date": date_str}))
    client.close()
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs


@st.cache_data(ttl=300)
def load_scores(date_str: str) -> list:
    client = MongoClient(MONGODB_URI)
    docs   = list(client[MONGODB_DB]["scores"].find({"date": date_str}))
    client.close()
    for d in docs:
        d["_id"]       = str(d["_id"])
        d["source_id"] = str(d.get("source_id", ""))
    return docs


@st.cache_data(ttl=300)
def load_signals(date_str: str) -> list:
    client = MongoClient(MONGODB_URI)
    docs   = list(client[MONGODB_DB]["signals"].find({"date": date_str}))
    client.close()
    for d in docs:
        d["_id"]       = str(d["_id"])
        d["source_id"] = str(d.get("source_id", ""))
    return docs


@st.cache_data(ttl=300)
def load_timeseries(date_str: str) -> list:
    client = MongoClient(MONGODB_URI)
    docs   = list(client[MONGODB_DB]["macro_timeseries"].find({"date": date_str}))
    client.close()
    for d in docs:
        d["_id"]       = str(d["_id"])
        d["source_id"] = str(d.get("source_id", ""))
    return docs

# ── Helpers ───────────────────────────────────────────────────────────────────

def _signal_label(avg: float) -> str:
    if avg > 30:  return "🟢 BULLISH"
    if avg < -30: return "🔴 BEARISH"
    return "🟡 NEUTRO"


def _fmt_value(value: float, unit: str) -> str:
    if value is None:
        return "—"
    if unit == "percent":
        return f"{value:.2f}%"
    if unit == "usd":
        return f"${value:,.2f}"
    if unit == "brl":
        return f"R${value:,.2f}"
    if unit == "milhoes_brl":
        if abs(value) >= 1000:
            return f"{value/1000:+.1f}bi"
        return f"{value:+.0f}mi"
    if unit == "pontos":
        return f"{value:,.2f}"
    if unit == "ratio":
        return f"{value:.2f}"
    return f"{value:g}"

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("📰 Narrative Engine")
st.sidebar.markdown("---")
selected_date = st.sidebar.date_input(
    "Data",
    value=datetime.date.today(),
    max_value=datetime.date.today(),
)
date_str = selected_date.strftime("%Y-%m-%d")

if st.sidebar.button("🔄 Atualizar"):
    st.cache_data.clear()
    st.rerun()

# ── Title ─────────────────────────────────────────────────────────────────────

st.title("📰 Narrative Engine")
st.caption(f"Dados de: **{date_str}**")
st.markdown("---")

# ── Load ──────────────────────────────────────────────────────────────────────

with st.spinner("Carregando dados do MongoDB…"):
    sources    = load_sources(date_str)
    scores     = load_scores(date_str)
    signals    = load_signals(date_str)
    timeseries = load_timeseries(date_str)

# ── BLOCO 1 — Fontes do Dia ───────────────────────────────────────────────────

st.subheader("📥 Fontes Processadas")

if not sources:
    st.info(f"Nenhum dado para {date_str}. Execute o pipeline para esta data.")
else:
    cols = st.columns(max(len(sources), 1))
    for i, src in enumerate(sources):
        with cols[i]:
            status = "✅" if src.get("status") == "processed" else "⏳"
            dur    = src.get("duration_min", 0)
            chunks = src.get("chunk_count", 0)
            vtype  = src.get("video_type", "—")
            lang   = src.get("language", "?")
            st.markdown(
                f"""
                <div style="background:#f8fafc;border:1px solid #e2e8f0;
                            border-radius:8px;padding:14px 16px;margin-bottom:8px">
                  <div style="font-size:1.05em;font-weight:600">{status} {src['channel']}</div>
                  <div style="color:#64748b;font-size:0.85em;margin-top:6px">
                    {vtype} &nbsp;·&nbsp; {chunks} chunks &nbsp;·&nbsp; {dur:.0f} min &nbsp;·&nbsp; lang={lang}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("---")

# ── BLOCO 2 — Consenso por Vertical ──────────────────────────────────────────

st.subheader("🎯 Consenso por Vertical")

if not scores:
    st.info("Nenhum score disponível para esta data.")
else:
    score_map: dict = {}
    for s in scores:
        key = (s.get("channel", ""), s.get("vertical", ""))
        score_map[key] = s

    short_names = [ch.split()[0] for ch in CHANNELS]
    table_rows  = []

    for vertical in VERTICAIS:
        row: dict = {"Vertical": vertical}
        ch_scores = []

        for ch, short in zip(CHANNELS, short_names):
            s = score_map.get((ch, vertical))
            if s:
                sent  = s.get("sentiment", "—")
                score = s.get("score", 0)
                row[short] = sent
                row[f"_sc_{short}"] = score
                ch_scores.append(score)
            else:
                row[short] = "—"
                row[f"_sc_{short}"] = None

        valid = [v for v in ch_scores if v is not None]
        if valid:
            avg          = sum(valid) / len(valid)
            row["Média"] = round(avg, 0)
            row["Sinal"] = _signal_label(avg)
        else:
            row["Média"] = None
            row["Sinal"] = "—"

        sentiments = [
            score_map.get((ch, vertical), {}).get("sentiment")
            for ch in CHANNELS
            if score_map.get((ch, vertical))
        ]
        unique = {s for s in sentiments if s}
        row["⚠️"] = "⚠️" if len(unique) > 1 and len(sentiments) >= 2 else ""

        table_rows.append(row)

    display_cols = ["Vertical"] + short_names + ["Média", "Sinal", "⚠️"]
    df_display   = pd.DataFrame(table_rows)[display_cols]

    def _style_consensus(row):
        styles = [""] * len(row)
        for i, col in enumerate(row.index):
            if col in short_names:
                styles[i] = SENTIMENT_COLOR.get(str(row[col]), "")
            elif col == "Média":
                v = row[col]
                if v is None:
                    pass
                elif v > 30:
                    styles[i] = "background-color:#bbf7d0;color:#166534;font-weight:600"
                elif v < -30:
                    styles[i] = "background-color:#fecaca;color:#991b1b;font-weight:600"
                else:
                    styles[i] = "background-color:#fef9c3;color:#92400e"
        return styles

    styled = (
        df_display.style
        .apply(_style_consensus, axis=1)
        .format({"Média": lambda v: f"{v:+.0f}" if v is not None else "—"}, na_rep="—")
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption("🟢 BULLISH (>+30) · 🔴 BEARISH (<−30) · 🟡 NEUTRO · ⚠️ divergência entre canais")

st.markdown("---")

# ── BLOCO 3 — Macro Timeseries ────────────────────────────────────────────────

st.subheader("📊 Dados Macro Capturados")

if not timeseries:
    st.info("Nenhum dado de macro timeseries para esta data.")
else:
    # Group by indicator
    ind_map: dict = {}
    for row in timeseries:
        ind = row.get("indicator", "")
        if ind not in ind_map:
            ind_map[ind] = []
        ind_map[ind].append({
            "value":  row.get("value"),
            "unit":   row.get("unit", "percent"),
            "source": row.get("source", ""),
        })

    def _render_ind_table(indicators: list):
        rows = []
        for ind in indicators:
            entries = ind_map.get(ind)
            if not entries:
                continue
            unit = entries[0]["unit"]
            vals = [e["value"] for e in entries if e["value"] is not None]
            srcs = [e["source"].split()[0] for e in entries]

            if not vals:
                val_str = "—"
            elif len(vals) == 1:
                val_str = _fmt_value(vals[0], unit)
            else:
                lo, hi = min(vals), max(vals)
                val_str = (
                    _fmt_value(lo, unit)
                    if abs(hi - lo) < 0.01
                    else f"{_fmt_value(lo, unit)} – {_fmt_value(hi, unit)}"
                )

            rows.append({
                "Indicador": ind,
                "Valor":     val_str,
                "Fonte":     " / ".join(srcs),
            })
        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Indicador": st.column_config.TextColumn(width="medium"),
                    "Valor":     st.column_config.TextColumn(width="small"),
                    "Fonte":     st.column_config.TextColumn(width="medium"),
                },
            )
        else:
            st.caption("Sem dados nesta categoria.")

    juros_present = sorted(k for k in ind_map if k in JUROS_INDICATORS)
    other_present = sorted(k for k in ind_map if k not in JUROS_INDICATORS)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**🏦 Juros & Inflação**")
        _render_ind_table(juros_present)
    with col_r:
        st.markdown("**🛢️ Fluxo & Commodities**")
        _render_ind_table(other_present)

st.markdown("---")

# ── BLOCO 4 — Alertas ─────────────────────────────────────────────────────────

st.subheader("⚠️ Alertas do Dia")

if not signals:
    st.info("Nenhum sinal disponível para esta data.")
else:
    all_alerts: list = []
    for sig in signals:
        ch = sig.get("channel", "")
        for alerta in sig.get("alertas", []):
            all_alerts.append((
                alerta.get("urgencia", "baixa"),
                alerta.get("tipo", ""),
                alerta.get("descricao", ""),
                ch,
            ))

    if not all_alerts:
        st.success("✅ Nenhum alerta registrado para esta data.")
    else:
        order = {"alta": 0, "media": 1, "baixa": 2}
        all_alerts.sort(key=lambda x: order.get(x[0], 3))

        for urgencia, tipo, desc, ch in all_alerts:
            icon  = {"alta": "🔴", "media": "🟡", "baixa": "🔵"}.get(urgencia, "ℹ️")
            short = ch.split()[0] if ch else "?"
            label = f"**[{tipo}]** {desc} _({short})_"
            if urgencia == "alta":
                st.error(f"{icon} {label}")
            elif urgencia == "media":
                st.warning(f"{icon} {label}")
            else:
                st.info(f"{icon} {label}")

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    f"equity_platform · Narrative Engine · "
    f"{len(sources)} fontes · {len(scores)} scores · "
    f"{len(timeseries)} indicadores macro · cache TTL 5 min"
)
