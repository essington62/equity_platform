"""
02_flow_engine.py — Capital Flow Intelligence Dashboard
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Flow Engine", layout="wide")

# ── Constants ─────────────────────────────────────────────────────────────────
BD = Path("/Volumes/LaCie/MLGeral/data_lake/crypto-market-state/01_raw/spot/business_day")
BENCHMARK = "acwi"

INVERT_SIGN = {
    "dxy": True, "uup": True,
    "usdbrl": True, "move_index": True,
}

BUCKETS = {
    "AI_Compute":        ["msft","meta","amzn","googl","nvda","orcl","soxx","smh","vrt","etn","pwr"],
    "China_Industrial":  ["copper","copx","fxi","kweb","oil_wti"],
    "Metals_HardAssets": ["gld","gdx","slv"],
    "Brasil_EM":         ["ewz","vwo","eem","ewy","ewt","ewj","inda","bovespa"],
    "Dollar_Stress":     ["dxy","uup","usdbrl","move_index"],
    "Credit_Duration":   ["tlt","high_yield_bonds"],
    "Crypto":            ["btc","bith11"],
    "Defensivo":         ["defense_etf","ura"],
    "Agro":              ["weat","corn","soyb","mos","ntr","dba"],
}

BUCKET_META = {
    "AI_Compute":        {"icon": "🤖", "color": "#6366f1"},
    "China_Industrial":  {"icon": "🏭", "color": "#ef4444"},
    "Metals_HardAssets": {"icon": "🪙", "color": "#f59e0b"},
    "Brasil_EM":         {"icon": "🇧🇷", "color": "#22c55e"},
    "Dollar_Stress":     {"icon": "💵", "color": "#8b5cf6"},
    "Credit_Duration":   {"icon": "📊", "color": "#06b6d4"},
    "Crypto":            {"icon": "₿",  "color": "#f97316"},
    "Defensivo":         {"icon": "🛡️", "color": "#64748b"},
    "Agro":              {"icon": "🌾", "color": "#84cc16"},
}

QUAD_BG = {
    "LEADING":   "#d1fae5",
    "WEAKENING": "#fef9c3",
    "IMPROVING": "#dbeafe",
    "LAGGING":   "#f1f5f9",
    "N/A":       "#ffffff",
}

# ── Utilities ─────────────────────────────────────────────────────────────────

def rolling_zscore(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    mp = min_periods if min_periods is not None else max(10, window // 4)
    roll = s.rolling(window, min_periods=mp)
    return (s - roll.mean()) / (roll.std() + 1e-9)


def cross_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sg = df.std(axis=1) + 1e-9
    return df.sub(mu, axis=0).div(sg, axis=0)


def last(series: pd.Series) -> float:
    try:
        return float(series.dropna().iloc[-1])
    except Exception:
        return np.nan


def quadrant(rz: float, mz: float) -> str:
    if np.isnan(rz) or np.isnan(mz):
        return "N/A"
    if rz > 0 and mz > 0:  return "LEADING"
    if rz > 0 and mz < 0:  return "WEAKENING"
    if rz < 0 and mz < 0:  return "LAGGING"
    return "IMPROVING"


def crowding_emoji(score: float) -> str:
    if np.isnan(score): return "─"
    if score > 2.0: return "🔴"
    if score > 1.5: return "🟠"
    if score > 1.0: return "🟡"
    return "🟢"


def accel_arrow(score: float) -> str:
    if np.isnan(score): return "─"
    if score >= 70: return "↑↑"
    if score >= 55: return "↑"
    if score >= 45: return "→"
    if score >= 30: return "↓"
    return "↓↓"


def hex_rgba(hex_color: str, alpha: float = 0.2) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_close(f: str) -> pd.Series | None:
    try:
        s = pd.read_parquet(BD / f).set_index("date")["close"]
        idx = pd.to_datetime(s.index)
        s.index = idx.tz_convert(None) if idx.tz is not None else idx.tz_localize(None)
        return s
    except Exception:
        return None


def _load_volume(f: str) -> pd.Series | None:
    try:
        s = pd.read_parquet(BD / f).set_index("date")["volume"]
        idx = pd.to_datetime(s.index)
        s.index = idx.tz_convert(None) if idx.tz is not None else idx.tz_localize(None)
        return s
    except Exception:
        return None


# ── Compute (cached) ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_and_compute() -> dict:
    if not BD.exists():
        return {"error": f"Data lake não encontrado: {BD}"}

    bm_price = _load_close(f"{BENCHMARK}.parquet")
    if bm_price is None:
        return {"error": f"Benchmark '{BENCHMARK}' não encontrado em {BD}"}
    if len(bm_price) < 60:
        return {"error": f"Histórico insuficiente ({len(bm_price)} dias < 60)"}

    bm_idx      = bm_price.index
    acwi_ret20  = bm_price.pct_change(20).clip(-0.25, 0.25)

    all_tickers = {t for tks in BUCKETS.values() for t in tks}
    prices: dict  = {}
    volumes: dict = {}
    for ticker in all_tickers:
        p = _load_close(f"{ticker}.parquet")
        v = _load_volume(f"{ticker}.parquet")
        if p is not None:
            prices[ticker]  = p.reindex(bm_idx, method="ffill")
        if v is not None:
            volumes[ticker] = v.reindex(bm_idx, method="ffill")

    ret5: dict = {}
    ret20: dict = {}
    ret60: dict = {}
    rv: dict = {}
    for ticker, p in prices.items():
        r5  = p.pct_change(5).clip(-0.25, 0.25)
        r20 = p.pct_change(20).clip(-0.25, 0.25)
        r60 = p.pct_change(60).clip(-0.25, 0.25)
        if ticker in INVERT_SIGN:
            r5, r20, r60 = -r5, -r20, -r60
        ret5[ticker]  = r5
        ret20[ticker] = r20
        ret60[ticker] = r60
        if ticker in volumes:
            vol_ma = volumes[ticker].rolling(20, min_periods=10).mean()
            rv[ticker] = volumes[ticker] / (vol_ma + 1e-9)

    bkt_r5: dict = {}
    bkt_r20: dict = {}
    bkt_r60: dict = {}
    bkt_rv: dict = {}
    for bkt, tickers in BUCKETS.items():
        r5s  = [ret5[t]  for t in tickers if t in ret5]
        r20s = [ret20[t] for t in tickers if t in ret20]
        r60s = [ret60[t] for t in tickers if t in ret60]
        rvs  = [rv[t]    for t in tickers if t in rv]
        if not r5s:
            continue
        bkt_r5[bkt]  = pd.concat(r5s,  axis=1).mean(axis=1)
        bkt_r20[bkt] = (pd.concat(r20s, axis=1).mean(axis=1)
                        if r20s else pd.Series(np.nan, index=bm_idx))
        bkt_r60[bkt] = (pd.concat(r60s, axis=1).mean(axis=1)
                        if r60s else pd.Series(np.nan, index=bm_idx))
        bkt_rv[bkt]  = (pd.concat(rvs,  axis=1).mean(axis=1)
                        if rvs  else pd.Series(np.nan, index=bm_idx))

    active = list(bkt_r5.keys())
    if not active:
        return {"error": "Nenhum bucket carregado."}

    r5_df  = pd.DataFrame({b: bkt_r5[b]  for b in active})
    r20_df = pd.DataFrame({b: bkt_r20[b] for b in active})
    r60_df = pd.DataFrame({b: bkt_r60[b] for b in active})
    rv_df  = pd.DataFrame({b: bkt_rv[b]  for b in active})

    # Engine 1 — Relative Strength
    z5  = cross_zscore(r5_df)
    z20 = cross_zscore(r20_df)
    z60 = cross_zscore(r60_df)
    z_w = 0.50 * z5 + 0.30 * z20 + 0.20 * z60
    rs_df      = (z_w.clip(-3, 3) + 3) / 6 * 100
    rs_vs_acwi = r20_df.sub(acwi_ret20, axis=0)
    persist_df = rs_df.rolling(20, min_periods=10).apply(
        lambda x: (x > 60).mean(), raw=True
    )

    # Engine 2 — Acceleration
    accel_df       = r5_df - r20_df
    accel_z_df     = accel_df.apply(lambda c: rolling_zscore(c, 60))
    accel_score_df = (accel_z_df.clip(-3, 3) + 3) / 6 * 100
    rel_accel_df   = cross_zscore(accel_df)

    # Engine 3 — RRG
    rs_ratio_raw = r20_df.div(acwi_ret20 + 1e-9, axis=0)
    rs_ratio_z   = rs_ratio_raw.apply(lambda c: rolling_zscore(c, 60))
    rs_mom_raw   = rs_ratio_z.diff(5).ewm(span=5, adjust=False).mean()
    rs_mom_z     = rs_mom_raw.apply(lambda c: rolling_zscore(c, 60))

    # Engine 4 — Crowding
    rs_z252  = rs_df.apply(lambda c: rolling_zscore(c, 252, min_periods=60))
    rv_z     = rv_df.apply(lambda c: rolling_zscore(c, 60))
    crowd_df = 0.7 * rs_z252 + 0.3 * rv_z

    # Engine 5 — Dispersion
    dispersion   = rs_df.std(axis=1)
    dispersion_z = rolling_zscore(dispersion, 60)

    # Collect snapshot
    rows = []
    for b in active:
        rz = last(rs_ratio_z[b])
        mz = last(rs_mom_z[b])
        rows.append({
            "bucket":    b,
            "rs":        last(rs_df[b]),
            "vs_acwi":   last(rs_vs_acwi[b]),
            "persist":   last(persist_df[b]),
            "crowd":     last(crowd_df[b]),
            "vel":       last(r5_df[b]),
            "mom":       last(r20_df[b]),
            "accel_sc":  last(accel_score_df[b]),
            "rel_accel": last(rel_accel_df[b]),
            "quad":      quadrant(rz, mz),
            "rz":        rz,
            "mz":        mz,
        })

    # RS Score history for sparklines (60 days)
    rs_history: dict = {}
    for b in active:
        s = rs_df[b].dropna()
        rs_history[b] = {
            "dates":  [d.strftime("%Y-%m-%d") for d in s.index[-60:]],
            "values": list(s.values[-60:]),
        }

    # RRG point history for arrows (6 days)
    rrg_history: dict = {}
    for b in active:
        df_pts = pd.DataFrame({"rz": rs_ratio_z[b], "mz": rs_mom_z[b]}).dropna()
        rrg_history[b] = [
            (float(row.rz), float(row.mz))
            for row in df_pts.iloc[-6:].itertuples()
        ]

    return {
        "error":       None,
        "rows":        rows,
        "by_rs":       sorted(rows, key=lambda x: -(x["rs"] if not np.isnan(x["rs"]) else -999)),
        "by_accel":    sorted(rows, key=lambda x: -(x["accel_sc"] if not np.isnan(x["accel_sc"]) else -999)),
        "rs_history":  rs_history,
        "rrg_history": rrg_history,
        "disp_now":    last(dispersion),
        "disp_z_now":  last(dispersion_z),
        "today_str":   bm_idx[-1].strftime("%Y-%m-%d"),
    }


# ── Render ────────────────────────────────────────────────────────────────────

def render(s: dict) -> None:
    rows     = s["rows"]
    by_rs    = s["by_rs"]
    by_accel = s["by_accel"]
    dn       = s["disp_now"]
    dz       = s["disp_z_now"]

    # ── Header ────────────────────────────────────────────────────────────────
    col_h, col_btn = st.columns([6, 1])
    with col_h:
        st.title("🌊 Flow Engine")
        st.caption(f"Atualizado: {s['today_str']}")
    with col_btn:
        st.write("")
        st.write("")
        if st.button("🔄 Atualizar dados"):
            st.cache_data.clear()
            st.rerun()

    # ── Bloco 1 — Market Pulse ────────────────────────────────────────────────
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    top_rs = by_rs[0] if by_rs else None
    top_ac = by_accel[0] if by_accel else None

    with col1:
        if top_rs:
            meta  = BUCKET_META.get(top_rs["bucket"], {"icon": "●"})
            rs_v  = top_rs["rs"]
            delta = f"RS {rs_v:.0f}/100" if not np.isnan(rs_v) else "N/A"
            st.metric("🏆 Tema Dominante", f"{meta['icon']} {top_rs['bucket']}", delta)

    with col2:
        if top_ac:
            meta  = BUCKET_META.get(top_ac["bucket"], {"icon": "●"})
            ac_v  = top_ac["accel_sc"]
            delta = f"Accel {ac_v:.0f}/100" if not np.isnan(ac_v) else "N/A"
            st.metric("📈 Tema Acelerando", f"{meta['icon']} {top_ac['bucket']}", delta)

    with col3:
        if not np.isnan(dz) and dz > 1.5:
            disp_label = "CONCENTRADO"
        elif not np.isnan(dz) and dz < -0.5:
            disp_label = "AMPLO"
        else:
            disp_label = "NEUTRO"
        delta_disp = f"z={dz:.1f}" if not np.isnan(dz) else "N/A"
        st.metric("🔀 Dispersão do Mercado", disp_label, delta_disp)

    # ── Bloco 2 — Flow Heatmap ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Flow Heatmap")

    hm_rows = []
    for r in by_rs:
        meta = BUCKET_META.get(r["bucket"], {"icon": "●"})
        vs   = r["vs_acwi"]
        pe   = r["persist"]
        hm_rows.append({
            "Bucket":    f"{meta['icon']} {r['bucket']}",
            "RS Score":  round(r["rs"], 1) if not np.isnan(r["rs"]) else 0.0,
            "vs ACWI":   f"{vs*100:+.1f}%" if not np.isnan(vs) else "N/A",
            "Accel":     accel_arrow(r["accel_sc"]),
            "Persist":   f"{pe*100:.0f}%" if not np.isnan(pe) else "N/A",
            "Crowding":  crowding_emoji(r["crowd"]),
            "Quadrante": r["quad"],
        })

    df_hm = pd.DataFrame(hm_rows)

    def _highlight_quad(row):
        bg = QUAD_BG.get(row["Quadrante"], "#ffffff")
        return [f"background-color: {bg}"] * len(row)

    styled_hm = df_hm.style.apply(_highlight_quad, axis=1)
    st.dataframe(
        styled_hm,
        column_config={
            "RS Score": st.column_config.ProgressColumn(
                "RS Score", min_value=0, max_value=100, format="%.0f"
            ),
            "Quadrante": st.column_config.TextColumn("Quadrante"),
        },
        use_container_width=True,
        hide_index=True,
    )

    # ── Bloco 3 — Leadership vs Rotation ─────────────────────────────────────
    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("🏆 Liderando Hoje")
        for r in by_rs:
            meta  = BUCKET_META.get(r["bucket"], {"icon": "●", "color": "#888888"})
            rs_v  = r["rs"] if not np.isnan(r["rs"]) else 0.0
            pct   = min(int(rs_v), 100)
            bar   = (
                f'<div style="margin-bottom:10px">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:3px">'
                f'<span>{meta["icon"]} <b>{r["bucket"]}</b></span>'
                f'<span style="color:#6b7280">{rs_v:.0f}/100</span></div>'
                f'<div style="background:#e5e7eb;border-radius:4px;height:8px">'
                f'<div style="background:{meta["color"]};width:{pct}%;height:8px;border-radius:4px">'
                f'</div></div></div>'
            )
            st.markdown(bar, unsafe_allow_html=True)

    with col_r:
        st.subheader("📈 Acelerando Agora")
        for r in by_accel:
            meta  = BUCKET_META.get(r["bucket"], {"icon": "●", "color": "#888888"})
            ac_v  = r["accel_sc"] if not np.isnan(r["accel_sc"]) else 0.0
            vel_v = r["vel"] if not np.isnan(r["vel"]) else 0.0
            vel_p = vel_v * 100
            arrow = accel_arrow(r["accel_sc"])
            sc    = "#22c55e" if vel_p >= 0 else "#ef4444"
            pct   = min(int(ac_v), 100)
            bar   = (
                f'<div style="margin-bottom:10px">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:3px">'
                f'<span>{meta["icon"]} <b>{r["bucket"]}</b></span>'
                f'<span style="color:{sc}">{arrow} {vel_p:+.2f}%</span></div>'
                f'<div style="background:#e5e7eb;border-radius:4px;height:8px">'
                f'<div style="background:{meta["color"]};width:{pct}%;height:8px;border-radius:4px">'
                f'</div></div></div>'
            )
            st.markdown(bar, unsafe_allow_html=True)

    # ── Bloco 4 — RRG Chart ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔄 Relative Rotation Graph")

    rrg_history = s["rrg_history"]
    all_rz = [r["rz"] for r in rows if not np.isnan(r["rz"])]
    all_mz = [r["mz"] for r in rows if not np.isnan(r["mz"])]
    for pts in rrg_history.values():
        all_rz += [p[0] for p in pts]
        all_mz += [p[1] for p in pts]

    xmax = max((abs(v) for v in all_rz), default=3.0) * 1.35
    ymax = max((abs(v) for v in all_mz), default=3.0) * 1.35
    xmax, ymax = max(xmax, 1.5), max(ymax, 1.5)

    fig_rrg = go.Figure()

    fig_rrg.update_layout(
        shapes=[
            # Quadrant fills
            dict(type="rect", x0=0, x1=xmax, y0=0, y1=ymax,
                 fillcolor="rgba(134,239,172,0.14)", line_width=0, layer="below"),
            dict(type="rect", x0=0, x1=xmax, y0=-ymax, y1=0,
                 fillcolor="rgba(253,224,71,0.14)", line_width=0, layer="below"),
            dict(type="rect", x0=-xmax, x1=0, y0=-ymax, y1=0,
                 fillcolor="rgba(203,213,225,0.18)", line_width=0, layer="below"),
            dict(type="rect", x0=-xmax, x1=0, y0=0, y1=ymax,
                 fillcolor="rgba(147,197,253,0.14)", line_width=0, layer="below"),
            # Axis dividers
            dict(type="line", x0=-xmax, x1=xmax, y0=0, y1=0,
                 line=dict(color="rgba(100,100,100,0.4)", width=1, dash="dot")),
            dict(type="line", x0=0, x1=0, y0=-ymax, y1=ymax,
                 line=dict(color="rgba(100,100,100,0.4)", width=1, dash="dot")),
        ]
    )

    for label, x, y in [
        ("LEADING",   xmax * 0.55,  ymax * 0.75),
        ("WEAKENING", xmax * 0.55, -ymax * 0.75),
        ("LAGGING",  -xmax * 0.55, -ymax * 0.75),
        ("IMPROVING",-xmax * 0.55,  ymax * 0.75),
    ]:
        fig_rrg.add_annotation(
            x=x, y=y, text=label, showarrow=False,
            font=dict(size=15, color="rgba(0,0,0,0.13)"),
        )

    # Trail lines + arrow annotations
    for b, pts in rrg_history.items():
        if len(pts) < 2:
            continue
        meta = BUCKET_META.get(b, {"color": "#888888"})
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        fig_rrg.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color=meta["color"], width=1.5, dash="dot"),
            opacity=0.45,
            showlegend=False,
            hoverinfo="skip",
        ))
        x0_a, y0_a = pts[-2]
        x1_a, y1_a = pts[-1]
        fig_rrg.add_annotation(
            x=x1_a, y=y1_a,
            ax=x0_a, ay=y0_a,
            axref="x", ayref="y",
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor=meta["color"],
            showarrow=True,
        )

    # Scatter points
    for r in rows:
        rz, mz = r["rz"], r["mz"]
        if np.isnan(rz) or np.isnan(mz):
            continue
        meta  = BUCKET_META.get(r["bucket"], {"icon": "●", "color": "#888888"})
        rs_v  = r["rs"] if not np.isnan(r["rs"]) else 30.0
        size  = max(14, min(44, rs_v * 0.44))
        fig_rrg.add_trace(go.Scatter(
            x=[rz], y=[mz],
            mode="markers+text",
            marker=dict(
                size=size, color=meta["color"], opacity=0.85,
                line=dict(color="white", width=1.5),
            ),
            text=[f"{meta['icon']} {r['bucket']}"],
            textposition="top center",
            textfont=dict(size=11),
            name=r["bucket"],
            hovertemplate=(
                f"<b>{meta['icon']} {r['bucket']}</b><br>"
                f"RS Ratio: {rz:.2f}<br>"
                f"RS Momentum: {mz:.2f}<br>"
                f"RS Score: {r['rs']:.0f}/100<br>"
                f"Quadrante: {r['quad']}<extra></extra>"
            ),
        ))

    fig_rrg.update_layout(
        xaxis=dict(
            title="RS Ratio (força relativa vs ACWI)",
            range=[-xmax, xmax], zeroline=False,
        ),
        yaxis=dict(
            title="RS Momentum (aceleração da força)",
            range=[-ymax, ymax], zeroline=False,
        ),
        height=560,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_rrg, use_container_width=True)

    # ── Bloco 5 — Sparklines ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 RS Score — Histórico 60 dias")

    rs_history  = s["rs_history"]
    bucket_list = list(BUCKETS.keys())
    n_cols = 3
    for i in range(0, len(bucket_list), n_cols):
        cols = st.columns(n_cols)
        for j, b in enumerate(bucket_list[i:i + n_cols]):
            hist  = rs_history.get(b, {})
            meta  = BUCKET_META.get(b, {"icon": "●", "color": "#888888"})
            r_row = next((x for x in rows if x["bucket"] == b), None)
            cur   = r_row["rs"] if r_row and not np.isnan(r_row["rs"]) else None
            with cols[j]:
                fig_sp = go.Figure()
                vals = hist.get("values", [])
                dts  = hist.get("dates", [])
                if vals and dts:
                    fig_sp.add_trace(go.Scatter(
                        x=dts, y=vals,
                        mode="lines",
                        fill="tozeroy",
                        fillcolor=hex_rgba(meta["color"], 0.15),
                        line=dict(color=meta["color"], width=2),
                        showlegend=False,
                        hovertemplate="%{x}: %{y:.1f}<extra></extra>",
                    ))
                    fig_sp.add_trace(go.Scatter(
                        x=[dts[-1]], y=[vals[-1]],
                        mode="markers",
                        marker=dict(
                            size=8, color=meta["color"],
                            line=dict(color="white", width=1.5),
                        ),
                        showlegend=False,
                        hoverinfo="skip",
                    ))
                title_txt = (
                    f"{meta['icon']} {b} — {cur:.0f}/100"
                    if cur is not None else f"{meta['icon']} {b}"
                )
                fig_sp.update_layout(
                    title=dict(text=title_txt, font=dict(size=12)),
                    height=165,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=35, b=0),
                    yaxis=dict(
                        showgrid=False, showticklabels=False,
                        range=[0, 100],
                    ),
                    xaxis=dict(showgrid=False, showticklabels=False),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_sp, use_container_width=True)

    # ── Bloco 6 — Alertas ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚠️ Alertas Ativos")

    alerts: list[tuple[str, str]] = []

    # Dispersão alta
    if not np.isnan(dz) and dz > 2.0:
        alerts.append(("warning", "⚠️ Capital concentrado — risco de rotação violenta"))

    for r in rows:
        b    = r["bucket"]
        q    = r["quad"]
        cr_v = r["crowd"]     if not np.isnan(r["crowd"])     else 0.0
        pe_v = r["persist"]   if not np.isnan(r["persist"])   else 1.0
        ra_v = r["rel_accel"] if not np.isnan(r["rel_accel"]) else 0.0
        meta = BUCKET_META.get(b, {"icon": "●"})

        # WEAKENING + crowding HIGH/EXTREME
        if q == "WEAKENING" and cr_v > 1.5:
            alerts.append((
                "error",
                f"🔴 {meta['icon']} {b} — WEAKENING + crowding elevado → exaustão detectada",
            ))

        # IMPROVING + accel positiva
        if q == "IMPROVING" and ra_v > 0:
            alerts.append((
                "info",
                f"📈 {meta['icon']} {b} ganhando momentum relativo",
            ))

        # Persistência baixa com IMPROVING
        if q == "IMPROVING" and pe_v < 0.3:
            alerts.append((
                "info",
                f"📊 {meta['icon']} {b} IMPROVING mas persistência baixa ({pe_v*100:.0f}%) — confirmar",
            ))

        # LEADING + crowding elevado
        if q == "LEADING" and cr_v > 1.5:
            alerts.append((
                "warning",
                f"🟠 {meta['icon']} {b} LEADING + crowding elevado — monitorar exaustão",
            ))

        # Quadrante mudou vs ontem
        hist_pts = rrg_history.get(b, [])
        if len(hist_pts) >= 2:
            rz_prev, mz_prev = hist_pts[-2]
            q_prev = quadrant(rz_prev, mz_prev)
            if q_prev not in ("N/A", q) and q != "N/A":
                alerts.append((
                    "warning",
                    f"🔄 {meta['icon']} {b} migrou de {q_prev} → {q}",
                ))

    if alerts:
        seen: set[str] = set()
        for kind, msg in alerts:
            if msg in seen:
                continue
            seen.add(msg)
            if kind == "error":
                st.error(msg)
            elif kind == "warning":
                st.warning(msg)
            else:
                st.info(msg)
    else:
        st.success("✅ Nenhum alerta ativo")


# ── Entry point ───────────────────────────────────────────────────────────────

with st.spinner("Carregando dados e calculando scores..."):
    scores = load_and_compute()

if scores.get("error"):
    st.error(f"Erro ao carregar dados: {scores['error']}")
    st.info(f"Verifique se o data lake está montado: `{BD}`")
    col_retry, _ = st.columns([1, 4])
    with col_retry:
        if st.button("🔄 Tentar novamente"):
            st.cache_data.clear()
            st.rerun()
else:
    render(scores)
