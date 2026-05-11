"""
flow_engine.py — Capital Flow Intelligence Engine

Detecta liderança, aceleração, persistência, concentração e rotação marginal
de capital entre temas globais competitivos.

NÃO prevê retornos absolutos. Detecta regime de fluxo.

Engines:
  1. Relative Strength   — quem domina cross-sectional (z-score ponderado)
  2. Acceleration        — quem está acelerando (velocity - momentum)
  3. RRG                 — quadrantes Leading/Weakening/Improving/Lagging
  4. Crowding Detector   — concentração histórica + volume relativo
  5. Flow Dispersion     — concentração vs broad participation

Output: Flow Dashboard com Leadership, Rotation, RRG, Sparklines, Alertas.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ── Data Source ───────────────────────────────────────────────────────────────
BD = Path("/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/01_raw/spot/business_day")
BENCHMARK = 'acwi'

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_bd(f):
    try:
        s = pd.read_parquet(BD / f).set_index('date')['close']
        idx = pd.to_datetime(s.index)
        s.index = idx.tz_convert(None) if idx.tz is not None else idx.tz_localize(None)
        return s
    except Exception:
        return None

def load_vol(f):
    try:
        s = pd.read_parquet(BD / f).set_index('date')['volume']
        idx = pd.to_datetime(s.index)
        s.index = idx.tz_convert(None) if idx.tz is not None else idx.tz_localize(None)
        return s
    except Exception:
        return None

# ── Sign Inversion ────────────────────────────────────────────────────────────
INVERT_SIGN = {
    'dxy':        True,
    'uup':        True,
    'usdbrl':     True,
    'move_index': True,
}

# ── Bucket Definitions ────────────────────────────────────────────────────────
BUCKETS = {
    'AI_Compute': [
        'msft', 'meta', 'amzn', 'googl', 'nvda', 'orcl',
        'soxx', 'smh', 'vrt', 'etn', 'pwr',
    ],
    'China_Industrial': [
        'copper', 'copx', 'fxi', 'kweb', 'oil_wti',
    ],
    'Metals_HardAssets': [
        'gld', 'gdx', 'slv',
    ],
    'Brasil_EM': [
        'ewz', 'vwo', 'eem', 'ewy', 'ewt', 'ewj',
        'inda', 'bovespa',
    ],
    'Dollar_Stress': [
        'dxy', 'uup', 'usdbrl', 'move_index',
    ],
    'Credit_Duration': [
        'tlt', 'high_yield_bonds',
    ],
    'Crypto': [
        'btc', 'bith11',
    ],
    'Defensivo': [
        'defense_etf', 'ura',
    ],
    'Agro': [
        'weat', 'corn', 'soyb', 'mos', 'ntr', 'dba',
    ],
}

# ── Utilities ─────────────────────────────────────────────────────────────────
BLOCKS = '▁▂▃▄▅▆▇█'


def rolling_zscore(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    mp = min_periods if min_periods is not None else max(10, window // 4)
    roll = s.rolling(window, min_periods=mp)
    return (s - roll.mean()) / (roll.std() + 1e-9)


def cross_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sg = df.std(axis=1) + 1e-9
    return df.sub(mu, axis=0).div(sg, axis=0)


def sparkline(values: np.ndarray, width: int = 20) -> str:
    vals = np.array(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return '─' * width
    vals = vals[-width:]
    pad = '─' * max(0, width - len(vals))
    vmin, vmax = vals.min(), vals.max()
    if vmax == vmin:
        return pad + '▄' * len(vals)
    normalized = (vals - vmin) / (vmax - vmin + 1e-9)
    chars = ''.join(BLOCKS[min(int(v * len(BLOCKS)), len(BLOCKS) - 1)] for v in normalized)
    return pad + chars


def safe(val, fmt: str = '.0f', fallback: str = 'N/A') -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return fallback
    return format(val, fmt)


def crowding_emoji(score: float) -> str:
    if np.isnan(score):
        return '─ '
    if score > 2.0:  return '🔴'
    if score > 1.5:  return '🟠'
    if score > 1.0:  return '🟡'
    return '🟢'


def crowding_label(score: float) -> str:
    if np.isnan(score):   return 'N/A'
    if score > 2.0:       return 'EXTREME  🔴'
    if score > 1.5:       return 'HIGH     🟠'
    if score > 1.0:       return 'ELEVATED 🟡'
    return 'NORMAL   🟢'


def quadrant(rz: float, mz: float) -> str:
    if np.isnan(rz) or np.isnan(mz):
        return 'N/A'
    if rz > 0 and mz > 0:  return 'LEADING'
    if rz > 0 and mz < 0:  return 'WEAKENING'
    if rz < 0 and mz < 0:  return 'LAGGING'
    return 'IMPROVING'


def last(series: pd.Series) -> float:
    try:
        return float(series.dropna().iloc[-1])
    except Exception:
        return np.nan


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 1. Load benchmark ────────────────────────────────────────────────────
    bm_price = load_bd(f'{BENCHMARK}.parquet')
    bm_vol   = load_vol(f'{BENCHMARK}.parquet')
    if bm_price is None:
        print(f"ERROR: benchmark '{BENCHMARK}' not found at {BD}")
        return

    bm_idx = bm_price.index
    if len(bm_idx) < 60:
        print(f"ERROR: insufficient history ({len(bm_idx)} days < 60)")
        return

    acwi_ret20 = bm_price.pct_change(20).clip(-0.25, 0.25)

    # ── 2. Load all assets ────────────────────────────────────────────────────
    all_tickers = {t for tickers in BUCKETS.values() for t in tickers}
    prices: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}

    for ticker in all_tickers:
        p = load_bd(f'{ticker}.parquet')
        v = load_vol(f'{ticker}.parquet')
        if p is not None:
            prices[ticker]  = p.reindex(bm_idx, method='ffill')
        if v is not None:
            volumes[ticker] = v.reindex(bm_idx, method='ffill')

    # ── 3. Per-asset returns and relative volume ──────────────────────────────
    ret5:  dict[str, pd.Series] = {}
    ret20: dict[str, pd.Series] = {}
    ret60: dict[str, pd.Series] = {}
    rv:    dict[str, pd.Series] = {}

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

    # ── 4. Bucket aggregation ─────────────────────────────────────────────────
    bkt_r5:  dict[str, pd.Series] = {}
    bkt_r20: dict[str, pd.Series] = {}
    bkt_r60: dict[str, pd.Series] = {}
    bkt_rv:  dict[str, pd.Series] = {}

    for bkt, tickers in BUCKETS.items():
        r5s  = [ret5[t]  for t in tickers if t in ret5]
        r20s = [ret20[t] for t in tickers if t in ret20]
        r60s = [ret60[t] for t in tickers if t in ret60]
        rvs  = [rv[t]    for t in tickers if t in rv]
        if not r5s:
            continue
        bkt_r5[bkt]  = pd.concat(r5s,  axis=1).mean(axis=1)
        bkt_r20[bkt] = pd.concat(r20s, axis=1).mean(axis=1) if r20s else pd.Series(np.nan, index=bm_idx)
        bkt_r60[bkt] = pd.concat(r60s, axis=1).mean(axis=1) if r60s else pd.Series(np.nan, index=bm_idx)
        bkt_rv[bkt]  = pd.concat(rvs,  axis=1).mean(axis=1) if rvs  else pd.Series(np.nan, index=bm_idx)

    active = list(bkt_r5.keys())
    if not active:
        print("ERROR: no bucket data loaded.")
        return

    r5_df  = pd.DataFrame({b: bkt_r5[b]  for b in active})
    r20_df = pd.DataFrame({b: bkt_r20[b] for b in active})
    r60_df = pd.DataFrame({b: bkt_r60[b] for b in active})
    rv_df  = pd.DataFrame({b: bkt_rv[b]  for b in active})

    # ── Engine 1: Relative Strength ───────────────────────────────────────────
    z5  = cross_zscore(r5_df)
    z20 = cross_zscore(r20_df)
    z60 = cross_zscore(r60_df)

    z_w = 0.50 * z5 + 0.30 * z20 + 0.20 * z60
    rs_df = (z_w.clip(-3, 3) + 3) / 6 * 100

    rs_vs_acwi = r20_df.sub(acwi_ret20, axis=0)

    persist_df = rs_df.rolling(20, min_periods=10).apply(
        lambda x: (x > 60).mean(), raw=True
    )

    # ── Engine 2: Acceleration ────────────────────────────────────────────────
    accel_df       = r5_df - r20_df
    accel_z_df     = accel_df.apply(lambda c: rolling_zscore(c, 60))
    accel_score_df = (accel_z_df.clip(-3, 3) + 3) / 6 * 100
    rel_accel_df   = cross_zscore(accel_df)

    # ── Engine 3: RRG ─────────────────────────────────────────────────────────
    rs_ratio_raw = r20_df.div(acwi_ret20 + 1e-9, axis=0)
    rs_ratio_z   = rs_ratio_raw.apply(lambda c: rolling_zscore(c, 60))

    rs_mom_raw = rs_ratio_z.diff(5).ewm(span=5, adjust=False).mean()
    rs_mom_z   = rs_mom_raw.apply(lambda c: rolling_zscore(c, 60))

    # ── Engine 4: Crowding ────────────────────────────────────────────────────
    rs_z252    = rs_df.apply(lambda c: rolling_zscore(c, 252, min_periods=60))
    rv_z       = rv_df.apply(lambda c: rolling_zscore(c, 60))
    crowd_df   = 0.7 * rs_z252 + 0.3 * rv_z

    # ── Engine 5: Dispersion ──────────────────────────────────────────────────
    dispersion   = rs_df.std(axis=1)
    dispersion_z = rolling_zscore(dispersion, 60)

    # ── Collect today's values ────────────────────────────────────────────────
    rows = []
    for b in active:
        rz  = last(rs_ratio_z[b])
        mz  = last(rs_mom_z[b])
        sp_vals = rs_df[b].values
        rows.append({
            'bucket':    b,
            'rs':        last(rs_df[b]),
            'vs_acwi':   last(rs_vs_acwi[b]),
            'persist':   last(persist_df[b]),
            'crowd':     last(crowd_df[b]),
            'vel':       last(r5_df[b]),
            'mom':       last(r20_df[b]),
            'accel_sc':  last(accel_score_df[b]),
            'rel_accel': last(rel_accel_df[b]),
            'quad':      quadrant(rz, mz),
            'rz':        rz,
            'mz':        mz,
            'spark':     sparkline(sp_vals, 20),
            'sp_first':  next((float(v) for v in sp_vals[-20:] if not np.isnan(v)), np.nan),
            'sp_last':   last(rs_df[b]),
        })

    by_rs    = sorted(rows, key=lambda x: -x['rs']    if not np.isnan(x['rs'])    else -999)
    by_accel = sorted(rows, key=lambda x: -x['accel_sc'] if not np.isnan(x['accel_sc']) else -999)

    disp_now   = last(dispersion)
    disp_z_now = last(dispersion_z)
    today_str  = bm_idx[-1].strftime('%Y-%m-%d')

    # ── Dashboard ─────────────────────────────────────────────────────────────
    W  = 62
    SW = 20   # sparkline width
    BR = '─' * W

    print()
    print('═' * W)
    print(f'  CAPITAL FLOW DASHBOARD — {today_str}')
    print('═' * W)

    disp_lbl = 'CONCENTRATED' if (not np.isnan(disp_z_now) and disp_z_now > 1.5) else 'BROAD'
    print(f'\n  MARKET DISPERSION: {safe(disp_now, ".1f")}  [{disp_lbl}]')
    print('  (alta = capital concentrado em poucos temas)')

    # ── Leadership ────────────────────────────────────────────────────────────
    print(f'\n{BR}')
    print('  LEADERSHIP (quem domina hoje)')
    print(f'{BR}')
    hdr = f'  {"Rank":<4} {"Bucket":<20} {"RS":>5} {"vs ACWI":>9} {"Persist":>8} {"Crowd":>4}'
    print(hdr)
    print(f'  {"─"*4} {"─"*20} {"─"*5} {"─"*9} {"─"*8} {"─"*4}')
    for i, r in enumerate(by_rs, 1):
        rs_s  = safe(r['rs'],    '.0f')
        va_s  = (f'{r["vs_acwi"]*100:+.1f}%' if not np.isnan(r['vs_acwi']) else 'N/A')
        pe_s  = (f'{r["persist"]*100:.0f}%'   if not np.isnan(r['persist']) else 'N/A')
        cr_em = crowding_emoji(r['crowd'])
        print(f'  {i:<4} {r["bucket"]:<20} {rs_s:>5} {va_s:>9} {pe_s:>8} {cr_em}')

    # ── Rotation ──────────────────────────────────────────────────────────────
    print(f'\n{BR}')
    print('  ROTATION (quem está acelerando)')
    print(f'{BR}')
    print(f'  {"Rank":<4} {"Bucket":<20} {"Accel":>6} {"Vel":>7} {"Mom":>7} {"RelAcc":>7}')
    print(f'  {"─"*4} {"─"*20} {"─"*6} {"─"*7} {"─"*7} {"─"*7}')
    for i, r in enumerate(by_accel, 1):
        ac_s = safe(r['accel_sc'], '.0f')
        vl_s = (f'{r["vel"]*100:+.1f}%'   if not np.isnan(r['vel'])       else 'N/A')
        mo_s = (f'{r["mom"]*100:+.1f}%'   if not np.isnan(r['mom'])       else 'N/A')
        ra_s = (f'{r["rel_accel"]:+.1f}'  if not np.isnan(r['rel_accel']) else 'N/A')
        print(f'  {i:<4} {r["bucket"]:<20} {ac_s:>6} {vl_s:>7} {mo_s:>7} {ra_s:>7}')

    # ── RRG Quadrants ─────────────────────────────────────────────────────────
    print(f'\n{BR}')
    print('  RRG QUADRANTES')
    print(f'{BR}')
    quads: dict[str, list[str]] = {'LEADING': [], 'WEAKENING': [], 'IMPROVING': [], 'LAGGING': [], 'N/A': []}
    for r in rows:
        quads[r['quad']].append(r['bucket'])

    def _qlist(q: str) -> str:
        return ', '.join(quads[q]) or '—'

    print(f'  LEADING    (forte+acelerando):  {_qlist("LEADING")}')
    lead_w = _qlist("WEAKENING")
    print(f'  WEAKENING  (forte+desaceler.):  {lead_w}{"  ⚠️" if quads["WEAKENING"] else ""}')
    lead_i = _qlist("IMPROVING")
    print(f'  IMPROVING  (fraco+acelerando):  {lead_i}{"  📈" if quads["IMPROVING"] else ""}')
    print(f'  LAGGING    (fraco+desaceler.):  {_qlist("LAGGING")}')

    # ── Sparklines ────────────────────────────────────────────────────────────
    print(f'\n{BR}')
    print(f'  SPARKLINES RS Score (últimos {SW} dias)')
    print(f'{BR}')
    for r in by_rs:
        sf = safe(r['sp_first'], '.0f', '?')
        sl = safe(r['sp_last'],  '.0f', '?')
        print(f'  {r["bucket"]:<20} {r["spark"]}  {sf}→{sl}')

    # ── Brasil/EM Context ─────────────────────────────────────────────────────
    br_em = next((r for r in rows if r['bucket'] == 'Brasil_EM'), None)
    if br_em:
        print(f'\n{BR}')
        print('  CONTEXTO MACRO BRASIL/EM')
        print(f'{BR}')
        print(f'  Brasil_EM RS Score:   {safe(br_em["rs"], ".0f")}/100')
        print(f'  Quadrante RRG:        {br_em["quad"]}')
        print(f'  Aceleração relativa:  {safe(br_em["rel_accel"], "+.1f")}')
        pe_v = f'{br_em["persist"]*100:.0f}%' if not np.isnan(br_em['persist']) else 'N/A'
        print(f'  Flow Persistence:     {pe_v}')
        print(f'  Crowding:             {crowding_label(br_em["crowd"])}')

        q   = br_em['quad']
        pe  = br_em['persist'] if not np.isnan(br_em['persist']) else 0.0
        cr  = br_em['crowd']   if not np.isnan(br_em['crowd'])   else 0.0
        ra  = br_em['rel_accel'] if not np.isnan(br_em['rel_accel']) else 0.0

        print()
        if q == 'IMPROVING' and pe < 0.5:
            print('  [IMPROVING com persistência baixa]')
            print('  → "EM ganhando momentum relativo')
            print('     mas ainda sem confirmação institucional"')
        elif q == 'LEADING' and cr > 2.0:
            print('  [LEADING com crowding EXTREME]')
            print('  → "EM em fluxo forte — monitorar exaustão"')
        elif q == 'LEADING' and pe >= 0.7:
            print('  [LEADING com alta persistência]')
            print('  → "EM com fluxo institucional confirmado"')
        elif q == 'WEAKENING':
            print('  [WEAKENING]')
            print('  → "EM perdendo momentum — fluxo marginal migrando"')
        elif q == 'LAGGING':
            print('  [LAGGING]')
            print('  → "EM sem suporte de fluxo — aguardar sinal de reversão"')
        elif q == 'IMPROVING' and ra > 0:
            print('  [IMPROVING com aceleração positiva]')
            print('  → "EM com momentum incipiente — acompanhar persistência"')
        else:
            print(f'  [{q}] — acompanhar evolução da persistência')

    # ── Alerts ────────────────────────────────────────────────────────────────
    print(f'\n{BR}')
    print('  ALERTAS')
    print(f'{BR}')

    alerts: list[str] = []

    ai = next((r for r in rows if r['bucket'] == 'AI_Compute'), None)
    if ai and ai['quad'] == 'WEAKENING' and not np.isnan(ai['crowd']) and ai['crowd'] > 2.0:
        alerts.append('  ⚠️  Tech exaustão — rotação estrutural possível')

    if br_em:
        ra_v = br_em['rel_accel'] if not np.isnan(br_em['rel_accel']) else 0.0
        if br_em['quad'] == 'IMPROVING' and ra_v > 0:
            alerts.append('  📈 EM acelerando — fluxo marginal melhorando')

    if not np.isnan(disp_z_now) and disp_z_now > 2.0:
        alerts.append('  🔴 Capital concentrado — risco de rotação violenta')

    seen: set[str] = set()
    for r in rows:
        b, q = r['bucket'], r['quad']
        cr_v = r['crowd'] if not np.isnan(r['crowd']) else 0.0
        pe_v = r['persist'] if not np.isnan(r['persist']) else 1.0

        key_w = f'WEAK_{b}'
        if q == 'WEAKENING' and cr_v > 2.0 and b != 'AI_Compute' and key_w not in seen:
            alerts.append(f'  ⚠️  {b} WEAKENING + crowding EXTREME')
            seen.add(key_w)

        key_i = f'IMP_{b}'
        if q == 'IMPROVING' and pe_v < 0.3 and key_i not in seen:
            alerts.append(f'  📊 {b} IMPROVING mas persistência baixa — confirmar')
            seen.add(key_i)

        if q == 'LEADING' and cr_v > 1.5:
            key_l = f'LEAD_CROWD_{b}'
            if key_l not in seen:
                alerts.append(f'  🟠 {b} LEADING + crowding elevado — monitorar exaustão')
                seen.add(key_l)

    if alerts:
        for a in alerts:
            print(a)
    else:
        print('  ✅ Sem alertas críticos ativos')

    print()
    print('═' * W)
    print()


if __name__ == '__main__':
    main()
