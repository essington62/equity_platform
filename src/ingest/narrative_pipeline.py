"""
narrative_pipeline.py
YouTube URL → transcript → MongoDB + ChromaDB → DeepSeek scoring
"""

import os
import re
import json
import datetime
import pathlib
import traceback

import yaml
import yt_dlp
from pymongo import MongoClient
import chromadb

# ── Config ──────────────────────────────────────────────────────────────────

MONGODB_URI = "mongodb+srv://edmundobrown_db_user:XktKCOIu54y7pRGN@equity-platform.snfkict.mongodb.net/?appName=equity-platform"
MONGODB_DB  = "equity_platform"
CHROMA_PATH = "/Users/brown/Documents/MLGeral/equity_platform/data/chromadb"
AUDIO_TMP   = "/tmp/equity_platform/"

_CREDS_PATH = pathlib.Path(__file__).parents[2] / "conf" / "credentials.yml"

def _load_deepseek_key() -> str:
    with open(_CREDS_PATH) as f:
        creds = yaml.safe_load(f)
    return creds["deepseek"]["api_key"]

VERTICAIS = {
    "AI_Compute":   ["CHIP11","BSOX39","MSFT34","GOGL34","AMZO34","ORCL34","NVDA","MSFT","GOOGL"],
    "China_Coreia": ["BEWY39","BEWJ39","BEEM39","DTCR39","FXI","EWY","EWJ"],
    "Brasil_EM":    ["WEGE3","WINFUT","BEWZ39","IVWO11","EWZ","BOVESPA"],
    "Commodities":  ["BCPX39","GLD","AURA33","COPPER","GDX"],
    "Crypto":       ["BITH11","BKCH39","BTC"],
    "Agro":         ["WSPM26","CCMU26","SJCN26","SOYB","CORN","WEAT"],
    "Macro":        ["DOLFUT","M1TA34","DXY","VIX","TLT"],
}

# ── Step 1 — Download + Transcription ───────────────────────────────────────

def transcribe_url(url: str, channel: str, video_type: str) -> dict:
    print(f"\n[STEP 1] Transcribing: {channel} | {url}")
    os.makedirs(AUDIO_TMP, exist_ok=True)

    today        = datetime.date.today().isoformat()
    safe_channel = channel.replace(" ", "_")
    audio_path   = os.path.join(AUDIO_TMP, f"{safe_channel}_{today}.mp3")

    # 1a+1b. Fetch metadata + download audio via yt_dlp Python API
    ydl_opts = {
        "format":       "bestaudio/best",
        "outtmpl":      audio_path,
        "postprocessors": [{
            "key":            "FFmpegExtractAudio",
            "preferredcodec": "mp3",
        }],
        "noplaylist":   True,
        "quiet":        True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info         = ydl.extract_info(url, download=True)
        title        = info.get("title", "unknown")
        duration_sec = info.get("duration", 0) or 0
        duration_min = round(duration_sec / 60, 1)

    # yt_dlp adds .mp3 suffix via postprocessor; actual file may need path fix
    if not os.path.exists(audio_path) and os.path.exists(audio_path + ".mp3"):
        audio_path = audio_path + ".mp3"

    print(f"  title: {title} | duration: {duration_min} min")
    print(f"  audio saved: {audio_path}")

    # 1c. Transcribe
    language   = "unknown"
    transcript = ""
    try:
        import mlx_whisper
        print("  transcribing with mlx_whisper…")
        result     = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo="mlx-community/whisper-medium-mlx",
            language="pt",
            initial_prompt="Morning call mercado financeiro Brasil análise ações",
        )
        transcript = result["text"]
        language   = result.get("language", "unknown")
    except Exception as e:
        print(f"  mlx_whisper failed ({e}), falling back to openai-whisper…")
        import whisper
        model      = whisper.load_model("medium")
        result     = model.transcribe(audio_path, language="pt",
                         initial_prompt="Morning call mercado financeiro Brasil análise ações")
        transcript = result["text"]
        language   = result.get("language", "unknown")

    print(f"  language: {language} | chars: {len(transcript)}")

    # 1d. Cleanup audio
    try:
        os.remove(audio_path)
        print(f"  audio deleted: {audio_path}")
    except Exception:
        pass

    return {
        "url":          url,
        "title":        title,
        "channel":      channel,
        "video_type":   video_type,
        "date":         today,
        "duration_min": duration_min,
        "language":     language,
        "transcript":   transcript,
    }

# ── Step 2 — Chunking ───────────────────────────────────────────────────────

def chunk_text(transcript: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    words  = transcript.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    print(f"[STEP 2] {len(chunks)} chunks created (size={chunk_size}, overlap={overlap})")
    return chunks

# ── Step 3 — MongoDB ─────────────────────────────────────────────────────────

def save_to_mongodb(meta: dict, chunks: list[str]):
    print(f"[STEP 3] Saving to MongoDB…")
    client = MongoClient(MONGODB_URI)
    db     = client[MONGODB_DB]
    doc    = {
        "date":         meta["date"],
        "channel":      meta["channel"],
        "title":        meta["title"],
        "url":          meta["url"],
        "video_type":   meta["video_type"],
        "language":     meta["language"],
        "duration_min": meta["duration_min"],
        "processed_at": datetime.datetime.utcnow(),
        "status":       "processing",
        "chunk_count":  len(chunks),
        "chroma_ids":   [],
    }
    result    = db["sources"].insert_one(doc)
    source_id = result.inserted_id
    print(f"  source_id: {source_id}")
    client.close()
    return source_id

# ── Step 4 — ChromaDB ────────────────────────────────────────────────────────

def save_to_chromadb(source_id, channel: str, date: str,
                     video_type: str, language: str, chunks: list[str]):
    print(f"[STEP 4] Saving {len(chunks)} chunks to ChromaDB…")
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection("transcripts")

    ids       = []
    documents = []
    metadatas = []
    safe_ch   = channel.replace(" ", "_")

    for i, chunk in enumerate(chunks):
        chunk_id = f"{safe_ch}_{date}_{video_type}_chunk_{i:03d}"
        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append({
            "source_id":   str(source_id),
            "date":        date,
            "channel":     channel,
            "video_type":  video_type,
            "chunk_index": i,
            "language":    language,
        })

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"  inserted {len(ids)} chunks | first_id: {ids[0]}")

    # Update MongoDB with chroma_ids
    mongo_client = MongoClient(MONGODB_URI)
    db           = mongo_client[MONGODB_DB]
    from bson import ObjectId
    db["sources"].update_one(
        {"_id": source_id},
        {"$set": {"chroma_ids": ids}}
    )
    mongo_client.close()
    return ids

# ── Steps 5+6 — DeepSeek Unified Analysis (CAMADA 1 + CAMADA 2) ─────────────

def _build_grade_str() -> str:
    lines = []
    for vertical, tickers in VERTICAIS.items():
        lines.append(f"  {vertical}: {', '.join(tickers)}")
    return "\n".join(lines)

_UNIFIED_SYSTEM_TEMPLATE = """Você é um analista de fluxo de capital.

Nossa carteira monitorada (NOSSA_GRADE):
{NOSSA_GRADE}

Extraia DOIS JSONs da transcrição.

===JSON_1===
Array com sentiment por vertical. Para cada vertical mencionada:
{{
  "vertical": "AI_Compute|China_Coreia|Brasil_EM|Commodities|Crypto|Agro|Macro",
  "sentiment": "BULLISH|BEARISH|NEUTRO",
  "score": número de -100 a +100,
  "confianca": 1 a 5,
  "argumento": "frase curta",
  "ativos_citados": ["lista"],
  "catalisadores": ["lista"],
  "mudanca_narrativa": true/false
}}
Se vertical não for mencionada, omitir.

===JSON_2===
Objeto com sinais críticos:
{{
  "fluxo_capital": [
    {{"tipo": "saida_estrangeiro|entrada_institucional|rotacao_setor",
      "valor": "com unidade",
      "periodo": "hoje|semana|etc",
      "fonte_dado": "B3|Bloomberg|Fed|etc",
      "impacto": "positivo|negativo|neutro",
      "trecho": "citação literal curta"}}
  ],
  "ativos_destaque": [
    {{"ativo": "nome",
      "ticker_grade": "ticker da NOSSA_GRADE se aplicável, senão ticker padrão",
      "mencao": "como foi citado",
      "direcao": "alta|baixa|lateral|indefinido",
      "nivel_citado": null ou número,
      "contexto": "catalisador mencionado",
      "relevancia": 1 a 5}}
  ],
  "posicionamento_analista": [
    {{"analista": "nome ou apresentador",
      "acao": "comprou|vendeu|reduziu_posicao|aumentou_posicao|recomenda",
      "ativo": "ativo ou vertical",
      "percentual": "string ou null",
      "racional": "frase curta"}}
  ],
  "dados_macro_citados": [
    {{"indicador": "snake_case",
      "valor": "com unidade",
      "periodo": "string ou null",
      "interpretacao": "bearish_emergentes|pressao_inflacionaria|etc"}}
  ],
  "alertas": [
    {{"tipo": "rotacao_capital|mudanca_narrativa|risco_sistemico|oportunidade",
      "descricao": "frase objetiva",
      "urgencia": "alta|media|baixa"}}
  ]
}}

REGRAS:
- Capture números exatos quando citados
- Se analista mudou posição própria → capture em posicionamento_analista
- Priorize ativos da NOSSA_GRADE em ticker_grade
- Nível de preço citado → capture sempre em nivel_citado
- Se seção vazia → lista []
- Não invente dados não mencionados

Retorne EXATAMENTE neste formato (sem markdown extra):
===JSON_1===
[array aqui]
===JSON_2===
{{objeto aqui}}"""


def _parse_unified_response(raw: str) -> tuple[list, dict]:
    raw = raw.strip()
    # Strip outer markdown fences if present
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]

    j1_marker, j2_marker = "===JSON_1===", "===JSON_2==="
    i1 = raw.find(j1_marker)
    i2 = raw.find(j2_marker)

    if i1 == -1 or i2 == -1:
        raise ValueError(f"Markers not found in response:\n{raw[:400]}")

    raw_j1 = raw[i1 + len(j1_marker): i2].strip()
    raw_j2 = raw[i2 + len(j2_marker):].strip()

    # Strip per-block fences
    def _strip_fence(s: str) -> str:
        if s.startswith("```"):
            lines = s.split("\n")
            s = "\n".join(lines[1:])
            if s.rstrip().endswith("```"):
                s = s[: s.rstrip().rfind("```")]
        return s.strip()

    scores  = json.loads(_strip_fence(raw_j1))
    signals = json.loads(_strip_fence(raw_j2))
    return scores, signals


def analyze_with_deepseek(source_id, channel: str, date: str,
                          transcript: str) -> tuple[list, dict]:
    print(f"[STEP 5+6] DeepSeek unified analysis (CAMADA 1 + 2)…")
    from openai import OpenAI

    system_prompt = _UNIFIED_SYSTEM_TEMPLATE.format(NOSSA_GRADE=_build_grade_str())
    api_key       = _load_deepseek_key()
    client        = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": transcript[:32_000]},
        ],
        temperature=0.1,
    )

    raw             = response.choices[0].message.content
    scores, signals = _parse_unified_response(raw)

    mongo_client = MongoClient(MONGODB_URI)
    db           = mongo_client[MONGODB_DB]

    for s in scores:
        db["scores"].insert_one({
            "source_id":         source_id,
            "date":              date,
            "channel":           channel,
            "vertical":          s.get("vertical"),
            "sentiment":         s.get("sentiment"),
            "score":             s.get("score"),
            "confianca":         s.get("confianca"),
            "argumento":         s.get("argumento"),
            "ativos_citados":    s.get("ativos_citados", []),
            "catalisadores":     s.get("catalisadores", []),
            "mudanca_narrativa": s.get("mudanca_narrativa", False),
        })

    db["signals"].insert_one({
        "source_id":               source_id,
        "date":                    date,
        "channel":                 channel,
        "fluxo_capital":           signals.get("fluxo_capital", []),
        "ativos_destaque":         signals.get("ativos_destaque", []),
        "posicionamento_analista": signals.get("posicionamento_analista", []),
        "dados_macro_citados":     signals.get("dados_macro_citados", []),
        "alertas":                 signals.get("alertas", []),
    })

    db["sources"].update_one(
        {"_id": source_id},
        {"$set": {"status": "processed"}}
    )
    mongo_client.close()

    print(f"  {len(scores)} verticals | signals: "
          f"fluxo={len(signals.get('fluxo_capital',[]))} "
          f"ativos={len(signals.get('ativos_destaque',[]))} "
          f"alertas={len(signals.get('alertas',[]))}")
    return scores, signals


def _print_signals(channel: str, s: dict):
    print(f"\n{'─'*60}")
    print(f"  SIGNALS — {channel}")
    print(f"{'─'*60}")

    if s.get("fluxo_capital"):
        print("  FLUXO:")
        for f in s["fluxo_capital"]:
            icon = "▼" if f.get("impacto") == "negativo" else ("▲" if f.get("impacto") == "positivo" else "→")
            print(f"    {icon} [{f.get('tipo','')}] {f.get('valor','')} ({f.get('periodo','')}) — {f.get('fonte_dado','')}")

    if s.get("ativos_destaque"):
        print("  ATIVOS:")
        for a in s["ativos_destaque"]:
            arrow = "↑" if a.get("direcao") == "alta" else ("↓" if a.get("direcao") == "baixa" else "→")
            nivel   = f" @ {a['nivel_citado']}" if a.get("nivel_citado") else ""
            ticker  = a.get("ticker_grade") or "—"
            print(f"    {arrow} {a.get('ativo',''):<12} [{ticker}]{nivel}  rel={a.get('relevancia','?')}  {a.get('contexto','')[:50]}")

    if s.get("posicionamento_analista"):
        print("  POSICIONAMENTO:")
        for p in s["posicionamento_analista"]:
            pct = f" ({p['percentual']})" if p.get("percentual") else ""
            print(f"    • {p.get('analista','?')} → {p.get('acao','')} {p.get('ativo','')}{pct}: {p.get('racional','')[:50]}")

    if s.get("dados_macro_citados"):
        print("  MACRO DATA:")
        for m in s["dados_macro_citados"]:
            periodo = f" [{m['periodo']}]" if m.get("periodo") else ""
            print(f"    • {m.get('indicador','')}: {m.get('valor','')}{periodo} → {m.get('interpretacao','')}")

    if s.get("alertas"):
        print("  ALERTAS:")
        for al in s["alertas"]:
            urg = {"alta": "!!!", "media": "!!", "baixa": "!"}.get(al.get("urgencia",""), "?")
            print(f"    {urg} [{al.get('tipo','')}] {al.get('descricao','')[:70]}")
    print()

# ── Step 7 — Macro Timeseries ────────────────────────────────────────────────

INDICATORS_MAP = {
    # Juros globais
    "treasury_10y_eua": {"unit": "percent",     "keywords": ["treasury 10", "10y", "10 anos eua", "10-year", "juros_americanos_10", "juros_10y", "10 anos americano", "juros_10_anos_americanos", "juros_americano_10"]},
    "treasury_30y_eua": {"unit": "percent",     "keywords": ["treasury 30", "30y", "30 anos eua", "30-year", "juros_americanos_30", "juros_30y", "30 anos americano", "30 anos eua"]},
    "treasury_2y_eua":  {"unit": "percent",     "keywords": ["treasury 2", "2y", "2 anos eua", "juros_2y"]},
    "juros_30y_japao":  {"unit": "percent",     "keywords": ["japão 30", "japan 30", "jgb 30", "japao_30", "juros_japoneses_30"]},
    "juros_10y_japao":  {"unit": "percent",     "keywords": ["japão 10", "japan 10", "jgb 10", "japao_10", "juros_japoneses_10"]},
    "juros_40y_japao":  {"unit": "percent",     "keywords": ["japão 40", "japan 40", "jgb 40", "japao_40", "juros_japoneses_40"]},
    "juros_10y_uk":     {"unit": "percent",     "keywords": ["uk 10", "reino unido 10", "gilt 10", "juros_ingleses_10"]},
    "juros_30y_uk":     {"unit": "percent",     "keywords": ["uk 30", "reino unido 30", "gilt 30", "juros_ingleses_30"]},
    # Brasil
    "selic_projecao":        {"unit": "percent",      "keywords": ["selic", "juros brasil", "di futuro", "selic_projecao", "selic_fim", "selic_2026", "selic_meta"]},
    "ipca_projecao":         {"unit": "percent",      "keywords": ["ipca", "inflação brasil", "cpi brasil", "ipca_projecao", "ipca_2026", "ipca_2027"]},
    "ntnb_longo":            {"unit": "percent",      "keywords": ["ntn-b", "ntnb", "ipca+", "ntnb_longo", "juro_real_ntnb"]},
    "juros_10y_br":          {"unit": "percent",      "keywords": ["juros brasil 10", "juros brasileiros 10", "juros_brasileiros_10", "di 10", "juro_10_anos_brasil", "juros_br_10"]},
    "fluxo_estrangeiro_b3":  {"unit": "milhoes_brl",  "keywords": ["estrangeiro", "gringo", "fluxo b3", "foreign flow", "fluxo_estrangeiro", "saida_estrangeiro", "entrada_estrangeiro", "captacao_estrangeira"]},
    "inflacao_servicos_br":  {"unit": "percent",      "keywords": ["inflação serviços", "servicos brasil", "inflacao_servicos"]},
    # EUA macro
    "cpi_eua":     {"unit": "percent",  "keywords": ["cpi eua", "cpi us", "inflação eua", "us inflation", "cpi_americano", "cpi_eua"]},
    "ppi_eua":     {"unit": "percent",  "keywords": ["ppi eua", "ppi us", "ppi_americano"]},
    "fed_funds":   {"unit": "percent",  "keywords": ["fed funds", "ffr", "taxa fed", "fed_funds"]},
    "pib_eua":     {"unit": "percent",  "keywords": ["pib eua", "gdp us", "pib_eua", "gdp_eua"]},
    "pib_china":   {"unit": "percent",  "keywords": ["pib china", "gdp china", "pib_china", "crescimento_china", "crescimento china"]},
    # Commodities
    "brent_price":  {"unit": "usd",     "keywords": ["brent", "petróleo brent", "crude", "brent_price", "preco_barril", "petroleo_brent"]},
    "wti_price":    {"unit": "usd",     "keywords": ["wti", "petróleo wti", "wti_price"]},
    "copper_price": {"unit": "usd",     "keywords": ["copper", "cobre", "hg=f", "copper_price"]},
    "gold_price":   {"unit": "usd",     "keywords": ["gold", "ouro", "xau", "gold_price", "preco_ouro"]},
    "iron_ore":     {"unit": "usd",     "keywords": ["minério", "iron ore", "minério de ferro", "iron_ore"]},
    # Alavancagem / Crowding
    "options_calls_abertura": {"unit": "milhoes_contratos", "keywords": ["call", "calls", "opções call", "open interest options"]},
    "margem_corretoras_eua":  {"unit": "trilhoes_usd",      "keywords": ["margin debt", "margin account", "conta margem", "alavancagem corretoras"]},
    "etf_alavancados_fluxo":  {"unit": "bilhoes_usd",       "keywords": ["leveraged etf", "etf alavancado", "3x etf"]},
    "put_call_ratio":         {"unit": "ratio",              "keywords": ["put/call", "put call ratio", "razão put call"]},
    # FX & índices
    "usdbrl":       {"unit": "brl",     "keywords": ["dólar real", "usdbrl", "dólar brasil", "real", "cambio", "usdbrl"]},
    "dxy":          {"unit": "pontos",  "keywords": ["dxy", "dollar index", "dxy_index"]},
    "vix":          {"unit": "pontos",  "keywords": ["vix", "volatilidade", "vix_index"]},
    "ibovespa":     {"unit": "pontos",  "keywords": ["ibovespa", "ibov", "bovespa", "ibovespa_pontos"]},
    "bitcoin":      {"unit": "usd",     "keywords": ["bitcoin", "btc", "bitcoin_price"]},
}


def parse_value(text: str, unit_hint: str = "percent") -> tuple:
    """Normaliza string → (float, unit). Retorna (None, None) se não parseable."""
    try:
        s = str(text).strip().lower()

        # Qualitative → skip
        for skip in ("fraco", "forte", "leve", "recente", "início", "acima", "abaixo",
                     "não especif", "flat", "estável", "revisado", "positivo", "negativo",
                     "neutro", "acelerado", "desaceler", "manutenção", "manter", "corte",
                     "queda", "alta generaliz", "perdeu força"):
            if skip in s:
                return None, None
        if not any(c.isdigit() for c in s):
            return None, None

        # Detect unit from text
        is_pct    = "%" in s
        is_bi     = "bi" in s and "b3" not in s
        is_mi     = "mi" in s and "min" not in s and "mil" not in s
        is_usd    = "$" in s and "r$" not in s and "brl" not in s
        is_brl    = "r$" in s or "brl" in s

        # Strip symbols
        clean = s.replace("~", "").replace("r$", "").replace("$", "")
        clean = clean.replace("%", "").replace("bi", "").replace("mi", "")
        clean = clean.replace("milhões", "").replace("milhoes", "")
        clean = clean.replace("bilhões", "").replace("bilhoes", "")
        clean = clean.replace("bilhão", "").replace("bilhao", "")
        clean = clean.replace(",", ".").strip()

        # Handle range "90-110" → midpoint (but keep negatives "-600")
        range_match = re.search(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)", clean)
        if range_match:
            val = (float(range_match.group(1)) + float(range_match.group(2))) / 2
        else:
            # Extract first number (handles "-600" correctly)
            match = re.search(r"[-]?\d+\.?\d*", clean)
            if not match:
                return None, None
            val = float(match.group())

        # Scale and assign unit
        if is_bi:
            return val * 1_000, "milhoes_brl"
        if is_mi:
            return val, "milhoes_brl"
        if is_pct:
            return val, "percent"
        if is_usd:
            return val, "usd"
        if is_brl:
            return val, "milhoes_brl"
        # Fallback to hint
        return val, unit_hint

    except Exception:
        return None, None


def _match_indicator(indicador: str) -> tuple[str, str]:
    """Returns (canonical_key, unit). Falls back to (indicador, 'percent')."""
    key = indicador.lower().replace(" ", "_")
    # Exact match
    if key in INDICATORS_MAP:
        return key, INDICATORS_MAP[key]["unit"]
    # Keyword scan
    for canon, cfg in INDICATORS_MAP.items():
        for kw in cfg["keywords"]:
            if kw in key or kw in indicador.lower():
                return canon, cfg["unit"]
    return indicador, "percent"


def save_timeseries(source_id, date: str, channel: str, signals: dict) -> list[str]:
    print(f"[STEP 7] Saving macro timeseries…")
    macro_data = signals.get("dados_macro_citados", [])
    if not macro_data:
        print("  no macro data to persist")
        return []

    mongo_client = MongoClient(MONGODB_URI)
    db           = mongo_client[MONGODB_DB]

    # Ensure unique index (idempotent)
    db["macro_timeseries"].create_index(
        [("date", 1), ("indicator", 1), ("source", 1)],
        unique=True, background=True
    )

    saved = []
    for item in macro_data:
        indicador_raw = item.get("indicador", "")
        valor_raw     = item.get("valor", "")
        if not indicador_raw or not valor_raw:
            continue

        canon, unit_hint = _match_indicator(indicador_raw)
        value, unit      = parse_value(valor_raw, unit_hint)
        if value is None:
            continue

        # Derive direction from interpretation or context
        interp   = item.get("interpretacao", "")
        dir_text = item.get("periodo", "") + " " + interp
        if any(w in dir_text for w in ("alta", "subindo", "crescendo", "up", "aumento")):
            direction = "up"
        elif any(w in dir_text for w in ("baixa", "caindo", "queda", "down", "redução", "corte")):
            direction = "down"
        else:
            direction = "stable"

        doc = {
            "date":           date,
            "indicator":      canon,
            "value":          value,
            "unit":           unit,
            "direction":      direction,
            "interpretation": interp,
            "source":         channel,
            "source_id":      source_id,
            "confidence":     3,
            "raw_text":       f"{indicador_raw}: {valor_raw}",
        }

        try:
            db["macro_timeseries"].update_one(
                {"date": date, "indicator": canon, "source": channel},
                {"$set": doc},
                upsert=True,
            )
            saved.append(canon)
        except Exception as e:
            print(f"  [WARN] upsert failed for {canon}: {e}")

    mongo_client.close()
    print(f"  saved {len(saved)} indicators: {saved}")
    return saved


def print_timeseries_table(date: str):
    mongo_client = MongoClient(MONGODB_URI)
    db           = mongo_client[MONGODB_DB]
    rows         = list(db["macro_timeseries"].find(
        {"date": date}, {"_id": 0, "indicator": 1, "value": 1, "unit": 1, "source": 1}
    ).sort([("indicator", 1), ("source", 1)]))
    total        = db["macro_timeseries"].count_documents({})
    mongo_client.close()

    if not rows:
        print(f"\n  No timeseries data for {date}")
        return

    c1, c2, c3, c4 = 24, 10, 10, 22
    sep = f"  ┌{'─'*c1}┬{'─'*c2}┬{'─'*c3}┬{'─'*c4}┐"
    hdr = f"  │{'Indicador':^{c1}}│{'Valor':^{c2}}│{'Unid':^{c3}}│{'Fonte':^{c4}}│"
    div = f"  ├{'─'*c1}┼{'─'*c2}┼{'─'*c3}┼{'─'*c4}┤"
    bot = f"  └{'─'*c1}┴{'─'*c2}┴{'─'*c3}┴{'─'*c4}┘"

    print(f"\n  MACRO TIMESERIES — {date}")
    print(sep)
    print(hdr)
    print(div)
    for r in rows:
        val_str = f"{r['value']:+.2f}" if r["value"] is not None else "—"
        src     = r["source"][:c4].center(c4)
        print(f"  │{r['indicator'][:c1]:<{c1}}│{val_str:>{c2}}│{r['unit'][:c3]:<{c3}}│{src}│")
    print(bot)
    print(f"  Total: {len(rows)} indicadores salvos hoje | "
          f"db.macro_timeseries total: {total}\n")

# ── Main ─────────────────────────────────────────────────────────────────────

VIDEOS = [
    {
        "url":        "https://www.youtube.com/watch?v=__mGJD9k4Ac",
        "channel":    "Genial Investimentos",
        "video_type": "abertura",
    },
    {
        "url":        "https://www.youtube.com/watch?v=gCYlrCNS09E",
        "channel":    "XP Investimentos",
        "video_type": "abertura",
    },
    {
        "url":        "https://www.youtube.com/watch?v=LAwArh9izYU",
        "channel":    "BTG Pactual",
        "video_type": "abertura",
    },
]


def _print_scores(channel: str, scores: list[dict]):
    print(f"\n{'═'*60}")
    print(f"  SCORES — {channel}")
    print(f"{'═'*60}")
    for s in scores:
        flag = "🔄" if s.get("mudanca_narrativa") else "  "
        print(f"  {flag} {s['vertical']:<15} {s['sentiment']:<8} "
              f"score={s['score']:+4d}  conf={s['confianca']}  "
              f"| {s.get('argumento','')[:60]}")
    print()


def process_video(v: dict):
    url        = v["url"]
    channel    = v["channel"]
    video_type = v["video_type"]

    print(f"\n{'═'*60}")
    print(f"  PROCESSING: {channel}")
    print(f"{'═'*60}")

    try:
        meta = transcribe_url(url, channel, video_type)
    except Exception:
        print(f"  [ERROR] transcribe_url failed:\n{traceback.format_exc()}")
        return

    try:
        chunks = chunk_text(meta["transcript"])
    except Exception:
        print(f"  [ERROR] chunk_text failed:\n{traceback.format_exc()}")
        return

    try:
        source_id = save_to_mongodb(meta, chunks)
    except Exception:
        print(f"  [ERROR] save_to_mongodb failed:\n{traceback.format_exc()}")
        return

    try:
        chroma_ids = save_to_chromadb(
            source_id, channel, meta["date"],
            video_type, meta["language"], chunks
        )
        print(f"  ChromaDB: {len(chroma_ids)} chunks confirmed")
    except Exception:
        print(f"  [ERROR] save_to_chromadb failed:\n{traceback.format_exc()}")

    signals = {}
    try:
        scores, signals = analyze_with_deepseek(
            source_id, channel, meta["date"], meta["transcript"]
        )
        _print_scores(channel, scores)
        _print_signals(channel, signals)
    except Exception:
        print(f"  [ERROR] analyze_with_deepseek failed:\n{traceback.format_exc()}")

    try:
        save_timeseries(source_id, meta["date"], channel, signals)
    except Exception:
        print(f"  [ERROR] save_timeseries failed:\n{traceback.format_exc()}")


if __name__ == "__main__":
    import time
    t0   = time.time()
    date = str(datetime.date.today())
    for video in VIDEOS:
        process_video(video)
    elapsed = round((time.time() - t0) / 60, 1)
    print(f"\n[DONE] All videos processed in {elapsed} min")

    # Macro timeseries table
    try:
        print_timeseries_table(date)
    except Exception:
        print(f"[WARN] print_timeseries_table failed:\n{traceback.format_exc()}")

    # Collection counts
    try:
        mongo_client = MongoClient(MONGODB_URI)
        db = mongo_client[MONGODB_DB]
        print(f"[MongoDB]  sources={db['sources'].count_documents({})}"
              f"  scores={db['scores'].count_documents({})}"
              f"  signals={db['signals'].count_documents({})}"
              f"  macro_timeseries={db['macro_timeseries'].count_documents({})}")
        mongo_client.close()
    except Exception:
        pass

    try:
        cc  = chromadb.PersistentClient(path=CHROMA_PATH)
        col = cc.get_or_create_collection("transcripts")
        print(f"[ChromaDB] transcripts collection count={col.count()}")
    except Exception:
        pass
