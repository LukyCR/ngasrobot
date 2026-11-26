import html
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import yfinance as yf

TZ = ZoneInfo("Europe/Prague")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

BULLISH_WORDS = [
    "cold", "freeze", "arctic", "chill", "heating demand", "draw", "larger draw",
    "bullish", "surge", "rally", "jump", "soar", "lng record", "outage", "deficit",
    "colder", "storm", "snow", "tight supply", "strong demand", "cooler"
]
BEARISH_WORDS = [
    "warm", "mild", "warmer", "bearish", "drop", "fall", "plunge", "tumble",
    "surplus", "build", "injection", "glut", "oversupply", "weak demand",
    "record production", "shutdown", "lng outage", "light demand", "comfortable"
]


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def strip_html(text: str) -> str:
    clean = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", html.unescape(clean)).strip()


def fetch_eia_storage():
    """Stáhne poslední oficiální týdenní report zásob zemního plynu z EIA."""
    data = {
        "available": False,
        "report_date": "Nezjištěno",
        "total_bcf": None,
        "weekly_change_text": "Nezjištěno",
        "weekly_change_bcf": None,
        "vs_5y_text": "Nezjištěno",
        "vs_ly_text": "Nezjištěno",
        "bias": "NEUTRAL",
        "score_impact": 0,
        "summary": "Nepodařilo se načíst automatický text EIA – zkontroluj přímý odkaz níže."
    }

    urls = ["https://ir.eia.gov/ngs/wngsr.txt", "https://ir.eia.gov/ngs/ngs.html"]
    raw_text = ""
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=12) as resp:
                raw_text = strip_html(resp.read().decode("utf-8", errors="ignore"))
            if "Working gas in storage" in raw_text:
                break
        except Exception:
            continue

    if not raw_text or "Working gas in storage" not in raw_text:
        return data

    m_total = re.search(r"Working gas in storage was ([\d,]+)\s*Bcf", raw_text, re.I)
    m_date = re.search(r"as of Friday,\s*([A-Za-z]+\s+\d+,\s+\d{4})", raw_text, re.I)
    m_chg = re.search(r"net (increase|decrease) of ([\d,]+)\s*Bcf", raw_text, re.I)
    m_5y = re.search(r"([\d,]+)\s*Bcf (above|below) the five-year average", raw_text, re.I)
    m_ly = re.search(r"([\d,]+)\s*Bcf (higher|less|more|lower|above|below) than last year", raw_text, re.I)

    if m_total:
        data["available"] = True
        data["total_bcf"] = int(m_total.group(1).replace(",", ""))
    if m_date:
        data["report_date"] = m_date.group(1)
    if m_chg:
        direction = m_chg.group(1).lower()
        val = int(m_chg.group(2).replace(",", ""))
        sign = "-" if direction == "decrease" else "+"
        cz_dir = "čerpání" if direction == "decrease" else "vtláčení"
        data["weekly_change_bcf"] = -val if direction == "decrease" else val
        data["weekly_change_text"] = f"{sign}{val} Bcf ({cz_dir})"
    if m_5y:
        val_5y = int(m_5y.group(1).replace(",", ""))
        pos = m_5y.group(2).lower()
        if pos == "below":
            data["vs_5y_text"] = f"-{val_5y} Bcf pod 5Y průměrem (Deficit)"
            data["bias"] = "BULLISH"
            data["score_impact"] = 8
        else:
            data["vs_5y_text"] = f"+{val_5y} Bcf nad 5Y průměrem (Přebytek)"
            data["bias"] = "BEARISH"
            data["score_impact"] = -8
    if m_ly:
        val_ly = int(m_ly.group(1).replace(",", ""))
        pos_ly = m_ly.group(2).lower()
        sign_ly = "+" if pos_ly in ("higher", "more", "above") else "-"
        data["vs_ly_text"] = f"{sign_ly}{val_ly} Bcf oproti loňsku"

    data["summary"] = (
        f"Zásoby k {data['report_date']}: {data['total_bcf']} Bcf | "
        f"Týdenní změna: {data['weekly_change_text']} | {data['vs_5y_text']}."
    )
    return data


def fetch_weather_sources():
    """Načte nejnovější výhled z NatGasWeather RSS (GFS/ECMWF) a vyhodnotí bias."""
    result = {
        "natgasweather_title": "Aktuální RSS nebylo dostupné – sleduj mapy NOAA níže.",
        "natgasweather_summary": "",
        "natgasweather_link": "https://natgasweather.com/",
        "natgasweather_date": "",
        "bias": "NEUTRAL",
        "score_impact": 0,
        "noaa_6_10_img": "https://www.cpc.ncep.noaa.gov/products/predictions/610day/610temp.new.gif",
        "noaa_8_14_img": "https://www.cpc.ncep.noaa.gov/products/predictions/814day/814temp.new.gif",
    }

    try:
        req = urllib.request.Request("https://natgasweather.com/feed/", headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            root = ET.fromstring(resp.read())
        item = root.find("./channel/item")
        if item is not None:
            title = item.findtext("title", "").strip()
            desc = strip_html(item.findtext("description", ""))[:320]
            link = item.findtext("link", "https://natgasweather.com/")
            pub = item.findtext("pubDate", "")
            combined = f"{title} {desc}".lower()

            b_hits = sum(1 for w in BULLISH_WORDS if w in combined)
            s_hits = sum(1 for w in BEARISH_WORDS if w in combined)

            result["natgasweather_title"] = title
            result["natgasweather_summary"] = desc + ("..." if len(desc) >= 320 else "")
            result["natgasweather_link"] = link
            result["natgasweather_date"] = pub

            if b_hits > s_hits:
                result["bias"] = "BULLISH"
                result["score_impact"] = 10
            elif s_hits > b_hits:
                result["bias"] = "BEARISH"
                result["score_impact"] = -10
    except Exception:
        pass

    return result


def fetch_news_sentiment():
    url = (
        "https://news.google.com/rss/search?"
        "q=%22natural+gas%22+OR+%22Henry+Hub%22+OR+%22NatGasWeather%22+OR+%22EIA+natural+gas%22"
        "&hl=en-US&gl=US&ceid=US:en"
    )
    items = []
    score = 0
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            root = ET.fromstring(resp.read())
        for item in root.findall("./channel/item")[:8]:
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            pub = item.findtext("pubDate", "")
            lower = title.lower()
            b_hits = sum(1 for w in BULLISH_WORDS if w in lower)
            s_hits = sum(1 for w in BEARISH_WORDS if w in lower)
            if b_hits > s_hits:
                tag = "BULLISH"
                score += 3
            elif s_hits > b_hits:
                tag = "BEARISH"
                score -= 3
            else:
                tag = "NEUTRAL"
            items.append({"title": title, "link": link, "date": pub, "sentiment": tag})
    except Exception as e:
        items.append({"title": f"Nepodařilo se načíst zprávy: {e}", "link": "#", "date": "", "sentiment": "NEUTRAL"})

    score = max(-15, min(15, score))
    return score, items


def get_risk_warnings(now: datetime):
    warnings = []
    wd = now.weekday()
    hr = now.hour

    if wd >= 5:
        warnings.append("Trh je o víkendu zavřený – zobrazují se poslední páteční data.")
    if wd == 3 and 12 <= hr <= 17:
        warnings.append("POZOR ČTVRTEK: Report zásob EIA v 16:30 SEČ – vysoké riziko prudkých výkyvů (whipsaw).")
    if wd == 4 and hr >= 19:
        warnings.append("Páteční večer: Zvýšené riziko víkendového gapu kvůli novým běhům modelů počasí (GFS/ECMWF).")
    if wd < 5 and 14 <= hr <= 16:
        warnings.append("Otevírá se hlavní US seance (14:30–16:00 SEČ) – očekávej vyšší volatilitu.")
    return warnings


def run_analysis():
    now = datetime.now(TZ)

    df_h1 = clean_df(yf.download("NG=F", period="60d", interval="1h", progress=False))
    df_d1 = clean_df(yf.download("NG=F", period="1y", interval="1d", progress=False))

    if df_h1.empty or df_d1.empty:
        raise RuntimeError("Nepodařilo se stáhnout cenová data NG=F z Yahoo Finance.")

    # Indikátory H1
    df_h1["EMA20"] = df_h1["Close"].ewm(span=20, adjust=False).mean()
    df_h1["EMA50"] = df_h1["Close"].ewm(span=50, adjust=False).mean()
    df_h1["EMA200"] = df_h1["Close"].ewm(span=200, adjust=False).mean()
    df_h1["RSI"] = calc_rsi(df_h1["Close"], 14)
    ema12 = df_h1["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df_h1["Close"].ewm(span=26, adjust=False).mean()
    df_h1["MACD"] = ema12 - ema26
    df_h1["MACD_SIGNAL"] = df_h1["MACD"].ewm(span=9, adjust=False).mean()
    df_h1["MACD_HIST"] = df_h1["MACD"] - df_h1["MACD_SIGNAL"]
    df_h1["ATR"] = calc_atr(df_h1, 14)

    # Indikátory D1
    df_d1["EMA50"] = df_d1["Close"].ewm(span=50, adjust=False).mean()

    last_h1 = df_h1.iloc[-1]
    prev_h1 = df_h1.iloc[-2]
    last_d1 = df_d1.iloc[-1]
    prev_d1 = df_d1.iloc[-2]

    price = float(last_h1["Close"])
    prev_close_d1 = float(prev_d1["Close"])
    change_pct = ((price - prev_close_d1) / prev_close_d1) * 100

    ema20 = float(last_h1["EMA20"])
    ema50 = float(last_h1["EMA50"])
    ema200 = float(last_h1["EMA200"])
    rsi = float(last_h1["RSI"])
    macd = float(last_h1["MACD"])
    macd_sig = float(last_h1["MACD_SIGNAL"])
    macd_hist = float(last_h1["MACD_HIST"])
    prev_hist = float(prev_h1["MACD_HIST"])
    atr = float(last_h1["ATR"])

    # Denní Pivoty
    p_h, p_l, p_c = float(prev_d1["High"]), float(prev_d1["Low"]), float(prev_d1["Close"])
    pivot = (p_h + p_l + p_c) / 3
    r1 = 2 * pivot - p_l
    s1 = 2 * pivot - p_h
    r2 = pivot + (p_h - p_l)
    s2 = pivot - (p_h - p_l)

    score = 0
    reasons = []

    # 1. Denní trend (D1)
    if price > float(last_d1["EMA50"]):
        score += 18
        reasons.append({"type": "bull", "text": f"Denní trend (D1) je býčí: cena ({price:.3f}) je nad D1 EMA50 ({float(last_d1['EMA50']):.3f})."})
    else:
        score -= 18
        reasons.append({"type": "bear", "text": f"Denní trend (D1) je medvědí: cena ({price:.3f}) je pod D1 EMA50 ({float(last_d1['EMA50']):.3f})."})

    # 2. Intradenní struktura (H1)
    if price > ema20 > ema50:
        score += 18
        reasons.append({"type": "bull", "text": f"Býčí intradenní struktura H1: cena > EMA20 ({ema20:.3f}) > EMA50 ({ema50:.3f})."})
    elif price < ema20 < ema50:
        score -= 18
        reasons.append({"type": "bear", "text": f"Medvědí intradenní struktura H1: cena < EMA20 ({ema20:.3f}) < EMA50 ({ema50:.3f})."})
    else:
        reasons.append({"type": "neutral", "text": "H1 klouzavé průměry EMA20/EMA50 jsou v konsolidaci."})

    # 3. Filtr H1 EMA200
    if price > ema200:
        score += 10
        reasons.append({"type": "bull", "text": f"Cena se drží nad H1 EMA200 ({ema200:.3f})."})
    else:
        score -= 10
        reasons.append({"type": "bear", "text": f"Cena je pod H1 EMA200 ({ema200:.3f})."})

    # 4. RSI (14)
    if 52 <= rsi <= 68:
        score += 12
        reasons.append({"type": "bull", "text": f"RSI (14) na H1 je {rsi:.1f} – zdravé nákupní momentum."})
    elif 32 <= rsi <= 48:
        score -= 12
        reasons.append({"type": "bear", "text": f"RSI (14) na H1 je {rsi:.1f} – převládá prodejní tlak."})
    elif rsi > 72:
        score -= 8
        reasons.append({"type": "bear", "text": f"RSI (14) je překoupené ({rsi:.1f}) – riziko korekce."})
    elif rsi < 28:
        score += 8
        reasons.append({"type": "bull", "text": f"RSI (14) je přeprodané ({rsi:.1f}) – šance na odraz nahoru."})
    else:
        reasons.append({"type": "neutral", "text": f"RSI (14) na H1 je neutrální ({rsi:.1f})."})

    # 5. MACD
    if macd > macd_sig and macd_hist > prev_hist:
        score += 12
        reasons.append({"type": "bull", "text": "MACD na H1 roste nad signální linií (nákupní momentum)."})
    elif macd < macd_sig and macd_hist < prev_hist:
        score -= 12
        reasons.append({"type": "bear", "text": "MACD na H1 klesá pod signální linií (prodejní momentum)."})
    else:
        reasons.append({"type": "neutral", "text": "MACD na H1 nedává jasné zrychlení."})

    # 6. Počasí (NatGasWeather)
    weather = fetch_weather_sources()
    score += weather["score_impact"]
    if weather["bias"] == "BULLISH":
        reasons.append({"type": "bull", "text": f"Počasí (NatGasWeather): chladnější/býčí signály (+{weather['score_impact']} b.)."})
    elif weather["bias"] == "BEARISH":
        reasons.append({"type": "bear", "text": f"Počasí (NatGasWeather): mírnější/medvědí výhled ({weather['score_impact']} b.)."})

    # 7. Zásoby EIA
    eia = fetch_eia_storage()
    score += eia["score_impact"]
    if eia["available"]:
        if eia["bias"] == "BULLISH":
            reasons.append({"type": "bull", "text": f"Zásoby EIA: {eia['vs_5y_text']} (+{eia['score_impact']} b.)."})
        elif eia["bias"] == "BEARISH":
            reasons.append({"type": "bear", "text": f"Zásoby EIA: {eia['vs_5y_text']} ({eia['score_impact']} b.)."})

    # 8. Zprávy
    news_score, news_items = fetch_news_sentiment()
    score += news_score
    score = max(-100, min(100, score))

    # Obchodní plán
    if score >= 30:
        signal = "LONG"
        action_cz = "OTEVŘÍT LONG (KOUPIT)"
        entry = price
        sl = price - 1.5 * atr
        tp1 = price + 2.0 * atr
        tp2 = price + 3.5 * atr
        plan_note = f"Hledej vstup poblíž aktuální ceny nebo při retestu EMA20 ({ema20:.3f})."
    elif score <= -30:
        signal = "SHORT"
        action_cz = "OTEVŘÍT SHORT (PRODAT)"
        entry = price
        sl = price + 1.5 * atr
        tp1 = price - 2.0 * atr
        tp2 = price - 3.5 * atr
        plan_note = f"Hledej vstup poblíž aktuální ceny nebo při odrazu od EMA20 ({ema20:.3f})."
    else:
        signal = "NEUTRAL"
        action_cz = "ČEKAT (NEOTEVÍRAT POZICI)"
        entry = price
        sl = s1
        tp1 = r1
        tp2 = r2
        plan_note = f"Trh nemá jasnou převahu. Čekej na průraz R1 ({r1:.3f}) pro Long nebo S1 ({s1:.3f}) pro Short."

    rrr = round(abs(tp1 - entry) / max(abs(entry - sl), 0.0001), 2) if signal != "NEUTRAL" else 0

    output = {
        "updated_at": now.strftime("%d.%m.%Y %H:%M SEČ"),
        "symbol": "NG=F (Henry Hub Futures)",
        "price": round(price, 3),
        "change_pct": round(change_pct, 2),
        "signal": signal,
        "action_cz": action_cz,
        "score": score,
        "plan": {
            "entry": round(entry, 3),
            "sl": round(sl, 3),
            "tp1": round(tp1, 3),
            "tp2": round(tp2, 3),
            "rrr": rrr,
            "note": plan_note,
        },
        "indicators": {
            "ema20": round(ema20, 3),
            "ema50": round(ema50, 3),
            "ema200": round(ema200, 3),
            "rsi": round(rsi, 1),
            "macd": round(macd, 4),
            "atr": round(atr, 3),
        },
        "levels": {
            "r2": round(r2, 3),
            "r1": round(r1, 3),
            "pivot": round(pivot, 3),
            "s1": round(s1, 3),
            "s2": round(s2, 3),
        },
        "weather": weather,
        "eia": eia,
        "reasons": reasons,
        "warnings": get_risk_warnings(now),
        "news": news_items,
    }

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Hotovo: {signal} (skóre {score}) @ {price:.3f} | EIA: {eia['available']}")


if __name__ == "__main__":
    run_analysis()
