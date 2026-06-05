from pathlib import Path
import streamlit as st
import pandas as pd
import anthropic
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field
from typing import List
import logging
import traceback
import base64
import threading

from amazon_scraper import MARKETPLACES, scrape_asin, scrape_idealo_ean

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_byte_length(text: str) -> int:
    return len(text.encode("utf-8")) if text else 0


# Optimal byte ranges (aligned with listing_agent)
TITLE_MIN, TITLE_MAX = 170, 200
BULLET_MIN, BULLET_MAX = 170, 200

# ── Pydantic Models ───────────────────────────────────────────────────────────

class CosmoEvaluation(BaseModel):
    model_config = {"extra": "ignore"}
    score_produktidentitaet: int = Field(default=0, description="0-2: Ist klar was das Produkt ist? Marke erkennbar?")
    score_eigenschaften: int = Field(default=0, description="0-2: Material, Komponenten, Qualität, Zertifikate genannt?")
    score_verwendungskontext: int = Field(default=0, description="0-2: Verwendungszweck, Umgebung, Aktivitäten beschrieben?")
    score_zielgruppe_anlass: int = Field(default=0, description="0-2: Zielgruppe, Anlass, Stil, Themen vorhanden?")
    score_format: int = Field(default=0, description="0-2: Titel-Struktur sinnvoll? Bullets mit Hook-Format? Byte-Längen im Zielbereich?")
    score_policy: int = Field(default=0, description="0-2: Amazon Policy konform? Keine subjektiven Claims ohne Beleg, keine Firmenhistorie, keine rhetorischen Fragen?")
    empfehlung: str = Field(default="", description="Zusammenfassende Handlungsempfehlung in 2-4 Sätzen: wichtigste Verbesserungen für Titel, Bullets, Backend und Gesamtstrategie.")


# ── COSMO Evaluation ──────────────────────────────────────────────────────────

COSMO_EVAL_PROMPT = """Du bist ein Amazon SEO-Experte und bewertest Produktlistings anhand der COSMO/RUFUS-Optimierungsregeln.

📦 PRODUKT-DATEN:
ASIN: {{asin}}
TITEL: {{title}}
BULLET 1: {{bullet1}}
BULLET 2: {{bullet2}}
BULLET 3: {{bullet3}}
BULLET 4: {{bullet4}}
BULLET 5: {{bullet5}}

📏 BYTE-ANALYSE (Umlaute = 2 Bytes):
Titel: {{title_bytes}} Bytes (Optimum: 170–200)
Bullet 1: {{b1_bytes}} Bytes | Bullet 2: {{b2_bytes}} Bytes | Bullet 3: {{b3_bytes}} Bytes
Bullet 4: {{b4_bytes}} Bytes | Bullet 5: {{b5_bytes}} Bytes
(Optimum je Bullet: 170–200 Bytes)

🎯 BEWERTUNGSSYSTEM — COSMO 15 Beziehungstypen (6 Dimensionen, je 0-2 Punkte, max. 12):

Bewerte jede Dimension mit:
- 0 = Fehlt komplett / schwerer Verstoß
- 1 = Teilweise vorhanden / leichte Mängel
- 2 = Gut bis sehr gut

**DIMENSION 1 – Produktidentität [0-2]**
Bezieht sich auf: is, has_brand
→ Ist klar erkennbar WAS das Produkt ist? Marke deutlich genannt?

**DIMENSION 2 – Eigenschaften & Material [0-2]**
Bezieht sich auf: has_property, made_of, has_component, has_quality, has_certification
→ Material, Komponenten, Qualitätsmerkmale, Zertifikate/Siegel vorhanden?

**DIMENSION 3 – Verwendungskontext [0-2]**
Bezieht sich auf: used_for, used_in, used_with, enables_activity
→ Verwendungszweck, Einsatzumgebung (z.B. Camping, Büro), Aktivitäten, Kombinationsprodukte beschrieben?

**DIMENSION 4 – Zielgruppe & Anlass [0-2]**
Bezieht sich auf: used_by, targets_audience, occasion, has_style, associated_with
→ Zielgruppe (z.B. Familien, Profis), Anlass (z.B. Alltag, Weihnachten), Stil, thematische Einordnung?

**DIMENSION 5 – Format, Struktur & Länge [0-2]**
→ Titel: Sinnvolle Struktur (Marke + Produktart + Eigenschaften)? Kein reines Keyword-Stuffing?
   Titellänge im Optimum (170–200 Bytes)? − 1 Punkt wenn deutlich zu kurz (<120B) oder zu lang (>230B)
→ Bullets: Beginnen mit informativen Hook in Großbuchstaben (HOOK: Satz)? Vollständige Sätze?
   Bullet-Längen überwiegend im Optimum (170–200 Bytes)?
- 2 = Struktur gut, Längen großteils im Zielbereich
- 1 = Leichte Abweichungen (Länge oder Format)
- 0 = Kein erkennbares Format und/oder Längen weit außerhalb

**DIMENSION 6 – Amazon Policy-Konformität [0-2]**
→ Prüfe Titel und Bullets auf Policy-Verstöße:
   ❌ Subjektive Behauptungen ohne Beleg (z.B. "HOCHWERTIG." oder "PREMIUM QUALITÄT." allein — ohne Material, Norm, Zertifikat)
   ❌ Firmengeschichte / Storytelling (z.B. "gegründet 1931", "seit X Jahren", "bekannt aus ...")
   ❌ Rhetorische Fragen an den Käufer (z.B. "Liebst du gutes Essen?")
   ❌ Lifestyle-Appelle / Call-to-Action statt Produktmerkmal (z.B. "Schmücke deinen Esstisch!", "Verwöhne deine Gäste!")
- 2 = Keine Policy-Verstöße erkennbar
- 1 = Leichte Grenzfälle (z.B. "hochwertig" mit schwacher Begründung)
- 0 = Klare Verstöße vorhanden

🔧 EMPFEHLUNG (empfehlung):
Fasse die wichtigsten Verbesserungsmaßnahmen in 2-4 prägnanten Sätzen zusammen.
Decke dabei die relevantesten Bereiche ab: Titel, Bullets, Backend-Felder und ggf. A+/Bilder.
Schreibe direkt und produktspezifisch — keine allgemeinen Floskeln.
Beispiel: "Der Titel sollte Material und Zielgruppe ergänzen. In den Bullets fehlen Angaben zur Spülmaschinenfestigkeit und Kompatibilität mit Induktionsherden. Die Backend-Felder Intended Use und Target Audience sollten befüllt werden. A+ Content würde die Conversion deutlich verbessern."

⚠️ Maximal 4 Sätze, präzise und umsetzbar.
"""


def _make_tool(name: str, description: str, model_class) -> dict:
    schema = model_class.model_json_schema()
    schema.pop("title", None)
    return {"name": name, "description": description, "input_schema": schema}


def evaluate_cosmo(client, model: str, asin: str, title: str, bullets: list) -> CosmoEvaluation:
    bullets_padded = (bullets + ["", "", "", "", ""])[:5]
    b = [bp or "" for bp in bullets_padded]
    prompt = (COSMO_EVAL_PROMPT
              .replace("{{asin}}", asin)
              .replace("{{title}}", title or "— kein Titel gecrawlt —")
              .replace("{{bullet1}}", b[0] or "— nicht vorhanden —")
              .replace("{{bullet2}}", b[1] or "— nicht vorhanden —")
              .replace("{{bullet3}}", b[2] or "— nicht vorhanden —")
              .replace("{{bullet4}}", b[3] or "— nicht vorhanden —")
              .replace("{{bullet5}}", b[4] or "— nicht vorhanden —")
              .replace("{{title_bytes}}", str(get_byte_length(title or "")))
              .replace("{{b1_bytes}}", str(get_byte_length(b[0])))
              .replace("{{b2_bytes}}", str(get_byte_length(b[1])))
              .replace("{{b3_bytes}}", str(get_byte_length(b[2])))
              .replace("{{b4_bytes}}", str(get_byte_length(b[3])))
              .replace("{{b5_bytes}}", str(get_byte_length(b[4]))))

    tool = _make_tool("cosmo_evaluation", "COSMO-Bewertung des Amazon-Listings", CosmoEvaluation)
    resp = client.messages.create(
        model=model,
        max_tokens=1500,
        system="Du bist ein Amazon SEO-Experte. Bewerte das Listing objektiv und streng anhand der COSMO-Kriterien.",
        tools=[tool],
        tool_choice={"type": "tool", "name": "cosmo_evaluation"},
        messages=[{"role": "user", "content": prompt}],
    )
    for block in resp.content:
        if block.type == "tool_use":
            return CosmoEvaluation.model_validate(block.input)
    raise ValueError("Keine Bewertung erhalten")


# ── Processing ────────────────────────────────────────────────────────────────

def process_item(asin: str, ean: str, base_url: str, client, model: str, fields: dict) -> dict:
    """
    asin: Amazon ASIN (10-stellig) oder leer → Amazon-Scraping
    ean:  EAN/GTIN (Ziffernfolge)  oder leer → Idealo-Abfrage
    """
    display_id = asin or ean
    logger.info(f"Verarbeite {'ASIN' if asin else 'EAN'}: {display_id}")

    result = {
        "display_id": display_id,
        "success": True,
        "title": "",
        "bullets": [],
        "gesamt_score": None,
    }
    if asin:
        result["ASIN"] = asin
    if ean:
        result["EAN"] = ean

    # ── Amazon-Scraping (nur wenn ASIN vorhanden) ─────────────────────────────
    if asin:
        scraped = scrape_asin(asin, base_url)
        if not scraped["success"]:
            return {"display_id": display_id, "ASIN": asin, "EAN": ean,
                    "success": False, "error": scraped["error"]}

        resolved_asin = scraped.get("resolved_asin", "") or asin
        bullets = (scraped["bullets"] + ["", "", "", "", ""])[:5]
        result["ASIN"] = resolved_asin
        result["title"] = scraped["title"]
        result["bullets"] = scraped["bullets"]

        if fields.get("title"):
            result["Live Titel"] = scraped["title"]
        if fields.get("bullets"):
            result["Live Bullet 1"] = bullets[0]
            result["Live Bullet 2"] = bullets[1]
            result["Live Bullet 3"] = bullets[2]
            result["Live Bullet 4"] = bullets[3]
            result["Live Bullet 5"] = bullets[4]
        if fields.get("desc"):
            result["Beschreibung"] = scraped["description"]
        if fields.get("ratings"):
            result["Anzahl Bewertungen"] = scraped["review_count"]
            result["Bewertungsschnitt"] = scraped["review_avg"]
        if fields.get("offer"):
            result["Preis"] = scraped["price"]
            result["Verkäufer"] = scraped["verkaeufer"]
        if fields.get("media"):
            result["Galeriebilder"] = scraped["image_count"]
            result["A+ Content"] = "Ja" if scraped["has_aplus"] else "Nein"

        if fields.get("cosmo"):
            try:
                evaluation = evaluate_cosmo(client, model, resolved_asin,
                                            scraped["title"], scraped["bullets"])
            except Exception as e:
                return {"display_id": display_id, "ASIN": asin, "EAN": ean,
                        "success": False, "error": f"Bewertungs-Fehler: {e}"}
            total = (evaluation.score_produktidentitaet + evaluation.score_eigenschaften +
                     evaluation.score_verwendungskontext + evaluation.score_zielgruppe_anlass +
                     evaluation.score_format + evaluation.score_policy)
            gesamt = max(1, min(12, total))
            result["evaluation"] = evaluation
            result["gesamt_score"] = gesamt
            result["COSMO Score"] = (
                f"{gesamt}/12 | "
                f"Produktidentität: {evaluation.score_produktidentitaet}/2 · "
                f"Eigenschaften: {evaluation.score_eigenschaften}/2 · "
                f"Verwendungskontext: {evaluation.score_verwendungskontext}/2 · "
                f"Zielgruppe & Anlass: {evaluation.score_zielgruppe_anlass}/2 · "
                f"Format & Länge: {evaluation.score_format}/2 · "
                f"Policy: {evaluation.score_policy}/2"
            )
            result["Empfehlungen"] = evaluation.empfehlung

    # ── Idealo-Abfrage (nur wenn EAN vorhanden und Checkbox aktiv) ────────────
    if fields.get("idealo") and ean:
        idealo = scrape_idealo_ean(ean)
        result["Idealo Preis"] = idealo["idealo_price"]
        result["Idealo Verkäufer"] = idealo["idealo_seller"]

    return result



# ── Streamlit App ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="ASIN Auditor by heyhome", page_icon="🔍", layout="wide")

_logo_path = Path(__file__).parent / "logo.png"
_logo_b64 = ""
if _logo_path.exists():
    _logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode()

st.markdown("""
<style>
.stButton > button, .stDownloadButton > button {
    background-color: #4d7b73 !important;
    color: #e7e137 !important;
    border: 1px solid #4d7b73 !important;
    font-weight: bold !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background-color: #3a6059 !important;
    color: #e7e137 !important;
}
</style>
""", unsafe_allow_html=True)


def main():
    if _logo_path.exists():
        st.sidebar.image(str(_logo_path), width=45)
    st.sidebar.title("ASIN Auditor")

    st.title("🔍 ASIN Auditor — COSMO Listing-Bewertung")
    st.caption("Crawlt Amazon-Produktseiten und bewertet Listings nach COSMO/RUFUS-Regeln (1–12): COSMO-Dimensionen, Byte-Längen & Amazon Policy")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("⚙️ Konfiguration")

        try:
            default_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            default_key = ""
        api_key = st.text_input("Anthropic API Key", value=default_key, type="password")

        model_options = {
            "Claude Sonnet 4.6 (Empfohlen)": "claude-sonnet-4-6",
            "Claude Haiku 4.5 (Schneller/Günstiger)": "claude-haiku-4-5-20251001",
            "Claude Opus 4.6 (Beste Qualität)": "claude-opus-4-6",
        }
        active_model = model_options[st.selectbox("Modell", list(model_options.keys()))]

        marketplace = st.selectbox("Marketplace", list(MARKETPLACES.keys()), index=0)
        base_url = MARKETPLACES[marketplace]

        parallel_workers = st.slider("Parallele Verarbeitung", 1, 3, 1,
                                     help="Max. 2 empfohlen um Amazon-Blocks zu vermeiden")

        st.subheader("📋 Output-Felder")
        field_title   = st.checkbox("Titel", value=True)
        field_bullets = st.checkbox("Bullet Points", value=True)
        field_desc    = st.checkbox("Beschreibung", value=True)
        field_ratings = st.checkbox("Bewertungen (Anzahl, Ø Sterne)", value=True)
        field_offer   = st.checkbox("Angebot (Preis, Verkäufer)", value=True)
        field_media   = st.checkbox("Präsentation (Galeriebilder, A+)", value=True)
        field_cosmo   = st.checkbox("COSMO-Analyse (Score + Empfehlungen)", value=True)
        field_idealo  = st.checkbox("Idealo Preisvergleich (nur EAN)", value=False,
                                    help="Sucht günstigsten Preis + Anbieter auf idealo.de. Erfordert: pip install playwright && playwright install chromium")

        fields = {
            "title":   field_title,
            "bullets": field_bullets,
            "desc":    field_desc,
            "ratings": field_ratings,
            "offer":   field_offer,
            "media":   field_media,
            "cosmo":   field_cosmo,
            "idealo":  field_idealo,
        }

    # ── Input: Freitext + File Upload ─────────────────────────────────────────
    st.subheader("📂 ASINs oder EANs eingeben")

    def _normalize_cols(frame):
        frame.columns = frame.columns.str.strip().str.lstrip('\ufeff')
        col_map = {c: c.upper() for c in frame.columns if c.upper() in ('ASIN', 'EAN')}
        if col_map:
            frame.rename(columns=col_map, inplace=True)
        return frame

    def _parse_freetext(text: str) -> list:
        result = []
        for line in text.splitlines():
            v = line.strip().upper()
            if not v:
                continue
            if not v.isdigit() and 8 <= len(v) <= 12:
                result.append({"asin": v, "ean": ""})
            elif v.isdigit() and 8 <= len(v) <= 14:
                result.append({"asin": "", "ean": v})
        return result

    with st.container(border=True):
        asin_text_raw = st.text_area(
            "ASINs oder EANs (eine pro Zeile)",
            placeholder="B09XMKL2RD\nB09XMKL3QS\nB09XMKL4PT",
            height=140,
        )
        st.caption("Oder CSV/XLSX mit ASIN- bzw. EAN-Spalte hochladen:")
        uploaded_file = st.file_uploader(
            "CSV oder XLSX hochladen",
            type=["csv", "xlsx"],
            label_visibility="collapsed",
        )

    items_from_text = _parse_freetext(asin_text_raw) if asin_text_raw.strip() else []

    items_from_file = []
    file_error = None
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                raw_bytes = uploaded_file.read()
                df = None
                for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
                    try:
                        df = _normalize_cols(pd.read_csv(io.BytesIO(raw_bytes), encoding=enc))
                        break
                    except (UnicodeDecodeError, ValueError):
                        continue
                if df is None:
                    raise ValueError("CSV-Encoding konnte nicht erkannt werden (versucht: utf-8, cp1252, latin-1)")
            else:
                xl = pd.ExcelFile(uploaded_file)
                df = None
                for sheet in xl.sheet_names:
                    candidate = _normalize_cols(xl.parse(sheet))
                    if "ASIN" in candidate.columns or "EAN" in candidate.columns:
                        df = candidate
                        break
                if df is None:
                    df = _normalize_cols(xl.parse(xl.sheet_names[0]))

            if "ASIN" not in df.columns and "EAN" not in df.columns:
                file_error = "❌ Keine Spalte 'ASIN' oder 'EAN' in der Datei gefunden."
            elif "ASIN" in df.columns and "EAN" in df.columns:
                for _, row in df.iterrows():
                    asin = str(row.get("ASIN", "") or "").strip().upper()
                    ean  = str(row.get("EAN",  "") or "").strip().upper()
                    asin = asin if (asin and not asin.isdigit() and 8 <= len(asin) <= 12) else ""
                    ean  = ean  if (ean  and ean.isdigit()      and 8 <= len(ean)  <= 14) else ""
                    if asin or ean:
                        items_from_file.append({"asin": asin, "ean": ean})
            elif "ASIN" in df.columns:
                for v in df["ASIN"].dropna().astype(str).str.strip().str.upper().unique():
                    if not v.isdigit() and 8 <= len(v) <= 12:
                        items_from_file.append({"asin": v, "ean": ""})
            else:
                for v in df["EAN"].dropna().astype(str).str.strip().str.upper().unique():
                    if v.isdigit() and 8 <= len(v) <= 14:
                        items_from_file.append({"asin": "", "ean": v})
        except Exception as e:
            file_error = f"Fehler beim Laden der Datei: {e}"

    if file_error:
        st.error(file_error)

    # Zusammenfuehren, Duplikate entfernen
    seen_keys: set = set()
    items = []
    for item in items_from_text + items_from_file:
        key = item["asin"] or item["ean"]
        if key not in seen_keys:
            seen_keys.add(key)
            items.append(item)

    if not items:
        if not asin_text_raw.strip() and not uploaded_file:
            st.info("ASINs oben eintragen oder eine Datei hochladen um zu starten.")
        return

    has_asins = any(i["asin"] for i in items)
    has_eans  = any(i["ean"]  for i in items)
    if has_asins and has_eans:
        mode = "both"
    elif has_asins:
        mode = "asin"
    else:
        mode = "ean"

    label     = {"asin": "ASINs",    "ean": "EANs",             "both": "Produkte"}[mode]
    mode_info = {
        "asin": "Nur ASINs erkannt → Amazon-Scraping",
        "ean":  "Nur EANs erkannt → Idealo-Abfrage (kein Amazon-Scraping)",
        "both": "ASINs + EANs erkannt → Amazon-Scraping (ASIN) + optionale Idealo-Abfrage (EAN)",
    }[mode]
    st.success(f"**{len(items)} {label}** geladen. {mode_info}")

    if len(items) == 1:
        num_to_scrape = st.number_input(
            f"Anzahl {label} analysieren",
            min_value=1,
            max_value=1,
            value=1,
            help=f"Wähle wie viele {label} (von oben) analysiert werden sollen.",
        )
    else:
        num_to_scrape = st.slider(
            f"Anzahl {label} analysieren",
            min_value=1,
            max_value=len(items),
            value=min(10, len(items)),
            help=f"Wähle wie viele {label} (von oben) analysiert werden sollen.",
        )
    items = items[:num_to_scrape]

    st.info(f"**{len(items)} {label}** werden auf **{marketplace}** analysiert.")

    if not fields.get("cosmo"):
        st.info("ℹ️ COSMO-Analyse deaktiviert — kein API Key erforderlich.")

    if not st.button(f"🚀 {len(items)} {label} analysieren",
                     disabled=(fields.get("cosmo") and not api_key)):
        if fields.get("cosmo") and not api_key:
            st.warning("Bitte API Key in der Sidebar eingeben (für COSMO-Analyse benötigt).")
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    client = anthropic.Anthropic(api_key=api_key or "dummy", max_retries=6)
    all_results = []
    results_lock = threading.Lock()
    completed = [0]

    progress = st.progress(0)
    status_text = st.empty()
    status_text.text(f"🚀 Starte Analyse ({parallel_workers} parallel)...")

    def on_result(r):
        with results_lock:
            all_results.append(r)
            completed[0] += 1
            progress.progress(completed[0] / len(items))
            status_text.text(f"✅ {completed[0]}/{len(items)} verarbeitet...")

        display_id = r.get("display_id", r.get("ASIN", r.get("EAN", "?")))
        if r["success"]:
            score = r.get("gesamt_score")
            score_label = f"{score}/12" if score is not None else "—"
            icon = "🟢" if isinstance(score, int) and score >= 9 else "🟡" if isinstance(score, int) and score >= 6 else ("🔴" if isinstance(score, int) else "⚪")
            title_preview = r.get("title", "")[:60]
            with st.expander(f"{icon} {display_id} — {title_preview}...{(' (Score: ' + score_label + ')') if score is not None else ''}"):
                cols = st.columns(4)
                if fields.get("cosmo"):
                    cols[0].metric("COSMO Score", score_label)
                if fields.get("ratings"):
                    cols[1].metric("Bewertungen", r.get("Anzahl Bewertungen", "—"))
                    cols[2].metric("Ø Sterne", r.get("Bewertungsschnitt", "—"))
                if fields.get("media"):
                    cols[3].metric("A+ Content", r.get("A+ Content", "—"))
                if fields.get("cosmo"):
                    st.write("**💡 Empfehlungen:**", r.get("Empfehlungen", ""))
        else:
            st.error(f"❌ {display_id}: {r.get('error', 'Unbekannter Fehler')}")

    try:
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {
                executor.submit(process_item, item["asin"], item["ean"], base_url, client, active_model, fields): item
                for item in items
            }
            for future in as_completed(futures):
                item = futures[future]
                item_id = item["asin"] or item["ean"]
                try:
                    on_result(future.result())
                except Exception as e:
                    logger.error(f"on_result error for {item_id}: {traceback.format_exc()}")
                    on_result({"display_id": item_id, "ASIN": item["asin"], "EAN": item["ean"],
                               "success": False, "error": str(e)})
    except Exception as e:
        logger.error(f"Executor error: {traceback.format_exc()}")
        st.error(f"Fehler bei der Analyse: {e}")
        return

    success_count = sum(1 for r in all_results if r["success"])
    fail_count = len(all_results) - success_count
    status_text.text(f"✅ Fertig! {success_count} erfolgreich, {fail_count} fehlgeschlagen.")

    # ── Export ────────────────────────────────────────────────────────────────
    success_results = [r for r in all_results if r["success"]]
    if not success_results:
        st.warning("Keine Ergebnisse zum Exportieren.")
        return

    has_ean_col  = any(r.get("EAN")  for r in success_results)
    has_asin_col = any(r.get("ASIN") for r in success_results)
    if has_ean_col and has_asin_col:
        first_cols = ["EAN", "ASIN"]
    elif has_ean_col:
        first_cols = ["EAN"]
    else:
        first_cols = ["ASIN"]
    optional_cols = []
    if fields.get("title"):
        optional_cols += ["Live Titel"]
    if fields.get("bullets"):
        optional_cols += ["Live Bullet 1", "Live Bullet 2", "Live Bullet 3",
                          "Live Bullet 4", "Live Bullet 5"]
    if fields.get("desc"):
        optional_cols += ["Beschreibung"]
    if fields.get("ratings"):
        optional_cols += ["Anzahl Bewertungen", "Bewertungsschnitt"]
    if fields.get("offer"):
        optional_cols += ["Preis", "Verkäufer"]
    if fields.get("media"):
        optional_cols += ["Galeriebilder", "A+ Content"]
    if fields.get("cosmo"):
        optional_cols += ["COSMO Score", "Empfehlungen"]
    if fields.get("idealo"):
        optional_cols += ["Idealo Preis", "Idealo Verkäufer"]

    export_cols = first_cols + optional_cols
    all_keys = set().union(*(r.keys() for r in success_results))
    export_cols = [c for c in export_cols if c in all_keys]
    logger.info(f"Export-Spalten: {export_cols}")

    try:
        export_data = [{k: v for k, v in r.items() if k in export_cols} for r in success_results]
        df_out = pd.DataFrame(export_data, columns=export_cols)
    except Exception as e:
        logger.error(f"DataFrame-Fehler: {traceback.format_exc()}")
        st.error(f"Fehler beim Erstellen der Tabelle: {e}")
        return

    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_out.to_excel(writer, index=False, sheet_name="ASIN Audit")
            ws = writer.sheets["ASIN Audit"]
            for col_cells in ws.columns:
                length = max((len(str(cell.value or "")) for cell in col_cells), default=10)
                ws.column_dimensions[col_cells[0].column_letter].width = min(length + 2, 60)
    except Exception as e:
        logger.error(f"Excel-Fehler: {traceback.format_exc()}")
        st.error(f"Fehler beim Erstellen der Excel-Datei: {e}")
        return

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"asin_audit_{marketplace.replace('.', '_')}_{timestamp}.xlsx"

    st.download_button(
        "📥 Excel-Report herunterladen",
        data=output.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()
