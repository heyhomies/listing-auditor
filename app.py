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

from amazon_scraper import MARKETPLACES, scrape_asin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Pydantic Models ───────────────────────────────────────────────────────────

class CosmoEvaluation(BaseModel):
    model_config = {"extra": "ignore"}
    score_produktidentitaet: int = Field(default=0, description="0-2: Ist klar was das Produkt ist? Marke erkennbar?")
    score_eigenschaften: int = Field(default=0, description="0-2: Material, Komponenten, Qualität, Zertifikate genannt?")
    score_verwendungskontext: int = Field(default=0, description="0-2: Verwendungszweck, Umgebung, Aktivitäten beschrieben?")
    score_zielgruppe_anlass: int = Field(default=0, description="0-2: Zielgruppe, Anlass, Stil, Themen vorhanden?")
    score_format: int = Field(default=0, description="0-2: Titel-Struktur sinnvoll? Bullets mit Hook-Format?")
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

🎯 BEWERTUNGSSYSTEM — COSMO 15 Beziehungstypen (5 Dimensionen, je 0-2 Punkte):

Bewerte jede Dimension mit:
- 0 = Fehlt komplett
- 1 = Teilweise vorhanden
- 2 = Gut bis sehr gut abgedeckt

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

**DIMENSION 5 – Format & Struktur [0-2]**
→ Titel: Sinnvolle Struktur (Marke + Produktart + Eigenschaften)? Kein reines Keyword-Stuffing?
→ Bullets: Beginnen mit einem informativen Hook? Vollständige nutzenorientierte Sätze? Kein reines Aufzählen?

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
    prompt = (COSMO_EVAL_PROMPT
              .replace("{{asin}}", asin)
              .replace("{{title}}", title or "— kein Titel gecrawlt —")
              .replace("{{bullet1}}", bullets_padded[0] or "— nicht vorhanden —")
              .replace("{{bullet2}}", bullets_padded[1] or "— nicht vorhanden —")
              .replace("{{bullet3}}", bullets_padded[2] or "— nicht vorhanden —")
              .replace("{{bullet4}}", bullets_padded[3] or "— nicht vorhanden —")
              .replace("{{bullet5}}", bullets_padded[4] or "— nicht vorhanden —"))

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

def process_asin(asin: str, base_url: str, client, model: str, fields: dict) -> dict:
    asin = asin.strip().upper()
    is_ean = asin.isdigit()
    logger.info(f"Verarbeite {'EAN' if is_ean else 'ASIN'}: {asin}")

    scraped = scrape_asin(asin, base_url)

    if not scraped["success"]:
        return {"asin": asin, "is_ean": is_ean, "success": False, "error": scraped["error"]}

    resolved_asin = scraped.get("resolved_asin", "") or asin
    bullets = (scraped["bullets"] + ["", "", "", "", ""])[:5]

    result = {
        "asin": asin,
        "is_ean": is_ean,
        "success": True,
        "title": scraped["title"],
        "bullets": scraped["bullets"],
        "gesamt_score": None,
        "ASIN": resolved_asin,
    }
    if is_ean:
        result["EAN"] = asin

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
            evaluation = evaluate_cosmo(client, model, resolved_asin, scraped["title"], scraped["bullets"])
        except Exception as e:
            return {"asin": asin, "is_ean": is_ean, "success": False,
                    "error": f"Bewertungs-Fehler: {e}", "scraped": scraped}
        total = (evaluation.score_produktidentitaet +
                 evaluation.score_eigenschaften +
                 evaluation.score_verwendungskontext +
                 evaluation.score_zielgruppe_anlass +
                 evaluation.score_format)
        gesamt = max(1, min(10, total))
        result["evaluation"] = evaluation
        result["gesamt_score"] = gesamt
        result["COSMO Score"] = (
            f"{gesamt}/10 | "
            f"Produktidentität: {evaluation.score_produktidentitaet}/2 · "
            f"Eigenschaften: {evaluation.score_eigenschaften}/2 · "
            f"Verwendungskontext: {evaluation.score_verwendungskontext}/2 · "
            f"Zielgruppe & Anlass: {evaluation.score_zielgruppe_anlass}/2 · "
            f"Format: {evaluation.score_format}/2"
        )
        result["Empfehlungen"] = evaluation.empfehlung

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
    st.caption("Crawlt Amazon-Produktseiten und bewertet Listings nach COSMO/RUFUS-Regeln (1–10)")

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

        fields = {
            "title":   field_title,
            "bullets": field_bullets,
            "desc":    field_desc,
            "ratings": field_ratings,
            "offer":   field_offer,
            "media":   field_media,
            "cosmo":   field_cosmo,
        }

    # ── File Upload ───────────────────────────────────────────────────────────
    st.subheader("📂 ASINs oder EANs hochladen")
    st.caption("Die Datei muss eine Spalte mit dem Header **ASIN** oder **EAN** enthalten.")
    uploaded_file = st.file_uploader("CSV oder XLSX hochladen", type=["csv", "xlsx"])

    if not uploaded_file:
        st.info("Lade eine Datei mit einer Spalte 'ASIN' oder 'EAN' hoch um zu starten.")
        return

    def _normalize_cols(frame):
        frame.columns = frame.columns.str.strip().str.lstrip('\ufeff')
        col_map = {c: c.upper() for c in frame.columns if c.upper() in ('ASIN', 'EAN')}
        if col_map:
            frame.rename(columns=col_map, inplace=True)
        return frame

    try:
        if uploaded_file.name.endswith(".csv"):
            df = _normalize_cols(pd.read_csv(uploaded_file))
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
    except Exception as e:
        st.error(f"Fehler beim Laden der Datei: {e}")
        return

    # ── EAN → ASIN conversion ─────────────────────────────────────────────────
    if "ASIN" not in df.columns and "EAN" not in df.columns:
        st.error("❌ Keine Spalte 'ASIN' oder 'EAN' gefunden.")
        return

    col = "ASIN" if "ASIN" in df.columns else "EAN"
    raw = df[col].dropna().astype(str).str.strip().str.upper().unique().tolist()

    asins = [v for v in raw if not v.isdigit() and 8 <= len(v) <= 12]
    eans  = [v for v in raw if v.isdigit() and 8 <= len(v) <= 14]
    identifiers = asins + eans

    if not identifiers:
        st.warning(f"Keine gültigen ASINs oder EANs in der Spalte '{col}' gefunden.")
        return

    label = "ASINs" if not eans else ("EANs" if not asins else "ASINs/EANs")

    st.success(f"**{len(identifiers)} {label}** geladen.")

    num_to_scrape = st.slider(
        f"Anzahl {label} scrapen",
        min_value=1,
        max_value=len(identifiers),
        value=min(10, len(identifiers)),
        help=f"Wähle wie viele {label} (aus der Liste, von oben) analysiert werden sollen.",
    )
    identifiers = identifiers[:num_to_scrape]

    st.info(f"**{len(identifiers)} {label}** werden auf **{marketplace}** gecrawlt.")

    if not fields.get("cosmo"):
        st.info("ℹ️ COSMO-Analyse deaktiviert — kein API Key erforderlich.")

    if not st.button(f"🚀 {len(identifiers)} {label} analysieren",
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
            progress.progress(completed[0] / len(identifiers))
            status_text.text(f"✅ {completed[0]}/{len(identifiers)} verarbeitet...")

        if r["success"]:
            score = r.get("gesamt_score")
            score_label = f"{score}/10" if score is not None else "—"
            icon = "🟢" if isinstance(score, int) and score >= 7 else "🟡" if isinstance(score, int) and score >= 4 else ("🔴" if isinstance(score, int) else "⚪")
            title_preview = r.get("title", "")[:60]
            with st.expander(f"{icon} {r['asin']} — {title_preview}...{(' (Score: ' + score_label + ')') if score is not None else ''}"):
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
            st.error(f"❌ {r['asin']}: {r.get('error', 'Unbekannter Fehler')}")

    try:
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {
                executor.submit(process_asin, ident, base_url, client, active_model, fields): ident
                for ident in identifiers
            }
            for future in as_completed(futures):
                ident = futures[future]
                try:
                    on_result(future.result())
                except Exception as e:
                    logger.error(f"on_result error for {ident}: {traceback.format_exc()}")
                    on_result({"asin": ident, "success": False, "error": str(e)})
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

    has_ean_input = any(r.get("is_ean") for r in success_results)
    first_cols = ["EAN", "ASIN"] if has_ean_input else ["ASIN"]
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
