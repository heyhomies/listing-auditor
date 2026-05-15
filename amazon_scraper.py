"""
Shared Amazon scraper module for listing_auditor and listing_agent.
"""

import requests
from bs4 import BeautifulSoup
import time
import random
import threading
import re
import json
import logging

logger = logging.getLogger(__name__)

# ── Marketplaces ──────────────────────────────────────────────────────────────

MARKETPLACES = {
    "amazon.de": "https://www.amazon.de/dp/",
    "amazon.com": "https://www.amazon.com/dp/",
    "amazon.co.uk": "https://www.amazon.co.uk/dp/",
    "amazon.fr": "https://www.amazon.fr/dp/",
    "amazon.it": "https://www.amazon.it/dp/",
    "amazon.es": "https://www.amazon.es/dp/",
}

# Accept-Language per marketplace — determines the language Amazon returns
MARKETPLACE_LANGUAGES = {
    "amazon.de":     "de-DE,de;q=0.9,en-US;q=0.8",
    "amazon.com":    "en-US,en;q=0.9",
    "amazon.co.uk":  "en-GB,en;q=0.9,en-US;q=0.8",
    "amazon.fr":     "fr-FR,fr;q=0.9,en;q=0.8",
    "amazon.it":     "it-IT,it;q=0.9,en;q=0.8",
    "amazon.es":     "es-ES,es;q=0.9,en;q=0.8",
}


def _accept_language(base_url: str) -> str:
    for domain, lang in MARKETPLACE_LANGUAGES.items():
        if domain in base_url:
            return lang
    return "en-US,en;q=0.9"

PROFILES = [
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
    },
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "sec-ch-ua": '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
    },
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
        "sec-ch-ua": '"Safari";v="17"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
    },
]

BASE_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

# One session per thread per domain — pre-warmed with homepage cookies
_session_local = threading.local()


def _session_key(base_url: str) -> str:
    """Thread-local attribute name keyed by domain."""
    domain = re.sub(r"[^a-z0-9]", "_", base_url.split("//")[-1].split("/")[0])
    return f"session_{domain}"


def _get_session(base_url: str) -> requests.Session:
    """Return a thread-local session for this marketplace, pre-warmed with homepage cookies."""
    key = _session_key(base_url)
    if not hasattr(_session_local, key):
        profile = random.choice(PROFILES)
        session = requests.Session()
        session.headers.update({
            **BASE_HEADERS,
            **profile,
            "Accept-Language": _accept_language(base_url),
        })
        homepage = base_url.replace("/dp/", "")
        try:
            session.get(homepage, timeout=15)
            time.sleep(random.uniform(0.8, 1.5))
        except Exception:
            pass
        setattr(_session_local, key, session)
    return getattr(_session_local, key)


def _reset_session(base_url: str) -> None:
    """Delete the cached session for this marketplace so it gets rebuilt on next request."""
    key = _session_key(base_url)
    if hasattr(_session_local, key):
        delattr(_session_local, key)


def _is_blocked(html: str) -> bool:
    indicators = [
        "captcha",
        "robot check",
        "validatecaptcha",
        "type the characters",
        "enter the characters",
        "not a robot",
        "automated access",
    ]
    lower = html.lower()
    return any(ind in lower for ind in indicators)


def _resolve_ean_to_product_url(ean: str, base_url: str, session) -> tuple:
    """
    Fetch Amazon search results for an EAN and return (product_url, error).
    Picks the first non-sponsored result. Returns (None, error_msg) if nothing found.
    """
    domain = base_url.rstrip("/").rsplit("/", 1)[0]
    search_url = f"{domain}/s?k={ean}"
    session.headers.update({"Referer": domain})
    time.sleep(random.uniform(2.0, 4.0))
    resp = session.get(search_url, timeout=20)

    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code} auf Suchseite"

    if _is_blocked(resp.text):
        _reset_session(base_url)
        return None, "Bot-Erkennung auf Suchseite"

    soup = BeautifulSoup(resp.content, "html.parser")

    page_text = soup.get_text(" ", strip=True).lower()
    no_results_phrases = [
        "keine ergebnisse für",
        "no results for",
        "hat keine ergebnisse",
        "did not match any products",
        "stimmt mit keinem produkt",
    ]
    if any(phrase in page_text for phrase in no_results_phrases):
        return None, "Kein Produkt für diese EAN verfügbar"

    for item in soup.select("[data-component-type='s-search-result']"):
        asin_val = item.get("data-asin", "").strip()
        if not asin_val:
            continue

        # Only use specific, reliable sponsored indicators (no broad text scan)
        is_sponsored = bool(
            item.select_one(".puis-sponsored-label-text")
            or item.select_one("[aria-label*='Gesponsert']")
            or item.select_one("[aria-label*='Sponsored']")
        )

        if not is_sponsored:
            return f"{domain}/dp/{asin_val}", ""

    return None, "EAN nicht auf Amazon gelistet"


def scrape_asin_once(asin: str, base_url: str) -> dict:
    result = {
        "asin": asin,
        "resolved_asin": "",
        "url": "",
        "success": False,
        "error": None,
        "title": "",
        "bullets": [],
        "description": "",
        "price": "",
        "verkaeufer": "",
        "review_count": "",
        "review_avg": "",
        "image_count": 0,
        "has_aplus": False,
    }

    try:
        session = _get_session(base_url)
        session.headers.update({"Referer": base_url.replace("/dp/", "")})

        if asin.isdigit():
            product_url, err = _resolve_ean_to_product_url(asin, base_url, session)
            if not product_url:
                result["error"] = err
                return result
            url = product_url
        else:
            url = f"{base_url}{asin}"

        result["url"] = url
        m = re.search(r"/dp/([A-Z0-9]{10})", url)
        if m:
            result["resolved_asin"] = m.group(1)
        time.sleep(random.uniform(2.0, 4.0))
        resp = session.get(url, timeout=20)

        if resp.status_code == 503 or resp.status_code == 500:
            _reset_session(base_url)
            result["error"] = f"HTTP {resp.status_code} (Bot-Block) — Session zurückgesetzt"
            return result

        if resp.status_code != 200:
            result["error"] = f"HTTP {resp.status_code}"
            return result

        if _is_blocked(resp.text):
            _reset_session(base_url)
            result["error"] = "Bot-Erkennung — Session zurückgesetzt"
            return result

        soup = BeautifulSoup(resp.content, "html.parser")

        title_el = soup.select_one("#productTitle")
        if title_el:
            result["title"] = title_el.get_text(strip=True)

        bullets = []
        for li in soup.select("#feature-bullets ul li span.a-list-item"):
            text = li.get_text(strip=True)
            if text and len(text) > 5 and not text.startswith("Stelle sicher") and not text.startswith("Make sure"):
                bullets.append(text)
        result["bullets"] = bullets[:5]

        desc_el = soup.select_one("#productDescription")
        if desc_el:
            for hidden in desc_el.select("[style*='display:none'], [style*='display: none']"):
                hidden.decompose()
            result["description"] = desc_el.get_text(" ", strip=True)

        # ── Availability state detection ─────────────────────────────────────────
        page_text_lower = soup.get_text().lower()
        availability_el = soup.select_one("#availability")
        av_text = (availability_el.get_text(strip=True) if availability_el else "").lower()

        is_unavailable = (
            "derzeit nicht verfügbar" in av_text
            or "currently unavailable" in av_text
            or ("derzeit nicht verfügbar" in page_text_lower and not availability_el)
        )
        is_suppressed = (not is_unavailable) and (
            "höherer preis als üblich" in page_text_lower
            or "higher price than usual" in page_text_lower
        )

        # ── Price extraction ──────────────────────────────────────────────────
        if not is_unavailable:
            buybox_price_selectors = [
                "#corePrice_feature_div .a-offscreen",
                "#desktop_buybox .a-price .a-offscreen",
                "#buybox .a-price .a-offscreen",
                "#priceblock_ourprice",
                "#priceblock_dealprice",
                "#apex_price_block .a-price .a-offscreen",
                "#apex_desktop .a-price .a-offscreen",
            ]
            for selector in buybox_price_selectors:
                price_el = soup.select_one(selector)
                if price_el:
                    val = price_el.get_text(strip=True)
                    if val:
                        result["price"] = val
                        break
            if not result["price"]:
                excluded = ("lpo_feature_div", "sims-", "sp_detail", "discovery", "inspiration")
                for price_el in soup.select(".a-price .a-offscreen"):
                    parent_ids = " ".join(a.get("id", "") for a in price_el.parents)
                    if not any(ex in parent_ids for ex in excluded):
                        val = price_el.get_text(strip=True)
                        if val and ("€" in val or "$" in val or "£" in val or any(c.isdigit() for c in val)):
                            result["price"] = val
                            break

            # For suppressed buybox: fetch mobile page to get unqualified offer price
            if is_suppressed and not result["price"]:
                try:
                    mobile_ua = (
                        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
                        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"
                    )
                    mobile_resp = session.get(url, timeout=15, headers={"User-Agent": mobile_ua})
                    if mobile_resp.status_code == 200:
                        m_soup = BeautifulSoup(mobile_resp.content, "html.parser")
                        for sel in [
                            "#unqualifiedBuyBox .a-price .a-offscreen",
                            "#unqualified_feature_div .a-price .a-offscreen",
                        ]:
                            unq = m_soup.select_one(sel)
                            if unq:
                                result["price"] = unq.get_text(strip=True)
                                break
                except Exception:
                    pass

        def _extract_seller(soup) -> str:
            seller_fragments = ("verk", "versend", "seller", "sold by", "verkauft")

            def _is_seller_label(text: str) -> bool:
                return any(frag in text.lower() for frag in seller_fragments)

            for section in soup.select('[data-csa-c-content-id="desktop-merchant-info"]'):
                wrap = section.select_one(".odf-popover-overflow-wrap")
                if wrap:
                    val = wrap.get_text(strip=True)
                    if val:
                        return val

            all_cells = soup.select(".tabular-buybox-text")
            for i, cell in enumerate(all_cells):
                if _is_seller_label(cell.get_text(strip=True)) and i + 1 < len(all_cells):
                    val = all_cells[i + 1].get_text(strip=True)
                    if val and not _is_seller_label(val):
                        return val

            merchant_el = soup.select_one("#merchant-info")
            if merchant_el:
                val = merchant_el.get_text(strip=True)
                if val:
                    return val

            seller_el = soup.select_one("#sellerProfileTriggerId")
            if seller_el:
                return seller_el.get_text(strip=True)

            return ""

        # ── Seller / availability override ────────────────────────────────────
        if is_unavailable:
            result["price"] = ""
            result["verkaeufer"] = "Derzeit nicht verfügbar"
        elif is_suppressed:
            result["verkaeufer"] = "nicht hervorgehoben"
        else:
            result["verkaeufer"] = _extract_seller(soup)

        review_count_el = soup.select_one("#acrCustomerReviewText")
        if review_count_el:
            text = review_count_el.get_text(strip=True)
            result["review_count"] = text.replace(" Bewertungen", "").replace(" ratings", "").replace(" Sternebewertungen", "").strip()

        for selector in ["#acrPopoverLink .a-size-base.a-color-base", "span[data-hook='rating-out-of-text']", "#averageCustomerReviews .a-icon-alt", "#acrPopoverLink span.a-size-base"]:
            avg_el = soup.select_one(selector)
            if avg_el:
                text = avg_el.get_text(strip=True).replace(" von 5 Sternen", "").replace(" out of 5 stars", "").strip()
                if text:
                    result["review_avg"] = text
                    break

        image_count = 0
        for li in soup.select("#altImages li.imageThumbnail"):
            cls = " ".join(li.get("class", []))
            if "videoThumbnail" in cls:
                continue
            if "overlayRestOfImages" in cls:
                more_el = li.select_one(".textMoreImages")
                if more_el:
                    try:
                        image_count += int(re.sub(r"[^\d]", "", more_el.get_text()))
                    except ValueError:
                        pass
            else:
                image_count += 1
        result["image_count"] = image_count

        aplus = (soup.select_one("#aplus") or
                 soup.select_one("#aplus3p_feature_div") or
                 soup.select_one("#aplusBrandStory_feature_div") or
                 soup.select_one("#aplus_feature_div"))
        result["has_aplus"] = aplus is not None

        result["success"] = True

    except requests.RequestException as e:
        result["error"] = f"Verbindungsfehler: {str(e)}"
    except Exception as e:
        result["error"] = f"Parse-Fehler: {str(e)}"

    return result


def scrape_asin(asin: str, base_url: str) -> dict:
    """Scrape with up to 3 retries on bot-block responses."""
    for attempt in range(3):
        result = scrape_asin_once(asin, base_url)
        if result["success"]:
            return result
        err = result.get("error", "")
        # Retry only on bot-block signals, not on genuine errors
        if "Bot" in err or "500" in err or "503" in err or "zurückgesetzt" in err:
            wait = (attempt + 1) * random.uniform(8.0, 15.0)
            logger.warning(f"Bot-Block für {asin} (Versuch {attempt+1}), warte {wait:.0f}s …")
            time.sleep(wait)
        else:
            return result
    return result


# ── Idealo Scraper (Playwright) ───────────────────────────────────────────────
# Requires: pip install playwright && playwright install chromium

_pw_instance = None
_idealo_browser = None


def _get_idealo_browser():
    """Return a persistent Chromium browser instance (created once per process)."""
    global _pw_instance, _idealo_browser
    if _idealo_browser is not None:
        try:
            if _idealo_browser.is_connected():
                return _idealo_browser, None
        except Exception:
            pass
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None, "playwright nicht installiert — bitte ausführen: pip install playwright && playwright install chromium"
    try:
        if _pw_instance is not None:
            try:
                _pw_instance.stop()
            except Exception:
                pass
        _pw_instance = sync_playwright().start()
        _idealo_browser = _pw_instance.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        return _idealo_browser, None
    except Exception as e:
        return None, f"Browser-Start fehlgeschlagen: {str(e)[:80]}"


def scrape_idealo_ean(ean: str) -> dict:
    """
    Search idealo.de for an EAN using a real Chromium browser (bypasses Cloudflare).
    Returns dict with keys: idealo_price, idealo_seller.
    """
    result = {"idealo_price": "", "idealo_seller": ""}

    browser, err = _get_idealo_browser()
    if err:
        result["idealo_seller"] = err
        return result

    try:
        context = browser.new_context(
            locale="de-DE",
            timezone_id="Europe/Berlin",
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        search_url = (
            f"https://www.idealo.de/preisvergleich/"
            f"MainSearchProductCategory.html?q={ean}"
        )
        page.goto(search_url, wait_until="domcontentloaded", timeout=30000)

        # Accept cookie/consent banner if present
        for btn in [
            "#onetrust-accept-btn-handler",
            "button[data-test='mf-accept-all-btn']",
            "button.consent-accept",
        ]:
            try:
                page.locator(btn).click(timeout=2000)
                page.wait_for_timeout(500)
                break
            except Exception:
                pass

        # Wait for JS-rendered offer list
        page.wait_for_timeout(2500)

        # ── 1. JSON-LD (fastest, most reliable) ──────────────────────────────
        ld_raw = page.evaluate("""() => {
            for (const s of document.querySelectorAll('script[type="application/ld+json"]')) {
                try {
                    const d = JSON.parse(s.textContent);
                    if (d['@type'] === 'Product' && d.offers)
                        return JSON.stringify(d.offers);
                } catch(e) {}
            }
            return null;
        }""")

        if ld_raw:
            offers_data = json.loads(ld_raw)
            if isinstance(offers_data, dict):
                offers_data = [offers_data]
            if isinstance(offers_data, list):
                valid = [o for o in offers_data if o.get("price")]
                if valid:
                    cheapest = min(valid, key=lambda o: float(str(o["price"]).replace(",", ".")))
                    currency = cheapest.get("priceCurrency", "EUR")
                    symbol = "€" if currency == "EUR" else currency
                    result["idealo_price"] = f"{cheapest['price']} {symbol}".strip()
                    seller = cheapest.get("seller", {})
                    result["idealo_seller"] = seller.get("name", "") if isinstance(seller, dict) else ""
                    context.close()
                    return result

        # ── 2. DOM selectors fallback ─────────────────────────────────────────
        for sel, key in [
            (".price__text",                     "idealo_price"),
            ("[data-test='offer-price']",         "idealo_price"),
            (".sc-price",                         "idealo_price"),
            (".shop__name",                       "idealo_seller"),
            ("[data-test='shop-name']",           "idealo_seller"),
            (".sc-shop-name",                     "idealo_seller"),
        ]:
            try:
                loc = page.locator(sel).first
                loc.wait_for(timeout=1500)
                val = loc.text_content() or ""
                if val.strip():
                    result[key] = val.strip()
            except Exception:
                pass

        # ── 3. No-result fallback ─────────────────────────────────────────────
        if not result["idealo_price"] and not result["idealo_seller"]:
            body = (page.content() or "").lower()
            if any(p in body for p in ["keine ergebnisse", "no results", "nichts gefunden"]):
                result["idealo_seller"] = "Nicht auf Idealo gelistet"
            else:
                result["idealo_seller"] = "Preis nicht auslesbar"

        context.close()

    except Exception as e:
        result["idealo_seller"] = f"Fehler: {str(e)[:80]}"

    return result
