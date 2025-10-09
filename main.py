import re, uuid, time, json, os
from typing import Optional, Dict, Any, List, Tuple

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import Response
import httpx
from urllib.parse import quote

from openai import AsyncOpenAI

app = FastAPI(title="ASIN → Keepa Lookup", version="2.0")

# Static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Key format validators ---
OPENAI_KEY_RE = re.compile(r"^sk-[A-Za-z0-9_\-]{20,}$")
KEEPA_KEY_RE  = re.compile(r"^[A-Za-z0-9]{64}$")

def looks_like_openai_key(k: Optional[str]) -> bool:
    return bool(k and OPENAI_KEY_RE.fullmatch(k.strip()))

def looks_like_keepa_key(k: Optional[str]) -> bool:
    return bool(k and KEEPA_KEY_RE.fullmatch(k.strip()))

ASIN_REGEX = re.compile(r"(?:/dp/|/gp/product/|/product/|ASIN=)([A-Z0-9]{10})", re.IGNORECASE)

# ---- Config knobs (env overrideable) ----
MIN_MARGIN_PCT = float(os.getenv("DEFAULT_MIN_MARGIN_THRESHOLD_PCT", "10.0"))
EBAY_FEE_PCT = float(os.getenv("DEFAULT_EBAY_SALES_FEE_PCT", "0.15"))
AD_FEE_PCT = float(os.getenv("DEFAULT_ADVERTISING_FEE_PCT", "0.04"))
UNDERCUT_AMOUNT = float(os.getenv("DEFAULT_UNDERCUT_AMOUNT", "0.05"))
DEFAULT_OPENAI_MODEL = os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o-mini")

# --- tiny in-memory TTL cache for one-shot result pages ---
_RESULT_CACHE: Dict[str, Dict[str, Any]] = {}
_RESULT_TTL_SECONDS = 120

# --- live progress storage ---
_PROGRESS_STORE: Dict[str, Dict[str, Any]] = {}
_EBAY_SUBMISSIONS: Dict[str, Dict[str, Any]] = {}

def _save_progress(rid: str, progress: Dict[str, Any], done: bool = False) -> None:
    _PROGRESS_STORE[rid] = {"progress": progress, "done": done}

def _get_progress(rid: str) -> Dict[str, Any]:
    return _PROGRESS_STORE.get(rid) or {"progress": None, "done": False}

def _finish_progress(rid: str, progress: Dict[str, Any]) -> None:
    _PROGRESS_STORE[rid] = {"progress": progress, "done": True}

_PROGRESS_STEPS_ORDER = [
    "Parse eBay payload",
    "Preprocess title (GPT)",
    "Query Product Finder (Keepa /query)",
    "Batch fetch products (Keepa /product)",
    "Phase-1 margin filter",
    "Phase-2 facet comparison",
    "Finalize results",
]
_PROGRESS_PERCENT = [5, 18, 40, 60, 78, 92, 100]

def _new_progress():
    return {"started": time.time(), "steps": [], "percent": 0}

def _log_step(progress: Dict[str, Any], label: str, summary: str, detail: str = "", rid: Optional[str] = None):
    ts = time.time()
    try:
        idx = _PROGRESS_STEPS_ORDER.index(label)
        pct = _PROGRESS_PERCENT[idx]
    except ValueError:
        pct = min(99, 5 + len(progress.get("steps", [])) * 10)
    progress.setdefault("steps", [])
    progress["steps"].append({"label": label, "summary": summary, "detail": detail, "ts": ts})
    progress["percent"] = max(progress.get("percent", 0), pct)
    if rid:
        _save_progress(rid, progress)

def _put_result(payload: Dict[str, Any], rid: Optional[str] = None) -> str:
    rid = rid or uuid.uuid4().hex
    _RESULT_CACHE[rid] = {"payload": payload, "ts": time.time()}
    return rid

def _get_result(rid: str) -> Optional[Dict[str, Any]]:
    item = _RESULT_CACHE.pop(rid, None)
    now = time.time()
    stale = [k for k, v in _RESULT_CACHE.items() if now - v["ts"] > _RESULT_TTL_SECONDS]
    for k in stale:
        _RESULT_CACHE.pop(k, None)
    return item["payload"] if item else None

def _peek_result(rid: str) -> Optional[Dict[str, Any]]:
    item = _RESULT_CACHE.get(rid)
    now = time.time()
    stale = [k for k, v in _RESULT_CACHE.items() if now - v["ts"] > _RESULT_TTL_SECONDS]
    for k in stale:
        _RESULT_CACHE.pop(k, None)
    return item["payload"] if item else None

# ---------- Helpers ----------
def cents_to_usd(v):
    return None if v in (-1, -2, None) else round(v / 100, 2)

def extract_asin(text: str) -> Optional[str]:
    if not text:
        return None
    m = ASIN_REGEX.search(text)
    if m:
        return m.group(1).upper()
    just_code = re.fullmatch(r"[A-Z0-9]{10}", text.strip(), re.IGNORECASE)
    return text.strip().upper() if just_code else None

def first_image_url(p: Dict[str, Any]) -> Optional[str]:
    """
    Return primary (hero) image URL from imagesCSV first token only.
    images[] fallback removed (legacy unused).
    """
    csv = p.get("imagesCSV")
    if not csv:
        return None
    token = (csv.split(",")[0] or "").strip()
    if not token:
        return None
    if token.startswith("http"):
        return token
    BASE = "https://m.media-amazon.com/images/I/"
    return BASE + quote(token, safe="/-_.~+()%")

def category_leaf_name(category_tree: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    if not category_tree or not isinstance(category_tree, list):
        return None
    leaf = category_tree[-1] if category_tree else {}
    return leaf.get("name")

def map_keepa_required_fields(p: Dict[str, Any]) -> Dict[str, Any]:
    """Map raw Keepa product to the trimmed structure used downstream.

    Pricing precedence (all cents ints -> USD):
      1. stats.current.buyBoxPrice
      2. stats.current.amazon
      3. stats.current.new
      4. stats.current.used
      else None with reason tag.

    Adds fields:
      - priceSource: one of buybox|amazon|new|used|none
      - noPriceReason: present only when priceSource == none
    """
    stats = p.get("stats") or {}
    current_raw = stats.get("current") or {}

    def norm_cents(val):
        try:
            if val in (-1, -2, None):
                return None
            return int(val)
        except Exception:
            return None

    buybox_c = None
    amazon_c = None
    new_c = None
    used_c = None

    # Preferred: named fields (if Keepa provides them without needing buybox=1 history arrays)
    if isinstance(current_raw, dict):
        buybox_c = norm_cents(current_raw.get("buyBoxPrice") or stats.get("buyBoxPrice") or p.get("buyBoxPrice"))
        amazon_c = norm_cents(current_raw.get("amazon"))
        new_c = norm_cents(current_raw.get("new"))
        used_c = norm_cents(current_raw.get("used"))
    elif isinstance(current_raw, list):
        # Fallback: index-based legacy array. Preserve old behavior (index 1 used previously) but document uncertainty.
        # Typical heuristic mapping (legacy): [0]=? (often not used here), [1]=amazon, [2]=new 3P, [3]=used
        amazon_c = norm_cents(current_raw[1]) if len(current_raw) > 1 else None
        new_c = norm_cents(current_raw[2]) if len(current_raw) > 2 else None
        used_c = norm_cents(current_raw[3]) if len(current_raw) > 3 else None
        # buyBoxPrice may still appear as a separate field in stats or product root
        buybox_c = norm_cents(stats.get("buyBoxPrice") or p.get("buyBoxPrice"))
    else:
        # Unknown shape; attempt global fallbacks
        buybox_c = norm_cents(stats.get("buyBoxPrice") or p.get("buyBoxPrice"))
        amazon_c = norm_cents(stats.get("amazon"))
        new_c = norm_cents(stats.get("new"))
        used_c = norm_cents(stats.get("used"))

    price_source = "none"
    price_cents = None
    if buybox_c is not None:
        price_cents = buybox_c; price_source = "buybox"
    elif amazon_c is not None:
        price_cents = amazon_c; price_source = "amazon"
    elif new_c is not None:
        price_cents = new_c; price_source = "new"
    elif used_c is not None:
        price_cents = used_c; price_source = "used"

    current_price_usd = cents_to_usd(price_cents) if price_cents is not None else None
    no_price_reason = None
    if price_source == "none":
        no_price_reason = "no_active_offers_or_buybox_suppressed"

    mapped = {
        # Identity
        "asin": p.get("asin"),
        "parentAsin": p.get("parentAsin"),

        # Descriptive basics
        "title": p.get("title"),
        "brand": p.get("brand"),
        "manufacturer": p.get("manufacturer"),
        "mpn": p.get("partNumber") or p.get("model"),

        # Identifiers
        "upcList": p.get("upcList") or [],
        "eanList": p.get("eanList") or [],

        # Category / Group
        "productGroup": p.get("productGroup"),
        "categoryTree": p.get("categoryTree"),
        "categoryLeaf": category_leaf_name(p.get("categoryTree")),

        # Variations / simple attributes
        "color": p.get("color"),
        "size": p.get("size"),
        "numberOfItems": p.get("numberOfItems"),
        "packageQuantity": p.get("packageQuantity"),

        # Flags
        "isAdultProduct": p.get("isAdultProduct"),

        # Descriptive copy
        "features": p.get("features") or [],
        "description": p.get("description"),

        # Physical details (descriptive)
        "itemLength": p.get("itemLength"),
        "itemWidth": p.get("itemWidth"),
        "itemHeight": p.get("itemHeight"),
        "itemWeight": p.get("itemWeight"),
        "packageLength": p.get("packageLength"),
        "packageWidth": p.get("packageWidth"),
        "packageHeight": p.get("packageHeight"),
        "packageWeight": p.get("packageWeight"),
        "unitCount": p.get("unitCount"),

        # Pricing
        "currentPriceUSD": current_price_usd,
        "priceSource": price_source,
        # Image
        "imageURL": first_image_url(p),
        "imagesCSV": p.get("imagesCSV"),
    }
    if no_price_reason:
        mapped["noPriceReason"] = no_price_reason
    return mapped


# ------------------ NEW: Title preprocessing ------------------

# ...existing code...

async def preprocess_ebay_title(raw_title: str, openai_key: str, model: str, strip_brand: bool = False) -> str:
    """
    Send the raw eBay title directly to GPT (no pre-stripping or normalization)
    to obtain a concise search string. If no OpenAI key, return it unchanged.
    """
    original = raw_title if raw_title is not None else ""

    if not openai_key:
        return original  # unchanged raw title when no key

    client = AsyncOpenAI(api_key=openai_key)

    system_msg = (
        "You are a concise product-query generator. "
        "Return only a short search string. No commentary."
    )
    extra_rule = "Remove any brand/manufacturer names (e.g. Apple, Samsung, Nike) entirely." if strip_brand else "Preserve brand only if it materially disambiguates; otherwise you may remove filler words."
    prompt = f"""
Raw eBay title (unmodified):

{original}

Produce the most concise useful search string for Keepa/Amazon.
Keep product type, pack/quantity if present, material, critical specs.
{extra_rule}
Remove filler, duplicates, irrelevant words.
Return ONLY the search string.
"""

    try:
        resp = await client.chat.completions.create(
            model=model or DEFAULT_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=50,
        )
        optimized = (resp.choices[0].message.content or "").strip()
        print("🔎 GPT-optimized search string (raw input):", optimized)
        return optimized or original
    except Exception as e:
        print("⚠️ GPT preprocessing failed, fallback to raw title:", e)
        return original

async def product_finder_asins(
    title: str,
    api_key: str,
    domain_id: int = 1,
    want_n: int = 25,
    openai_key: Optional[str] = None,
    openai_model: Optional[str] = None,
    strip_brand: bool = False,
) -> Tuple[List[str], str]:
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing Keepa API key")

    # Raw title → GPT (no pre-strip)
    gpt_query = await preprocess_ebay_title(title, openai_key, openai_model, strip_brand=strip_brand)
    # Minimal trim for Keepa selection
    keepa_query = (gpt_query or "").strip().strip('"').strip("'")

    selection = {
        "title": keepa_query,
        "titleSearch": 1,
        "perPage": max(50, want_n),
        "page": 0,
    }

    params = {
        "key": api_key,
        "domain": domain_id,
        "selection": json.dumps(selection, separators=(",", ":")),
    }
    url = "https://api.keepa.com/query"

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params=params)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code,
                                detail=f"Keepa Product Finder error: {r.text}")
        data = r.json()

    asin_list = data.get("asinList") or []
    print(f"🔎 Keepa query '{keepa_query}' → {len(asin_list)} ASINs")
    return asin_list[:want_n], keepa_query


async def fetch_keepa_products(asins: List[str], api_key: str, domain_id: int = 1) -> List[Dict[str, Any]]:
    """
    Batch-fetch multiple products from Keepa by ASINs.
    """
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing Keepa API key")

    asins = [a.strip().upper() for a in asins if a and a.strip()]
    if not asins:
        return []

    url = "https://api.keepa.com/product"
    params = {
        "key": api_key,
        "domain": domain_id,
        "asin": ",".join(asins),
        "stats": 90,
        "history": 0,
        "buybox": 0,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params=params)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=f"Keepa error: {r.text}")
        data = r.json()

    out: List[Dict[str, Any]] = []
    for p in (data.get("products") or []):
        if not p.get("asin"):
            continue
        out.append(map_keepa_required_fields(p))
    return out

async def fetch_keepa_products_batched(
    asins: List[str],
    api_key: str,
    domain_id: int = 1,
    batch_size: int = 20,
    concurrency: int = 3,
    max_retries: int = 3,
    progress: Optional[Dict[str, Any]] = None,
    rid: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch Keepa products in parallel batches.

    Preserves original ASIN order in the returned list (skips missing).
    Logs per-batch start/finish if progress provided.
    """
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing Keepa API key")
    clean_asins = [a.strip().upper() for a in asins if a and a.strip()]
    if not clean_asins:
        return []
    batches: List[List[str]] = [clean_asins[i:i+batch_size] for i in range(0, len(clean_asins), batch_size)]
    total_batches = len(batches)
    sem = asyncio.Semaphore(concurrency)
    results_map: Dict[str, Dict[str, Any]] = {}

    async def fetch_batch(idx: int, batch: List[str]):
        label = f"Keepa /product batch {idx+1}/{total_batches}" if total_batches > 1 else "Keepa /product batch"
        if progress is not None:
            _log_step(progress, label, f"Requesting {len(batch)} ASIN(s)", detail=','.join(batch), rid=rid)
        attempt = 0
        backoff = 1.0
        url = "https://api.keepa.com/product"
        while attempt < max_retries:
            attempt += 1
            try:
                async with sem:
                    params = {
                        "key": api_key,
                        "domain": domain_id,
                        "asin": ",".join(batch),
                        "stats": 90,
                        "history": 0,
                        "buybox": 0,
                    }
                    async with httpx.AsyncClient(timeout=30) as client:
                        r = await client.get(url, params=params)
                        if r.status_code != 200:
                            raise HTTPException(status_code=r.status_code, detail=f"Keepa error: {r.text}")
                        data = r.json()
                prods = data.get("products") or []
                for p in prods:
                    if p.get("asin"):
                        results_map[p["asin"].upper()] = map_keepa_required_fields(p)
                if progress is not None:
                    _log_step(progress, label, f"Success ({len(prods)} products)", detail=f"attempt={attempt}", rid=rid)
                return
            except Exception as e:
                if attempt >= max_retries:
                    if progress is not None:
                        _log_step(progress, label, f"Failed after {attempt} attempt(s)", detail=str(e), rid=rid)
                    return
                if progress is not None:
                    _log_step(progress, label, f"Retry {attempt}/{max_retries-1}", detail=str(e), rid=rid)
                await asyncio.sleep(backoff)
                backoff *= 2

    await asyncio.gather(*[fetch_batch(i, b) for i, b in enumerate(batches)])

    # Preserve original order
    ordered: List[Dict[str, Any]] = []
    for a in clean_asins:
        p = results_map.get(a)
        if p:
            ordered.append(p)
    return ordered

# ---------- Price/Margin helpers (Phase 1) ----------

def compute_profit_and_margin_with_fees(
    ebay_list_price: Optional[float],
    amazon_price: Optional[float],
    undercut_amount: float = UNDERCUT_AMOUNT,
    ebay_fee_pct: float = EBAY_FEE_PCT,
    ad_fee_pct: float = AD_FEE_PCT,
) -> Tuple[Optional[float], Optional[float], Dict[str, float]]:
    """
    Sell Price = eBay input price - undercut_amount
    Fees = Sell Price * (eBay fee % + Ad fee %)
    Net Revenue = Sell Price - Fees
    Profit = Net Revenue - Amazon Price
    Margin % = Profit / Sell Price * 100
    Returns (profit_usd, margin_pct, breakdown_dict)
    """
    if ebay_list_price is None or amazon_price is None or ebay_list_price <= 0:
        return None, None, {
            "sellPrice": None, "feesTotal": None, "ebayFee": None, "adFee": None, "netRevenue": None
        }

    sell_price = max(0.0, round(float(ebay_list_price) - float(undercut_amount), 2))
    ebay_fee = round(sell_price * float(ebay_fee_pct), 2)
    ad_fee = round(sell_price * float(ad_fee_pct), 2)
    fees_total = round(ebay_fee + ad_fee, 2)
    net_revenue = round(sell_price - fees_total, 2)

    profit = round(net_revenue - float(amazon_price), 2)
    margin_pct = round((profit / sell_price) * 100.0, 2) if sell_price > 0 else None

    return profit, margin_pct, {
        "sellPrice": sell_price,
        "ebayFee": ebay_fee,
        "adFee": ad_fee,
        "feesTotal": fees_total,
        "netRevenue": net_revenue
    }

"""Phase-2 Replacement Helpers: facet-based comparison

We remove the legacy yes/no + score based LLM decision. Instead the LLM returns
ONLY structured facet mappings per ASIN. Local code then:
  * normalizes weights for facets present in eBay title
  * maps status colors → numeric multipliers
  * computes a final percentage score
  * applies a user-configurable minimum score threshold

Facet keys (canonical internal names):
  product_type, functional_feature, compatibility, material,
  brand, color, quantity, vague_descriptor, marketing
"""

from scoring import compute_phase2_scores, FACET_WEIGHTS  # reuse weights for any display/meta if needed
import asyncio
import re
import json

PHASE2_FACET_ORDER = [
    "product_type","functional_feature","compatibility","material",
    "brand","color","quantity","vague_descriptor","marketing"
]

# JSON Schema used for structured facet extraction (OpenAI response_format)
PHASE2_FACET_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Phase 2 Facet Comparison",
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["asin", "facets"],
                "properties": {
                    "asin": {"type": "string"},
                    "facets": {
                        "type": "object",
                        "properties": {
                            "product_type": {"$ref": "#/definitions/facet"},
                            "functional_feature": {"$ref": "#/definitions/facet"},
                            "compatibility": {"$ref": "#/definitions/facet"},
                            "material": {"$ref": "#/definitions/facet"},
                            "brand": {"$ref": "#/definitions/facet"},
                            "color": {"$ref": "#/definitions/facet"},
                            "quantity": {"$ref": "#/definitions/facet"},
                            "vague_descriptor": {"$ref": "#/definitions/facet"},
                            "marketing": {"$ref": "#/definitions/facet"}
                        },
                        "additionalProperties": False
                    }
                },
                "additionalProperties": True
            }
        },
        "error": {"type": "string"}
    },
    "required": ["results"],
    "additionalProperties": True,
    "definitions": {
        "facet": {
            "type": "object",
            "properties": {
                "ebay": {"type": ["string", "null"]},
                "amazon": {"type": ["string", "null"]},
                "status": {"type": "string", "enum": ["green", "yellow", "red", "gray"]},
                "note": {"type": "string"}
            },
            "required": ["status"],
            "additionalProperties": False
        }
    }
}

def _safe_json_extract(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r'\{.*\}|\[.*\]', s, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise

def _build_phase2_messages(ebay_title: str, products_meta: List[Dict[str,str]]):
    system_msg = (
        "You are an assistant that extracts comparison facets between an eBay title and multiple Amazon products. "
        "Return ONLY the JSON that conforms to the provided JSON Schema. Use null for any missing eBay facet."
    )
    user_payload = {
        "ebay_title": ebay_title,
        "amazon_candidates": products_meta,
        "facet_keys": PHASE2_FACET_ORDER,
        "status_semantics": {
            "green": "exact or clear synonym match",
            "yellow": "close variant/approximate",
            "red": "contradiction or required facet missing",
            "gray": "facet absent from eBay title (neutral)"
        }
    }
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
    ]

async def llm_phase2_facets(
    ebay_title: str,
    products_meta: List[Dict[str,str]],
    openai_key: str,
    model: str,
) -> Tuple[Dict[str, Any], str]:
    if not openai_key:
        raise HTTPException(status_code=400, detail="OpenAI key required for facet comparison.")
    client = AsyncOpenAI(api_key=openai_key)
    messages = _build_phase2_messages(ebay_title, products_meta)
    # Log truncated user JSON for debugging
    for m in messages:
        if m["role"] == "user":
            print("📝 Phase-2 facet user payload (trunc):", m["content"][:1800], "…")
    try:
        resp = await client.chat.completions.create(
            model=model or DEFAULT_OPENAI_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=2200,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "phase2_facets",
                    "schema": PHASE2_FACET_JSON_SCHEMA
                }
            }
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as net_e:
        # Network / API failure: return empty results with error metadata
        print("⚠️ Phase-2 facet LLM request failed:", net_e)
        return {"results": [], "error": f"llm_request_failed: {net_e}"}, ""
    # Direct parse (schema-enforced). Provide graceful fallback on parse error.
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("top-level not object")
    except Exception as e:
        print("⚠️ Phase-2 facet JSON parse failed despite schema mode:", e)
        return {"results": [], "error": f"json_parse_failed: {e}"}, raw
    return parsed, raw

# ---------- No-cache headers (restored) ----------
@app.middleware("http")
async def no_cache(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "error": None,
        "batch": None,
        "summary": None,
        "MIN_MARGIN_PCT": MIN_MARGIN_PCT,
        "EBAY_FEE_PCT": EBAY_FEE_PCT,
        "AD_FEE_PCT": AD_FEE_PCT,
        "UNDERCUT_AMOUNT": UNDERCUT_AMOUNT,
    })

from fastapi.responses import JSONResponse
from fastapi import Query

@app.get("/progress")
async def progress_api(rid: str):
    pr = _get_progress(rid)
    prog = pr.get("progress") or {"percent": 0, "steps": []}
    steps = prog.get("steps") or []
    last = steps[-1] if steps else {}
    headline = last.get("label") or "Working…"
    subheadline = last.get("summary") or ""

    return JSONResponse({
        "done": bool(pr.get("done")),
        "progress": prog,                    # keep full object for the collapsible log
        "headline": headline,                # short line for the bar area
        "subheadline": subheadline,          # optional second line if you want it
        "has_result": _peek_result(rid) is not None
    })

@app.post("/reset")
async def reset_api(rid: str = Query(None, description="Run id to reset (optional)")):
    """Clear progress + result caches for a given rid (or all if none)."""
    cleared: Dict[str, Any] = {"progress": 0, "results": 0}
    if rid:
        if rid in _PROGRESS_STORE:
            _PROGRESS_STORE.pop(rid, None)
            cleared["progress"] = 1
        if rid in _RESULT_CACHE:
            _RESULT_CACHE.pop(rid, None)
            cleared["results"] = 1
    else:
        cleared["progress"] = len(_PROGRESS_STORE)
        cleared["results"] = len(_RESULT_CACHE)
        _PROGRESS_STORE.clear()
        _RESULT_CACHE.clear()
    return JSONResponse({"status": "ok", "cleared": cleared})

import asyncio

async def _analyze_worker(
    rid: str,
    ebay_data: str,
    keepa_key: str,
    openai_key: str,
    openai_model: str,
    amazon_links: str,
    want_n: int,
    ebay_fee_pct: float,
    ad_fee_pct: float,
    min_margin_pct: float,
    undercut_amount: float,
    min_phase2_score: float,
    yellow_multiplier: float,
    strip_brand: bool = False,
):
    # copy your existing analyze() body HERE, but:
    #   - remove the @app.post decorator/signature
    #   - use the local variables from params
    #   - use _log_step(..., rid=rid) or _save_progress(rid, progress) after each step
    #   - at the very end:
    #         _finish_progress(rid, progress)
    #         _put_result({...})   (keep exactly what you had)
    # DO NOT return a Response from this function.
    progress = _new_progress()
    _save_progress(rid, progress)

    # ---- BEGIN: your current analyze body (trimmed to the key edits) ----
    try:
        # Wrap the entire worker logic so any unhandled exception is converted to a graceful result
        try:
            ebay_payload = json.loads(ebay_data)
            ebay_title = (ebay_payload.get("title") or "").strip()
            ebay_price = float(ebay_payload.get("price")) if ebay_payload.get("price") is not None else None
            _log_step(progress, "Parse eBay payload", "Parsed title and price from eBay input",
                      detail=json.dumps(ebay_payload, ensure_ascii=False), rid=rid)
        except Exception:
            _put_result({
                "error": "Invalid eBay data JSON. Expect: {\"title\":\"...\",\"price\":12.34,\"itemNumber\":\"...\"}",
                "batch": [], "summary": {"checked": 0, "passed": 0, "want_n": 0},
                "progress": progress
            }, rid=rid)
            _finish_progress(rid, progress)
            return

        if not ebay_title or ebay_price is None:
            _put_result({
                "error": "Missing required fields in eBay data. Need both title and price.",
                "batch": [], "summary": {"checked": 0, "passed": 0, "want_n": 0},
                "progress": progress
            }, rid=rid)
            _finish_progress(rid, progress)
            return

        want_n = max(1, min(50, int(want_n)))
        lines = [ln.strip() for ln in (amazon_links or "").splitlines() if ln.strip()]
        candidates: List[Tuple[str, str]] = []

        if lines:
            for raw in lines:
                asin = extract_asin(raw)
                candidates.append((raw, asin if asin else ""))
            _log_step(progress, "Manual ASIN/links supplied",
                      f"Using {len(candidates)} manual ASIN/link(s)",
                      detail="\n".join(lines), rid=rid)
        else:
            if not openai_key:
                _put_result({
                    "error": "Missing OpenAI API key. Please enter your OpenAI key to enable Product Finder.",
                    "batch": [], "summary": {"checked": 0, "passed": 0, "want_n": want_n},
                    "progress": progress
                }, rid=rid)
                _finish_progress(rid, progress)
                return

            _log_step(progress, "Preprocess title (GPT)",
                      "Generating concise Keepa query from eBay title",
                      detail=ebay_title, rid=rid)
            try:
                asins, keepa_query = await product_finder_asins(
                    title=ebay_title, api_key=keepa_key, domain_id=1,
                    want_n=want_n, openai_key=openai_key, openai_model=openai_model,
                    strip_brand=strip_brand,
                )
                _log_step(progress, "Query Product Finder (Keepa /query)",
                          f"Keepa returned {len(asins)} ASIN(s)",
                          detail=f'query="{keepa_query}"\nasins={asins}', rid=rid)
                if not asins:
                    _put_result({
                        "error": "No matches found in Product Finder for that eBay title.",
                        "batch": [], "summary": {"checked": 0, "passed": 0, "want_n": want_n},
                        "progress": progress
                    }, rid=rid)
                    _finish_progress(rid, progress)
                    return
                for a in asins:
                    candidates.append((ebay_title, a))
            except HTTPException as e:
                _put_result({
                    "error": e.detail, "batch": [],
                    "summary": {"checked": 0, "passed": 0, "want_n": want_n},
                    "progress": progress
                }, rid=rid)
                _finish_progress(rid, progress)
                return

        all_checked: List[Dict[str, Any]] = []
        valid_asins = [asin for _, asin in candidates if asin]
        # Use batched Keepa fetch for resilience & parallelism
        products = await fetch_keepa_products_batched(
            valid_asins, keepa_key, domain_id=1, batch_size=20, concurrency=3,
            max_retries=3, progress=progress, rid=rid
        )
        _log_step(progress, "Batch fetch products (Keepa /product)",
                  f"Fetched {len(products)} product(s) from Keepa (requested={len(valid_asins)})",
                  detail=f"requested={len(valid_asins)} asins={valid_asins}", rid=rid)

        by_asin = {p["asin"]: p for p in products if p.get("asin")}
        rejected_phase1: List[str] = []  # <-- collect only ASINs rejected in Phase-1

        rejected_price_map: Dict[str, Optional[float]] = {}
        for input_label, asin in candidates:
            if not asin:
                all_checked.append({
                    "input": input_label,
                    "asin": None,
                    "error": "Invalid or missing ASIN",
                    "result": None
                })
                continue

            product = by_asin.get(asin)
            if not product:
                all_checked.append({
                    "input": input_label,
                    "asin": asin,
                    "error": "ASIN not returned by Keepa batch",
                    "result": None
                })
                continue

        # ---- Phase 1: Price/Margin filter w/ fees + undercut ----
            amazon_price = product.get("currentPriceUSD")
            profit, margin_pct, breakdown = compute_profit_and_margin_with_fees(
            ebay_list_price=ebay_price,
            amazon_price=amazon_price,
            undercut_amount=undercut_amount,
            ebay_fee_pct=ebay_fee_pct,
            ad_fee_pct=ad_fee_pct,
        )

            phase1_reasons = []
            price_pass = True

            if amazon_price is None:
                price_pass = False
                phase1_reasons.append("Missing Amazon current price ✗")
            elif profit is None or margin_pct is None:
                price_pass = False
                phase1_reasons.append("Invalid price inputs ✗")
            elif margin_pct < min_margin_pct:
                price_pass = False
                phase1_reasons.append(f"Margin {margin_pct:.2f}% < {min_margin_pct:.2f}% ✗")
            else:
                phase1_reasons.append(
                    f"Selling at ${breakdown['sellPrice']:.2f} (undercut ${undercut_amount:.2f}); "
                    f"fees: eBay {int(ebay_fee_pct*100)}% + ads {int(ad_fee_pct*100)}% = ${breakdown['feesTotal']:.2f}; "
                    f"net revenue ${breakdown['netRevenue']:.2f}; profit ${profit:.2f}; margin {margin_pct:.2f}% ✓"
                )
                rec = {
                    "input": input_label,
                    "asin": asin,
                    "error": None,
                    "result": product,
                    "pricing": {
                        "ebayListPriceUSD": round(float(ebay_price), 2),
                        "sellPriceUSD": breakdown["sellPrice"],
                        "amazonPriceUSD": amazon_price,
                        "fees": {
                            "ebayFeeUSD": breakdown["ebayFee"],
                            "adFeeUSD": breakdown["adFee"],
                            "totalUSD": breakdown["feesTotal"],
                            "ebayFeePct": ebay_fee_pct,
                            "adFeePct": ad_fee_pct
                        },
                        "netRevenueUSD": breakdown["netRevenue"],
                        "profitUSD": profit,
                        "marginPct": margin_pct,
                        "reasons": phase1_reasons,
                        "passed": True,
                    },
                    "match": { "passed": False, "score": 0.0, "reasons": ["Pending LLM match."] }
                }
                all_checked.append(rec)

            if not price_pass:
                rejected_phase1.append(asin)  # track ASIN
                rejected_price_map[asin] = amazon_price
                all_checked.append({
                    "input": input_label,
                    "asin": asin,
                    "error": None,
                    "result": product,
                    "pricing": {
                        "ebayListPriceUSD": round(float(ebay_price), 2),
                        "sellPriceUSD": breakdown.get("sellPrice"),
                        "amazonPriceUSD": amazon_price,
                        "fees": {
                            "ebayFeeUSD": breakdown.get("ebayFee"),
                            "adFeeUSD": breakdown.get("adFee"),
                            "totalUSD": breakdown.get("feesTotal"),
                            "ebayFeePct": ebay_fee_pct,
                            "adFeePct": ad_fee_pct
                        },
                        "netRevenueUSD": breakdown.get("netRevenue"),
                        "profitUSD": profit,
                        "marginPct": margin_pct,
                        "reasons": phase1_reasons,
                        "passed": False
                    },
                    "match": {
                        "passed": False,
                        "score": 0.0,
                        "reasons": ["Skipped title check due to margin filter."]
                    }
                })

        # --- Phase-1 consolidated progress log (passes + discards) ---
        total_evaluated = len(all_checked)
        passed_count = sum(1 for r in all_checked if r.get("pricing", {}).get("passed"))
        discarded_count = len(rejected_phase1)
        # Build list with prices: prefer amazon price (cost) fallback to currentPriceUSD in result mapping
        if rejected_phase1:
            decorated = []
            for asin_r in rejected_phase1:
                p = rejected_price_map.get(asin_r)
                if p is None:
                    decorated.append(f"{asin_r} (n/a)")
                else:
                    decorated.append(f"{asin_r} (${p:.2f})")
            discarded_list = ", ".join(decorated)
        else:
            discarded_list = "None"
        # Include effective config values (fees are fractions already)
        # Attempt to capture a representative sell price: use first passed record if available
        # Representative sell price can be computed regardless of any Phase-1 passes
        # (user wants this even when 0 products pass). Previously we only showed it
        # when at least one product passed (derived from first passed record). Now we
        # compute directly from eBay list price - undercut.
        rep_sell_price = max(0.0, round(float(ebay_price) - float(undercut_amount), 2))
        # Build config lines with original eBay list price and representative sell price
        cfg_lines = [
            f"Effective eBay fee fraction: {ebay_fee_pct:.4f}",
            f"Effective Advertising fee fraction: {ad_fee_pct:.4f}",
            f"Effective Undercut amount: ${undercut_amount:.2f}",
            f"Effective Min margin %: {min_margin_pct:.2f}",
            f"eBay list price: ${float(ebay_price):.2f}",
        ]
        # Always include the representative sell price line (independent of pass/fail)
        cfg_lines.append(f"Representative sell price (list - undercut): ${rep_sell_price:.2f}")
        detail = ("\n".join(cfg_lines) + "\n\nDiscarded ASINs:\n" + f"{discarded_list}\nDiscarded={discarded_count}")
        _log_step(
            progress,
            "Phase-1 margin filter",
            f"{passed_count} passed, {discarded_count} discarded (threshold {min_margin_pct:.2f}%)",
            detail=detail,
            rid=rid
        )

        phase1_passed_rows = [r for r in all_checked if r.get("pricing", {}).get("passed")]
        if not phase1_passed_rows:
            summary = {"checked": len(all_checked), "passed": 0, "want_n": want_n}
            _log_step(progress, "Finalize results",
                      "0 final matches (no Phase-1 passes)",
                      detail=json.dumps({"summary": summary}, ensure_ascii=False), rid=rid)
            _put_result({
                "error": None, "batch": [], "summary": summary,
                "MIN_MARGIN_PCT": min_margin_pct, "EBAY_FEE_PCT": ebay_fee_pct,
                "AD_FEE_PCT": ad_fee_pct, "UNDERCUT_AMOUNT": undercut_amount,
                "progress": progress
            }, rid=rid)
            _finish_progress(rid, progress)
            return
        if not openai_key:
            _put_result({
                "error": "Missing OpenAI API key for Phase-2 LLM matching.",
                "batch": [], "summary": {"checked": len(all_checked), "passed": 0, "want_n": want_n},
                "MIN_MARGIN_PCT": min_margin_pct, "EBAY_FEE_PCT": ebay_fee_pct,
                "AD_FEE_PCT": ad_fee_pct, "UNDERCUT_AMOUNT": undercut_amount,
                "progress": progress
            }, rid=rid)
            _finish_progress(rid, progress)
            return

        # Phase-2: build products_meta with FULL Amazon titles (no truncation per user request)
        products_meta = [{"asin": r["asin"], "title": (r["result"].get("title") or "")}
                         for r in phase1_passed_rows]

        # Emit strict JSON (object) for easier debugging/parsing downstream
        phase2_payload = {
            "eBay_title": ebay_title,
            "phase2_candidate_count": len(phase1_passed_rows),
            "candidates": products_meta,
        }
        phase2_detail = json.dumps(phase2_payload, ensure_ascii=False, indent=2)
        _log_step(
            progress,
            "Phase-2 facet comparison",
            f"Extracting facets for {len(phase1_passed_rows)} candidate(s)",
            detail=phase2_detail,
            rid=rid
        )

        # --- Phase-2 facet extraction (batched if many candidates) ---
        llm_raw_all: List[str] = []
        if len(products_meta) <= 5:
            facet_json, llm_raw = await llm_phase2_facets(
                ebay_title=ebay_title,
                products_meta=products_meta,
                openai_key=openai_key,
                model=openai_model,
            )
            if facet_json.get("error"):
                fallback_detail = json.dumps({
                    "error": facet_json.get("error"),
                    "phase2_payload": phase2_payload
                }, ensure_ascii=False)
                _log_step(
                    progress,
                    "Phase-2 facet comparison (error)",
                    f"Facet extraction failed for {len(phase1_passed_rows)} candidate(s); continuing with zero facet scores",
                    detail=fallback_detail,
                    rid=rid
                )
            llm_raw_all.append(llm_raw)
        else:
            # Batching
            batch_size = 5
            batches_pm = [products_meta[i:i+batch_size] for i in range(0, len(products_meta), batch_size)]
            total_batches = len(batches_pm)
            _log_step(
                progress,
                "Phase-2 batching start",
                f"Processing {len(products_meta)} candidates in {total_batches} LLM batch(es) (size={batch_size})",
                detail=f"candidate_count={len(products_meta)} batch_size={batch_size}",
                rid=rid
            )
            sem_llm = asyncio.Semaphore(2)  # concurrency limit for LLM calls
            batch_results: List[Tuple[int, Dict[str, Any], str]] = []  # (index, facet_json, raw)

            async def run_facet_batch(bi: int, subset: List[Dict[str, str]]):
                label = f"Phase-2 facet batch {bi+1}/{total_batches}"
                _log_step(progress, label, f"Starting ({len(subset)} candidates)", rid=rid, detail="asins=" + ",".join([p['asin'] for p in subset]))
                async with sem_llm:
                    fj, raw = await llm_phase2_facets(
                        ebay_title=ebay_title,
                        products_meta=subset,
                        openai_key=openai_key,
                        model=openai_model,
                    )
                status_msg = "ok" if not fj.get("error") else f"error: {fj.get('error')}"
                detail = f"raw_len={len(raw)}" + (f"\nerror={fj.get('error')}" if fj.get("error") else "")
                _log_step(progress, label, f"Completed ({status_msg})", rid=rid, detail=detail)
                batch_results.append((bi, fj, raw))

            await asyncio.gather(*[run_facet_batch(i, sub) for i, sub in enumerate(batches_pm)])
            # Merge preserving original order
            batch_results.sort(key=lambda t: t[0])
            merged: List[Dict[str, Any]] = []
            any_error = None
            for _, fj, raw in batch_results:
                llm_raw_all.append(raw)
                rs = fj.get("results") or []
                merged.extend(rs)
                if fj.get("error") and not any_error:
                    any_error = fj["error"]
            facet_json = {"results": merged}
            if any_error:
                facet_json["error"] = any_error
                _log_step(progress, "Phase-2 facet comparison (error)",
                          f"One or more batches had errors; merged {len(merged)} result entries",
                          detail=str(any_error), rid=rid)
        llm_raw = "\n\n---\n".join([r for r in llm_raw_all if r])
        facet_scores = compute_phase2_scores(
            facet_json,
            min_phase2_score,
            yellow_multiplier=yellow_multiplier,
            override_brand_weight=1 if 'strip_brand_flag' in globals() and strip_brand_flag else None
        )

        final_rows: List[Dict[str, Any]] = []
        for row in phase1_passed_rows:
            asin = row["asin"]
            score_block = facet_scores.get(asin) or {"score_pct": 0.0, "passed": False, "facets": {}}
            row["match"] = {
                "passed": score_block["passed"],
                "score": score_block["score_pct"],
                "reasons": [f"Score {score_block['score_pct']:.2f}% (threshold {min_phase2_score:.2f}%)"],
                "facets": score_block["facets"],
            }
            if score_block["passed"]:
                final_rows.append(row)

        final_rows.sort(key=lambda r: (r["result"].get("currentPriceUSD") or 1e9, r["asin"]))
        summary = {"checked": len(all_checked), "passed": len(final_rows), "want_n": want_n}

        finalize_meta = {
            "effective_yellow_multiplier": yellow_multiplier,
            "effective_min_phase2_score": min_phase2_score,
            "phase2_candidate_count": len(phase1_passed_rows)
        }
        finalize_detail = json.dumps({"config": finalize_meta, "summary": summary}, ensure_ascii=False)
        _log_step(
            progress,
            "Finalize results",
            f"{len(final_rows)} final match(es) after LLM (Phase-2 candidates: {len(phase1_passed_rows)})",
            detail=finalize_detail,
            rid=rid
        )

        _put_result({
            "error": None,
            "batch": final_rows,
            # All Phase-2 candidates with facet data (pass or fail) for System B visualization
            "phase2_all_rows": [
                {
                    "asin": r["asin"],
                    "result": r.get("result"),
                    "match": r.get("match"),  # contains score, passed, facets
                    "facetComparison": (r.get("match") or {}).get("facets"),
                    # expose pricing block for client-side sorting convenience
                    "pricing": r.get("pricing")
                } for r in phase1_passed_rows
            ],
            "summary": summary,
            "MIN_MARGIN_PCT": min_margin_pct,
            "EBAY_FEE_PCT": ebay_fee_pct,
            "AD_FEE_PCT": ad_fee_pct,
            "UNDERCUT_AMOUNT": undercut_amount,
            "keepa_query": keepa_query if 'keepa_query' in locals() else None,
            "llm_raw_output": llm_raw,
            "phase2_facets": facet_json,
            "MIN_PHASE2_SCORE": min_phase2_score,
            "YELLOW_MULTIPLIER": yellow_multiplier,
            "progress": progress
        }, rid=rid)
        _finish_progress(rid, progress)
    except Exception as e:
        # Final catch-all: ensure a result exists and progress finishes
        _log_step(progress, "Analysis failed", "Unhandled exception in worker",
                  detail=str(e), rid=rid)
        _put_result({
            "error": f"Unhandled worker exception: {e}",
            "batch": [],
            "summary": {"checked": 0, "passed": 0, "want_n": want_n},
            "progress": progress
        }, rid=rid)
        _finish_progress(rid, progress)
    # ---- END worker ----

@app.post("/analyze")
async def analyze(
    ebay_data: str = Form(...),
    keepa_key: str = Form(...),
    openai_key: str = Form(""),
    openai_model: str = Form(DEFAULT_OPENAI_MODEL),
    amazon_links: str = Form(""),
    want_n: int = Form(25),
    ebay_fee_pct: float = Form(None),  # user supplied percent (e.g. 15 for 15%)
    ad_fee_pct: float = Form(None),    # user supplied percent
    min_margin_pct: float = Form(None),  # user supplied min margin percent (-10 to 200)
    undercut_amount: float = Form(None),  # user supplied undercut amount 0.00 - 1.00
    min_phase2_score: float = Form(70.0),  # NEW: user supplied min Phase-2 score percent (40-95)
    yellow_multiplier: float = Form(0.5),  # NEW: dynamic yellow facet multiplier (0-1)
    min_score_pct: float = Form(None),     # Alternate field name used by template
    strip_brand: Optional[str] = Form(None),  # checkbox ("1" when on)
):
    
     # 1) Keepa format (32 hex)
    if not looks_like_keepa_key(keepa_key):
        # Return a small, harmless redirect to the result page with a clean error (will rarely be hit)
        rid = uuid.uuid4().hex
        progress = _new_progress()
        _save_progress(rid, progress)
        _put_result({
            "error": "Invalid Keepa API key. Expected 64 hex characters.",
            "batch": [],
            "summary": {"checked": 0, "passed": 0, "want_n": 0},
            "progress": progress
        }, rid=rid)
        _finish_progress(rid, progress)
        return RedirectResponse(url=f"/result?rid={rid}", status_code=303)

    # 2) If using Product Finder (no manual links), OpenAI key must look valid
    links_present = any(ln.strip() for ln in (amazon_links or "").splitlines())
    if not links_present and not looks_like_openai_key(openai_key):
        rid = uuid.uuid4().hex
        progress = _new_progress()
        _save_progress(rid, progress)
        _put_result({
            "error": "Invalid or missing OpenAI API key. Provide a valid key to use Product Finder.",
            "batch": [],
            "summary": {"checked": 0, "passed": 0, "want_n": 0},
            "progress": progress
        }, rid=rid)
        _finish_progress(rid, progress)
        return RedirectResponse(url=f"/result?rid={rid}", status_code=303)

    # 3) eBay JSON sanity
    try:
        obj = json.loads(ebay_data)
        if not obj.get("title") or not isinstance(obj.get("price"), (int, float)):
            raise ValueError("Missing title or numeric price")
    except Exception:
        rid = uuid.uuid4().hex
        progress = _new_progress()
        _save_progress(rid, progress)
        _put_result({
            "error": "Invalid eBay data JSON. Example: {\"title\":\"…\",\"price\":12.34,\"itemNumber\":\"…\"}",
            "batch": [],
            "summary": {"checked": 0, "passed": 0, "want_n": 0},
            "progress": progress
        }, rid=rid)
        _finish_progress(rid, progress)
        return RedirectResponse(url=f"/result?rid={rid}", status_code=303)
        
    rid = uuid.uuid4().hex
    progress = _new_progress()
    _save_progress(rid, progress)  # so the page has something to show immediately
    # Stash eBay submission for phase-2 sidebar card (survives textarea clearing after redirect)
    try:
        obj = json.loads(ebay_data)
        _EBAY_SUBMISSIONS[rid] = {
            "title": (obj.get("title") or "").strip(),
            "itemNumber": obj.get("itemNumber"),
            "price": obj.get("price")
        }
    except Exception:
        _EBAY_SUBMISSIONS[rid] = {"title": "", "itemNumber": None, "price": None}

    # Clamp + normalize fee percentages (convert from whole % to fraction)
    def norm(val, dflt, lo, hi):
        if val is None: return dflt
        try:
            v = float(val)
        except Exception:
            return dflt
        v = max(lo, min(hi, v))
        return round(v / 100.0, 4)  # store as fraction

    eff_ebay_fee = norm(ebay_fee_pct, EBAY_FEE_PCT*100, 5, 30)
    eff_ad_fee = norm(ad_fee_pct, AD_FEE_PCT*100, 0, 20)
    # Min margin: allow a little negative (for experimentation), convert to raw percent (not fraction) for comparisons later
    def norm_margin(val, dflt, lo, hi):
        if val is None: return dflt
        try:
            v = float(val)
        except Exception:
            return dflt
        return max(lo, min(hi, v))
    eff_min_margin = norm_margin(min_margin_pct, MIN_MARGIN_PCT, -10, 200)
    # Undercut amount clamp (no conversion needed)
    def norm_under(val, dflt, lo, hi):
        if val is None: return dflt
        try:
            v = float(val)
        except Exception:
            return dflt
        return max(lo, min(hi, v))
    eff_undercut = norm_under(undercut_amount, UNDERCUT_AMOUNT, 0.0, 1.0)

    # Fire the worker in the background with effective fees (already fractions)
    # Clamp phase-2 score (range 40-95 per spec)
    # Accept either min_score_pct (template) or min_phase2_score (legacy) with precedence to min_score_pct
    chosen_min_score = min_score_pct if min_score_pct is not None else min_phase2_score
    try:
        m2 = float(chosen_min_score)
    except Exception:
        m2 = 70.0
    min_phase2_score_eff = max(40.0, min(95.0, m2))

    # Clamp yellow multiplier (0-1)
    try:
        ym = float(yellow_multiplier)
    except Exception:
        ym = 0.5
    ym_eff = max(0.0, min(1.0, ym))

    # Interpret strip_brand checkbox
    strip_brand_flag = bool(strip_brand)

    asyncio.create_task(_analyze_worker(
        rid, ebay_data, keepa_key, openai_key, openai_model, amazon_links, want_n,
        eff_ebay_fee, eff_ad_fee, eff_min_margin, eff_undercut, min_phase2_score_eff, ym_eff,
        strip_brand=strip_brand_flag
    ))
    return RedirectResponse(url=f"/result?rid={rid}", status_code=303)



@app.get("/result", response_class=HTMLResponse)
async def result(request: Request, rid: str):
    payload = _peek_result(rid) or {}
    pr = _get_progress(rid)
    ctx = {
        "request": request,
        "rid": rid,
        "error": payload.get("error"),
        "batch": payload.get("batch"),
        "phase2_all_rows": payload.get("phase2_all_rows"),
        "summary": payload.get("summary"),
        "progress": pr.get("progress"),
        "MIN_MARGIN_PCT": payload.get("MIN_MARGIN_PCT", MIN_MARGIN_PCT),
        "EBAY_FEE_PCT": payload.get("EBAY_FEE_PCT", EBAY_FEE_PCT),
        "AD_FEE_PCT": payload.get("AD_FEE_PCT", AD_FEE_PCT),
        "UNDERCUT_AMOUNT": payload.get("UNDERCUT_AMOUNT", UNDERCUT_AMOUNT),
        "keepa_query": payload.get("keepa_query"),  # no legacy fallback
        "llm_raw_output": payload.get("llm_raw_output"),
        "phase2_facets": payload.get("phase2_facets"),
        "MIN_PHASE2_SCORE": payload.get("MIN_PHASE2_SCORE", 70.0),
        "YELLOW_MULTIPLIER": payload.get("YELLOW_MULTIPLIER", 0.5),
        "EBAY_SUB": _EBAY_SUBMISSIONS.get(rid)
    }
    # Provide phase2_meta aggregated object for template compatibility
    try:
        ctx["phase2_meta"] = {
            "yellow_multiplier": float(ctx["YELLOW_MULTIPLIER"]),
            "min_score_pct": float(ctx["MIN_PHASE2_SCORE"])
        }
    except Exception:
        ctx["phase2_meta"] = {"yellow_multiplier": 0.5, "min_score_pct": 70.0}
    import json
    ctx["progress_json"] = json.dumps(ctx["progress"] or {})
    ctx["waiting"] = ctx["batch"] is None
    return templates.TemplateResponse("index.html", ctx)


