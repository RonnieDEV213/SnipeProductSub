import re, uuid, time, json
from typing import Optional, Dict, Any, List, Tuple

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import Response
import httpx
from urllib.parse import quote

app = FastAPI(title="ASIN → Keepa Lookup", version="1.8")

# Static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

ASIN_REGEX = re.compile(r"(?:[/dp/]|ASIN=)([A-Z0-9]{10})", re.IGNORECASE)

# --- tiny in-memory TTL cache for one-shot result pages ---
_RESULT_CACHE: Dict[str, Dict[str, Any]] = {}
_RESULT_TTL_SECONDS = 120  # keep for 2 minutes, then purge

def _put_result(payload: Dict[str, Any]) -> str:
    rid = uuid.uuid4().hex
    _RESULT_CACHE[rid] = {"payload": payload, "ts": time.time()}
    return rid

def _get_result(rid: str) -> Optional[Dict[str, Any]]:
    item = _RESULT_CACHE.pop(rid, None)  # pop so refresh clears results
    # also purge old items
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
    Return a usable image URL from Keepa product payload.
    Prefers full URLs if provided; otherwise builds a URL from the filename.
    """
    # Prefer imagesCSV first
    csv = p.get("imagesCSV")
    token = None
    if csv:
        token = (csv.split(",")[0] or "").strip()

    # Fallback to structured images[] if present
    if not token:
        imgs = p.get("images")
        if isinstance(imgs, list) and imgs:
            token = (imgs[0] or "").strip()

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
    # Current price from stats.current[1] (cents → USD)
    stats = p.get("stats") or {}
    current_arr = stats.get("current") or []
    current_price_cents = current_arr[1] if len(current_arr) > 1 else None
    price_current = cents_to_usd(current_price_cents)

    return {
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
        "currentPriceUSD": price_current,

        # Image
        "imageURL": first_image_url(p),
        "imagesCSV": p.get("imagesCSV"),
    }

# ------------------ Phase-1 Filtration ------------------

# Canonical vocab and synonyms (minimal, extend as you go)
CANON_TYPES = {
    "carabiner", "keychain", "clip", "hook", "snap", "ring", "d-ring", "dring"
}
TYPE_SYNONYMS = {
    "snap hook": "carabiner",
    "spring hook": "carabiner",
    "spring clip": "carabiner",
    "key ring": "keychain",
    "keyring": "keychain",
    "key chain": "keychain",
    "d ring": "d-ring",
    "dring": "d-ring",
}
SHAPE_SYNONYMS = {
    "pear": "gourd",
    "tear drop": "gourd",
    "teardrop": "gourd",
}
COLOR_SYNONYMS = {
    "multi color": "assorted",
    "multicolor": "assorted",
    "various colors": "assorted",
    "mixed color": "assorted",
}
MATERIALS = {"aluminum", "aluminium", "stainless", "stainless steel", "steel", "zinc", "zinc alloy", "alloy"}
COLORS    = {"black","silver","gray","grey","gold","blue","red","green","white","assorted"}
QUALIFIERS= {"spring","locking","lock","heavy duty","mini","small","large","key","buckle"}

STOPWORDS = {"for","with","and","the","a","an","of","to","in","on"}

PACK_RE = re.compile(r"\b(\d+)\s*(?:pcs?|pack|count|ct|pairs?)\b", re.I)
# sizes: 12 mm / 0.5 in / 1/4"
SIZE_UNITS_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(mm|cm|in|\"|inch|inches)\b", re.I)
FRACTION_IN_RE = re.compile(r"\b(\d+)\s*/\s*(\d+)\s*(?:in|\"|inch|inches)?\b", re.I)
METRIC_THREAD_RE = re.compile(r"\bM(\d+(?:\.\d+)?)\b", re.I)  # M6, M8.5

def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    # normalize quotes and dashes
    s = s.replace("’", "'").replace("“","\"").replace("”","\"")
    # canonicalize synonyms (multiword first)
    for k, v in {**TYPE_SYNONYMS, **SHAPE_SYNONYMS, **COLOR_SYNONYMS}.items():
        s = re.sub(rf"\b{re.escape(k)}\b", v, s)
    # keep letters, numbers, unit markers and spaces
    s = re.sub(r"[^a-z0-9\.\-\"/\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _singularize(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token

def _mm_from_size(value: float, unit: str) -> float:
    unit = unit.lower().strip('"')
    if unit in ("mm",):
        return value
    if unit in ("cm",):
        return value * 10.0
    if unit in ("in", "inch", "inches", ""):
        return value * 25.4
    return value

def _parse_sizes_mm(text: str) -> List[float]:
    out: List[float] = []
    for m in SIZE_UNITS_RE.finditer(text):
        v = float(m.group(1)); u = m.group(2)
        out.append(_mm_from_size(v, u))
    for m in FRACTION_IN_RE.finditer(text):
        num = float(m.group(1)); den = float(m.group(2))
        out.append(_mm_from_size(num/den, "in"))
    for m in METRIC_THREAD_RE.finditer(text):
        out.append(float(m.group(1)))  # treat M6 as ~6 mm
    # de-dup within ~0.1 mm
    out_sorted = []
    for v in sorted(out):
        if not out_sorted or abs(out_sorted[-1] - v) > 0.1:
            out_sorted.append(v)
    return out_sorted

def _tokens(text: str) -> List[str]:
    toks = [t for t in _normalize_text(text).split(" ") if t and t not in STOPWORDS]
    return [_singularize(t) for t in toks]

def extract_attributes_from_text(text: str) -> Dict[str, Any]:
    norm = _normalize_text(text)
    toks = _tokens(norm)
    joined = " " .join(toks)

    # types (presence based)
    types = set()
    for t in CANON_TYPES:
        if re.search(rf"\b{re.escape(t)}\b", joined):
            types.add(t)

    # materials
    materials = set()
    for m in MATERIALS:
        if re.search(rf"\b{re.escape(m)}\b", joined):
            materials.add(m)

    # colors
    colors = set()
    for c in COLORS:
        if re.search(rf"\b{re.escape(c)}\b", joined):
            colors.add(c)

    # shapes / qualifiers
    shapes = set()
    for k, canon in SHAPE_SYNONYMS.items():
        if re.search(rf"\b{re.escape(canon)}\b", joined):
            shapes.add(canon)
    qualifiers = set()
    for q in QUALIFIERS:
        if re.search(rf"\b{re.escape(q)}\b", joined):
            qualifiers.add(q)

    # pack qty
    pack_qty = None
    m = PACK_RE.search(joined)
    if m:
        pack_qty = int(m.group(1))

    # sizes
    sizes_mm = _parse_sizes_mm(joined)

    return {
        "types": types,
        "materials": materials,
        "colors": colors,
        "shapes": shapes,
        "qualifiers": qualifiers,
        "pack_qty": pack_qty,
        "sizes_mm": sizes_mm,
        "normalized": norm
    }

def extract_ebay_attributes(ebay_title: str) -> Dict[str, Any]:
    return extract_attributes_from_text(ebay_title)

def extract_product_attributes(prod: Dict[str, Any]) -> Dict[str, Any]:
    title = prod.get("title") or ""
    feats = " ".join(prod.get("features") or [])
    base = extract_attributes_from_text(title + " " + feats)

    # overlay structured fields if present
    # pack qty
    pq = prod.get("packageQuantity")
    noi = prod.get("numberOfItems")
    base["pack_qty"] = pq if isinstance(pq, int) and pq > 0 else (noi if isinstance(noi, int) and noi > 0 else base["pack_qty"])
    # color (structured)
    if prod.get("color"):
        base["colors"].add(_singularize(_normalize_text(prod["color"])))
    # add obvious type hints from group/category if missing
    if not base["types"]:
        pg = (prod.get("productGroup") or "").lower()
        if "ring" in pg: base["types"].add("ring")
        if "hook" in pg: base["types"].add("hook")
        if "clip" in pg: base["types"].add("clip")
        if "key" in pg:  base["types"].add("keychain")
    return base

def compare_attributes(
    ebay: Dict[str, Any],
    prod: Dict[str, Any],
    size_tolerance_mm: float = 1.0,
    threshold: float = 0.7
) -> Tuple[bool, float, List[str]]:
    """
    Deterministic comparison with gates + weighted score.
    Returns: (passed, score, reasons)
    """
    reasons: List[str] = []
    score = 0.0

    # ---- GATES ----
    # 1) core type overlap
    if ebay["types"] and prod["types"]:
        if ebay["types"].intersection(prod["types"]):
            score += 0.4
            reasons.append("Type overlap ✓")
        else:
            reasons.append("Type mismatch ✗")
            return False, 0.0, reasons
    # If eBay has no detectable type, skip gate (be permissive)

    # 2) pack quantity (if ebay specified)
    if ebay["pack_qty"] is not None:
        if prod["pack_qty"] is None:
            reasons.append(f"Pack qty missing on Amazon (need ≥ {ebay['pack_qty']}) ✗")
            return False, score, reasons
        if prod["pack_qty"] >= ebay["pack_qty"]:
            score += 0.2
            reasons.append(f"Pack qty OK (Amazon {prod['pack_qty']} ≥ eBay {ebay['pack_qty']}) ✓")
        else:
            reasons.append(f"Pack qty too small (Amazon {prod['pack_qty']} < eBay {ebay['pack_qty']}) ✗")
            return False, score, reasons

    # 3) size (if ebay specified)
    if ebay["sizes_mm"]:
        if prod["sizes_mm"]:
            ok = any(
                abs(se - sp) <= size_tolerance_mm
                for se in ebay["sizes_mm"] for sp in prod["sizes_mm"]
            )
            if ok:
                score += 0.2
                reasons.append("Size within tolerance ✓")
            else:
                reasons.append("Size mismatch beyond tolerance ✗")
                return False, score, reasons
        else:
            reasons.append("Amazon has no size parsed ✗")
            return False, score, reasons

    # ---- EXTRAS (contribute to score but not hard gates) ----
    # material
    if ebay["materials"] and prod["materials"]:
        if ebay["materials"].intersection(prod["materials"]):
            score += 0.2
            reasons.append("Material match ✓")
        else:
            reasons.append("Material mismatch ◦")

    # color
    if ebay["colors"] and prod["colors"]:
        if "assorted" in ebay["colors"] or "assorted" in prod["colors"] or ebay["colors"].intersection(prod["colors"]):
            score += 0.1
            reasons.append("Color compatible ✓")
        else:
            reasons.append("Color mismatch ◦")

    # shape
    if ebay["shapes"] and prod["shapes"] and ebay["shapes"].intersection(prod["shapes"]):
        score += 0.1
        reasons.append("Shape match ✓")

    # qualifiers (mild)
    if ebay["qualifiers"] and prod["qualifiers"] and ebay["qualifiers"].intersection(prod["qualifiers"]):
        score += 0.05
        reasons.append("Qualifier overlap ✓")

    passed = (score >= threshold)
    if not passed:
        reasons.append(f"Score below threshold ({score:.2f} < {threshold})")
    else:
        reasons.append(f"Passed with score {score:.2f}")
    return passed, score, reasons

# ------------------ Keepa calls ------------------

async def fetch_keepa_product(asin: str, api_key: str, domain_id: int = 1) -> Dict[str, Any]:
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing Keepa API key")

    url = "https://api.keepa.com/product"
    params = {
        "key": api_key,
        "domain": domain_id,   # 1 = amazon.com (US)
        "asin": asin,
        "stats": 90,           # include stats for current price only
        "history": 0,          # skip heavy arrays
        "buybox": 0,           # not needed for the fields you requested
    }

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params=params)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=f"Keepa error: {r.text}")
        data = r.json()

    if not data.get("products"):
        raise HTTPException(status_code=404, detail=f"No product found for ASIN {asin}")

    p = data["products"][0]
    p["asin"] = asin
    return map_keepa_required_fields(p)

async def product_finder_asins(
    title: str,
    api_key: str,
    domain_id: int = 1,
    want_n: int = 25,
) -> List[str]:
    """
    Use Keepa Product Finder to get ASINs by title, sorted by current_NEW ascending.
    POST to /query with `selection` body. Some docs note perPage minimum 50; we request 50
    then slice to the first `want_n`.
    """
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing Keepa API key")

    selection = {
        "title": title,
        "sort": [["current_NEW", "asc"]],
        "perPage": 50,   # safe minimum; we'll slice to 25 afterward
        "page": 0
    }

    url = "https://api.keepa.com/query"
    params = {"key": api_key, "domain": domain_id}
    body = {"selection": json.dumps(selection)}

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, params=params, data=body)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=f"Keepa Product Finder error: {r.text}")
        data = r.json()

    asin_list = data.get("asinList") or []
    return asin_list[:want_n]

# ---------- No-cache headers so we never reuse stale HTML ----------
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
    })

@app.post("/analyze")
async def analyze(
    ebay_title: str = Form(...),            # REQUIRED
    keepa_key: str = Form(...),             # REQUIRED
    amazon_links: str = Form("")            # OPTIONAL list
):
    lines = [ln.strip() for ln in (amazon_links or "").splitlines() if ln.strip()]
    all_checked: List[Dict[str, Any]] = []
    passed_items: List[Dict[str, Any]] = []

    # Extract eBay attributes once
    ebay_attrs = extract_ebay_attributes(ebay_title)

    # Get candidate ASINs/products either from links or Product Finder
    candidates: List[Tuple[str, str]] = []  # list of (input_label, asin)

    if lines:
        for raw in lines:
            asin = extract_asin(raw)
            candidates.append((raw, asin if asin else ""))
    else:
        try:
            asins = await product_finder_asins(title=ebay_title, api_key=keepa_key, domain_id=1, want_n=25)
            if not asins:
                rid = _put_result({
                    "error": "No matches from Product Finder for that title.",
                    "batch": [],
                    "summary": {"checked": 0, "passed": 0}
                })
                return RedirectResponse(url=f"/result?rid={rid}", status_code=303)
            for a in asins:
                candidates.append((ebay_title, a))
        except HTTPException as e:
            rid = _put_result({"error": e.detail, "batch": [], "summary": {"checked": 0, "passed": 0}})
            return RedirectResponse(url=f"/result?rid={rid}", status_code=303)

    # Fetch and filter
    for input_label, asin in candidates:
        if not asin:
            all_checked.append({"input": input_label, "asin": None, "error": "Invalid or missing ASIN", "result": None})
            continue
        try:
            product = await fetch_keepa_product(asin, keepa_key)
            # Extract product attributes and compare
            prod_attrs = extract_product_attributes(product)
            passed, score, reasons = compare_attributes(ebay_attrs, prod_attrs, size_tolerance_mm=1.0, threshold=0.7)

            row = {
                "input": input_label,
                "asin": asin,
                "error": None,
                "result": product,
                "match": {
                    "passed": passed,
                    "score": round(score, 3),
                    "reasons": reasons
                }
            }
            all_checked.append(row)
            if passed:
                passed_items.append(row)

        except HTTPException as e:
            all_checked.append({"input": input_label, "asin": asin, "error": e.detail, "result": None})

    summary = {"checked": len(all_checked), "passed": len(passed_items)}
    rid = _put_result({"error": None, "batch": passed_items, "summary": summary})
    return RedirectResponse(url=f"/result?rid={rid}", status_code=303)

@app.get("/result", response_class=HTMLResponse)
async def result(request: Request, rid: str):
    payload = _get_result(rid)
    ctx = payload or {"error": None, "batch": None, "summary": None}
    return templates.TemplateResponse("index.html", {"request": request, **ctx})
