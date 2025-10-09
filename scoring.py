from typing import Any, Dict

# Extracted scoring logic so tests can import without initializing FastAPI app

FACET_WEIGHTS = {
    "product_type": 8,
    "functional_feature": 8,
    "compatibility": 6,
    "material": 5,
    "brand": 5,
    "color": 4,
    "quantity": 4,
    "vague_descriptor": 2,
    "marketing": 1,
}

STATUS_MULTIPLIERS = {
    "green": 1.0,
    "yellow": 0.5,
    "red": 0.0,
}

def compute_phase2_scores(
    facet_json: Dict[str, Any],
    min_score_threshold: float,
    yellow_multiplier: float = 0.5,
    override_brand_weight: int | None = None,
) -> Dict[str, Dict[str, Any]]:
    results = facet_json.get("results") if isinstance(facet_json, dict) else None
    if not isinstance(results, list):
        return {}

    DEBUG_ADJUST = False
    NULLISH_STRINGS = {"", "null", "none", "n/a", "na", "n.a.", "-", "—", "–", "_", "n\u2014", "n\u2013"}

    def _is_nullish(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, (int, float)):
            return False
        if isinstance(v, str):
            s = v.strip().lower()
            return s in NULLISH_STRINGS
        return False

    def adjust_facets(facets: Dict[str, Any]):
        if not isinstance(facets, dict) or not facets:
            return
        for fk, fv in list(facets.items()):
            if not isinstance(fv, dict):
                continue
            ebay_v = fv.get("ebay")
            if _is_nullish(ebay_v):
                old = fv.get("status")
                if old != "gray":
                    fv["status"] = "gray"
                    if DEBUG_ADJUST:
                        print(f"[adjust_facets] -> gray facet={fk} ebay={ebay_v!r} amazon={fv.get('amazon')!r} old_status={old}")
                fv["ebay"] = None

    out: Dict[str, Dict[str, Any]] = {}
    for row in results:
        asin = row.get("asin")
        facets = row.get("facets") or {}
        if not asin:
            continue
        adjust_facets(facets)
        present = [k for k,v in facets.items() if isinstance(v, dict) and v.get("ebay") not in (None, "")]
        # Dynamic brand weight override (e.g., when brand stripped from search)
        def eff_weight(facet_key: str) -> int:
            if facet_key == "brand" and override_brand_weight is not None:
                return override_brand_weight
            return FACET_WEIGHTS.get(facet_key, 0)

        weight_sum = sum(eff_weight(k) for k in present) or 1.0
        score_acc = 0.0
        for k in present:
            status = (facets.get(k) or {}).get("status")
            if status == "yellow":
                mult = max(0.0, min(1.0, float(yellow_multiplier)))
            else:
                mult = STATUS_MULTIPLIERS.get(status, 0.0 if status == "red" else 0.0)
            w = eff_weight(k)
            score_acc += w * mult
        norm_score = (score_acc / weight_sum) * 100.0
        passed = norm_score >= min_score_threshold
        out[asin] = {
            "score_pct": round(norm_score, 2),
            "passed": passed,
            "facets": facets,
        }
    return out
