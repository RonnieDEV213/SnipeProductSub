import pytest, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scoring import compute_phase2_scores

def test_phase2_basic_scoring():
    facet_json = {
        "results": [
            {
                "asin": "B001TEST01",
                "facets": {
                    "product_type": {"ebay": "widget", "amazon": "widget", "status": "green"},
                    "color": {"ebay": "red", "amazon": "crimson", "status": "yellow"},
                    "brand": {"ebay": "Acme", "amazon": None, "status": "red"},
                    "marketing": {"ebay": None, "amazon": "premium", "status": "gray"}
                }
            }
        ]
    }
    scores = compute_phase2_scores(facet_json, min_score_threshold=50.0, yellow_multiplier=0.5)
    assert "B001TEST01" in scores
    row = scores["B001TEST01"]
    assert 0 <= row["score_pct"] <= 100
    # product_type green weight 8 + color yellow 4*0.5 = 2 over denominator (product_type + color + brand) weights 8+4+5=17
    # brand red contributes 0. So expected raw = (8 + 2)/17 *100 ≈ 58.82
    assert abs(row["score_pct"] - 58.82) < 1.0
    assert row["passed"] is True


def test_phase2_nullish_normalization():
    facet_json = {
        "results": [
            {
                "asin": "B00NULLISH01",
                "facets": {
                    # product_type present and green
                    "product_type": {"ebay": "Lunch Box", "amazon": "Lunch Box", "status": "green"},
                    # functional_feature has placeholder ebay values that should become gray
                    "functional_feature": {"ebay": "null", "amazon": "4 Compartments", "status": "red"},
                    "compatibility": {"ebay": "N/A", "amazon": "Kids", "status": "yellow"},
                    "brand": {"ebay": "-", "amazon": "QQKO", "status": "red"},
                    "color": {"ebay": " ", "amazon": "Green", "status": "yellow"},
                }
            }
        ]
    }
    scores = compute_phase2_scores(facet_json, min_score_threshold=10.0, yellow_multiplier=0.5)
    row = scores["B00NULLISH01"]
    # Only product_type should remain present (others gray excluded)
    # Score should therefore be 100.
    assert abs(row["score_pct"] - 100.0) < 0.01
    assert row["passed"] is True
