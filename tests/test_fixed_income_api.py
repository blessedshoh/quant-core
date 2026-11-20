"""The /fixed-income routes: bondlab's lab service behind the existing API."""

import json

import pytest

pytest.importorskip("fastapi", reason='install with: pip install -e ".[api]"')
pytest.importorskip("bondlab", reason="install bondlab: pip install -e ../bondlab")
from fastapi.testclient import TestClient  # noqa: E402

from api import main  # noqa: E402
from bondlab.ui.service import handle  # noqa: E402

client = TestClient(main.app)

CURVE = {
    "day": "2026-11-18",
    "scheme": "MonotoneConvex",
    "yields": {"1 Mo": 4.10, "1.5 Month": 4.24, "2 Mo": 4.27, "3 Mo": 4.34, "4 Mo": 4.35, "6 Mo": 4.43, "1 Yr": 4.62,
               "2 Yr": 4.80, "3 Yr": 4.84, "5 Yr": 4.89, "7 Yr": 4.96, "10 Yr": 5.07, "20 Yr": 5.44, "30 Yr": 5.41},
}
BOND = {"curve": CURVE, "bond": {"preset": "UST", "coupon": 4.0, "maturity": "2036-11-15"}, "mode": "curve", "spreadBp": 0}

CASES = {
    "treasury-curve": CURVE,
    "bond": BOND,
    "hedge": {"curve": CURVE, "position": BOND["bond"], "faceAmount": 1_000_000, "mode": "dv01", "hedgeTenors": ["10 Yr"]},
    "basics": {"model": "price", "face": 100, "coupon": 0, "years": 20, "frequency": 1, "ytm": 10},
    "rate-tree": {"volatility": 10, "lowestRates": [3.5, 4.07361919503, 3.24627558493], "face": 100, "coupon": 4,
                  "years": 3, "option": "call", "strike": 100, "firstExerciseYear": 1},
    "convertible": {"face": 1000, "coupon": 10, "conversionRatio": 15, "years": 3, "couponOnConversion": True,
                    "stockPrice": 100, "volatility": 20, "rate": 5, "dividendYield": 9},
}


@pytest.mark.parametrize("route", list(CASES))
def test_route_answers_with_the_engine_values(route):
    response = client.post(f"/fixed-income/{route}", json=CASES[route])
    assert response.status_code == 200, response.text
    # Exactly the engine's answer, after the same JSON round trip the HTTP response makes.
    assert response.json() == json.loads(json.dumps(dict(handle(route, CASES[route]))))


def test_workbook_numbers_come_through():
    assert client.post("/fixed-income/basics", json=CASES["basics"]).json()["price"] == pytest.approx(14.8643628024, abs=1e-8)
    assert client.post("/fixed-income/rate-tree", json=CASES["rate-tree"]).json()["value"] == pytest.approx(99.8150009406, abs=1e-9)
    assert client.post("/fixed-income/convertible", json=CASES["convertible"]).json()["value"] == pytest.approx(1504.0110866921, abs=1e-8)


def test_unknown_route_is_404():
    assert client.post("/fixed-income/uzbekistan", json={}).status_code == 404
    assert client.post("/fixed-income/nope", json={}).status_code == 404


def test_bad_payload_is_422_with_the_engine_message():
    response = client.post("/fixed-income/bond", json={**BOND, "bond": {"preset": "UST", "coupon": 4.0, "maturity": "2066-11-15"}})
    assert response.status_code == 422
    assert "after the curve's last pillar" in response.json()["detail"]
    response = client.post("/fixed-income/basics", json={"model": "nope"})
    assert response.status_code == 422
    assert "known" in response.json()["detail"]


def test_body_must_be_an_object():
    response = client.post("/fixed-income/basics", json=[1, 2, 3])
    assert response.status_code == 422


def test_oversized_body_is_413():
    big = {"model": "price", "padding": "x" * 210_000}
    response = client.post("/fixed-income/basics", json=big)
    assert response.status_code == 413


def test_cors_allows_the_bonds_site():
    response = client.options(
        "/fixed-income/bond",
        headers={"Origin": "https://bonds.blessedshoh.dev", "Access-Control-Request-Method": "POST",
                 "Access-Control-Request-Headers": "content-type"},
    )
    assert response.headers.get("access-control-allow-origin") == "https://bonds.blessedshoh.dev"


def test_a_declared_oversized_body_is_refused_before_it_is_read():
    response = client.post(
        "/fixed-income/bond", content=b"{}", headers={"content-type": "application/json", "content-length": "5000000"}
    )
    assert response.status_code == 413


def test_inputs_the_model_cannot_answer_are_422_with_cors_not_500():
    # A bond a week from maturity at 90: no yield brackets it. Before the fix this escaped as a
    # server error without CORS headers, which the site could only call "unreachable".
    payload = {**BOND, "bond": {**BOND["bond"], "maturity": "2026-11-25"}, "mode": "price", "cleanPrice": 90}
    payload.pop("spreadBp")
    response = client.post("/fixed-income/bond", json=payload, headers={"origin": "https://bonds.blessedshoh.dev"})
    assert response.status_code == 422
    assert response.headers.get("access-control-allow-origin") == "https://bonds.blessedshoh.dev"


def test_oversized_work_is_refused_quickly():
    response = client.post("/fixed-income/convertible", json={**CASES["convertible"], "years": 3000})
    assert response.status_code == 422
    assert "years" in response.json()["detail"]
