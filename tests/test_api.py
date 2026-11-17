"""The HTTP API: shape of the answer, agreement with the engine, input limits and browser access rules."""

import importlib

import numpy as np
import pytest

pytest.importorskip("fastapi", reason='install with: pip install -e ".[api]"')
from fastapi.testclient import TestClient  # noqa: E402

import quantcore as qc  # noqa: E402
from api import main  # noqa: E402

client = TestClient(main.app)

LONG_CALL = {
    "spot": 166.0,
    "rate": 0.04,
    "legs": [{"kind": "call", "quantity": 10, "strike": 180, "days": 30, "vol": 0.64}],
}
ENGINE_LONG_CALL = qc.Position(spot=166.0, rate=0.04, legs=[qc.Leg("call", 10, strike=180, days=30, vol=0.64)])


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "engine": qc.__version__}


def test_grid_has_the_promised_shape():
    body = client.post("/options/grid", json=LONG_CALL).json()
    v, d, p = body["profit"]["shape"]

    assert (v, d, p) == (len(body["axes"]["vol_scales"]), len(body["axes"]["days"]), len(body["axes"]["prices"]))
    assert len(body["profit"]["values"]) == v * d * p
    for name in ("delta", "gamma", "theta_per_day", "vega_per_point"):
        assert np.shape(body["greeks"][name]) == (v, d)


def test_grid_values_are_the_engine_values():
    body = client.post("/options/grid", json=LONG_CALL).json()
    cube = np.array(body["profit"]["values"]).reshape(body["profit"]["shape"])
    engine = qc.profit_grid(ENGINE_LONG_CALL)

    np.testing.assert_allclose(cube, engine.profit, atol=0.006)
    np.testing.assert_allclose(body["axes"]["prices"], engine.prices, atol=1e-4)
    np.testing.assert_allclose(body["axes"]["days"], engine.days, atol=1e-6)


def test_summary_matches_the_engine():
    summary = client.post("/options/grid", json=LONG_CALL).json()["summary"]
    cost = qc.entry_cost(ENGINE_LONG_CALL)

    assert summary["entry_cost"] == pytest.approx(cost, abs=0.01)
    assert summary["first_expiry_days"] == 30
    assert summary["breakevens"] == pytest.approx(qc.breakevens(ENGINE_LONG_CALL), abs=1e-3)
    assert summary["max_profit"] is None  # unlimited
    assert summary["max_loss"] == pytest.approx(-cost, abs=0.01)
    assert summary["legs"][0]["entry_price"] == pytest.approx(cost / 1000, abs=1e-4)


def test_unlimited_loss_is_reported_as_null():
    short_call = {"spot": 166.0, "legs": [{"kind": "call", "quantity": -1, "strike": 180, "days": 30, "vol": 0.64}]}
    summary = client.post("/options/grid", json=short_call).json()["summary"]
    assert summary["max_loss"] is None
    assert summary["max_profit"] > 0


def test_the_range_setting_widens_the_price_axis():
    narrow = client.post("/options/grid", json={**LONG_CALL, "range_pct": 0.10}).json()["axes"]["prices"]
    wide = client.post("/options/grid", json={**LONG_CALL, "range_pct": 0.50}).json()["axes"]["prices"]
    assert narrow[0] == pytest.approx(166 * 0.9) and narrow[-1] == pytest.approx(166 * 1.1)
    assert wide[0] == pytest.approx(166 * 0.5) and wide[-1] == pytest.approx(166 * 1.5)


@pytest.mark.parametrize(
    "bad",
    [
        {"spot": -5, "legs": LONG_CALL["legs"]},
        {"spot": 166, "legs": []},
        {"spot": 166, "legs": LONG_CALL["legs"] * 9},
        {"spot": 166, "legs": [{"kind": "call", "quantity": 1, "days": 30, "vol": 0.5}]},
        {"spot": 166, "legs": [{"kind": "call", "quantity": 0, "strike": 180, "days": 30, "vol": 0.5}]},
        {"spot": 166, "legs": [{"kind": "swap", "quantity": 1}]},
        {"spot": 166, "legs": [{"kind": "call", "quantity": 1, "strike": 180, "days": 30, "vol": 9.0}]},
        {"spot": 166, "legs": LONG_CALL["legs"], "range_pct": 5},
    ],
)
def test_bad_requests_are_refused(bad):
    assert client.post("/options/grid", json=bad).status_code == 422


def test_large_answers_are_compressed():
    response = client.post("/options/grid", json=LONG_CALL, headers={"accept-encoding": "gzip"})
    assert response.headers["content-encoding"] == "gzip"


def test_only_the_site_may_call_from_a_browser():
    def allowed(origin):
        response = client.get("/health", headers={"origin": origin})
        return response.headers.get("access-control-allow-origin") == origin

    assert allowed("https://options.blessedshoh.dev")
    assert allowed("https://blessedshoh.dev")
    assert not allowed("https://evil.example")
    assert not allowed("https://blessedshoh.dev.evil.example")
    assert not allowed("http://localhost:4321")


def test_local_development_origins_need_an_explicit_switch(monkeypatch):
    monkeypatch.setenv("QUANTCORE_DEV_ORIGINS", "1")
    dev_client = TestClient(importlib.reload(main).app)
    try:
        response = dev_client.get("/health", headers={"origin": "http://localhost:4321"})
        assert response.headers.get("access-control-allow-origin") == "http://localhost:4321"
    finally:
        monkeypatch.delenv("QUANTCORE_DEV_ORIGINS")
        importlib.reload(main)


# --- risk ---------------------------------------------------------------------


def test_risk_matches_the_engine():
    from quantcore import probability, scenarios

    body = client.post("/options/risk", json=LONG_CALL).json()
    assert body["chance_of_profit"] == pytest.approx(probability.chance_of_profit(ENGINE_LONG_CALL), abs=1e-6)
    assert body["expected_profit"] == pytest.approx(probability.expected_profit(ENGINE_LONG_CALL), abs=0.01)
    assert body["chance_of_profit_with_drift"] is None
    np.testing.assert_allclose(body["scenarios"]["profit"], scenarios.scenario_grid(ENGINE_LONG_CALL).profit, atol=0.006)

    dist = body["distribution"]
    assert dist["days"] == 30 and len(dist["prices"]) == len(dist["density"])
    inside = np.trapezoid(dist["density"], dist["prices"])
    assert inside + dist["below_range"] + dist["above_range"] == pytest.approx(1.0, abs=2e-3)


def test_risk_with_a_view_and_a_horizon():
    body = client.post("/options/risk", json={**LONG_CALL, "drift": 0.4, "horizon_days": 10}).json()
    assert body["chance_of_profit_with_drift"] > body["chance_of_profit"]
    assert body["scenarios"]["horizon_days"] == 10
    assert body["scenarios"]["profit"][2][4] < 0  # ten days of decay, nothing else moved


def test_risk_for_stock_only_has_no_option_numbers():
    body = client.post("/options/risk", json={"spot": 50, "legs": [{"kind": "stock", "quantity": 100}]}).json()
    assert body["chance_of_profit"] is None and body["distribution"] is None
    assert body["scenarios"]["profit"][2][8] == pytest.approx(1000.0)  # +20% on $5,000 of stock


@pytest.mark.parametrize("bad", [{"drift": 3}, {"horizon_days": -1}, {"legs": []}])
def test_risk_refuses_bad_requests(bad):
    assert client.post("/options/risk", json={**LONG_CALL, **bad}).status_code == 422


# --- explain ------------------------------------------------------------------


def test_explain_matches_the_engine_and_adds_up():
    from quantcore import scenarios

    body = client.post(
        "/options/explain", json={**LONG_CALL, "price_move_pct": 0.1, "days_passed": 7, "vol_shift": -0.05}
    ).json()
    engine = scenarios.explain(ENGINE_LONG_CALL, 16.6, 7, -0.05)
    assert body["price_move"] == pytest.approx(16.6)
    for name in ("delta", "gamma", "theta", "vega", "unexplained", "actual"):
        assert body[name] == pytest.approx(getattr(engine, name), abs=0.006)
    total = body["delta"] + body["gamma"] + body["theta"] + body["vega"] + body["unexplained"]
    assert total == pytest.approx(body["actual"], abs=0.03)


@pytest.mark.parametrize("bad", [{"price_move_pct": -1}, {"vol_shift": 2}, {"days_passed": -3}])
def test_explain_refuses_bad_requests(bad):
    assert client.post("/options/explain", json={**LONG_CALL, **bad}).status_code == 422


# --- American exercise and the binomial explorer ---------------------------------

PUT = {"spot": 166.0, "rate": 0.04, "legs": [{"kind": "put", "quantity": 1, "strike": 200, "days": 90, "vol": 0.5}]}
ONE_OPTION = {"spot": 100, "strike": 105, "days": 182.5, "vol": 0.3, "rate": 0.05}


def test_american_exercise_raises_the_value_of_a_deep_put():
    european = client.post("/options/grid", json=PUT).json()["summary"]["entry_cost"]
    american = client.post("/options/grid", json={**PUT, "exercise": "american"}).json()
    assert american["summary"]["entry_cost"] > european
    assert american["profit"]["shape"][0] == 9  # the coarser American grid


def test_tree_route():
    body = client.post("/options/tree", json={**ONE_OPTION, "kind": "put", "steps": 4}).json()
    assert len(body["prices"]) == 5 and len(body["prices"][4]) == 5
    assert body["american_value"] > body["european_value"]
    assert body["exercise"][4][0] is True  # the lowest price at expiry: the put pays


def test_convergence_route():
    body = client.post("/options/convergence", json={**ONE_OPTION, "kind": "call", "american": False, "max_steps": 40}).json()
    assert body["steps"] == list(range(1, 41))
    assert abs(body["leisen_reimer"][-1] - body["black_scholes"]) < 0.001
    assert body["reference"] == pytest.approx(body["black_scholes"], abs=1e-4)


@pytest.mark.parametrize("bad", [{"steps": 40}, {"days": 0}, {"kind": "straddle"}])
def test_tree_refuses_bad_requests(bad):
    assert client.post("/options/tree", json={**ONE_OPTION, **bad}).status_code == 422
