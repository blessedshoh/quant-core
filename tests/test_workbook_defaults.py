"""The fixed income lab's coursework defaults must carry the same workbook numbers as bondlab's
golden file. Both live in sibling repositories; the check is skipped when they are absent."""

import csv
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = ROOT / "site" / "bonds" / "src" / "data" / "basics-defaults.json"
GOLDEN = ROOT / "bondlab" / "tests" / "golden" / "workbook_basics.csv"

# Workbook cells that bondlab checks in its test files rather than the golden CSV, with the
# values those tests assert (test_textbook.py, test_kwf_tree.py, test_convertible.py).
FROM_TESTS = {
    "FixedIncomeDeriv_ All_Models.xlsm, Yield Change!H23": 41.0911155553,
    "FixedIncomeDeriv_ All_Models.xlsm, Yield Change!H30": 100.0,
    "FixedIncomeDeriv_ All_Models.xlsm, Binomial Tree!H18": 99.8150009406,
    "FixedIncomeDeriv_ All_Models.xlsm, Convertible Bond!H4": 1504.0110866921,
    "FixedIncomeDeriv_ All_Models.xlsm, Convertible Bond!C20": 0.3527900730,
    "FI_final_ShohruhM.xlsx, Q1!C21": 982.2212369016,
}

pytestmark = pytest.mark.skipif(
    not (DEFAULTS.exists() and GOLDEN.exists()), reason="site and bondlab are not checked out beside quant-core"
)


def _cell(text):
    match = re.match(r"^(.*?, [^!]+![A-Z]+\d+)", text)
    assert match, text
    return match.group(1)


def _entries():
    data = json.loads(DEFAULTS.read_text(encoding="utf-8"))
    for model, entry in data.items():
        for w in entry["workbook"]:
            yield model, w
        for preset in entry.get("presets", []):
            for w in preset["workbook"]:
                yield model, w


def test_every_workbook_number_on_the_site_matches_bondlab():
    golden = {}
    for row in csv.DictReader(GOLDEN.read_text(encoding="utf-8").splitlines()):
        golden.setdefault(_cell(row["provenance"]), []).append(row)
    checked = 0
    for model, w in _entries():
        cell = _cell(w["cell"])
        if cell in golden:
            # Spot rates are decimals in the golden file and percent on the site.
            scale = 100.0 if model == "bootstrap" else 1.0
            expected = [float(r["expected"]) * scale for r in golden[cell]]
            assert any(v == pytest.approx(w["value"], abs=1e-9) for v in expected), (cell, w["value"], expected)
        else:
            assert cell in FROM_TESTS, f"{cell} is neither in the golden file nor in bondlab's tests"
            assert w["value"] == pytest.approx(FROM_TESTS[cell], abs=1e-9), cell
        checked += 1
    assert checked == 23
