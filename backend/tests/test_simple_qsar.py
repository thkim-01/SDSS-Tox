import json
import os
import subprocess
import sys

import pytest

from app.services.simple_qsar import SimpleQSAR, RDKitNotAvailable
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def rdkit_available():
    try:
        import rdkit  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not rdkit_available(), reason="RDKit not available")
def test_compute_descriptors_smiles():
    qsar = SimpleQSAR()
    desc = qsar.compute_descriptors("CCO")
    assert isinstance(desc, dict)
    assert "MW" in desc and desc["MW"] > 0
    assert "logP" in desc


def test_predict_from_descriptors():
    qsar = SimpleQSAR()
    descriptors = {"MW": 180.16, "logP": 0.89, "TPSA": 63.6}
    out = qsar.predict(descriptors)
    assert set(["score", "confidence", "label", "details"]).issubset(out.keys())
    assert 0.0 <= out["score"] <= 1.0


def test_qsar_cli_integration(tmp_path):
    # Ensure CLI script exists
    script = os.path.join(os.path.dirname(__file__), "..", "..", "src", "python", "qsar_cli.py")
    script = os.path.abspath(script)
    if not os.path.exists(script):
        pytest.skip("qsar_cli.py not present")

    # Write SMILES file
    sf = tmp_path / "smiles.txt"
    sf.write_text("CCO\n")

    proc = subprocess.run([sys.executable, script, str(sf)], capture_output=True, text=True)
    assert proc.returncode == 0
    out = proc.stdout.strip().splitlines()
    assert any(line.startswith("QSAR:") for line in out)
    # Parse JSON payload
    for line in out:
        if line.startswith("QSAR:"):
            _, payload = line.split("|", 1)
            parsed = json.loads(payload)
            assert "descriptors" in parsed and "prediction" in parsed


def test_qsar_endpoint_from_descriptors():
    payload = {"descriptors": {"MW": 180.16, "logP": 0.89, "TPSA": 63.6}}
    r = client.post("/qsar/predict", json=payload)
    assert r.status_code == 200
    j = r.json()
    assert "prediction" in j and "descriptors" in j
