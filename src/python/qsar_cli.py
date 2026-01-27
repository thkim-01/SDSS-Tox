"""
Simple QSAR CLI used by the JavaFX GUI for local descriptor calculation.
Reads SMILES lines from a file and prints lines of the form:

QSAR:<SMILES>|<JSON>

where JSON contains {"descriptors": {...}, "prediction": {...}}

This is intentionally lightweight and only uses RDKit directly.
"""
import json
import sys
import os

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
except Exception as e:
    RDKit_AVAILABLE = False
else:
    RDKit_AVAILABLE = True


def compute_descriptors(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    desc = {
        "MW": float(Descriptors.MolWt(mol)),
        "logKow": float(Descriptors.MolLogP(mol)),
        "HBD": int(rdMolDescriptors.CalcNumHBD(mol)),
        "HBA": int(rdMolDescriptors.CalcNumHBA(mol)),
        "nRotB": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "Aromatic_Rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "Heteroatom_Count": int(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (6, 1))),
        "Heavy_Atom_Count": int(mol.GetNumHeavyAtoms()),
        "logP": float(Descriptors.MolLogP(mol)),
    }
    return desc


def predict_from_descriptors(desc: dict):
    mw = desc.get("MW", 0.0)
    logp = desc.get("logP", desc.get("logKow", 0.0))
    tpsa = desc.get("TPSA", 0.0)

    mw_score = min(max((mw - 100.0) / 500.0, 0.0), 1.0)
    logp_score = min(max((logp + 2.0) / 7.0, 0.0), 1.0)
    tpsa_score = 1.0 - min(max(tpsa / 200.0, 0.0), 1.0)

    score = 0.45 * mw_score + 0.35 * logp_score + 0.20 * tpsa_score
    score = float(min(max(score, 0.0), 1.0))

    conf = 0.0
    conf += 1.0 if 100.0 <= mw <= 500.0 else 0.0
    conf += 1.0 if -2.0 <= logp <= 5.0 else 0.0
    conf += 1.0 if 0.0 <= tpsa <= 140.0 else 0.0
    confidence = float(conf / 3.0)

    label = "Toxic" if score > 0.6 else ("Safe" if score < 0.3 else "Moderate")

    return {"score": score, "confidence": confidence, "label": label}


def main():
    if len(sys.argv) < 2:
        print("Usage: python qsar_cli.py <smiles_file>")
        sys.exit(1)

    if not RDKit_AVAILABLE:
        print("ERROR: RDKit not available")
        sys.exit(2)

    smiles_file = sys.argv[1]
    with open(smiles_file, 'r') as f:
        smiles_list = [ln.strip() for ln in f if ln.strip()]

    for s in smiles_list:
        try:
            desc = compute_descriptors(s)
            pred = predict_from_descriptors(desc)
            payload = {"descriptors": desc, "prediction": pred}
            print(f"QSAR:{s}|{json.dumps(payload, ensure_ascii=False)}")
        except Exception as e:
            print(f"QSAR:{s}|{{\"error\":\"{str(e)}\"}}")


if __name__ == '__main__':
    main()
