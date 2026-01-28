import subprocess
import os
import argparse
import sys
import json

def run_java_backend(command, query, dto_path, jar_path):
    """
    Call Java backend and parse JSON response.
    command: 'stats', 'search', or 'targets'
    query: search term or disease name
    dto_path: path to dto.rdf
    jar_path: path to the built JAR file
    """
    print(f"[Java] Calling: {command} with query: '{query}'")
    
    if not os.path.exists(jar_path):
        print(f"Error: Java JAR not found at {jar_path}")
        print("Please build the project using 'mvn package'")
        return None

    try:
        cmd = ["java", "-jar", jar_path, dto_path, command]
        if query:
            cmd.append(query)
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"[Java] Error: {result.stderr}")
            return None
        
        # Parse JSON from stdout
        output_lines = result.stdout.strip().split('\n')
        # Find JSON object (skip any log lines)
        json_str = ""
        brace_count = 0
        in_json = False
        for line in output_lines:
            if '{' in line:
                in_json = True
            if in_json:
                json_str += line + "\n"
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0:
                    break
        
        if json_str:
            response = json.loads(json_str)
            return response
        else:
            print("[Java] No JSON output found")
            return None
            
    except subprocess.TimeoutExpired:
        print("[Java] Timeout: Ontology loading took too long")
        return None
    except json.JSONDecodeError as e:
        print(f"[Java] JSON parse error: {e}")
        print(f"Raw output: {result.stdout[:500]}")
        return None
    except Exception as e:
        print(f"[Java] Error: {e}")
        return None

def run_python_prediction(targets, smiles_list=None):
    """
    Run prediction for molecules targeting the given targets.
    """
    print(f"\n[Python] Running predictions for {len(targets)} targets...")
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "python"))
    from data_loader import DataLoader
    from model import MoleculeModel
    
    # Load data and train model (in production, load pre-trained model)
    data_path = os.path.join(os.path.dirname(__file__), "data")
    loader = DataLoader(data_path)
    
    try:
        X, y, tasks = loader.get_data('tox21')
    except Exception as e:
        print(f"[Python] Error loading data: {e}")
        return None
    
    # Train a quick model
    model = MoleculeModel(n_estimators=50)
    model.train(X, y, task_name='NR-AhR')
    
    # Predict for sample molecules
    if smiles_list is None:
        # Use some example molecules
        smiles_list = [
            "CCO",  # Ethanol
            "C=CC(=O)OC",  # Methyl acrylate
            "c1ccccc1",  # Benzene
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        ]
    
    print("\n[Python] Predictions:")
    print("-" * 60)
    results = []
    for smiles in smiles_list:
        pred = model.predict(smiles)
        if pred is not None:
            prob = pred[0]
            risk = "High" if prob > 0.5 else "Low"
            print(f"  SMILES: {smiles[:30]:<30} | Toxicity: {prob:.3f} ({risk})")
            results.append({"smiles": smiles, "toxicity_prob": prob, "risk": risk})
        else:
            print(f"  SMILES: {smiles[:30]:<30} | Error: Invalid molecule")
    print("-" * 60)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='In Silico DSS - Drug Target Decision Support System')
    parser.add_argument('--query', type=str, default='', help='Disease or Target query (e.g., "Breast Cancer")')
    parser.add_argument('--dto', type=str, default='data/ontology/dto.rdf', help='Path to DTO file')
    parser.add_argument('--jar', type=str, default='target/sdss-tox-1.0-SNAPSHOT.jar', help='Path to Java JAR')
    parser.add_argument('--command', type=str, default='targets', choices=['stats', 'search', 'targets'], help='Java backend command')
    parser.add_argument('--predict', action='store_true', help='Run Python predictions')
    args = parser.parse_args()

    print("=" * 60)
    print("       In Silico DSS - Drug Target Decision Support")
    print("=" * 60)
    
    # Step 1: Query Java backend for targets
    response = run_java_backend(args.command, args.query, args.dto, args.jar)
    
    if response and response.get('status') == 'success':
        data = response.get('data', [])
        
        if args.command == 'stats':
            print("\n[Ontology Statistics]")
            for key, value in data.items():
                print(f"  {key}: {value}")
        else:
            print(f"\n[Found {response.get('count', 0)} results]")
            if isinstance(data, list):
                for i, item in enumerate(data[:10]):  # Show top 10
                    label = item.get('label', 'N/A')
                    iri = item.get('iri', 'N/A')
                    print(f"  {i+1}. {label}")
                    print(f"     IRI: {iri}")
                if len(data) > 10:
                    print(f"  ... and {len(data) - 10} more")
        
        # Step 2: Run predictions if requested
        if args.predict:
            run_python_prediction(data)
    else:
        print("\n[Java backend returned no results or error]")
        if response:
            print(f"  Message: {response.get('message', 'Unknown error')}")
        
        # Still run predictions with sample data
        if args.predict:
            run_python_prediction([])

    print("\n" + "=" * 60)
    print("                        Done")
    print("=" * 60)

if __name__ == '__main__':
    main()
