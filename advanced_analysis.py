import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from sklearn.linear_model import LinearRegression, RANSACRegressor
import sys

def main():
    # Load dataset
    try:
        df = pd.read_csv("data/esol/esol.csv")
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 1. RDKit Processing
    print("Calculating molecular properties...")
    
    properties = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        smiles = row.get('smiles') # Adjust column name if needed
        if not isinstance(smiles, str):
            continue
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                props = {
                    'MW': Descriptors.MolWt(mol),
                    'LogP': Descriptors.MolLogP(mol),
                    'H-Donors': Descriptors.NumHDonors(mol),
                    'H-Acceptors': Descriptors.NumHAcceptors(mol),
                    'TPSA': Descriptors.TPSA(mol),
                    'Rotatable Bonds': Descriptors.NumRotatableBonds(mol),
                    'Heavy Atom Count': Descriptors.HeavyAtomCount(mol),
                    'Ring Count': Descriptors.RingCount(mol)
                }
                properties.append(props)
                valid_indices.append(idx)
            else:
                print(f"Invalid SMILES at index {idx}: {smiles}")
        except Exception as e:
            print(f"Error processing SMILES at index {idx}: {e}")

    if not properties:
        print("No valid molecules found.")
        return

    # Create DataFrame from properties
    prop_df = pd.DataFrame(properties)
    # Merge with original data if needed, but for now we just use calculated props
    
    # 2. Regression Analysis (X=LogP, Y=MW)
    X = prop_df[['LogP']].values
    y = prop_df['MW'].values
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred_lr = lr.predict(X)
    
    # RANSAC Regressor
    ransac = RANSACRegressor()
    ransac.fit(X, y)
    y_pred_ransac = ransac.predict(X)
    
    print("Regression analysis complete.")
    
    # 3. Visualization (Plotly)
    print("Generating plot...")
    
    fig = px.scatter(
        prop_df, 
        x='LogP', 
        y='MW', 
        color='TPSA',
        title='Molecular Weight vs LogP (Colored by TPSA)',
        labels={'LogP': 'LogP', 'MW': 'Molecular Weight (MW)'},
        hover_data=['H-Donors', 'H-Acceptors', 'Ring Count']
    )
    
    # Add Linear Regression Line
    # Sort X for line plotting
    sort_idx = np.argsort(X.flatten())
    X_sorted = X[sort_idx]
    
    fig.add_trace(go.Scatter(
        x=X_sorted.flatten(), 
        y=lr.predict(X_sorted), 
        mode='lines', 
        name='Linear Regression',
        line=dict(color='red', dash='dash')
    ))
    
    # Add RANSAC Regression Line
    fig.add_trace(go.Scatter(
        x=X_sorted.flatten(), 
        y=ransac.predict(X_sorted), 
        mode='lines', 
        name='RANSAC Regressor',
        line=dict(color='green', width=2)
    ))
    
    fig.update_layout(
        template='plotly_white',
        width=1000,
        height=800
    )
    
    fig.show()
    print("Plot generated and opened in browser.")

if __name__ == "__main__":
    main()
