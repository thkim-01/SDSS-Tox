import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.linear_model import LinearRegression, RANSACRegressor

# Mock request data
results = [
    {"smiles": "CCO", "molecule_name": "Ethanol", "combined_score": 0.1},
    {"smiles": "c1ccccc1", "molecule_name": "Benzene", "combined_score": 0.2},
    {"smiles": "CC(=O)O", "molecule_name": "Acetic Acid", "combined_score": 0.3},
    {"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "molecule_name": "Caffeine", "combined_score": 0.4},
    {"smiles": "C1CCCCC1", "molecule_name": "Cyclohexane", "combined_score": 0.5}
]
x_descriptor = "LogP"
y_descriptor = "MW"

# Prepare data for plotting
data = []
for i, r in enumerate(results):
    item = {
        'index': i,
        'smiles': r.get('smiles', ''),
        'molecule_name': r.get('molecule_name', 'Unknown'),
        'ml_prediction': r.get('ml_prediction', 0),
        'ontology_score': r.get('ontology_score', 0),
        'combined_score': r.get('combined_score', 0),
        'confidence': r.get('confidence', 0),
        'risk_level': r.get('risk_level', 'Unknown')
    }
    
    # Calculate additional properties if needed
    smiles = item['smiles']
    if smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                item['LogP'] = Descriptors.MolLogP(mol)
                item['MW'] = Descriptors.MolWt(mol)
                item['TPSA'] = Descriptors.TPSA(mol)
        except:
            pass
    
    data.append(item)

df = pd.DataFrame(data)

# Get X and Y columns
x_col = x_descriptor
y_col = y_descriptor

print(f"Initial x_col: {x_col}, y_col: {y_col}")
print(f"Columns: {df.columns.tolist()}")

if x_col not in df.columns:
    print(f"x_col {x_col} not in columns, resetting")
    x_col = 'combined_score'
if y_col not in df.columns:
    print(f"y_col {y_col} not in columns, resetting")
    y_col = 'combined_score'
    
print(f"Final x_col: {x_col}, y_col: {y_col}")

# Determine color column
color_col = 'risk_level'
if 'TPSA' in df.columns and (x_col == 'LogP' or y_col == 'MW'):
    color_col = 'TPSA'

# Create scatter plot
fig = px.scatter(
    df,
    x=x_col,
    y=y_col,
    color=color_col,
    title=f'Analysis: {x_col} vs {y_col}'
)

# Add Regression Lines if LogP vs MW
if x_col == 'LogP' and y_col == 'MW' and len(df) > 1:
    print("Entering regression block")
    # Drop NaNs for regression
    reg_df = df[[x_col, y_col]].dropna()
    print(f"reg_df length: {len(reg_df)}")
    
    if len(reg_df) > 1:
        X = reg_df[[x_col]].values
        y = reg_df[y_col].values
        
        # Linear Regression
        lr = LinearRegression()
        lr.fit(X, y)
        print("Linear Regression fitted")
        
        # RANSAC Regressor
        try:
            ransac = RANSACRegressor()
            ransac.fit(X, y)
            has_ransac = True
            print("RANSAC fitted")
        except:
            has_ransac = False
        
        # Sort for plotting
        sort_idx = np.argsort(X.flatten())
        X_sorted = X[sort_idx]
        
        fig.add_trace(go.Scatter(
            x=X_sorted.flatten(), 
            y=lr.predict(X_sorted), 
            mode='lines', 
            name='Linear Regression',
            line=dict(color='red', dash='dash')
        ))
        
        if has_ransac:
            fig.add_trace(go.Scatter(
                x=X_sorted.flatten(),
                y=ransac.predict(X_sorted), 
                mode='lines', 
                name='RANSAC Regressor',
                line=dict(color='green', width=2)
            ))

html_content = fig.to_html(full_html=True)
if "Linear Regression" in html_content:
    print("SUCCESS: Linear Regression found in HTML")
else:
    print("FAILURE: Linear Regression NOT found in HTML")
