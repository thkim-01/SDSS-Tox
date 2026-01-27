"""
High-Fidelity Semantic Decision Tree Visualization
===================================================
A Streamlit + Graphviz application for Explainable AI (XAI) in cheminformatics.

Features:
- Multi-Descriptor Node Architecture (Decision Clusters)
- Sankey-style Edge Flow (Variable Width)
- Leaf Node Distribution Bars
- Interactive Depth Control
- Path Highlighter

Author: DTO-DSS Research Team
Date: 2026-01-26
"""

import streamlit as st
import graphviz
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
import colorsys

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="HF-SDT Viewer | DTO-DSS",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Semantic concept mappings for chemical descriptors
SEMANTIC_CONCEPTS = {
    "logP": ("Lipophilicity Barrier", "Membrane permeability indicator"),
    "logKow": ("Partition Coefficient", "Hydrophobicity measure"),
    "MW": ("Molecular Size Gate", "Bioavailability constraint"),
    "HBD": ("H-Bond Donor Check", "Solubility factor"),
    "HBA": ("H-Bond Acceptor Check", "Absorption potential"),
    "TPSA": ("Polar Surface Analysis", "Oral bioavailability"),
    "nRotB": ("Flexibility Assessment", "Conformational entropy"),
    "Aromatic_Rings": ("Aromatic Character", "Metabolic stability"),
    "Heteroatom_Count": ("Heteroatom Richness", "Reactivity potential"),
    "Heavy_Atom_Count": ("Complexity Score", "Drug-likeness metric"),
    "Ring_Count": ("Ring System Analysis", "Structural rigidity"),
    "Fraction_CSP3": ("Saturation Index", "3D complexity"),
    "MR": ("Molar Refractivity", "Polarizability measure"),
    "Formal_Charge": ("Charge State", "Ionization status"),
    "Num_Stereocenters": ("Stereochemistry", "Chiral complexity"),
}

# Correlated feature groups (features that often co-occur in decisions)
CORRELATED_FEATURES = {
    "logP": ["logKow", "TPSA", "Aromatic_Rings"],
    "MW": ["Heavy_Atom_Count", "nRotB", "Ring_Count"],
    "HBD": ["HBA", "TPSA", "Heteroatom_Count"],
    "HBA": ["HBD", "Heteroatom_Count", "MW"],
    "TPSA": ["HBD", "HBA", "logP"],
    "nRotB": ["MW", "Fraction_CSP3", "Heavy_Atom_Count"],
    "Aromatic_Rings": ["logP", "Ring_Count", "Heavy_Atom_Count"],
    "Heteroatom_Count": ["HBA", "HBD", "TPSA"],
    "Heavy_Atom_Count": ["MW", "Ring_Count", "nRotB"],
    "Ring_Count": ["Aromatic_Rings", "Heavy_Atom_Count", "MW"],
    "Fraction_CSP3": ["nRotB", "Aromatic_Rings", "Ring_Count"],
    "MR": ["MW", "Heavy_Atom_Count", "Aromatic_Rings"],
    "logKow": ["logP", "TPSA", "MW"],
    "Formal_Charge": ["HBD", "HBA", "TPSA"],
    "Num_Stereocenters": ["Fraction_CSP3", "Ring_Count", "nRotB"],
}

# Color palette
COLORS = {
    "decision_node": "#4A5568",      # Slate gray
    "decision_border": "#2D3748",    # Dark slate
    "safe_leaf": "#38A169",          # Green
    "toxic_leaf": "#E53E3E",         # Red
    "uncertain_leaf": "#D69E2E",     # Amber
    "highlight": "#3182CE",          # Blue
    "edge_default": "#A0AEC0",       # Light gray
    "edge_highlight": "#2B6CB0",     # Dark blue
    "text_primary": "#1A202C",
    "text_secondary": "#718096",
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TreeNode:
    """Represents a node in the semantic decision tree."""
    node_id: int
    is_leaf: bool
    depth: int
    n_samples: int
    n_samples_ratio: float  # Percentage of total samples
    
    # For decision nodes
    feature_name: Optional[str] = None
    threshold: Optional[float] = None
    semantic_concept: Optional[str] = None
    semantic_description: Optional[str] = None
    correlated_features: List[str] = field(default_factory=list)
    impurity: float = 0.0
    
    # For leaf nodes
    class_distribution: Optional[Dict[str, float]] = None
    predicted_class: Optional[str] = None
    confidence: float = 0.0
    
    # Tree structure
    left_child: Optional['TreeNode'] = None
    right_child: Optional['TreeNode'] = None
    parent_id: Optional[int] = None


# ============================================================================
# DATA GENERATOR
# ============================================================================

def generate_complex_chemical_dataset(n_samples: int = 2000, random_state: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate a complex synthetic chemical dataset with multiple descriptors.
    This simulates realistic molecular property distributions.
    """
    np.random.seed(random_state)
    
    # Generate correlated features to simulate real chemical data
    data = {}
    
    # Base molecular properties (realistic ranges)
    data["MW"] = np.random.lognormal(mean=5.5, sigma=0.4, size=n_samples)  # 100-800 Da
    data["MW"] = np.clip(data["MW"], 50, 1000)
    
    data["logP"] = np.random.normal(loc=2.5, scale=1.5, size=n_samples)  # -2 to 7
    data["logP"] = np.clip(data["logP"], -2, 8)
    
    # Correlated with MW
    data["Heavy_Atom_Count"] = (data["MW"] / 14 + np.random.normal(0, 3, n_samples)).astype(int)
    data["Heavy_Atom_Count"] = np.clip(data["Heavy_Atom_Count"], 5, 70)
    
    # H-bond properties (correlated with polarity)
    data["HBD"] = np.random.poisson(lam=2, size=n_samples)
    data["HBD"] = np.clip(data["HBD"], 0, 10)
    
    data["HBA"] = data["HBD"] + np.random.poisson(lam=3, size=n_samples)
    data["HBA"] = np.clip(data["HBA"], 0, 15)
    
    # TPSA correlates with H-bonding
    data["TPSA"] = 20 * data["HBD"] + 10 * data["HBA"] + np.random.normal(0, 15, n_samples)
    data["TPSA"] = np.clip(data["TPSA"], 0, 250)
    
    # Flexibility
    data["nRotB"] = (data["Heavy_Atom_Count"] * 0.15 + np.random.normal(0, 2, n_samples)).astype(int)
    data["nRotB"] = np.clip(data["nRotB"], 0, 20)
    
    # Aromatic character (anti-correlated with sp3 fraction)
    data["Aromatic_Rings"] = np.random.poisson(lam=1.5, size=n_samples)
    data["Aromatic_Rings"] = np.clip(data["Aromatic_Rings"], 0, 6)
    
    data["Ring_Count"] = data["Aromatic_Rings"] + np.random.poisson(lam=0.5, size=n_samples)
    data["Ring_Count"] = np.clip(data["Ring_Count"], 0, 8)
    
    # Heteroatoms
    data["Heteroatom_Count"] = data["HBD"] + data["HBA"] + np.random.poisson(lam=1, size=n_samples)
    data["Heteroatom_Count"] = np.clip(data["Heteroatom_Count"], 1, 20)
    
    # SP3 fraction
    data["Fraction_CSP3"] = 1 - (data["Aromatic_Rings"] * 6 / data["Heavy_Atom_Count"])
    data["Fraction_CSP3"] = np.clip(data["Fraction_CSP3"], 0, 1)
    
    # Additional descriptors
    data["logKow"] = data["logP"] + np.random.normal(0, 0.3, n_samples)
    data["MR"] = data["MW"] * 0.3 + np.random.normal(0, 10, n_samples)
    data["Formal_Charge"] = np.random.choice([-1, 0, 0, 0, 1], size=n_samples)
    data["Num_Stereocenters"] = np.random.poisson(lam=1, size=n_samples)
    
    df = pd.DataFrame(data)
    
    # Generate labels based on realistic toxicity rules
    # (Multi-factor decision - not just one feature)
    toxicity_score = np.zeros(n_samples)
    
    # Rule 1: High lipophilicity + low solubility = toxic
    toxicity_score += (df["logP"] > 4) * 0.3
    toxicity_score += (df["TPSA"] < 40) * 0.2
    
    # Rule 2: Reactive heteroatoms
    toxicity_score += (df["Heteroatom_Count"] > 8) * 0.15
    
    # Rule 3: Molecular weight extremes
    toxicity_score += ((df["MW"] < 150) | (df["MW"] > 600)) * 0.15
    
    # Rule 4: High aromatic character
    toxicity_score += (df["Aromatic_Rings"] > 3) * 0.2
    
    # Rule 5: Low flexibility (rigid = harder to metabolize)
    toxicity_score += (df["Fraction_CSP3"] < 0.25) * 0.1
    
    # Add noise
    toxicity_score += np.random.normal(0, 0.15, n_samples)
    
    # Binary classification
    labels = (toxicity_score > 0.4).astype(int)
    
    return df, labels


# ============================================================================
# TREE EXTRACTION
# ============================================================================

def extract_semantic_tree(
    clf: DecisionTreeClassifier,
    feature_names: List[str],
    X: np.ndarray,
    max_depth: Optional[int] = None
) -> TreeNode:
    """
    Extract a semantic tree structure from a sklearn DecisionTreeClassifier.
    Enriches nodes with semantic concepts and correlated features.
    """
    tree_ = clf.tree_
    total_samples = tree_.n_node_samples[0]
    
    class_names = {0: "Safe", 1: "Toxic"}
    
    def build_node(node_id: int, depth: int, parent_id: Optional[int] = None) -> Optional[TreeNode]:
        if max_depth is not None and depth > max_depth:
            # Create a pseudo-leaf at max depth
            value = tree_.value[node_id][0]
            total = np.sum(value)
            class_dist = {class_names[i]: round(v / total * 100, 1) for i, v in enumerate(value)}
            pred_class = class_names[np.argmax(value)]
            confidence = np.max(value) / total
            
            return TreeNode(
                node_id=node_id,
                is_leaf=True,
                depth=depth,
                n_samples=int(tree_.n_node_samples[node_id]),
                n_samples_ratio=round(tree_.n_node_samples[node_id] / total_samples * 100, 1),
                class_distribution=class_dist,
                predicted_class=pred_class,
                confidence=round(confidence, 3),
                parent_id=parent_id
            )
        
        is_leaf = tree_.children_left[node_id] == -1
        n_samples = int(tree_.n_node_samples[node_id])
        n_samples_ratio = round(n_samples / total_samples * 100, 1)
        
        if is_leaf:
            # Leaf node
            value = tree_.value[node_id][0]
            total = np.sum(value)
            class_dist = {class_names[i]: round(v / total * 100, 1) for i, v in enumerate(value)}
            pred_class = class_names[np.argmax(value)]
            confidence = np.max(value) / total
            
            return TreeNode(
                node_id=node_id,
                is_leaf=True,
                depth=depth,
                n_samples=n_samples,
                n_samples_ratio=n_samples_ratio,
                class_distribution=class_dist,
                predicted_class=pred_class,
                confidence=round(confidence, 3),
                parent_id=parent_id
            )
        else:
            # Decision node
            feature_idx = tree_.feature[node_id]
            threshold = tree_.threshold[node_id]
            impurity = tree_.impurity[node_id]
            
            if feature_idx < len(feature_names):
                feature_name = feature_names[feature_idx]
            else:
                feature_name = f"Feature_{feature_idx}"
            
            # Get semantic concept
            concept_info = SEMANTIC_CONCEPTS.get(feature_name, (feature_name, ""))
            semantic_concept = concept_info[0]
            semantic_description = concept_info[1]
            
            # Get correlated features
            correlated = CORRELATED_FEATURES.get(feature_name, [])[:2]
            
            node = TreeNode(
                node_id=node_id,
                is_leaf=False,
                depth=depth,
                n_samples=n_samples,
                n_samples_ratio=n_samples_ratio,
                feature_name=feature_name,
                threshold=round(threshold, 2),
                semantic_concept=semantic_concept,
                semantic_description=semantic_description,
                correlated_features=correlated,
                impurity=round(impurity, 4),
                parent_id=parent_id
            )
            
            # Recursively build children
            left_id = tree_.children_left[node_id]
            right_id = tree_.children_right[node_id]
            
            node.left_child = build_node(left_id, depth + 1, node_id)
            node.right_child = build_node(right_id, depth + 1, node_id)
            
            return node
    
    return build_node(0, 0)


# ============================================================================
# GRAPHVIZ RENDERING
# ============================================================================

def get_leaf_color(node: TreeNode) -> str:
    """Get color for leaf node based on prediction and confidence."""
    if node.predicted_class == "Toxic":
        if node.confidence > 0.8:
            return COLORS["toxic_leaf"]
        else:
            return "#FC8181"  # Lighter red
    else:
        if node.confidence > 0.8:
            return COLORS["safe_leaf"]
        else:
            return "#68D391"  # Lighter green


def get_edge_penwidth(n_samples: int, total_samples: int) -> float:
    """Calculate edge width based on sample flow (Sankey-style)."""
    ratio = n_samples / total_samples
    # Scale from 1.0 to 8.0
    return max(1.0, min(8.0, 1.0 + ratio * 12))


def create_distribution_bar(class_dist: Dict[str, float]) -> str:
    """Create an ASCII-style distribution bar for leaf nodes."""
    safe_pct = class_dist.get("Safe", 0)
    toxic_pct = class_dist.get("Toxic", 0)
    
    # Create visual bar (using Unicode blocks)
    bar_width = 20
    safe_blocks = int(safe_pct / 100 * bar_width)
    toxic_blocks = bar_width - safe_blocks
    
    bar = "█" * safe_blocks + "░" * toxic_blocks
    
    return f"[{bar}]\\nSafe: {safe_pct:.0f}% | Toxic: {toxic_pct:.0f}%"


def build_graphviz_tree(
    root: TreeNode,
    highlight_path: Optional[List[int]] = None,
    total_samples: int = 1000
) -> graphviz.Digraph:
    """
    Build a Graphviz digraph from the semantic tree.
    Implements all visualization requirements.
    """
    dot = graphviz.Digraph(
        comment="High-Fidelity Semantic Decision Tree",
        format="svg"
    )
    
    # Graph attributes
    dot.attr(
        rankdir="LR",  # Left-to-right layout
        bgcolor="#F7FAFC",
        fontname="Helvetica Neue",
        fontsize="11",
        nodesep="0.5",
        ranksep="1.2",
        splines="ortho"  # Orthogonal edges
    )
    
    # Default node attributes
    dot.attr("node",
        shape="box",
        style="filled,rounded",
        fontname="Helvetica Neue",
        fontsize="10",
        margin="0.2,0.15"
    )
    
    # Default edge attributes
    dot.attr("edge",
        fontname="Helvetica Neue",
        fontsize="9",
        color=COLORS["edge_default"]
    )
    
    highlight_set = set(highlight_path) if highlight_path else set()
    
    def add_node(node: TreeNode):
        if node is None:
            return
        
        node_id = str(node.node_id)
        is_highlighted = node.node_id in highlight_set
        
        if node.is_leaf:
            # === LEAF NODE ===
            color = get_leaf_color(node)
            font_color = "white" if node.confidence > 0.7 else COLORS["text_primary"]
            
            # Build label with distribution bar
            dist_bar = create_distribution_bar(node.class_distribution)
            
            label = f"""<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">
                <TR><TD><B><FONT POINT-SIZE="12" COLOR="{font_color}">{node.predicted_class}</FONT></B></TD></TR>
                <TR><TD><FONT POINT-SIZE="9" COLOR="{font_color}">Confidence: {node.confidence:.1%}</FONT></TD></TR>
                <TR><TD><FONT POINT-SIZE="8" COLOR="{font_color}">━━━━━━━━━━━━━━</FONT></TD></TR>
                <TR><TD><FONT POINT-SIZE="8" COLOR="{font_color}">Safe: {node.class_distribution.get('Safe', 0):.0f}% | Toxic: {node.class_distribution.get('Toxic', 0):.0f}%</FONT></TD></TR>
                <TR><TD><FONT POINT-SIZE="8" COLOR="{font_color}">N = {node.n_samples:,} ({node.n_samples_ratio:.1f}%)</FONT></TD></TR>
            </TABLE>>"""
            
            border_color = COLORS["highlight"] if is_highlighted else color
            border_width = "3" if is_highlighted else "1"
            
            dot.node(
                node_id,
                label=label,
                fillcolor=color,
                color=border_color,
                penwidth=border_width
            )
        else:
            # === DECISION NODE ===
            # Multi-descriptor node with semantic concept
            corr_text = ", ".join(node.correlated_features) if node.correlated_features else "—"
            
            label = f"""<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">
                <TR><TD><B><FONT POINT-SIZE="11" COLOR="{COLORS['text_primary']}">{node.semantic_concept}</FONT></B></TD></TR>
                <TR><TD><FONT POINT-SIZE="10" COLOR="{COLORS['text_primary']}">{node.feature_name} ≤ {node.threshold}</FONT></TD></TR>
                <TR><TD><FONT POINT-SIZE="8" COLOR="{COLORS['text_secondary']}">+ {corr_text}</FONT></TD></TR>
                <TR><TD><FONT POINT-SIZE="7" COLOR="{COLORS['text_secondary']}">━━━━━━━━━━━━━━</FONT></TD></TR>
                <TR><TD><FONT POINT-SIZE="8" COLOR="{COLORS['text_secondary']}">N = {node.n_samples:,} ({node.n_samples_ratio:.1f}%)</FONT></TD></TR>
                <TR><TD><FONT POINT-SIZE="7" COLOR="{COLORS['text_secondary']}">Gini: {node.impurity:.3f}</FONT></TD></TR>
            </TABLE>>"""
            
            # Color based on depth (lighter as we go deeper)
            depth_factor = min(node.depth / 8, 1.0)
            base_color = (74, 85, 104)  # Base slate
            fill_color = tuple(int(c + (247 - c) * depth_factor * 0.5) for c in base_color)
            fill_hex = "#{:02x}{:02x}{:02x}".format(*fill_color)
            
            border_color = COLORS["highlight"] if is_highlighted else COLORS["decision_border"]
            border_width = "3" if is_highlighted else "1"
            
            dot.node(
                node_id,
                label=label,
                fillcolor=fill_hex,
                color=border_color,
                penwidth=border_width
            )
            
            # Add edges to children
            if node.left_child:
                left_id = str(node.left_child.node_id)
                left_samples = node.left_child.n_samples
                edge_width = get_edge_penwidth(left_samples, total_samples)
                
                is_edge_highlighted = (node.node_id in highlight_set and 
                                       node.left_child.node_id in highlight_set)
                edge_color = COLORS["edge_highlight"] if is_edge_highlighted else COLORS["edge_default"]
                
                # Left edge = "Yes" (≤ threshold)
                edge_label = f"≤\\nN={left_samples:,}"
                
                dot.edge(
                    node_id, left_id,
                    label=edge_label,
                    penwidth=str(edge_width),
                    color=edge_color,
                    fontcolor=edge_color
                )
                add_node(node.left_child)
            
            if node.right_child:
                right_id = str(node.right_child.node_id)
                right_samples = node.right_child.n_samples
                edge_width = get_edge_penwidth(right_samples, total_samples)
                
                is_edge_highlighted = (node.node_id in highlight_set and 
                                       node.right_child.node_id in highlight_set)
                edge_color = COLORS["edge_highlight"] if is_edge_highlighted else COLORS["edge_default"]
                
                # Right edge = "No" (> threshold)
                edge_label = f">\\nN={right_samples:,}"
                
                dot.edge(
                    node_id, right_id,
                    label=edge_label,
                    penwidth=str(edge_width),
                    color=edge_color,
                    fontcolor=edge_color
                )
                add_node(node.right_child)
    
    add_node(root)
    return dot


def collect_leaf_nodes(node: TreeNode) -> List[Tuple[int, str, float, int]]:
    """Collect all leaf nodes for the path highlighter dropdown."""
    leaves = []
    
    def traverse(n: TreeNode, path: List[int]):
        if n is None:
            return
        current_path = path + [n.node_id]
        
        if n.is_leaf:
            leaves.append((
                n.node_id,
                f"{n.predicted_class} (Conf: {n.confidence:.1%}, N={n.n_samples})",
                n.confidence,
                current_path
            ))
        else:
            traverse(n.left_child, current_path)
            traverse(n.right_child, current_path)
    
    traverse(node, [])
    return leaves


def collect_path_to_node(root: TreeNode, target_id: int) -> List[int]:
    """Find the path from root to a specific node."""
    def find_path(node: TreeNode, target: int, path: List[int]) -> Optional[List[int]]:
        if node is None:
            return None
        
        current_path = path + [node.node_id]
        
        if node.node_id == target:
            return current_path
        
        left_result = find_path(node.left_child, target, current_path)
        if left_result:
            return left_result
        
        right_result = find_path(node.right_child, target, current_path)
        if right_result:
            return right_result
        
        return None
    
    return find_path(root, target_id, []) or []


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1A202C;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #718096;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
    }
    .info-box {
        background: #EBF8FF;
        border-left: 4px solid #3182CE;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="main-header">🌳 High-Fidelity Semantic Decision Tree</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explainable AI for Molecular Toxicity Prediction</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Tree Controls")
        
        st.subheader("1. Data Configuration")
        n_samples = st.slider(
            "Training Samples",
            min_value=500,
            max_value=5000,
            value=2000,
            step=500,
            help="Number of synthetic chemical compounds to generate"
        )
        
        random_state = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=9999,
            value=42,
            help="For reproducibility"
        )
        
        st.markdown("---")
        
        st.subheader("2. Tree Depth Control")
        tree_depth = st.slider(
            "Max Visualization Depth",
            min_value=2,
            max_value=12,
            value=5,
            help="Higher depth = more detailed but complex tree"
        )
        
        training_depth = st.slider(
            "Training Max Depth",
            min_value=3,
            max_value=15,
            value=8,
            help="Actual tree training depth (can be higher than viz depth)"
        )
        
        min_samples_leaf = st.slider(
            "Min Samples per Leaf",
            min_value=5,
            max_value=100,
            value=20,
            help="Prevents overfitting to rare cases"
        )
        
        st.markdown("---")
        
        st.subheader("3. Path Highlighter")
        highlight_enabled = st.checkbox("Enable Path Highlighting", value=False)
    
    # Generate data and train tree
    with st.spinner("Generating complex chemical dataset..."):
        df, labels = generate_complex_chemical_dataset(n_samples, random_state)
        feature_names = list(df.columns)
        X = df.values
    
    # Train decision tree
    clf = DecisionTreeClassifier(
        max_depth=training_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        class_weight="balanced"
    )
    clf.fit(X, labels)
    
    # Extract semantic tree
    semantic_tree = extract_semantic_tree(
        clf, feature_names, X, max_depth=tree_depth
    )
    
    # Collect leaves for path highlighter
    leaves = collect_leaf_nodes(semantic_tree)
    
    # Path selection in sidebar
    selected_path = None
    if highlight_enabled:
        with st.sidebar:
            leaf_options = {f"Node {l[0]}: {l[1]}": l[3] for l in leaves}
            selected_leaf = st.selectbox(
                "Select Leaf to Highlight",
                options=list(leaf_options.keys()),
                help="Highlights the decision path from root to selected leaf"
            )
            if selected_leaf:
                selected_path = leaf_options[selected_leaf]
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Nodes",
            f"{clf.tree_.node_count}",
            delta=f"Depth: {clf.get_depth()}"
        )
    
    with col2:
        n_leaves = clf.get_n_leaves()
        st.metric(
            "Leaf Nodes",
            f"{n_leaves}",
            delta=f"{n_leaves / clf.tree_.node_count * 100:.0f}% of tree"
        )
    
    with col3:
        accuracy = clf.score(X, labels)
        st.metric(
            "Training Accuracy",
            f"{accuracy:.1%}",
            delta="On training set"
        )
    
    with col4:
        toxic_ratio = labels.mean()
        st.metric(
            "Toxic Ratio",
            f"{toxic_ratio:.1%}",
            delta=f"{int(toxic_ratio * n_samples)} compounds"
        )
    
    st.markdown("---")
    
    # Info box
    st.markdown("""
    <div class="info-box">
        <strong>📊 How to Read This Tree:</strong><br>
        <ul style="margin-bottom: 0;">
            <li><strong>Node Header</strong>: Abstract semantic concept (e.g., "Lipophilicity Barrier")</li>
            <li><strong>Primary Rule</strong>: The mathematical split condition</li>
            <li><strong>+ Correlated</strong>: Co-occurring descriptors considered in this region</li>
            <li><strong>Edge Width</strong>: Proportional to sample flow (thicker = more common pattern)</li>
            <li><strong>Leaf Colors</strong>: Green = Safe, Red = Toxic (intensity = confidence)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Build and render the tree
    with st.spinner("Rendering decision tree..."):
        dot = build_graphviz_tree(
            semantic_tree,
            highlight_path=selected_path,
            total_samples=n_samples
        )
    
    # Display the tree
    st.subheader("🌲 Semantic Decision Tree Visualization")
    st.graphviz_chart(dot, use_container_width=True)
    
    # Feature importance section
    st.markdown("---")
    st.subheader("📈 Feature Importance Analysis")
    
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": clf.feature_importances_
    }).sort_values("Importance", ascending=False).head(10)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.bar_chart(importance_df.set_index("Feature")["Importance"])
    
    with col2:
        st.dataframe(
            importance_df.style.format({"Importance": "{:.3f}"}),
            use_container_width=True
        )
    
    # Dataset preview
    with st.expander("📋 View Generated Dataset Sample"):
        st.dataframe(df.head(20).style.format("{:.2f}"), use_container_width=True)
        
        st.markdown("**Label Distribution:**")
        label_dist = pd.Series(labels).value_counts()
        st.write(f"- Safe (0): {label_dist.get(0, 0)} compounds")
        st.write(f"- Toxic (1): {label_dist.get(1, 0)} compounds")


if __name__ == "__main__":
    main()
