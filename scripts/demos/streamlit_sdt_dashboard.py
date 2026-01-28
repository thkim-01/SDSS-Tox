"""Interactive SDT dashboard backed by FastAPI /analysis/sdt-tree.
Run with: streamlit run backend/streamlit_sdt_dashboard.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import requests
import streamlit as st
from graphviz import Digraph

# ---------------------------
# Data models
# ---------------------------


@dataclass
class TreeNode:
    node_id: str
    parent: Optional[str]
    is_leaf: bool
    name: str
    rule: str
    value: Optional[List[float]]
    decision: str
    is_active: bool


@dataclass
class TreePayload:
    nodes: List[TreeNode]
    active_path: List[str]


# ---------------------------
# Config
# ---------------------------
DEFAULT_API = "http://127.0.0.1:8000/analysis/sdt-tree"

# ---------------------------
# Helpers
# ---------------------------


def severity_color(prob: float) -> str:
    prob = max(0.0, min(1.0, prob))
    r = int(255 * prob)
    g = int(255 * (1 - prob))
    return f"#{r:02x}{g:02x}66"


def parse_payload(raw: Dict) -> TreePayload:
    nodes: List[TreeNode] = []
    for n in raw.get("nodes", []):
        nodes.append(
            TreeNode(
                node_id=str(n.get("id")),
                parent=(
                    str(n["parent"]) if n.get("parent") is not None else None
                ),
                is_leaf=bool(n.get("is_leaf")),
                name=str(n.get("name", "")),
                rule=str(n.get("rule", "")),
                value=n.get("value"),
                decision=str(n.get("decision", "")),
                is_active=bool(n.get("is_active")),
            )
        )
    active_path = [str(x) for x in raw.get("active_path", [])]
    return TreePayload(nodes=nodes, active_path=active_path)


def fetch_tree(api_url: str, smiles: Optional[str]) -> TreePayload:
    params = {"smiles": smiles} if smiles else {}
    resp = requests.get(api_url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(data.get("error"))
    return parse_payload(data)


def node_prob(node: TreeNode) -> float:
    if node.value and sum(node.value) > 0:
        return max(node.value) / sum(node.value)
    return 0.3 if not node.is_leaf else 0.0


def build_graph(tree: TreePayload) -> Digraph:
    dot = Digraph(engine="dot")
    dot.attr(rankdir="TB", nodesep="0.35", ranksep="0.45", splines="polyline")

    for n in tree.nodes:
        prob = node_prob(n)
        color = severity_color(prob)
        border = "3" if n.node_id in tree.active_path else "1.2"
        label = f"""<
        <b>{n.name}</b><br/>
        {n.rule}
        >"""
        dot.node(
            n.node_id,
            label=label,
            style="filled",
            fillcolor=color,
            penwidth=border,
            shape="box",
            fontsize="12",
        )

    for n in tree.nodes:
        if n.parent is None:
            continue
        dot.edge(n.parent, n.node_id, label=n.decision)

    return dot


def summarize(tree: TreePayload) -> Dict[str, str]:
    leaves = [n for n in tree.nodes if n.is_leaf]
    active_leaf = tree.active_path[-1] if tree.active_path else "N/A"
    return {
        "total_nodes": str(len(tree.nodes)),
        "leaf_nodes": str(len(leaves)),
        "active_len": str(len(tree.active_path)),
        "active_leaf": str(active_leaf),
    }


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="SDT Dashboard", layout="wide")
st.title("Semantic Decision Tree (Live)")
st.caption("백엔드 /analysis/sdt-tree 연동")

with st.sidebar:
    st.subheader("입력")
    api_url = st.text_input("API URL", value=DEFAULT_API)
    smiles = st.text_input("SMILES (선택)")
    refresh = st.button("새로고침")

if refresh:
    st.experimental_rerun()

status = st.empty()
try:
    tree = fetch_tree(api_url, smiles if smiles else None)
    dot = build_graph(tree)
    stats = summarize(tree)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 노드", stats["total_nodes"])
    c2.metric("리프", stats["leaf_nodes"])
    c3.metric("Active Path", stats["active_len"])
    c4.metric("Active Leaf", stats["active_leaf"])

    st.graphviz_chart(dot, use_container_width=True)

    with st.expander("경로/설명", expanded=True):
        active_display = tree.active_path if tree.active_path else "없음"
        st.write(f"Active path: {active_display}")
except Exception as exc:  # noqa: BLE001
    status.error(f"트리를 불러오지 못했습니다: {exc}")
    if st.button("다시 시도"):
        st.experimental_rerun()
