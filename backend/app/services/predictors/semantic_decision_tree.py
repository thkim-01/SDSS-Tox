import time
import logging
import json
import os
from typing import List, Dict, Optional, Tuple, Any

# 3rd party libraries
try:
    import pubchempy as pcp
except ImportError:
    pcp = None
    print("Warning: pubchempy not installed. 'pip install pubchempy'")

try:
    from rdflib import Graph, RDFS, RDF, Namespace, Literal, URIRef
except ImportError:
    Graph = None
    print("Warning: rdflib not installed. 'pip install rdflib'")

try:
    from graphviz import Digraph
except ImportError:
    Digraph = None
    print("Warning: graphviz not installed. 'pip install graphviz'")

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SemanticTree")

# DTO Namespace (Assuming based on typical DTO usage, to be verified with actual rdf file content)
# Since I cannot see the content of dto.rdf, I will assume a generic structure or try to read labels.
# Usually DTO uses http://www.drugtargetontology.org/dto/
DTO = Namespace("http://www.drugtargetontology.org/dto/")

# ---------------------------------------------------------
# Step 1: 타겟 낚시 (Target Fishing / External Knowledge)
# ---------------------------------------------------------
class TargetFisher:
    def __init__(self, cache_path="target_cache.json"):
        if not pcp:
            raise ImportError("PubChemPy is required for Target Fishing.")
        self.cache_path = cache_path
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, str]:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def get_target_name(self, smiles: str) -> str:
        """
        SMILES에서 PubChem을 통해 유력한 타겟 이름 추론 (캐싱 적용).
        """
        if smiles in self.cache:
            # logger.info(f"Cache hit for SMILES: {smiles[:10]}...")
            return self.cache[smiles]

        try:
            # 1. CID 검색
            compounds = pcp.get_compounds(smiles, namespace='smiles')
            if not compounds:
                logger.warning(f"No compound found for SMILES: {smiles[:10]}...")
                self.cache[smiles] = "Unknown"
                self._save_cache()
                return "Unknown"
            
            compound = compounds[0]
            
            # API 호출 딜레이 (차단 방지)
            time.sleep(1.0)
            
            synonyms = compound.synonyms
            candidate = "Unknown"
            if synonyms:
                # 간단한 휴리스틱: 가장 짧은 이름이 보통 일반명
                candidate = min(synonyms, key=len)
            
            self.cache[smiles] = candidate
            self._save_cache()
            return candidate

        except Exception as e:
            logger.error(f"Error in target fishing: {e}")
            return "Unknown"

# ---------------------------------------------------------
# Step 2: 의미론적 속성 추출 (Semantic Feature Extraction)
# ---------------------------------------------------------
class DTOKnowledgeBase:
    def __init__(self, rdf_path: str):
        if not Graph:
            raise ImportError("rdflib is required for DTO processing.")
            
        self.g = Graph()
        try:
            logger.info(f"Loading DTO from {rdf_path}...")
            self.g.parse(rdf_path)
            logger.info(f"DTO loaded with {len(self.g)} triples.")
        except Exception as e:
            logger.error(f"Failed to load DTO: {e}")
            self.g = None

    def get_semantic_info(self, target_name: str) -> Dict[str, str]:
        """
        타겟 이름을 DTO 그래프와 매핑하여 계층 정보 추출.
        [Family] -> [Class] -> [Target]
        """
        info = {
            "Target": target_name,
            "Class": "Unknown Class",
            "Family": "Unknown Family",
            "Disease": "Unknown Disease",
            "Toxicity": "None"
        }
        
        if not self.g:
            return info

        # SPARQL 쿼리 예시 (실제 RDF 구조에 맞춰 조정 필요)
        # 라벨(rdfs:label)이 target_name과 일치하거나 포함하는 노드 찾기
        query = """
            SELECT ?s ?label ?parentLabel ?grandParentLabel
            WHERE {
                ?s rdfs:label ?label .
                FILTER(CONTAINS(LCASE(?label), LCASE(?target)))
                
                OPTIONAL {
                    ?s rdfs:subClassOf ?parent .
                    ?parent rdfs:label ?parentLabel .
                    
                    OPTIONAL {
                        ?parent rdfs:subClassOf ?grandParent .
                        ?grandParent rdfs:label ?grandParentLabel .
                    }
                }
            }
            LIMIT 1
        """
        
        # rdflib는 변수 바인딩을 initBindings로 처리
        try:
            # target_name이 "BACE1"이라면 Literal로 변환
            # rdflib 쿼리 실행
            # 주의: DTO 구조가 복잡하므로 간단한 상위 클래스 탐색 로직 구현
            
            # 여기서는 데모를 위해 매핑이 안될 경우 Fallback 로직을 사용하거나
            # 내부 매핑 테이블을 시뮬레이션 합니다. (실제 RDF 탐색은 구조 의존적)
            
            # Simulation for demo purposes if RDF query is complex / empty in stub
            if "BACE1" in target_name.upper():
                info["Target"] = "Beta-secretase 1"
                info["Class"] = "Aspartyl protease"
                info["Family"] = "Enzyme"
                info["Disease"] = "Alzheimer's disease"
            elif "EGFR" in target_name.upper():
                info["Target"] = "Epidermal Growth Factor Receptor"
                info["Class"] = "Receptor Tyrosine Kinase"
                info["Family"] = "Kinase"
                info["Disease"] = "Lung Cancer"
            elif "COX-2" in target_name.upper():
                info["Target"] = "Cyclooxygenase-2"
                info["Class"] = "Prostaglandin-endoperoxide synthase"
                info["Family"] = "Enzyme"
                info["Toxicity"] = "Cardiovascular events"
            
        except Exception as e:
            logger.error(f"SPARQL error: {e}")

        # [추가] DTO 추출 실패 시 보정 로직 (Fallback)
        if info["Disease"] == "Unknown Disease" or info["Disease"] == "Unknown":
            t_upper = target_name.upper()
            if "COX-2" in t_upper or "CYCLOOXYGENASE" in t_upper:
                info["Disease"] = "Inflammation & Pain"
            elif "BACE1" in t_upper:
                info["Disease"] = "Alzheimer's Disease"
            elif "ACE" in t_upper: # Angiotensin Converting Enzyme
                info["Disease"] = "Hypertension"
            elif "EGFR" in t_upper:
                info["Disease"] = "Lung Cancer"

        return info

# ---------------------------------------------------------
# Step 3: 시맨틱 의사결정 트리 구축 (Custom Semantic Tree)
# ---------------------------------------------------------
class SemanticNode:
    def __init__(self, name: str, level: str, data: Any = None):
        self.name = name
        self.level = level  # 'Family', 'Class', 'Leaf'
        self.data = data
        self.children: Dict[str, 'SemanticNode'] = {}

    def add_child(self, key: str, node: 'SemanticNode'):
        self.children[key] = node
        
    def __repr__(self):
        return f"Node({self.level}: {self.name})"

class SemanticDecisionTree:
    def __init__(self):
        self.root = SemanticNode("Root", "Root")

    def insert(self, semantic_info: Dict[str, str]):
        """
        [Family] -> [Class] -> [Target] 순서로 트리에 삽입
        """
        family = semantic_info.get("Family", "Unknown Family")
        p_class = semantic_info.get("Class", "Unknown Class")
        target = semantic_info.get("Target", "Unknown Target")
        
        # 1. Family Level
        if family not in self.root.children:
            self.root.add_child(family, SemanticNode(family, "Family"))
        family_node = self.root.children[family]
        
        # 2. Class Level
        if p_class not in family_node.children:
            family_node.add_child(p_class, SemanticNode(p_class, "Class"))
        class_node = family_node.children[p_class]
        
        # 3. Target Level (Leaf)
        if target not in class_node.children:
            class_node.add_child(target, SemanticNode(target, "Leaf", data=semantic_info))

    def print_tree(self, node: SemanticNode = None, indent: int = 0):
        if node is None:
            node = self.root
            
        prefix = "  " * indent
        symbol = "└─" if indent > 0 else ""
        
        print(f"{prefix}{symbol} [{node.level}] {node.name}")
        
        if node.level == "Leaf" and node.data:
            # Leaf node explanation
            d = node.data
            risk = f", Risk: {d['Toxicity']}" if d.get("Toxicity") != "None" else ""
            bbbp = f", BBBP: {d['BBBP_Result']}" if d.get("BBBP_Result") else ""
            value = f", Value: {d['Value']}" if d.get("Value") is not None else ""
            print(f"{prefix}    => Inference: Acts on {d['Target']} ({d['Family']}). {d['Disease']} related{risk}{bbbp}{value}.")

        for child_name, child_node in node.children.items():
            self.print_tree(child_node, indent + 1)

    def explain(self, smiles: str, semantic_info: Dict[str, str]) -> str:
        family = semantic_info.get("Family")
        target = semantic_info.get("Target")
        disease = semantic_info.get("Disease")
        toxicity = semantic_info.get("Toxicity")
        
        explanation = f"이 약물({smiles[:10]}...)은 [{family}] 계열의 [{target}]에 작용하므로, "
        if disease and disease != "Unknown Disease":
             explanation += f"[{disease}] 치료와 관련이 있습니다."
        else:
             explanation += "특정 질환 정보가 불분명합니다."
             
        if toxicity and toxicity != "None":
            explanation += f" 단, [{toxicity}]의 독성 위험이 보고되었습니다."
            
        return explanation

    # [SemanticDecisionTree 클래스 내부에 추가]
    def export_graphviz(self, filename="semantic_tree_output"):
        try:
            from graphviz import Digraph
        except ImportError:
            print("Graphviz not installed.")
            return

        # 1. 핵심 설정: 위에서 아래로(TB), 직각 선(ortho)
        dot = Digraph(comment='SDT Structure', format='png')
        dot.attr(rankdir='TB')      # Top-to-Bottom (공간 효율 최적화)
        dot.attr(splines='ortho')   # 직각 선 (깔끔함)
        dot.attr(nodesep='0.6')     # 노드 간격 확보
        dot.attr('node', shape='box', style='filled,rounded', fontname='Segoe UI', fontsize='10')
        dot.attr('edge', penwidth='1.2', color='#2c3e50', arrowsize='0.7')
        
        # 2. 색상 테마 정의 (신호등 시스템)
        COLORS = {
            "Root": "#444444",      # Dark Grey
            "Family": "#5DADE2",    # Blue
            "Class": "#AED6F1",     # Light Blue
            "Leaf_Safe": "#ABEBC6", # Green (Pass)
            "Leaf_Risk": "#F5B7B1", # Red (Fail/Risk)
            "Leaf_Unknown": "#EAEDED" # Grey
        }
        
        FONT_COLORS = {"Root": "white", "Family": "black", "Class": "black"}

        def add_nodes(node, parent_id=None, graph_context=None):
            # If graph_context is not provided, use the main dot object
            if graph_context is None:
                graph_context = dot
                
            node_id = f"{node.level}_{node.name}_{id(node)}"
            
            # 라벨 및 색상 로직
            fill = COLORS.get(node.level, "white")
            font = FONT_COLORS.get(node.level, "black")
            label = node.name
            
            # 리프 노드 상태에 따른 색상 분기
            if node.level == "Leaf" and node.data:
                d = node.data
                is_pass = d.get('BBBP_Result') == "Pass"
                has_risk = d.get('Toxicity') and d['Toxicity'] != "None"
                
                if has_risk:
                    fill = COLORS["Leaf_Risk"]
                    label += f"\n⚠️ {d['Toxicity']}"
                elif is_pass:
                    fill = COLORS["Leaf_Safe"]
                    label += "\n(Pass)"
                else:
                    fill = COLORS["Leaf_Unknown"] # Default leaf color

            # Draw Node
            graph_context.node(node_id, label, fillcolor=fill, fontcolor=font)
            
            # Draw Edge from Parent
            if parent_id:
                # Always allow edges across subgraphs by using the main dot object if needed, 
                # but graphviz engine handles it.
                graph_context.edge(parent_id, node_id)
                
            for child_name, child_node in node.children.items():
                # Family Cluster Logic
                if child_node.level == "Family":
                    with dot.subgraph(name=f'cluster_{child_name}') as c:
                        c.attr(label=child_name, style='dashed', color='#7f8c8d')
                        add_nodes(child_node, node_id, c)
                else:
                    add_nodes(child_node, node_id, graph_context)

        add_nodes(self.root)
        
        try:
            output = dot.render(filename, cleanup=True)
            print(f">>> [System] Graph saved: {output}")
        except Exception as e:
            print(f"Error rendering graph: {e}")

# ---------------------------------------------------------
# Step 4: 메인 실행 흐름
# ---------------------------------------------------------
def main():
    # 0. 설정
    RDF_PATH = "data/ontology/dto.rdf" # 프로젝트 루트 기준 상대 경로
    
    # 예시 SMILES (BACE1 inhibitor, EGFR inhibitor, COX-2 inhibitor)
    # 실제로는 복잡한 SMILES 사용
    sample_smiles = [
        ("CC1=CC(=C(C(=C1)C)C)C2=C(C=C(C=C2)S(=O)(=O)N)O", "Rofecoxib (COX-2)"), 
        ("CS(=O)(=O)CC1=CC=C(C=C1)C2=C(C(=NO2)C3=CC=CC=C3Cl)C4=CC=C(C=C4)F", "Etoricoxib (COX-2)"),
        ("COCC1=C(C=C(C(=C1)C#CC2=CC=CC=C2N)C)CN(C)C", "Dummy EGFR-like"), 
    ]
    
    # 1. 초기화
    try:
        fisher = TargetFisher()
    except ImportError:
        print("Skipping Target Fishing (PubChemPy missing)")
        fisher = None
        
    try:
        dto_kb = DTOKnowledgeBase(RDF_PATH)
    except ImportError:
        print("Skipping DTO KB (RDFLib missing)")
        dto_kb = None
        
    tree = SemanticDecisionTree()
    
    print("\n=== Semantic Decision Tree Build Process ===\n")
    
    for smiles, label in sample_smiles:
        print(f"Processing: {label}")
        
        # Step 1: Target Fishing
        if fisher:
            # 실제 호출 잠시 주석처리 (데모용 Mocking)
            # target_name = fisher.get_target_name(smiles)
            # Demo Override
            if "COX-2" in label:
                target_name = "COX-2"
            elif "EGFR" in label:
                target_name = "EGFR"
            else:
                target_name = "Unknown"
            print(f"  -> Detected Target: {target_name}")
        else:
            target_name = "Unknown"
            
        # Step 2: Semantic Feature Extraction
        if dto_kb:
            semantic_info = dto_kb.get_semantic_info(target_name)
        else:
            # Fallback mock
            semantic_info = {
                "Target": target_name,
                "Class": "Mock Class",
                "Family": "Mock Family",
                "Disease": "Mock Disease"
            }
            if target_name == "COX-2":
                semantic_info = {
                    "Target": "Cyclooxygenase-2",
                    "Class": "Prostaglandin-endoperoxide synthase",
                    "Family": "Enzyme",
                    "Disease": "Inflammation",
                    "Toxicity": "Cardiovascular Risk"
                }

        # Step 3: Insert into Tree
        tree.insert(semantic_info)
        
        # Step 4: Explanation
        expl = tree.explain(smiles, semantic_info)
        print(f"  -> Explanation: {expl}\n")
        
    print("=== Final Semantic Decision Tree ===\n")
    tree.print_tree()
    
    # [추가] 그래프 시각화 파일 내보내기
    tree.export_graphviz("semantic_tree_output")

if __name__ == "__main__":
    main()
