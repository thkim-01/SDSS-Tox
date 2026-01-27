"""Drug Target Ontology RDF 파서.

Phase 4.1: DTO RDF 파서
- RDF/OWL 파일 파싱 (rdflib 사용)
- 클래스, 프로퍼티, 인스턴스 추출
- SPARQL 쿼리 지원
- 화학물질-타겟 매핑

Author: DTO-DSS Team
Date: 2026-01-19
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field

try:
    from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL
    from rdflib.namespace import XSD
    RDFLIB_AVAILABLE = True
except ImportError:
    RDFLIB_AVAILABLE = False

# 로깅 설정
logger = logging.getLogger(__name__)


class RDFLibNotAvailableError(ImportError):
    """rdflib가 설치되지 않았을 때 발생하는 예외."""
    pass


class OntologyNotLoadedError(RuntimeError):
    """온톨로지가 로드되지 않았을 때 발생하는 예외."""
    pass


@dataclass
class OntologyClass:
    """온톨로지 클래스 정보."""
    uri: str
    name: str
    label: Optional[str] = None
    description: Optional[str] = None
    superclasses: List[str] = field(default_factory=list)
    subclasses: List[str] = field(default_factory=list)


@dataclass
class OntologyProperty:
    """온톨로지 프로퍼티 정보."""
    uri: str
    name: str
    property_type: str  # "ObjectProperty" or "DataProperty"
    domain: Optional[str] = None
    range: Optional[str] = None
    label: Optional[str] = None


@dataclass
class OntologyStatistics:
    """온톨로지 통계."""
    classes: int
    properties: int
    individuals: int
    axioms: int
    object_properties: int
    data_properties: int


class DTOParser:
    """Drug Target Ontology RDF 파서.

    rdflib를 사용하여 RDF/OWL 파일을 파싱하고,
    클래스, 프로퍼티, 인스턴스를 추출합니다.

    Attributes:
        graph: rdflib Graph 인스턴스.
        loaded: 온톨로지 로드 여부.
        file_path: 로드된 파일 경로.

    Example:
        >>> parser = DTOParser()
        >>> parser.load("dto.rdf")
        >>> classes = parser.get_classes()
        >>> toxicity_classes = parser.search_classes("toxic")
    """

    # DTO 관련 네임스페이스
    DTO = Namespace("http://www.drugtargetontology.org/dto/")
    OBOREL = Namespace("http://purl.obolibrary.org/obo/")

    def __init__(self) -> None:
        """DTOParser를 초기화한다."""
        if not RDFLIB_AVAILABLE:
            raise RDFLibNotAvailableError(
                "rdflib is not installed. Install with: pip install rdflib"
            )

        self.graph: Optional[Graph] = None
        self.loaded = False
        self.file_path: Optional[str] = None

        # 캐시
        self._classes_cache: Dict[str, OntologyClass] = {}
        self._properties_cache: Dict[str, OntologyProperty] = {}

        logger.info("DTOParser initialized")

    def load(self, file_path: str, format: str = "xml") -> OntologyStatistics:
        """RDF/OWL 파일을 로드한다.

        Args:
            file_path: RDF/OWL 파일 경로.
            format: 파일 포맷 ("xml", "turtle", "n3", "nt").

        Returns:
            온톨로지 통계 정보.

        Raises:
            FileNotFoundError: 파일이 존재하지 않을 때.
            RuntimeError: 파싱 중 에러가 발생했을 때.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Ontology file not found: {file_path}")

        logger.info(f"Loading ontology from: {file_path}")

        try:
            self.graph = Graph()
            self.graph.parse(str(path), format=format)
            self.loaded = True
            self.file_path = str(path)

            # 캐시 클리어
            self._classes_cache.clear()
            self._properties_cache.clear()

            stats = self.get_statistics()
            logger.info(f"Ontology loaded: {stats.classes} classes, {stats.properties} properties")

            return stats

        except Exception as e:
            logger.error(f"Failed to load ontology: {e}")
            raise RuntimeError(f"Failed to load ontology: {e}") from e

    def get_statistics(self) -> OntologyStatistics:
        """온톨로지 통계를 반환한다."""
        self._check_loaded()

        # 클래스 수
        classes = set(self.graph.subjects(RDF.type, OWL.Class))
        
        # 프로퍼티 수
        object_props = set(self.graph.subjects(RDF.type, OWL.ObjectProperty))
        data_props = set(self.graph.subjects(RDF.type, OWL.DatatypeProperty))
        annotation_props = set(self.graph.subjects(RDF.type, OWL.AnnotationProperty))
        
        # 개체 수 (Named Individual)
        individuals = set(self.graph.subjects(RDF.type, OWL.NamedIndividual))
        
        # 전체 트리플 수 (axioms 참고)
        axioms = len(self.graph)

        return OntologyStatistics(
            classes=len(classes),
            properties=len(object_props) + len(data_props) + len(annotation_props),
            individuals=len(individuals),
            axioms=axioms,
            object_properties=len(object_props),
            data_properties=len(data_props)
        )

    def get_classes(self, limit: int = 100) -> List[OntologyClass]:
        """모든 클래스 목록을 반환한다."""
        self._check_loaded()

        classes = []
        count = 0

        for class_uri in self.graph.subjects(RDF.type, OWL.Class):
            if count >= limit:
                break

            cls = self._parse_class(class_uri)
            if cls:
                classes.append(cls)
                count += 1

        return classes

    def search_classes(self, query: str, limit: int = 50) -> List[OntologyClass]:
        """쿼리 문자열로 클래스를 검색한다.

        Args:
            query: 검색어 (클래스 이름이나 레이블에 포함된 문자열).
            limit: 최대 결과 수.

        Returns:
            매칭되는 클래스 목록.
        """
        self._check_loaded()

        query_lower = query.lower()
        results = []

        for class_uri in self.graph.subjects(RDF.type, OWL.Class):
            if len(results) >= limit:
                break

            # URI에서 이름 추출
            name = self._get_local_name(class_uri)
            
            # 레이블 가져오기
            label = self._get_label(class_uri)

            # 검색어 매칭
            if (query_lower in name.lower() or 
                (label and query_lower in label.lower())):
                cls = self._parse_class(class_uri)
                if cls:
                    results.append(cls)

        return results

    def get_class_by_name(self, name: str) -> Optional[OntologyClass]:
        """이름으로 클래스를 조회한다."""
        self._check_loaded()

        for class_uri in self.graph.subjects(RDF.type, OWL.Class):
            local_name = self._get_local_name(class_uri)
            if local_name.lower() == name.lower():
                return self._parse_class(class_uri)

        return None

    def get_properties(self, limit: int = 100) -> List[OntologyProperty]:
        """모든 프로퍼티 목록을 반환한다."""
        self._check_loaded()

        properties = []
        count = 0

        # Object Properties
        for prop_uri in self.graph.subjects(RDF.type, OWL.ObjectProperty):
            if count >= limit:
                break
            prop = self._parse_property(prop_uri, "ObjectProperty")
            if prop:
                properties.append(prop)
                count += 1

        # Data Properties
        for prop_uri in self.graph.subjects(RDF.type, OWL.DatatypeProperty):
            if count >= limit:
                break
            prop = self._parse_property(prop_uri, "DataProperty")
            if prop:
                properties.append(prop)
                count += 1

        return properties

    def get_subclasses(self, class_name: str) -> List[str]:
        """클래스의 하위 클래스 목록을 반환한다."""
        self._check_loaded()

        subclasses = []

        for class_uri in self.graph.subjects(RDF.type, OWL.Class):
            local_name = self._get_local_name(class_uri)
            if local_name.lower() == class_name.lower():
                # 이 클래스의 subclass 찾기
                for sub_uri in self.graph.subjects(RDFS.subClassOf, class_uri):
                    sub_name = self._get_local_name(sub_uri)
                    subclasses.append(sub_name)
                break

        return subclasses

    def get_superclasses(self, class_name: str) -> List[str]:
        """클래스의 상위 클래스 목록을 반환한다."""
        self._check_loaded()

        superclasses = []

        for class_uri in self.graph.subjects(RDF.type, OWL.Class):
            local_name = self._get_local_name(class_uri)
            if local_name.lower() == class_name.lower():
                # 이 클래스의 superclass 찾기
                for super_uri in self.graph.objects(class_uri, RDFS.subClassOf):
                    if isinstance(super_uri, URIRef):
                        super_name = self._get_local_name(super_uri)
                        superclasses.append(super_name)
                break

        return superclasses

    def query_sparql(self, sparql: str) -> List[Dict[str, Any]]:
        """SPARQL 쿼리를 실행한다.

        Args:
            sparql: SPARQL 쿼리 문자열.

        Returns:
            쿼리 결과 리스트.
        """
        self._check_loaded()

        results = []
        try:
            for row in self.graph.query(sparql):
                result = {}
                for var, value in zip(row.labels, row):
                    if value:
                        result[str(var)] = str(value)
                results.append(result)
        except Exception as e:
            logger.error(f"SPARQL query failed: {e}")
            raise RuntimeError(f"SPARQL query failed: {e}") from e

        return results

    def get_toxic_related_classes(self) -> List[OntologyClass]:
        """독성 관련 클래스를 검색한다."""
        toxic_keywords = ["toxic", "toxicity", "carcinogen", "mutagen", "hepato", "nephro", "neuro"]
        results = []

        for keyword in toxic_keywords:
            for cls in self.search_classes(keyword, limit=20):
                if cls not in results:
                    results.append(cls)

        return results

    def _check_loaded(self) -> None:
        """온톨로지가 로드되었는지 확인한다."""
        if not self.loaded or self.graph is None:
            raise OntologyNotLoadedError("Ontology not loaded. Call load() first.")

    def _parse_class(self, class_uri: URIRef) -> Optional[OntologyClass]:
        """클래스 URI를 파싱하여 OntologyClass 객체로 변환한다."""
        try:
            name = self._get_local_name(class_uri)
            label = self._get_label(class_uri)
            description = self._get_description(class_uri)

            # Superclasses
            superclasses = []
            for super_uri in self.graph.objects(class_uri, RDFS.subClassOf):
                if isinstance(super_uri, URIRef):
                    superclasses.append(self._get_local_name(super_uri))

            return OntologyClass(
                uri=str(class_uri),
                name=name,
                label=label,
                description=description,
                superclasses=superclasses
            )
        except Exception as e:
            logger.debug(f"Failed to parse class {class_uri}: {e}")
            return None

    def _parse_property(self, prop_uri: URIRef, prop_type: str) -> Optional[OntologyProperty]:
        """프로퍼티 URI를 파싱하여 OntologyProperty 객체로 변환한다."""
        try:
            name = self._get_local_name(prop_uri)
            label = self._get_label(prop_uri)

            # Domain
            domain = None
            for d in self.graph.objects(prop_uri, RDFS.domain):
                if isinstance(d, URIRef):
                    domain = self._get_local_name(d)
                    break

            # Range
            range_ = None
            for r in self.graph.objects(prop_uri, RDFS.range):
                if isinstance(r, URIRef):
                    range_ = self._get_local_name(r)
                    break

            return OntologyProperty(
                uri=str(prop_uri),
                name=name,
                property_type=prop_type,
                domain=domain,
                range=range_,
                label=label
            )
        except Exception as e:
            logger.debug(f"Failed to parse property {prop_uri}: {e}")
            return None

    def _get_local_name(self, uri: URIRef) -> str:
        """URI에서 로컬 이름을 추출한다."""
        uri_str = str(uri)
        
        # # 또는 / 이후의 이름
        if '#' in uri_str:
            return uri_str.split('#')[-1]
        else:
            return uri_str.split('/')[-1]

    def _get_label(self, uri: URIRef) -> Optional[str]:
        """엔티티의 rdfs:label을 가져온다."""
        for label in self.graph.objects(uri, RDFS.label):
            return str(label)
        return None

    def _get_description(self, uri: URIRef) -> Optional[str]:
        """엔티티의 rdfs:comment를 가져온다."""
        for comment in self.graph.objects(uri, RDFS.comment):
            return str(comment)
        return None


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)

    # 테스트
    parser = DTOParser()
    
    # dto.rdf 로드
    try:
        stats = parser.load("dto.rdf")
        print(f"\n=== Ontology Statistics ===")
        print(f"Classes: {stats.classes}")
        print(f"Properties: {stats.properties}")
        print(f"Individuals: {stats.individuals}")
        print(f"Axioms: {stats.axioms}")

        # 클래스 검색
        toxic_classes = parser.search_classes("toxic", limit=10)
        print(f"\n=== Toxic-related classes ({len(toxic_classes)}) ===")
        for cls in toxic_classes[:5]:
            print(f"  - {cls.name}: {cls.label}")

    except FileNotFoundError:
        print("dto.rdf not found, skipping test")
