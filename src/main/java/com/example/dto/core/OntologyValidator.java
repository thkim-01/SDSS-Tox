package com.example.dto.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Validates ML predictions against ontology axioms for XAI.
 * Detects potential misclassifications by comparing ML output with 
 * ontology knowledge.
 */
@SuppressWarnings({"unused", "null"})
public class OntologyValidator {
    private final DtoLoader loader;
    
    // Known toxic-related classes in DTO
    private static final Set<String> TOXIC_KEYWORDS = Set.of(
        "toxic", "toxicity", "carcinogen", "mutagen", "teratogen",
        "hepatotoxic", "nephrotoxic", "neurotoxic", "cytotoxic"
    );
    
    private static final Set<String> SAFE_KEYWORDS = Set.of(
        "drug", "therapeutic", "approved", "safe", "nutrient", "vitamin"
    );

    public OntologyValidator(DtoLoader loader) {
        this.loader = loader;
    }

    /**
     * Validate ML prediction against ontology axioms.
     * Returns validation result with explanation.
     */
    public ValidationResult validate(String smiles, double mlPrediction, String mlRisk) {
        ValidationResult result = new ValidationResult();
        result.smiles = smiles;
        result.mlPrediction = mlPrediction;
        result.mlRisk = mlRisk;
        
        // Search for related classes in ontology
        List<Map<String, String>> relatedClasses = findRelatedClasses(smiles);
        result.relatedClasses = relatedClasses;
        
        // Analyze ontology evidence
        OntologyEvidence evidence = analyzeOntologyEvidence(relatedClasses);
        result.ontologyEvidence = evidence;
        
        // Compare ML prediction with ontology knowledge
        result.consistencyCheck = checkConsistency(mlRisk, evidence);
        
        // Generate explanation
        result.explanation = generateExplanation(result);
        
        return result;
    }
    
    private List<Map<String, String>> findRelatedClasses(String smiles) {
        List<Map<String, String>> related = new ArrayList<>();
        
        // Search for molecular patterns in ontology
        // In real implementation, would use ChEBI mapping or SMARTS patterns
        // For now, search by common substructures
        
        String[] patterns = extractPatterns(smiles);
        for (String pattern : patterns) {
            List<Map<String, String>> found = loader.searchClassesByLabel(pattern);
            for (Map<String, String> cls : found) {
                cls.put("matchedPattern", pattern);
                related.add(cls);
            }
            if (related.size() >= 10) break;
        }
        
        return related;
    }
    
    private String[] extractPatterns(String smiles) {
        // Extract searchable patterns from SMILES
        List<String> patterns = new ArrayList<>();
        
        // Common functional groups
        if (smiles.contains("N")) patterns.add("amine");
        if (smiles.contains("O=C")) patterns.add("carbonyl");
        if (smiles.contains("c1ccccc1")) patterns.add("benzene");
        if (smiles.contains("Cl")) patterns.add("chloro");
        if (smiles.contains("F")) patterns.add("fluoro");
        if (smiles.contains("S")) patterns.add("sulfur");
        if (smiles.contains("[N+]") || smiles.contains("[O-]")) patterns.add("nitro");
        if (smiles.contains("O=N")) patterns.add("nitroso");
        
        if (patterns.isEmpty()) {
            patterns.add("chemical");
        }
        
        return patterns.toArray(new String[0]);
    }
    
    private OntologyEvidence analyzeOntologyEvidence(List<Map<String, String>> relatedClasses) {
        OntologyEvidence evidence = new OntologyEvidence();
        evidence.toxicIndicators = new ArrayList<>();
        evidence.safeIndicators = new ArrayList<>();
        evidence.neutralIndicators = new ArrayList<>();
        
        for (Map<String, String> cls : relatedClasses) {
            String label = cls.get("label").toLowerCase();
            String iri = cls.get("iri");
            
            boolean isToxic = TOXIC_KEYWORDS.stream().anyMatch(label::contains);
            boolean isSafe = SAFE_KEYWORDS.stream().anyMatch(label::contains);
            
            if (isToxic) {
                evidence.toxicIndicators.add(cls.get("label") + " (" + cls.get("matchedPattern") + ")");
            } else if (isSafe) {
                evidence.safeIndicators.add(cls.get("label") + " (" + cls.get("matchedPattern") + ")");
            } else {
                evidence.neutralIndicators.add(cls.get("label"));
            }
        }
        
        // Determine ontology-based risk
        if (!evidence.toxicIndicators.isEmpty()) {
            evidence.ontologyRisk = "High";
        } else if (!evidence.safeIndicators.isEmpty()) {
            evidence.ontologyRisk = "Low";
        } else {
            evidence.ontologyRisk = "Unknown";
        }
        
        return evidence;
    }
    
    private ConsistencyCheck checkConsistency(String mlRisk, OntologyEvidence evidence) {
        ConsistencyCheck check = new ConsistencyCheck();
        
        if (evidence.ontologyRisk.equals("Unknown")) {
            check.status = "INSUFFICIENT_DATA";
            check.message = "온톨로지에 충분한 정보가 없습니다.";
            check.isConsistent = true;  // Can't contradict
        } else if (mlRisk.equalsIgnoreCase(evidence.ontologyRisk)) {
            check.status = "CONSISTENT";
            check.message = "ML 예측과 온톨로지 지식이 일치합니다.";
            check.isConsistent = true;
        } else {
            check.status = "MISMATCH";
            check.message = String.format("불일치 감지: ML=%s, 온톨로지=%s", mlRisk, evidence.ontologyRisk);
            check.isConsistent = false;
        }
        
        return check;
    }
    
    private String generateExplanation(ValidationResult result) {
        StringBuilder sb = new StringBuilder();
        
        sb.append("분석 결과:\n");
        sb.append(String.format("  ML 예측: %s (%.2f%%)\n", result.mlRisk, result.mlPrediction * 100));
        sb.append(String.format("  온톨로지 판단: %s\n\n", result.ontologyEvidence.ontologyRisk));
        
        if (!result.ontologyEvidence.toxicIndicators.isEmpty()) {
            sb.append("독성 관련 온톨로지 클래스:\n");
            for (String ind : result.ontologyEvidence.toxicIndicators) {
                sb.append("  - ").append(ind).append("\n");
            }
        }
        
        if (!result.ontologyEvidence.safeIndicators.isEmpty()) {
            sb.append("안전성 관련 온톨로지 클래스:\n");
            for (String ind : result.ontologyEvidence.safeIndicators) {
                sb.append("  - ").append(ind).append("\n");
            }
        }
        
        sb.append("\n").append(result.consistencyCheck.message);
        
        if (!result.consistencyCheck.isConsistent) {
            sb.append("\n\n오분류 가능성: ML 모델의 예측이 온톨로지 지식과 일치하지 않습니다. ");
            sb.append("추가 검토가 필요합니다.");
        }
        
        return sb.toString();
    }
    
    // Inner classes for structured results
    public static class ValidationResult {
        public String smiles;
        public double mlPrediction;
        public String mlRisk;
        public List<Map<String, String>> relatedClasses;
        public OntologyEvidence ontologyEvidence;
        public ConsistencyCheck consistencyCheck;
        public String explanation;
    }
    
    public static class OntologyEvidence {
        public List<String> toxicIndicators;
        public List<String> safeIndicators;
        public List<String> neutralIndicators;
        public String ontologyRisk;
    }
    
    public static class ConsistencyCheck {
        public String status;  // CONSISTENT, MISMATCH, INSUFFICIENT_DATA
        public String message;
        public boolean isConsistent;
    }
}
