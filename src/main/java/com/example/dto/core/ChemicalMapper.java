package com.example.dto.core;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Maps SMILES patterns to DTO ontology classes using ChEBI and structural patterns.
 */
public class ChemicalMapper {
    private final DtoLoader loader;
    
    // SMILES pattern to ontology class mappings
    private static final Map<String, String[]> STRUCTURE_MAPPINGS = new LinkedHashMap<>();
    
    static {
        // Aromatic compounds
        STRUCTURE_MAPPINGS.put("c1ccccc1", new String[]{"benzene", "aromatic", "phenyl"});
        STRUCTURE_MAPPINGS.put("c1ccncc1", new String[]{"pyridine", "heterocyclic"});
        STRUCTURE_MAPPINGS.put("c1ccoc1", new String[]{"furan", "heterocyclic"});
        STRUCTURE_MAPPINGS.put("c1ccsc1", new String[]{"thiophene", "heterocyclic"});
        
        // Functional groups - Toxic indicators
        STRUCTURE_MAPPINGS.put("[N+](=O)[O-]", new String[]{"nitro", "toxic", "mutagenic"});
        STRUCTURE_MAPPINGS.put("N=N", new String[]{"azo", "carcinogenic"});
        STRUCTURE_MAPPINGS.put("C=O", new String[]{"aldehyde", "reactive"});
        STRUCTURE_MAPPINGS.put("Cl", new String[]{"chloro", "halogen"});
        STRUCTURE_MAPPINGS.put("Br", new String[]{"bromo", "halogen"});
        STRUCTURE_MAPPINGS.put("I", new String[]{"iodo", "halogen"});
        STRUCTURE_MAPPINGS.put("F", new String[]{"fluoro", "halogen"});
        STRUCTURE_MAPPINGS.put("[As]", new String[]{"arsenic", "toxic", "heavy metal"});
        STRUCTURE_MAPPINGS.put("[Hg]", new String[]{"mercury", "toxic", "heavy metal"});
        STRUCTURE_MAPPINGS.put("[Pb]", new String[]{"lead", "toxic", "heavy metal"});
        
        // Functional groups - Generally safe
        STRUCTURE_MAPPINGS.put("O", new String[]{"alcohol", "hydroxyl"});
        STRUCTURE_MAPPINGS.put("C(=O)O", new String[]{"carboxylic acid", "organic acid"});
        STRUCTURE_MAPPINGS.put("N", new String[]{"amine", "nitrogen"});
        STRUCTURE_MAPPINGS.put("C(=O)N", new String[]{"amide", "peptide bond"});
        STRUCTURE_MAPPINGS.put("S", new String[]{"thiol", "sulfur"});
        STRUCTURE_MAPPINGS.put("P", new String[]{"phosphate", "phosphorus"});
        
        // Drug-like structures
        STRUCTURE_MAPPINGS.put("c1ccc2c(c1)", new String[]{"naphthalene", "polycyclic"});
        STRUCTURE_MAPPINGS.put("C1CCCCC1", new String[]{"cyclohexane", "cyclic"});
        STRUCTURE_MAPPINGS.put("C1CCNCC1", new String[]{"piperidine", "alkaloid"});
        STRUCTURE_MAPPINGS.put("c1c[nH]c2c1cccc2", new String[]{"indole", "tryptophan"});
    }
    
    // Toxic substructure alerts (simplified)
    private static final Set<String> TOXIC_ALERTS = Set.of(
        "[N+](=O)[O-]",  // Nitro group
        "N=N",           // Azo group
        "[As]", "[Hg]", "[Pb]",  // Heavy metals
        "C(Cl)(Cl)Cl",   // Chloroform-like
        "CC(=O)OC",      // Ester (can hydrolyze)
        "C#N"            // Nitrile
    );
    
    private static final Set<String> SAFE_PATTERNS = Set.of(
        "CCO",           // Ethanol
        "C(=O)O",        // Carboxylic acid
        "OCC(O)CO"       // Glycerol
    );

    public ChemicalMapper(DtoLoader loader) {
        this.loader = loader;
    }

    /**
     * Analyze SMILES and return ontology-based toxicity assessment.
     */
    public MappingResult mapToOntology(String smiles) {
        MappingResult result = new MappingResult();
        result.smiles = smiles;
        result.matchedPatterns = new ArrayList<>();
        result.ontologyClasses = new ArrayList<>();
        result.toxicAlerts = new ArrayList<>();
        result.safeIndicators = new ArrayList<>();
        
        // Step 1: Find matching structural patterns
        for (Map.Entry<String, String[]> entry : STRUCTURE_MAPPINGS.entrySet()) {
            if (smiles.contains(entry.getKey())) {
                result.matchedPatterns.add(entry.getKey());
                for (String keyword : entry.getValue()) {
                    // Search ontology for this keyword
                    List<Map<String, String>> found = loader.searchClassesByLabel(keyword);
                    for (Map<String, String> cls : found) {
                        cls.put("source", "structural_pattern");
                        cls.put("pattern", entry.getKey());
                        result.ontologyClasses.add(cls);
                    }
                }
            }
        }
        
        // Step 2: Check toxic alerts
        for (String alert : TOXIC_ALERTS) {
            if (smiles.contains(alert)) {
                result.toxicAlerts.add(alert);
            }
        }
        
        // Step 3: Check safe patterns
        for (String safe : SAFE_PATTERNS) {
            if (smiles.equals(safe) || smiles.startsWith(safe)) {
                result.safeIndicators.add(safe);
            }
        }
        
        // Step 4: Determine ontology-based risk
        if (!result.toxicAlerts.isEmpty()) {
            result.ontologyRisk = "High";
            result.confidence = 0.8;
            result.explanation = "독성 구조 알림 감지: " + String.join(", ", result.toxicAlerts);
        } else if (!result.safeIndicators.isEmpty() && result.toxicAlerts.isEmpty()) {
            result.ontologyRisk = "Low";
            result.confidence = 0.7;
            result.explanation = "안전한 구조 패턴 감지: " + String.join(", ", result.safeIndicators);
        } else if (!result.matchedPatterns.isEmpty()) {
            // Analyze matched patterns
            boolean hasToxicKeyword = result.ontologyClasses.stream()
                    .anyMatch(c -> c.get("label").toLowerCase().contains("toxic") ||
                                  c.get("label").toLowerCase().contains("mutagen") ||
                                  c.get("label").toLowerCase().contains("carcinogen"));
            
            if (hasToxicKeyword) {
                result.ontologyRisk = "High";
                result.confidence = 0.6;
                result.explanation = "온톨로지 클래스에서 독성 관련 용어 발견";
            } else {
                result.ontologyRisk = "Low";
                result.confidence = 0.5;
                result.explanation = "구조 패턴 분석 결과 명확한 독성 위험 없음";
            }
        } else {
            result.ontologyRisk = "Unknown";
            result.confidence = 0.3;
            result.explanation = "알려진 구조 패턴과 일치하지 않음";
        }
        
        return result;
    }
    
    /**
     * Get all known structure patterns for display.
     */
    public Map<String, String[]> getAllPatterns() {
        return new LinkedHashMap<>(STRUCTURE_MAPPINGS);
    }
    
    // Result class
    public static class MappingResult {
        public String smiles;
        public List<String> matchedPatterns;
        public List<Map<String, String>> ontologyClasses;
        public List<String> toxicAlerts;
        public List<String> safeIndicators;
        public String ontologyRisk;
        public double confidence;
        public String explanation;
    }
}
