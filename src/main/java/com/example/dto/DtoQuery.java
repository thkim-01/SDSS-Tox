package com.example.dto;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.semanticweb.owlapi.model.IRI;
import org.semanticweb.owlapi.model.OWLClass;
import org.semanticweb.owlapi.model.OWLDataFactory;
import org.semanticweb.owlapi.model.OWLOntology;
import org.semanticweb.owlapi.reasoner.NodeSet;
import org.semanticweb.owlapi.reasoner.OWLReasoner;

public class DtoQuery {
    private DtoLoader loader;

    public DtoQuery(DtoLoader loader) {
        this.loader = loader;
    }

    public List<Map<String, String>> findTargetsForDisease(String diseaseName) {
        List<Map<String, String>> targets = new ArrayList<>();
        
        // Search for classes containing the disease name
        List<Map<String, String>> diseaseClasses = loader.searchClassesByLabel(diseaseName);
        
        if (diseaseClasses.isEmpty()) {
            System.err.println("No disease classes found for: " + diseaseName);
            // Return some sample protein targets as fallback
            return findProteinTargets(10);
        }
        
        // For each disease class, find related targets via object properties
        OWLOntology ontology = loader.getOntology();
        OWLDataFactory df = loader.getDataFactory();
        
        for (Map<String, String> diseaseClass : diseaseClasses) {
            String iri = diseaseClass.get("iri");
            OWLClass cls = df.getOWLClass(IRI.create(iri));
            
            // Get subclasses as potential targets
            NodeSet<OWLClass> subClasses = loader.getReasoner().getSubClasses(cls, false);
            for (OWLClass sub : subClasses.getFlattened()) {
                if (!sub.isOWLNothing()) {
                    Map<String, String> entry = new HashMap<>();
                    entry.put("iri", sub.getIRI().toString());
                    entry.put("label", loader.getLabel(sub));
                    entry.put("relation", "subclass_of_disease");
                    targets.add(entry);
                    if (targets.size() >= 20) break;
                }
            }
            if (targets.size() >= 20) break;
        }
        
        // If no targets found through disease, return protein targets
        if (targets.isEmpty()) {
            return findProteinTargets(10);
        }
        
        return targets;
    }
    
    public List<Map<String, String>> findProteinTargets(int limit) {
        List<Map<String, String>> targets = new ArrayList<>();
        OWLOntology ontology = loader.getOntology();
        
        // Search for protein-related classes
        String[] proteinKeywords = {"protein", "kinase", "receptor", "enzyme", "transporter"};
        
        for (String keyword : proteinKeywords) {
            List<Map<String, String>> found = loader.searchClassesByLabel(keyword);
            for (Map<String, String> entry : found) {
                entry.put("type", "protein_target");
                targets.add(entry);
                if (targets.size() >= limit) break;
            }
            if (targets.size() >= limit) break;
        }
        
        return targets;
    }
    
    public List<Map<String, String>> listSubclasses(String classIri, int limit) {
        List<Map<String, String>> results = new ArrayList<>();
        OWLDataFactory df = loader.getDataFactory();
        OWLClass cls = df.getOWLClass(IRI.create(classIri));
        
        OWLReasoner reasoner = loader.getReasoner();
        NodeSet<OWLClass> subClasses = reasoner.getSubClasses(cls, true);
        
        for (OWLClass sub : subClasses.getFlattened()) {
            if (!sub.isOWLNothing()) {
                Map<String, String> entry = new HashMap<>();
                entry.put("iri", sub.getIRI().toString());
                entry.put("label", loader.getLabel(sub));
                results.add(entry);
                if (results.size() >= limit) break;
            }
        }
        
        return results;
    }
}
