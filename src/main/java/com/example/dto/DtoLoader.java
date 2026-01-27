package com.example.dto;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.OWLAnnotation;
import org.semanticweb.owlapi.model.OWLAnnotationValue;
import org.semanticweb.owlapi.model.OWLClass;
import org.semanticweb.owlapi.model.OWLDataFactory;
import org.semanticweb.owlapi.model.OWLEntity;
import org.semanticweb.owlapi.model.OWLLiteral;
import org.semanticweb.owlapi.model.OWLOntology;
import org.semanticweb.owlapi.model.OWLOntologyCreationException;
import org.semanticweb.owlapi.model.OWLOntologyManager;
import org.semanticweb.owlapi.reasoner.OWLReasoner;
import org.semanticweb.owlapi.reasoner.OWLReasonerFactory;
import org.semanticweb.owlapi.reasoner.structural.StructuralReasonerFactory;
import org.semanticweb.owlapi.search.EntitySearcher;

public class DtoLoader {
    private OWLOntologyManager manager;
    private OWLOntology ontology;
    private OWLReasoner reasoner;
    private OWLDataFactory dataFactory;

    public DtoLoader() {
        this.manager = OWLManager.createOWLOntologyManager();
    }

    public void loadOntology(String filePath) throws OWLOntologyCreationException {
        File file = new File(filePath);
        System.err.println("Loading ontology from: " + file.getAbsolutePath());
        this.ontology = manager.loadOntologyFromOntologyDocument(file);
        this.dataFactory = manager.getOWLDataFactory();
        System.err.println("Loaded ontology: " + ontology.getOntologyID());
        
        OWLReasonerFactory reasonerFactory = new StructuralReasonerFactory();
        this.reasoner = reasonerFactory.createReasoner(ontology);
    }

    public OWLOntology getOntology() {
        return ontology;
    }

    public OWLReasoner getReasoner() {
        return reasoner;
    }

    public OWLDataFactory getDataFactory() {
        return dataFactory;
    }

    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("axioms", ontology.getAxiomCount());
        stats.put("classes", ontology.getClassesInSignature().size());
        stats.put("individuals", ontology.getIndividualsInSignature().size());
        stats.put("objectProperties", ontology.getObjectPropertiesInSignature().size());
        return stats;
    }

    public String getLabel(OWLEntity entity) {
        for (OWLAnnotation annotation : EntitySearcher.getAnnotations(entity, ontology, dataFactory.getRDFSLabel()).collect(Collectors.toList())) {
            OWLAnnotationValue val = annotation.getValue();
            if (val instanceof OWLLiteral) {
                return ((OWLLiteral) val).getLiteral();
            }
        }
        return entity.getIRI().getShortForm();
    }
    
    public List<Map<String, String>> searchClassesByLabel(String searchTerm) {
        List<Map<String, String>> results = new ArrayList<>();
        String lowerSearchTerm = searchTerm.toLowerCase();
        
        for (OWLClass cls : ontology.getClassesInSignature()) {
            String label = getLabel(cls);
            if (label.toLowerCase().contains(lowerSearchTerm)) {
                Map<String, String> entry = new HashMap<>();
                entry.put("iri", cls.getIRI().toString());
                entry.put("label", label);
                results.add(entry);
            }
        }
        return results;
    }
}
