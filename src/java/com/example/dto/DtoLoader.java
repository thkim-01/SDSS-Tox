package com.example.dto;

import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.reasoner.OWLReasoner;
import org.semanticweb.owlapi.reasoner.OWLReasonerFactory;
import org.semanticweb.owlapi.reasoner.structural.StructuralReasonerFactory;

import java.io.File;
import java.util.Set;
import java.util.stream.Collectors;

public class DtoLoader {
    private OWLOntologyManager manager;
    private OWLOntology ontology;
    private OWLReasoner reasoner;

    public DtoLoader() {
        this.manager = OWLManager.createOWLOntologyManager();
    }

    public void loadOntology(String filePath) throws OWLOntologyCreationException {
        File file = new File(filePath);
        this.ontology = manager.loadOntologyFromOntologyDocument(file);
        System.out.println("Loaded ontology: " + ontology.getOntologyID());
        
        OWLReasonerFactory reasonerFactory = new StructuralReasonerFactory();
        this.reasoner = reasonerFactory.createReasoner(ontology);
    }

    public OWLOntology getOntology() {
        return ontology;
    }

    public OWLReasoner getReasoner() {
        return reasoner;
    }

    public void printStatistics() {
        System.out.println("Axioms: " + ontology.getAxiomCount());
        System.out.println("Classes: " + ontology.getClassesInSignature().size());
        System.out.println("Individuals: " + ontology.getIndividualsInSignature().size());
    }
}
