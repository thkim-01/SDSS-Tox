package com.example.dto;

import org.semanticweb.owlapi.model.OWLOntologyCreationException;

public class Main {
    public static void main(String[] args) {
        System.out.println("Starting DTO-DSS Java Backend...");
        
        String dtoPath = "dto.rdf"; // Assuming it's in the root or passed as arg
        if (args.length > 0) {
            dtoPath = args[0];
        }

        DtoLoader loader = new DtoLoader();
        try {
            loader.loadOntology(dtoPath);
            loader.printStatistics();
            
            DtoQuery query = new DtoQuery(loader);
            // Example query
            // query.listSubclasses("http://www.drugtargetontology.org/dto/DTO_00000001"); // Example root
            
        } catch (OWLOntologyCreationException e) {
            System.err.println("Failed to load ontology: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
