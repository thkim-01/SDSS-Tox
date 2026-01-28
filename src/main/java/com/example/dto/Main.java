package com.example.dto;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.semanticweb.owlapi.model.OWLOntologyCreationException;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.example.dto.core.DtoLoader;
import com.example.dto.core.DtoQuery;

public class Main {
    public static void main(String[] args) {
        // Arguments: <dto_path> <command> [query]
        // Commands: stats, search, targets
        
        if (args.length < 2) {
            System.err.println("Usage: java -jar sdss-tox.jar <dto_path> <command> [query]");
            System.err.println("Commands:");
            System.err.println("  stats - Show ontology statistics");
            System.err.println("  search <term> - Search classes by label");
            System.err.println("  targets <disease> - Find targets for disease");
            System.exit(1);
        }

        String dtoPath = args[0];
        String command = args[1];
        String query = args.length > 2 ? args[2] : "";

        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        Map<String, Object> response = new HashMap<>();

        DtoLoader loader = new DtoLoader();
        try {
            loader.loadOntology(dtoPath);
            
            switch (command.toLowerCase()) {
                case "stats":
                    response.put("status", "success");
                    response.put("data", loader.getStatistics());
                    break;
                    
                case "search":
                    if (query.isEmpty()) {
                        response.put("status", "error");
                        response.put("message", "Search term required");
                    } else {
                        List<Map<String, String>> results = loader.searchClassesByLabel(query);
                        response.put("status", "success");
                        response.put("count", results.size());
                        response.put("data", results);
                    }
                    break;
                    
                case "targets":
                    DtoQuery dtoQuery = new DtoQuery(loader);
                    List<Map<String, String>> targets;
                    if (query.isEmpty()) {
                        targets = dtoQuery.findProteinTargets(20);
                    } else {
                        targets = dtoQuery.findTargetsForDisease(query);
                    }
                    response.put("status", "success");
                    response.put("count", targets.size());
                    response.put("data", targets);
                    break;
                    
                default:
                    response.put("status", "error");
                    response.put("message", "Unknown command: " + command);
            }
            
        } catch (OWLOntologyCreationException e) {
            response.put("status", "error");
            response.put("message", "Failed to load ontology: " + e.getMessage());
        } catch (Exception e) {
            response.put("status", "error");
            response.put("message", "Error: " + e.getMessage());
        }
        
        // Output JSON to stdout (for Python to parse)
        System.out.println(gson.toJson(response));
    }
}
