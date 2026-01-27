package com.example.dto.visualization.manager;

import java.util.HashMap;
import java.util.Map;

import com.example.dto.visualization.api.ChartType;
import com.example.dto.visualization.api.VisualizationRenderer;
import com.example.dto.visualization.renderers.DecisionBoundaryRenderer;
import com.example.dto.visualization.renderers.DecisionTreeRenderer;
import com.example.dto.visualization.renderers.JavaFXNativeRenderer;

import javafx.scene.Node;
import javafx.scene.control.Label;

public class VisualizationManager {
    private static VisualizationManager instance;
    private final Map<String, VisualizationRenderer> renderers = new HashMap<>();
    private String defaultRendererName = "JavaFX Native";

    private VisualizationManager() {
        registerRenderer(new JavaFXNativeRenderer());
        registerRenderer(new DecisionTreeRenderer());
        registerRenderer(new DecisionBoundaryRenderer());
    }

    public static VisualizationManager getInstance() {
        if (instance == null) {
            instance = new VisualizationManager();
        }
        return instance;
    }

    public void registerRenderer(VisualizationRenderer renderer) {
        renderers.put(renderer.getLibraryName(), renderer);
        System.out.println("Registered renderer: " + renderer.getLibraryName());
    }

    public Node render(String modelName, ChartType chartType, Map<String, Object> data) {
        // Logic to select renderer based on chart type
        VisualizationRenderer renderer = null;

        if (chartType == ChartType.DECISION_TREE_STRUCTURE) {
            renderer = renderers.get("JavaFX Native Tree"); 
        } else if (chartType == ChartType.SCATTER_PLOT_2D) {
            renderer = renderers.get("JavaFX Native Boundary");
        } else {
            renderer = renderers.get(defaultRendererName);
        }

        if (renderer != null && renderer.isSupported(chartType)) {
             try {
                if (chartType == ChartType.FEATURE_IMPORTANCE) {
                    return renderer.createFeatureImportance(data);
                } else if (chartType == ChartType.SHAP_SUMMARY) {
                    return renderer.createExplanationChart(data);
                } else if (chartType == ChartType.DECISION_TREE_STRUCTURE) {
                    return renderer.createSummaryChart(data); 
                } else if (chartType == ChartType.SCATTER_PLOT_2D) {
                    return renderer.createSummaryChart(data); 
                }
             } catch (Exception e) {
                 e.printStackTrace();
                 return new Label("Error rendering chart: " + e.getMessage());
             }
        }
        
        return new Label("Visualization not available for " + chartType);
    }
}
