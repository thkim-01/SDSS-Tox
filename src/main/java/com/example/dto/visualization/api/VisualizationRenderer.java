package com.example.dto.visualization.api;

import java.util.Map;

import javafx.scene.Node;

public interface VisualizationRenderer {
    /**
     * Creates a feature importance chart.
     * @param data Model-specific data (e.g., feature names and scores).
     * @return A JavaFX Node containing the visualization.
     */
    Node createFeatureImportance(Map<String, Object> data);

    /**
     * Creates a summary chart (e.g., SHAP summary).
     * @param data Model-specific data.
     * @return A JavaFX Node containing the visualization.
     */
    Node createSummaryChart(Map<String, Object> data);

    /**
     * Creates an explanation chart for a specific prediction (local interpretation).
     * @param data Model-specific data.
     * @return A JavaFX Node containing the visualization.
     */
    Node createExplanationChart(Map<String, Object> data);

    /**
     * Returns the library name (e.g., "JavaFX Native", "SHAP", "Matplotlib").
     */
    String getLibraryName();

    /**
     * Checks if a specific chart type is supported by this renderer.
     */
    boolean isSupported(ChartType type);
}
