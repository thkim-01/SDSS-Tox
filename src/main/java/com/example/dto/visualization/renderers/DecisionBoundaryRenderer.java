package com.example.dto.visualization.renderers;

import java.util.Map;

import com.example.dto.api.DssApiClient;
import com.example.dto.visualization.api.ChartType;
import com.example.dto.visualization.api.VisualizationRenderer;

import javafx.application.Platform;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.scene.web.WebView;

public class DecisionBoundaryRenderer implements VisualizationRenderer {

    private final DssApiClient apiClient = new DssApiClient();
    private WebView boundaryWebView;
    private ComboBox<String> xAxisCombo;
    private ComboBox<String> yAxisCombo;
    private Label statusLabel;
    private Integer sampleSize = 500;
    
    private static final String[] DESCRIPTORS = {
        "LogP", "MW", "TPSA", "HBD", "HBA", "nRotB"
    };
    
    private String currentDataset = null;
    private String currentModel = "random_forest";

    @Override
    public Node createSummaryChart(Map<String, Object> data) {
        VBox root = new VBox(10);
        root.setPadding(new javafx.geometry.Insets(10));
        root.setStyle("-fx-background-color: transparent;");

        // Control Bar
        HBox controlBar = new HBox(10);
        controlBar.setAlignment(Pos.CENTER_LEFT);
        controlBar.setStyle("-fx-padding: 5; -fx-background-color: white; -fx-background-radius: 5; -fx-border-color: #ddd; -fx-border-radius: 5;");
        
        Label xLabel = new Label("X:");
        xAxisCombo = new ComboBox<>();
        xAxisCombo.getItems().addAll(DESCRIPTORS);
        xAxisCombo.setValue("MW");
        
        Label yLabel = new Label("Y:");
        yAxisCombo = new ComboBox<>();
        yAxisCombo.getItems().addAll(DESCRIPTORS);
        yAxisCombo.setValue("LogP");

        // Re-render when axes change
        xAxisCombo.valueProperty().addListener((obs, oldV, newV) -> loadDecisionBoundary());
        yAxisCombo.valueProperty().addListener((obs, oldV, newV) -> loadDecisionBoundary());
        
        // loadBtn logic removed
        
        statusLabel = new Label("Ready");
        statusLabel.setStyle("-fx-text-fill: #666;");
        
        controlBar.getChildren().addAll(xLabel, xAxisCombo, yLabel, yAxisCombo, statusLabel);
        
        // WebView
        boundaryWebView = new WebView();
        VBox.setVgrow(boundaryWebView, Priority.ALWAYS);
        
        // Initial Placeholder
        boundaryWebView.getEngine().loadContent(
            "<html><body style='display:flex;justify-content:center;align-items:center;height:100%;font-family:sans-serif;background:#f8f9fa;color:#666;'>" +
            "<div><h2 style='color:#2c3e50;'>Decision Boundary</h2><p>Generating visualization, please wait...</p></div>" +
            "</body></html>"
        );

        root.getChildren().addAll(controlBar, boundaryWebView);
        
        // Check for injected data
        if (data != null && data.containsKey("results")) {
            java.util.List<?> results = (java.util.List<?>) data.get("results");
            statusLabel.setText("Benchmark Data Linked (" + results.size() + " samples)");
            statusLabel.setStyle("-fx-text-fill: #2ecc71; -fx-font-weight: bold;");
        }
        
        if (data != null && data.containsKey("dataset")) {
            this.currentDataset = (String) data.get("dataset");
        }

        if (data != null && data.containsKey("sampleSize")) {
            Object ss = data.get("sampleSize");
            if (ss instanceof Number) {
                this.sampleSize = ((Number) ss).intValue();
            }
        }
        
        if (data != null && data.containsKey("model")) {
            String modelName = (String) data.get("model");
            if (modelName.contains("Decision Tree")) this.currentModel = "decision_tree";
            else if (modelName.contains("SDT")) this.currentModel = "sdt";
            else this.currentModel = "random_forest";
        }
        
        // Auto-run if data is present OR just by default since button is gone
        if (data != null) {
            Platform.runLater(this::loadDecisionBoundary);
        }
        
        return root;
    }

    private void loadDecisionBoundary() {
        statusLabel.setText("Generating...");
        String xFeature = xAxisCombo.getValue();
        String yFeature = yAxisCombo.getValue();

        apiClient.getDecisionBoundary2D(xFeature, yFeature, currentDataset, currentModel, sampleSize)
            .thenAccept(html -> {
                Platform.runLater(() -> {
                    boundaryWebView.getEngine().loadContent(html);
                    statusLabel.setText("Done.");
                });
            }).exceptionally(ex -> {
                Platform.runLater(() -> statusLabel.setText("Error: " + ex.getMessage()));
                return null;
            });
    }

    @Override
    public Node createFeatureImportance(Map<String, Object> data) {
        return null;
    }

    @Override
    public Node createExplanationChart(Map<String, Object> data) {
        return null;
    }

    @Override
    public String getLibraryName() {
        return "JavaFX Native Boundary";
    }

    @Override
    public boolean isSupported(ChartType type) {
        return type == ChartType.SCATTER_PLOT_2D;
    }
}
