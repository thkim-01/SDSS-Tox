package com.example.dto.visualization.renderers;

import java.util.Map;

import com.example.dto.api.DssApiClient;
import com.example.dto.visualization.api.ChartType;
import com.example.dto.visualization.api.VisualizationRenderer;

import javafx.application.Platform;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.scene.web.WebView;

public class DecisionTreeRenderer implements VisualizationRenderer {
    
    private final DssApiClient apiClient = new DssApiClient();
    private WebView treeWebView;
    private ComboBox<Integer> maxDepthCombo;

    private Label statusLabel;

    @Override
    public Node createSummaryChart(Map<String, Object> data) {
         // Create a VBox that matches the previous Window layout
         VBox root = new VBox(10);
         root.setPadding(new javafx.geometry.Insets(10));
         root.setStyle("-fx-background-color: transparent;");

         // Control Bar
         HBox controlBar = new HBox(10);
         controlBar.setAlignment(Pos.CENTER_LEFT);
         controlBar.setStyle("-fx-padding: 5; -fx-background-color: white; -fx-background-radius: 5; -fx-border-color: #ddd; -fx-border-radius: 5;");
         
         Label depthLabel = new Label("Max Depth:");
         maxDepthCombo = new ComboBox<>();
         maxDepthCombo.getItems().addAll(3, 4, 5, 6, 7, 8, 10);
         maxDepthCombo.setValue(5);
         
         // CheckBox removed as requested
         // semanticCheck = new CheckBox("Use Semantic Rules");
         
         Button loadBtn = new Button("Generate Tree");
         loadBtn.setOnAction(e -> loadDecisionTree());
         
         statusLabel = new Label("Ready");
         statusLabel.setStyle("-fx-text-fill: #666;");
         
         controlBar.getChildren().addAll(depthLabel, maxDepthCombo, loadBtn, statusLabel);
         
         // WebView
         treeWebView = new WebView();
         VBox.setVgrow(treeWebView, Priority.ALWAYS);
         
         // Initial Placeholder
         treeWebView.getEngine().loadContent(
             "<html><body style='display:flex;justify-content:center;align-items:center;height:100%;font-family:sans-serif;color:#666;'>" +
             "<div><h2>Decision Tree</h2><p>Click 'Generate Tree' to visualize.</p></div>" +
             "</body></html>"
         );

         root.getChildren().addAll(controlBar, treeWebView);
         
         // Check for injected data (Benchmark Linkage)
         if (data != null && data.containsKey("results")) {
            java.util.List<?> results = (java.util.List<?>) data.get("results");
            statusLabel.setText("Dataset Linked (" + results.size() + " samples)");
            statusLabel.setStyle("-fx-text-fill: #2ecc71; -fx-font-weight: bold;");
         }
         
         // If data provides initial params, load automatically (optional)
         
         return root;
    }

    private void loadDecisionTree() {
        statusLabel.setText("Generating...");
        int maxDepth = maxDepthCombo.getValue();
        // Force semantic to false or true based on implicit requirement? User said "Use Semantic Rules 기능은 SDT를 만들었기 때문에 필요 없어보여 제거". 
        // This implies for standard DT we don't use it, but for SDT model it's inherent. 
        // Let's assume false for the generic visualization toggles, or just pass false.
        boolean useSemantic = false; 
        
        apiClient.getDecisionTreePlot(maxDepth, useSemantic)
            .thenAccept(html -> {
                Platform.runLater(() -> {
                    treeWebView.getEngine().loadContent(html);
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
        return "JavaFX Native Tree";
    }

    @Override
    public boolean isSupported(ChartType type) {
        return type == ChartType.DECISION_TREE_STRUCTURE;
    }
}
