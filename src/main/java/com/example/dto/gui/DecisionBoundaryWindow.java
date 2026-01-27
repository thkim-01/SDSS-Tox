package com.example.dto.gui;

import com.example.dto.api.DssApiClient;

import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.scene.web.WebView;
import javafx.stage.Modality;
import javafx.stage.Stage;

/**
 * 2D 결정 경계 시각화 윈도우
 * (3D 기능은 사용자 요청에 의해 제거됨)
 */
public class DecisionBoundaryWindow extends Stage {
    
    private final DssApiClient apiClient;
    private WebView boundaryWebView;
    private ComboBox<String> xAxisCombo;
    private ComboBox<String> yAxisCombo;
    private Label statusLabel;
    
    private static final String[] DESCRIPTORS = {
        "LogP", "MW", "TPSA", "HBD", "HBA", "nRotB"
    };
    
    public DecisionBoundaryWindow(boolean unused) {
        // Parameter kept for compatibility if needed, but logic is strict 2D now
        this.apiClient = new DssApiClient();
        
        setTitle("2D Decision Boundary");
        initModality(Modality.NONE);
        setWidth(1000);
        setHeight(700);
        
        initUI();
    }
    
    // Default constructor
    public DecisionBoundaryWindow() {
        this(false);
    }
    
    private void initUI() {
        VBox root = new VBox(10);
        root.setPadding(new Insets(15));
        root.setStyle("-fx-background-color: #f5f5f5;");
        
        // Title
        Label titleLabel = new Label("2D Decision Boundary");
        titleLabel.setStyle("-fx-font-size: 18px; -fx-font-weight: bold;");
        
        // Control Bar
        HBox controlBar = new HBox(10);
        controlBar.setStyle("-fx-padding: 10; -fx-background-color: white; -fx-background-radius: 5;");
        
        Label xLabel = new Label("X-Axis:");
        xAxisCombo = new ComboBox<>();
        xAxisCombo.getItems().addAll(DESCRIPTORS);
        xAxisCombo.setValue("LogP");
        
        Label yLabel = new Label("Y-Axis:");
        yAxisCombo = new ComboBox<>();
        yAxisCombo.getItems().addAll(DESCRIPTORS);
        yAxisCombo.setValue("MW");
        
        Button loadBtn = new Button("Generate Boundary");
        loadBtn.setStyle("-fx-background-color: #27ae60; -fx-text-fill: white;");
        loadBtn.setOnAction(e -> loadDecisionBoundary());
        
        statusLabel = new Label("Ready...");
        statusLabel.setStyle("-fx-text-fill: #666;");
        
        controlBar.getChildren().addAll(xLabel, xAxisCombo, yLabel, yAxisCombo, loadBtn, statusLabel);
        
        // WebView for boundary visualization
        boundaryWebView = new WebView();
        VBox.setVgrow(boundaryWebView, Priority.ALWAYS);
        
        // Initial placeholder
        boundaryWebView.getEngine().loadContent(
            "<html><body style='display:flex;justify-content:center;align-items:center;height:100%;font-family:sans-serif;color:#666;'>" +
            "<div><h2>2D Decision Boundary</h2><p>Click 'Generate Boundary' to visualize.</p></div>" +
            "</body></html>"
        );
        
        root.getChildren().addAll(titleLabel, controlBar, boundaryWebView);
        
        Scene scene = new Scene(root);
        setScene(scene);
    }
    
    private void loadDecisionBoundary() {
        statusLabel.setText("Generating boundary...");
        
        String xFeature = xAxisCombo.getValue();
        String yFeature = yAxisCombo.getValue();
        
        apiClient.getDecisionBoundary2D(xFeature, yFeature, null, "random_forest", null)
            .thenAccept(html -> {
                javafx.application.Platform.runLater(() -> {
                    boundaryWebView.getEngine().loadContent(html);
                    statusLabel.setText("Done!");
                });
            })
            .exceptionally(e -> {
                javafx.application.Platform.runLater(() -> {
                    statusLabel.setText("Error: " + e.getMessage());
                });
                return null;
            });
    }
}
