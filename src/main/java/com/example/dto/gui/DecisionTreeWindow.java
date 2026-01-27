package com.example.dto.gui;

import com.example.dto.api.DssApiClient;

import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.scene.web.WebView;
import javafx.stage.Modality;
import javafx.stage.Stage;



/**
 * Decision Tree 시각화 윈도우
 * scikit-learn DecisionTreeClassifier 기반 시각화
 */
public class DecisionTreeWindow extends Stage {
    
    private final DssApiClient apiClient;
    private WebView treeWebView;
    private ComboBox<Integer> maxDepthCombo;
    private Label statusLabel;
    
    public DecisionTreeWindow() {
        this.apiClient = new DssApiClient();
        
        setTitle("Decision Tree Visualization");
        initModality(Modality.NONE);
        setWidth(1000);
        setHeight(700);
        
        initUI();
    }
    
    private CheckBox semanticCheck; // Add CheckBox

    private void initUI() {
        VBox root = new VBox(10);
        root.setPadding(new Insets(15));
        root.setStyle("-fx-background-color: #f5f5f5;");
        
        // Title
        Label titleLabel = new Label("Decision Tree Visualization");
        titleLabel.setStyle("-fx-font-size: 18px; -fx-font-weight: bold;");
        
        // Control Bar
        HBox controlBar = new HBox(10);
        controlBar.setStyle("-fx-padding: 10; -fx-background-color: white; -fx-background-radius: 5;");
        
        Label depthLabel = new Label("Max Depth:");
        maxDepthCombo = new ComboBox<>();
        maxDepthCombo.getItems().addAll(3, 4, 5, 6, 7, 8, 10);
        maxDepthCombo.setValue(5);
        
        // Semantic Checkbox
        semanticCheck = new CheckBox("Use Semantic Rules");
        semanticCheck.setStyle("-fx-font-weight: bold; -fx-text-fill: #2c3e50;");
        
        Button loadBtn = new Button("Generate Tree");
        loadBtn.setStyle("-fx-background-color: #3498db; -fx-text-fill: white;");
        loadBtn.setOnAction(e -> loadDecisionTree());
        
        statusLabel = new Label("Ready...");
        statusLabel.setStyle("-fx-text-fill: #666;");
        
        controlBar.getChildren().addAll(depthLabel, maxDepthCombo, semanticCheck, loadBtn, statusLabel);
        
        // WebView for tree visualization
        treeWebView = new WebView();
        VBox.setVgrow(treeWebView, Priority.ALWAYS);
        
        // Initial placeholder
        treeWebView.getEngine().loadContent(
            "<html><body style='display:flex;justify-content:center;align-items:center;height:100%;font-family:sans-serif;color:#666;'>" +
            "<div><h2>Decision Tree</h2><p>Click 'Generate Tree' to visualize.</p></div>" +
            "</body></html>"
        );
        
        root.getChildren().addAll(titleLabel, controlBar, treeWebView);
        
        Scene scene = new Scene(root);
        setScene(scene);
    }
    
    private void loadDecisionTree() {
        statusLabel.setText("Generating tree...");
        int maxDepth = maxDepthCombo.getValue();
        boolean useSemantic = semanticCheck.isSelected();
        
        apiClient.getDecisionTreePlot(maxDepth, useSemantic)
            .thenAccept(html -> {
                javafx.application.Platform.runLater(() -> {
                    treeWebView.getEngine().loadContent(html);
                    statusLabel.setText("Done!");
                });
            })
            .exceptionally(e -> {
                javafx.application.Platform.runLater(() -> {
                    statusLabel.setText("Error: " + e.getMessage());
                    treeWebView.getEngine().loadContent(
                        "<html><body style='color:red;padding:20px;'>" +
                        "<h3>Error Occurred</h3><p>" + e.getMessage() + "</p></body></html>"
                    );
                });
                return null;
            });
    }
}
