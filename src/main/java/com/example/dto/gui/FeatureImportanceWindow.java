package com.example.dto.gui;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.example.dto.api.DssApiClient;

import javafx.collections.FXCollections;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.CategoryAxis;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.layout.VBox;
import javafx.stage.Modality;
import javafx.stage.Stage;

/**
 * RF Feature Importance 시각화 윈도우
 * 벤치마크 분석에서 분리되어 별도 윈도우로 표시
 */
public class FeatureImportanceWindow extends Stage {
    
    private final DssApiClient apiClient;
    private BarChart<Number, String> importanceChart;
    
    public FeatureImportanceWindow() {
        this.apiClient = new DssApiClient();
        
        setTitle("RF Feature Importance");
        initModality(Modality.NONE);
        setWidth(800);
        setHeight(600);
        
        initUI();
        loadFeatureImportance();
    }
    
    private void initUI() {
        VBox root = new VBox(10);
        root.setPadding(new Insets(15));
        root.setStyle("-fx-background-color: #f5f5f5;");
        
        // Title
        Label titleLabel = new Label("Random Forest Feature Importance");
        titleLabel.setStyle("-fx-font-size: 18px; -fx-font-weight: bold;");
        
        // Description
        Label descLabel = new Label("Shows the contribution of each molecular descriptor to the toxicity prediction.");
        descLabel.setStyle("-fx-font-size: 12px; -fx-text-fill: #666;");
        
        // Chart
        NumberAxis xAxis = new NumberAxis();
        xAxis.setLabel("Importance Score");
        
        CategoryAxis yAxis = new CategoryAxis();
        yAxis.setLabel("Feature");
        
        importanceChart = new BarChart<>(xAxis, yAxis);
        importanceChart.setTitle("Feature Importance");
        importanceChart.setAnimated(false);
        importanceChart.setLegendVisible(false);
        importanceChart.setMinHeight(400);
        importanceChart.setPrefHeight(500);
        
        ScrollPane scrollPane = new ScrollPane(importanceChart);
        scrollPane.setFitToWidth(true);
        scrollPane.setFitToHeight(true);
        
        root.getChildren().addAll(titleLabel, descLabel, scrollPane);
        
        Scene scene = new Scene(root);
        setScene(scene);
    }
    
    private void loadFeatureImportance() {
        new Thread(() -> {
            try {
                Map<String, Object> data = apiClient.getRfFeatureImportance();
                updateChart(data);
            } catch (Exception e) {
                System.err.println("Failed to load feature importance: " + e.getMessage());
            }
        }).start();
    }
    
    @SuppressWarnings("unchecked")
    private void updateChart(Map<String, Object> data) {
        javafx.application.Platform.runLater(() -> {
            if (data == null || !data.containsKey("feature_importance")) {
                System.err.println("No feature importance data");
                return;
            }
            
            Map<String, Double> importance = (Map<String, Double>) data.get("feature_importance");
            
            XYChart.Series<Number, String> series = new XYChart.Series<>();
            series.setName("Importance");
            
            // Sort by importance value (descending)
            List<Map.Entry<String, Double>> sortedEntries = new ArrayList<>(importance.entrySet());
            sortedEntries.sort((e1, e2) -> Double.compare(e2.getValue(), e1.getValue())); // Descending order
            
            // Limit to top 20 features
            int limit = Math.min(sortedEntries.size(), 20);
            List<Map.Entry<String, Double>> topEntries = sortedEntries.subList(0, limit);
            
            // Reverse for display (so highest is at top if axis fills from bottom, or adjust as needed)
            // For BarChart, usually the first item added appears at the bottom? Let's check.
            // Actually, let's keep descending order but maybe reverse if UI shows it upside down.
            // Let's stick to descending logic in list, but if we want largest at top, and axis starts from 0 (bottom), 
            // valid categories are usually laid out bottom-to-top or top-to-bottom. 
            // Let's try adding largest first.
            
            List<String> categories = new ArrayList<>();
            // Note: If we add largest first, and it renders bottom-up, largest will be at bottom. 
            // Often we want largest at top. Let's reverse for display order if needed.
            // Standard Chart: CategoryAxis usually plots in order added?
            // Let's iterate reverse to put largest at top (if Y axis grows down? No, Y grows up).
            // Actually, let's just use the top 20.
            
            for (int i = topEntries.size() - 1; i >= 0; i--) {
                Map.Entry<String, Double> entry = topEntries.get(i);
                series.getData().add(new XYChart.Data<>(entry.getValue(), entry.getKey()));
                categories.add(entry.getKey());
            }
            
            CategoryAxis yAxis = (CategoryAxis) importanceChart.getYAxis();
            yAxis.setCategories(FXCollections.observableArrayList(categories));
            
            importanceChart.getData().clear();
            importanceChart.getData().add(series);
            
            // Style the bars
            for (XYChart.Data<Number, String> item : series.getData()) {
                if (item.getNode() != null) {
                    item.getNode().setStyle("-fx-bar-fill: #3498db;");
                }
            }
        });
    }
}
