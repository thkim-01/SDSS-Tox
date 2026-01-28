package com.example.dto.visualization.renderers;

import java.util.List;
import java.util.Map;

import com.example.dto.visualization.api.ChartType;
import com.example.dto.visualization.api.VisualizationRenderer;

import javafx.scene.Node;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.CategoryAxis;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Label;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import javafx.geometry.Insets;

@SuppressWarnings({"unchecked", "unused"})
public class JavaFXNativeRenderer implements VisualizationRenderer {

    @Override
    public Node createFeatureImportance(Map<String, Object> data) {
        if (data == null || !data.containsKey("features") 
            || !data.containsKey("importance")) {
            return new Label("No feature importance data available.");
        }

        try {
            List<String> features = (List<String>) data.get("features");
            List<Double> importance = (List<Double>) data.get("importance");
            
            if (features.size() != importance.size()) {
                 return new Label("Data mismatch: feature count != importance count.");
            }

            CategoryAxis xAxis = new CategoryAxis();
            NumberAxis yAxis = new NumberAxis();
            xAxis.setLabel("Feature");
            yAxis.setLabel("Importance");

            BarChart<String, Number> barChart = new BarChart<>(xAxis, yAxis);
            barChart.setTitle("Feature Importance");
            barChart.setLegendVisible(false);

            XYChart.Series<String, Number> series = new XYChart.Series<>();
            for (int i = 0; i < features.size(); i++) {
                series.getData().add(new XYChart.Data<>(features.get(i), importance.get(i)));
            }

            barChart.getData().add(series);
            return barChart;
        } catch (ClassCastException e) {
            return new Label("Error parsing data for Feature Importance.");
        }
    }

    @Override
    public Node createSummaryChart(Map<String, Object> data) {
        return new Label("Summary Chart not supported in Native renderer yet.");
    }

    @Override
    public Node createExplanationChart(Map<String, Object> data) {
        if (data == null || !data.containsKey("shap_features")) {
            return new Label("No SHAP data available.");
        }

        try {
            List<Map<String, Object>> shapFeatures = (List<Map<String, Object>>) data.get("shap_features");
            
            if (shapFeatures == null || shapFeatures.isEmpty()) {
                return new Label("No SHAP features to display.");
            }

            // Create custom diverging bar chart using VBox and custom rectangles
            VBox container = new VBox();
            container.setPadding(new Insets(10));
            container.setSpacing(15);
            
            Label titleLabel = new Label("LOCAL FEATURE CONTRIBUTION (SHAP)");
            titleLabel.setStyle("-fx-font-size: 14; -fx-font-weight: bold;");
            container.getChildren().add(titleLabel);

            // Find max absolute SHAP value for scaling
            double maxAbsValue = 0;
            for (Map<String, Object> feature : shapFeatures) {
                double shapValue = ((Number) feature.get("shap_value")).doubleValue();
                maxAbsValue = Math.max(maxAbsValue, Math.abs(shapValue));
            }

            // Add bars for each feature
            for (Map<String, Object> feature : shapFeatures) {
                String featureName = (String) feature.get("feature");
                double shapValue = ((Number) feature.get("shap_value")).doubleValue();
                
                // Create bar container
                VBox barContainer = new VBox();
                barContainer.setStyle("-fx-border-color: #e0e0e0; -fx-border-width: 0 0 0.5 0;");
                barContainer.setPadding(new Insets(5, 0, 5, 0));
                
                // Feature name label
                Label featureLabel = new Label(featureName);
                featureLabel.setStyle("-fx-font-size: 10;");
                
                // Bar and value container
                javafx.scene.layout.HBox barRow = new javafx.scene.layout.HBox();
                barRow.setSpacing(10);
                barRow.setPadding(new Insets(2, 0, 2, 0));
                
                // Determine color and width
                Color barColor = shapValue >= 0 ? Color.web("#e74c3c") : Color.web("#3498db"); // Red for positive, blue for negative
                double barWidth = (Math.abs(shapValue) / (maxAbsValue > 0 ? maxAbsValue : 1)) * 300; // Scale to max 300px
                
                Rectangle bar = new Rectangle(barWidth, 20);
                bar.setFill(barColor);
                bar.setStyle("-fx-arc-width: 2; -fx-arc-height: 2;");
                
                // Value label
                Label valueLabel = new Label(String.format("%.4f", shapValue));
                valueLabel.setStyle("-fx-font-size: 9; -fx-text-fill: #333;");
                valueLabel.setMinWidth(60);
                
                barRow.getChildren().addAll(bar, valueLabel);
                
                barContainer.getChildren().addAll(featureLabel, barRow);
                container.getChildren().add(barContainer);
            }

            return container;
        } catch (Exception e) {
            e.printStackTrace();
            return new Label("Error creating SHAP chart: " + e.getMessage());
        }
    }

    @Override
    public String getLibraryName() {
        return "JavaFX Native";
    }

    @Override
    public boolean isSupported(ChartType type) {
        return type == ChartType.FEATURE_IMPORTANCE || type == ChartType.SHAP_SUMMARY;
    }
}
