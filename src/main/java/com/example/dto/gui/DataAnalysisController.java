package com.example.dto.gui;

import java.util.List;

import com.example.dto.api.DssApiClient;
import com.example.dto.api.DssApiClient.CombinedPredictionResponse;

import javafx.fxml.FXML;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.web.WebView;
import netscape.javascript.JSObject;

public class DataAnalysisController {

    @FXML private Label datasetLabel;
    @FXML private ComboBox<String> xAxisCombo;
    @FXML private ComboBox<String> yAxisCombo;
    @FXML private ComboBox<String> colorCombo;
    @FXML private CheckBox logXCheck;
    @FXML private CheckBox logYCheck;
    @FXML private CheckBox linearRegCheck;
    @FXML private CheckBox ransacCheck;
    @FXML private WebView plotWebView;
    
    private List<CombinedPredictionResponse> data;
    private final DssApiClient apiClient = new DssApiClient();
    
    @FXML
    public void initialize() {
        // Initialize Combo Boxes
        xAxisCombo.getItems().addAll("LogP", "MW", "TPSA", "HBD", "HBA", "nRotB", "n_heavy", "n_rings");
        xAxisCombo.setValue("MW");
        
        yAxisCombo.getItems().addAll("LogP", "MW", "TPSA", "HBD", "HBA", "nRotB", "n_heavy", "n_rings");
        yAxisCombo.setValue("LogP");
        
        colorCombo.getItems().addAll("TPSA", "risk_level", "confidence", "LogP", "MW");
        colorCombo.setValue("TPSA");
        
        // Add listeners to auto-update? Or just use "Update Plot" button?
        // User asked for "Data Analysis Tool", manual update is fine but auto is smoother.
        // Let's rely on the Update Plot button for explicit action + listeners for convenience.
        xAxisCombo.valueProperty().addListener((o, old, val) -> updatePlot());
        yAxisCombo.valueProperty().addListener((o, old, val) -> updatePlot());
        colorCombo.valueProperty().addListener((o, old, val) -> updatePlot());
        logXCheck.selectedProperty().addListener((o, old, val) -> updatePlot());
        logYCheck.selectedProperty().addListener((o, old, val) -> updatePlot());
        linearRegCheck.selectedProperty().addListener((o, old, val) -> updatePlot());
        ransacCheck.selectedProperty().addListener((o, old, val) -> updatePlot());
    }
    
    public void setData(String datasetName, List<CombinedPredictionResponse> data) {
        this.data = data;
        datasetLabel.setText("Dataset: " + datasetName + " (" + (data != null ? data.size() : 0) + " samples)");
        updatePlot();
    }
    
    @FXML
    private void onUpdatePlot() {
        updatePlot();
    }
    
    private void updatePlot() {
        if (data == null || data.isEmpty()) return;
        
        String xDescriptor = xAxisCombo.getValue();
        String yDescriptor = yAxisCombo.getValue();
        String coloring = colorCombo.getValue();
        boolean xLog = logXCheck.isSelected();
        boolean yLog = logYCheck.isSelected();
        boolean showLinear = linearRegCheck.isSelected();
        boolean showRansac = ransacCheck.isSelected();
        
        if (xDescriptor == null || yDescriptor == null) return;
        
        apiClient.getBenchmarkPlot(data, xDescriptor, yDescriptor, coloring, xLog, yLog, showLinear, showRansac)
            .thenAccept(html -> {
                javafx.application.Platform.runLater(() -> {
                     plotWebView.getEngine().getLoadWorker().stateProperty().addListener((obs, oldState, newState) -> {
                        if (newState == javafx.concurrent.Worker.State.SUCCEEDED) {
                             try {
                                JSObject window = (JSObject) plotWebView.getEngine().executeScript("window");
                                window.setMember("javabridge", new JavaBridge());
                             } catch (Exception e) {
                                e.printStackTrace();
                             }
                        }
                     });
                     plotWebView.getEngine().loadContent(html);
                });
            })
            .exceptionally(e -> {
                e.printStackTrace();
                return null;
            });
    }
    
    public class JavaBridge {
        public void onPointClick(int index, String smiles) {
            javafx.application.Platform.runLater(() -> {
                if (data != null && index >= 0 && index < data.size()) {
                    CombinedPredictionResponse result = data.get(index);
                    showMoleculeDetailPopup(result);
                }
            });
        }
    }
    
    private void showMoleculeDetailPopup(CombinedPredictionResponse result) {
         javafx.stage.Stage popup = new javafx.stage.Stage();
         popup.initModality(javafx.stage.Modality.NONE); // Allow interaction with main window
         popup.setTitle("Molecule Details - " + result.smiles);
         
         javafx.scene.layout.VBox content = new javafx.scene.layout.VBox(15);
         content.setStyle("-fx-padding: 20; -fx-background-color: white;");
         
         // Basic Info
         Label smilesLabel = new Label("SMILES: " + result.smiles);
         smilesLabel.setStyle("-fx-font-weight: bold;");
         Label nameLabel = new Label("Name: " + (result.molecule_name != null ? result.molecule_name : "Unknown"));
         
         // Image
         javafx.scene.image.ImageView imageView = new javafx.scene.image.ImageView();
         imageView.setFitWidth(280);
         imageView.setFitHeight(200);
         imageView.setPreserveRatio(true);
         
         java.util.concurrent.CompletableFuture.supplyAsync(() -> {
             try {
                 return apiClient.getSmilesImage(result.smiles);
             } catch (Exception e) {
                 throw new RuntimeException(e);
             }
         }).thenAccept(is -> {
             javafx.application.Platform.runLater(() -> imageView.setImage(new javafx.scene.image.Image(is)));
         }).exceptionally(e -> {
             System.err.println("Failed to load image: " + e.getMessage());
             return null;
         });
         
         // Descriptors Table (Simplified version)
         TableView<DescriptorEntry> table = new TableView<>();
         TableColumn<DescriptorEntry, String> pCol = new TableColumn<>("Property");
         pCol.setCellValueFactory(new PropertyValueFactory<>("name"));
         TableColumn<DescriptorEntry, String> vCol = new TableColumn<>("Value");
         vCol.setCellValueFactory(new PropertyValueFactory<>("value"));
         table.getColumns().addAll(pCol, vCol);
         
         if (result.all_descriptors != null) {
             result.all_descriptors.forEach((k, v) -> {
                 table.getItems().add(new DescriptorEntry(k, v.toString()));
             });
         }
         
         content.getChildren().addAll(nameLabel, smilesLabel, imageView, table);
         
         javafx.scene.Scene scene = new javafx.scene.Scene(content, 400, 600);
         popup.setScene(scene);
         popup.show();
    }
    
    public static class DescriptorEntry {
        private final String name;
        private final String value;
        
        public DescriptorEntry(String name, String value) {
            this.name = name;
            this.value = value;
        }
        
        public String getName() { return name; }
        public String getValue() { return value; }
    }
}
