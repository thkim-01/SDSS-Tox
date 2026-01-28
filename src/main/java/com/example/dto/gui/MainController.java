package com.example.dto.gui;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.ResourceBundle;

import com.example.dto.core.ChemicalMapper;
import com.example.dto.core.ChemicalMapper.MappingResult;
import com.example.dto.core.DtoLoader;
import com.example.dto.core.DtoQuery;
import com.example.dto.api.DssApiClient;
import com.example.dto.api.DssApiClient.CombinedPredictionResponse;
import com.example.dto.api.DssApiClient.TriggeredRule;
import com.example.dto.api.MolecularDescriptors;
import com.example.dto.data.DatasetLoader;
import com.example.dto.data.DatasetLoader.DatasetInfo;
import com.example.dto.data.DatasetLoader.DatasetSample;

import com.example.dto.visualization.manager.VisualizationManager;
import com.example.dto.visualization.api.ChartType;
import javafx.scene.layout.StackPane;
import java.util.Arrays;

import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.concurrent.Task;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.ListView;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.TableCell;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.TextArea;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.web.WebView;
import netscape.javascript.JSObject;

@SuppressWarnings({"unchecked", "unused", "null"})
public class MainController implements Initializable {
    
    @FXML private TextArea smilesInput;
    @FXML private Button analyzeButton;
    @FXML private Button ontologyBrowserBtn;
    @FXML private TableView<AnalysisResult> resultTable;
    @FXML private TableColumn<AnalysisResult, String> smilesColumn;
    @FXML private TableColumn<AnalysisResult, String> mlPredColumn;
    @FXML private TableColumn<AnalysisResult, String> ontoPredColumn;
    @FXML private TableColumn<AnalysisResult, String> statusColumn;
    @FXML private TextArea explanationArea;
    @FXML private ListView<String> ontologyListView;
    @FXML private Label statusLabel;
    @FXML private ProgressBar progressBar;
    @FXML private Label statsLabel;
    @FXML private Label patternsLabel;

    private final DssApiClient apiClient = new DssApiClient();

    // Benchmark Tab Components
    // Benchmark Tab Components (Definitions moved below)
    @FXML private ProgressBar confidenceBar;
    @FXML private Label confidenceLabel;
    @FXML private Label agreementLabel;
    @FXML private Label triggeredRulesCountLabel;
    @FXML private ListView<String> triggeredRulesListView;
    
    // Updated Plot/XAI Area
    @FXML private WebView benchWebView;
    @FXML private StackPane visualizationContainer;
    @FXML private javafx.scene.layout.HBox visualizationButtonContainer;
    // @FXML private BarChart<Number, String> rfImportanceChart; // Replaced by dynamic visualization
    
    // Store benchmark results for plot
    private List<CombinedPredictionResponse> benchmarkResults = new ArrayList<>();
    
    private DtoLoader dtoLoader;
    private DtoQuery dtoQuery;
    private ChemicalMapper chemicalMapper;
    private boolean ontologyLoaded = false;
    
    private List<FullAnalysisResult> currentResults = new ArrayList<>();
    private OntologyGraphWindow ontologyWindow;
    
    @FXML private ComboBox<DatasetInfo> datasetCombo;
    @FXML private ComboBox<Integer> sampleSizeCombo;
    @FXML private ComboBox<String> modelSelector;
    @FXML private Button runBenchmarkBtn;
    @FXML private Button refreshDatasetsBtn;

    @FXML private TableView<CombinedPredictionResponse> benchmarkTable;
    @FXML private TableColumn<CombinedPredictionResponse, String> benchNameCol;
    @FXML private TableColumn<CombinedPredictionResponse, String> benchSmilesCol;
    @FXML private TableColumn<CombinedPredictionResponse, Double> benchMlCol;
    @FXML private TableColumn<CombinedPredictionResponse, Integer> benchBinaryPredCol;
    @FXML private TableColumn<CombinedPredictionResponse, Double> benchOntoCol;
    @FXML private TableColumn<CombinedPredictionResponse, Double> benchCombinedCol;
    @FXML private TableColumn<CombinedPredictionResponse, Double> benchConfCol;
    @FXML private TableColumn<CombinedPredictionResponse, String> benchRiskCol;

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        // Setup table columns
        smilesColumn.setCellValueFactory(new PropertyValueFactory<>("smiles"));
        mlPredColumn.setCellValueFactory(new PropertyValueFactory<>("mlPrediction"));
        ontoPredColumn.setCellValueFactory(new PropertyValueFactory<>("ontoPrediction"));
        statusColumn.setCellValueFactory(new PropertyValueFactory<>("status"));
        
        // Style status column based on consistency
        statusColumn.setCellFactory(column -> new TableCell<>() {
            @Override
            protected void updateItem(String item, boolean empty) {
                super.updateItem(item, empty);
                if (empty || item == null) {
                    setText(null);
                    setStyle("");
                } else {
                    setText(item);
                    if (item.contains("Match")) {
                        setStyle("-fx-background-color: #d5f4e6; -fx-text-fill: #27ae60; -fx-font-weight: bold;");
                    } else if (item.contains("Mismatch")) {
                        setStyle("-fx-background-color: #fadbd8; -fx-text-fill: #e74c3c; -fx-font-weight: bold;");
                    } else {
                        setStyle("-fx-background-color: #fef9e7; -fx-text-fill: #f39c12;");
                    }
                }
            }
        });
        
        // Table selection listener
        resultTable.getSelectionModel().selectedIndexProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal.intValue() >= 0 && newVal.intValue() < currentResults.size()) {
                showExplanation(currentResults.get(newVal.intValue()));
            }
        });
        
        // Load ontology in background
        loadOntologyAsync();
        
        // Initialize Benchmark Tab
        initializeBenchmarkTab();
        
        // Initialize Advanced Plot
        // onRefreshAdvancedPlot(); // Removed as per user request
        
        // Setup Java Bridge for Benchmark Plot
        benchWebView.getEngine().setJavaScriptEnabled(true);
        benchWebView.getEngine().getLoadWorker().stateProperty().addListener((obs, oldState, newState) -> {
            if (newState == javafx.concurrent.Worker.State.SUCCEEDED) {
                JSObject window = (JSObject) benchWebView.getEngine().executeScript("window");
                window.setMember("javabridge", new JavaBridge());
                window.setMember("javabridge", new JavaBridge());
            }
        });
        
        // Copy/Paste support temporarily disabled (TableUtils not available)
    }
    private void loadOntologyAsync() {
        statusLabel.setText("Loading Ontology...");
        progressBar.setProgress(-1);
        analyzeButton.setDisable(true);
        ontologyBrowserBtn.setDisable(true);
        
        Task<Void> loadTask = new Task<>() {
            @Override
            protected Void call() throws Exception {
                dtoLoader = new DtoLoader();
                dtoLoader.loadOntology("data/ontology/dto.rdf");
                dtoQuery = new DtoQuery(dtoLoader);
                chemicalMapper = new ChemicalMapper(dtoLoader);
                return null;
            }
        };
        
        loadTask.setOnSucceeded(e -> {
            ontologyLoaded = true;
            Map<String, Object> stats = dtoLoader.getStatistics();
            statsLabel.setText(String.format("Classes: %s | Axioms: %s | Individuals: %s",
                    stats.get("classes"), stats.get("axioms"), stats.get("individuals")));
            statusLabel.setText("Ready - Enter SMILES and Run Analysis");
            progressBar.setProgress(1);
            analyzeButton.setDisable(false);
            ontologyBrowserBtn.setDisable(false);
        });
        
        loadTask.setOnFailed(e -> {
            statusLabel.setText("Ontology Load Failed: " + loadTask.getException().getMessage());
            progressBar.setProgress(0);
        });
        
        new Thread(loadTask).start();
    }
    
    @FXML
    private void onOpenOntologyBrowser() {
        if (!ontologyLoaded) {
            showAlert("Notice", "Ontology is still loading.");
            return;
        }
        
        if (ontologyWindow == null) {
            ontologyWindow = new OntologyGraphWindow(dtoLoader);
        }
        ontologyWindow.show();
        ontologyWindow.toFront();
    }
    
    @FXML
    private void onAnalyze() {
        if (!ontologyLoaded) {
            showAlert("Notice", "Ontology is still loading.");
            return;
        }
        
        String smilesText = smilesInput.getText().trim();
        if (smilesText.isEmpty()) {
            showAlert("Notice", "Please enter SMILES.");
            return;
        }
        
        String[] smilesList = smilesText.split("\n");
        statusLabel.setText("Running Integrated Analysis...");
        progressBar.setProgress(-1);
        analyzeButton.setDisable(true);
        
        Task<List<FullAnalysisResult>> analyzeTask = new Task<>() {
            @Override
            protected List<FullAnalysisResult> call() throws Exception {
                return runIntegratedAnalysis(smilesList);
            }
        };
        
        analyzeTask.setOnSucceeded(e -> {
            currentResults = analyzeTask.getValue();
            displayResults(currentResults);
            statusLabel.setText("Analysis Complete - " + currentResults.size() + " molecules processed");
            progressBar.setProgress(1);
            analyzeButton.setDisable(false);
        });
        
        analyzeTask.setOnFailed(e -> {
            statusLabel.setText("Analysis Failed: " + analyzeTask.getException().getMessage());
            progressBar.setProgress(0);
            analyzeButton.setDisable(false);
        });
        
        new Thread(analyzeTask).start();
    }
    
    private List<FullAnalysisResult> runIntegratedAnalysis(String[] smilesList) throws Exception {
        List<FullAnalysisResult> results = new ArrayList<>();
        
        // Step 1: Get ML predictions from Python
        Map<String, double[]> mlPredictions = runMLPrediction(smilesList);
        
        // Step 2: Validate each prediction against ontology using ChemicalMapper
        for (String smiles : smilesList) {
            smiles = smiles.trim();
            if (smiles.isEmpty()) continue;
            
            FullAnalysisResult result = new FullAnalysisResult();
            result.smiles = smiles;
            
            // ML prediction
            double[] pred = mlPredictions.getOrDefault(smiles, new double[]{0.5});
            result.mlToxicity = pred[0];
            result.mlRisk = result.mlToxicity > 0.5 ? "High" : "Low";
            
            // Ontology mapping using ChemicalMapper
            MappingResult mapping = chemicalMapper.mapToOntology(smiles);
            result.mappingResult = mapping;
            result.ontoRisk = mapping.ontologyRisk;
            result.confidence = mapping.confidence;
            
            // Consistency check
            if (mapping.ontologyRisk.equals("Unknown")) {
                result.isConsistent = true;  // Can't contradict
                result.status = "Insufficient Info";
            } else if (result.mlRisk.equals(mapping.ontologyRisk)) {
                result.isConsistent = true;
                result.status = "Match";
            } else {
                result.isConsistent = false;
                result.status = "Mismatch";
            }
            
            // Generate explanation
            result.explanation = generateReport(result);
            
            results.add(result);
        }
        
        return results;
    }
    
    private String generateReport(FullAnalysisResult result) {
        StringBuilder sb = new StringBuilder();
        
        sb.append("═══════════════════════════════════════\n");
        sb.append("           Analysis Report\n");
        sb.append("═══════════════════════════════════════\n\n");
        
                sb.append("Molecule: ").append(result.smiles).append("\n\n");
        
        sb.append("── ML Prediction ──\n");
        sb.append(String.format("  Toxicity Probability: %.1f%%\n", result.mlToxicity * 100));
        sb.append(String.format("  Risk: %s\n\n", result.mlRisk.equals("High") ? "High" : "Low"));
        
        sb.append("── Ontology-based Analysis ──\n");
        sb.append(String.format("  Judgment: %s\n", 
                result.ontoRisk.equals("High") ? "High" : 
                result.ontoRisk.equals("Low") ? "Low" : "Unknown"));
        sb.append(String.format("  Confidence: %.0f%%\n", result.confidence * 100));
        sb.append("  Evidence: ").append(result.mappingResult.explanation).append("\n\n");
        
        if (!result.mappingResult.toxicAlerts.isEmpty()) {
            sb.append("Toxic Structural Alerts:\n");
            for (String alert : result.mappingResult.toxicAlerts) {
                sb.append("  ").append(alert).append("\n");
            }
            sb.append("\n");
        }
        
        sb.append("── Consistency ──\n");
        sb.append("  ").append(result.status).append("\n");
        
        if (!result.isConsistent && !result.ontoRisk.equals("Unknown")) {
            sb.append("\nMismatch Warning:\n");
            sb.append("  ML model and Ontology knowledge disagree.\n");
            sb.append("  Additional review is recommended.\n");
        }
        
        return sb.toString();
    }
    
    private Map<String, double[]> runMLPrediction(String[] smilesList) throws Exception {
        Map<String, double[]> predictions = new HashMap<>();
        
        // Write SMILES to temp file
        String tempFile = System.getProperty("java.io.tmpdir") + "/smiles_input.txt";
        java.nio.file.Files.write(java.nio.file.Paths.get(tempFile), 
                String.join("\n", smilesList).getBytes());
        
        // Run Python prediction script
        ProcessBuilder pb = new ProcessBuilder(
                "python", "src/python/predict_smiles.py", tempFile);
        pb.redirectErrorStream(true);
        Process process = pb.start();
        
        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        String line;
        while ((line = reader.readLine()) != null) {
            if (line.startsWith("RESULT:")) {
                String[] parts = line.substring(7).split("\\|");
                if (parts.length >= 2) {
                    try {
                        double tox = Double.parseDouble(parts[1]);
                        predictions.put(parts[0], new double[]{tox});
                    } catch (NumberFormatException ex) {
                        predictions.put(parts[0], new double[]{0.5});
                    }
                }
            }
        }
        process.waitFor();
        
        return predictions;
    }
    
    private void displayResults(List<FullAnalysisResult> results) {
        ObservableList<AnalysisResult> items = FXCollections.observableArrayList();
        
        for (FullAnalysisResult r : results) {
            String mlPred = String.format("%.0f%% (%s)", r.mlToxicity * 100, 
                    r.mlRisk.equals("High") ? "High" : "Low");
            String ontoPred = r.ontoRisk.equals("High") ? "High" : 
                              r.ontoRisk.equals("Low") ? "Low" : "Unknown";
            
            items.add(new AnalysisResult(r.smiles, mlPred, ontoPred, r.status));
        }
        
        resultTable.setItems(items);
        
        // Show first result explanation
        if (!results.isEmpty()) {
            resultTable.getSelectionModel().selectFirst();
        }
    }
    
    private void showExplanation(FullAnalysisResult result) {
        explanationArea.setText(result.explanation);
        
        // Show matched patterns
        if (result.mappingResult.matchedPatterns.isEmpty()) {
            patternsLabel.setText("None");
        } else {
            patternsLabel.setText(String.join(", ", result.mappingResult.matchedPatterns));
        }
        
        // Show related ontology classes
        ObservableList<String> classes = FXCollections.observableArrayList();
        for (Map<String, String> cls : result.mappingResult.ontologyClasses) {
            String source = cls.getOrDefault("source", "unknown");
            String pattern = cls.getOrDefault("pattern", "");
            classes.add(cls.get("label") + " [" + pattern + "]");
        }
        
        if (classes.isEmpty()) {
            classes.add("(No mapped classes)");
        }
        
        ontologyListView.setItems(classes);
    }
    
    private void showAlert(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle(title);
        alert.setContentText(message);
        alert.showAndWait();
    }
    
    private void initializeBenchmarkTab() {
        // Setup Combo Boxes
        sampleSizeCombo.setItems(FXCollections.observableArrayList(10, 20, 50, 100, 10000)); // 10000 represents "Full"
        sampleSizeCombo.setConverter(new javafx.util.StringConverter<Integer>() {
            @Override
            public String toString(Integer object) {
                if (object == null) return null;
                return object >= 10000 ? "Full" : object.toString();
            }
            @Override
            public Integer fromString(String string) {
                if ("Full".equals(string)) return 10000;
                return Integer.valueOf(string);
            }
        });
        sampleSizeCombo.setValue(10);
        
        // Setup Dataset Combo with explicit converter and cell factory
        datasetCombo.setConverter(new javafx.util.StringConverter<DatasetInfo>() {
            @Override
            public String toString(DatasetInfo object) {
                if (object == null) return null;
                return String.format("%s (%d samples)", object.name, object.size);
            }
            @Override
            public DatasetInfo fromString(String string) {
                return null; // Not needed for read-only combo
            }
        });
        
        datasetCombo.setCellFactory(param -> new javafx.scene.control.ListCell<DatasetInfo>() {
            @Override
            protected void updateItem(DatasetInfo item, boolean empty) {
                super.updateItem(item, empty);
                if (empty || item == null) {
                    setText(null);
                } else {
                    setText(String.format("%s (%s) - %d samples", item.name, item.type, item.size));
                }
            }
        });
        
        // Setup Model Selector
        modelSelector.setItems(FXCollections.observableArrayList("Semantic Decision Tree (SDT)", "Decision Tree", "Random Forest"));
        modelSelector.setValue("Semantic Decision Tree (SDT)");
        
        // Load datasets from Backend via ApiClient
        loadDatasets();
        
        // Setup Table Columns
        benchNameCol.setCellValueFactory(new PropertyValueFactory<>("molecule_name"));
        benchSmilesCol.setCellValueFactory(new PropertyValueFactory<>("smiles"));
        benchMlCol.setCellValueFactory(new PropertyValueFactory<>("ml_prediction"));
        benchBinaryPredCol.setCellValueFactory(new PropertyValueFactory<>("binary_prediction"));
        benchOntoCol.setCellValueFactory(new PropertyValueFactory<>("ontology_score"));
        benchCombinedCol.setCellValueFactory(new PropertyValueFactory<>("combined_score"));
        benchConfCol.setCellValueFactory(new PropertyValueFactory<>("confidence"));
        benchRiskCol.setCellValueFactory(new PropertyValueFactory<>("risk_level"));
        
        // Format doubles
        benchMlCol.setCellFactory(col -> new TableCell<>() {
            @Override
            protected void updateItem(Double item, boolean empty) {
                super.updateItem(item, empty);
                if (empty || item == null) {
                    setText(null);
                } else {
                    // Display as probability (0.000 - 1.000)
                    setText(String.format("%.3f", item));
                }
            }
        });
        benchOntoCol.setCellFactory(col -> new TableCell<>() {
            @Override
            protected void updateItem(Double item, boolean empty) {
                super.updateItem(item, empty);
                if (empty || item == null) {
                    setText(null);
                } else {
                    setText(String.format("%.3f", item));
                }
            }
        });
        benchCombinedCol.setCellFactory(col -> new TableCell<>() {
            @Override
            protected void updateItem(Double item, boolean empty) {
                super.updateItem(item, empty);
                if (empty || item == null) {
                    setText(null);
                } else {
                    setText(String.format("%.3f", item));
                }
            }
        });
        benchConfCol.setCellFactory(col -> new TableCell<>() {
            @Override
            protected void updateItem(Double item, boolean empty) {
                super.updateItem(item, empty);
                if (empty || item == null) {
                    setText(null);
                } else {
                    setText(String.format("%.0f%%", item * 100));
                }
            }
        });
        
        // Setup Button Action
        // Setup Button Action
        runBenchmarkBtn.setOnAction(e -> onRunBenchmark());
        if (refreshDatasetsBtn != null) {
            refreshDatasetsBtn.setOnAction(e -> onRefreshDatasets());
        }
        
        // Setup Table Selection
        benchmarkTable.getSelectionModel().selectedItemProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal != null) {
                updateVisualization(newVal);
            }
        });
    }

    private void updateBenchPlot() {
        String selectedModel = modelSelector.getValue();
        if (selectedModel == null) selectedModel = "Random Forest"; // Default safety
        
        // Clear containers
        if (visualizationContainer != null) visualizationContainer.getChildren().clear();
        if (visualizationButtonContainer != null) visualizationButtonContainer.getChildren().clear();

        // Automated Visualization Logic (No Manual Buttons)
        if ("Random Forest".equals(selectedModel)) {
            // Priority 1: Feature Importance
            triggerAutomatedVisualization(selectedModel, ChartType.FEATURE_IMPORTANCE);
        } else if (selectedModel.contains("Decision Tree") || selectedModel.contains("SDT")) {
            // Priority 1: Decision Boundary (Automated per request)
            triggerAutomatedVisualization(selectedModel, ChartType.SCATTER_PLOT_2D);
        } else {
             // Fallback
             if (benchWebView != null) {
                visualizationContainer.getChildren().add(benchWebView);
                benchWebView.setVisible(true);
                benchWebView.getEngine().loadContent("<html><body><h3 style='text-align:center;margin-top:50px;'>No visualization available for " + selectedModel + "</h3></body></html>");
             }
        }
    }

    private void triggerAutomatedVisualization(String model, ChartType type) {
        // Generate dummy data if needed
        Map<String, Object> data = new HashMap<>();
        data.put("model", model);
        if (model.equals("Random Forest") && type == ChartType.FEATURE_IMPORTANCE) {
             data.put("features", Arrays.asList("MolWt", "LogP", "TPSA", "H-Donors", "H-Acceptors", "RotBonds", "Rings"));
             data.put("importance", Arrays.asList(0.35, 0.25, 0.15, 0.10, 0.08, 0.05, 0.02));
        }

        if (sampleSizeCombo != null && sampleSizeCombo.getValue() != null) {
            data.put("sampleSize", sampleSizeCombo.getValue());
        }
        
        // Pass Benchmark Results if available
        // Use benchmarkResults first if available, otherwise currentResults (fallback)
        if (benchmarkResults != null && !benchmarkResults.isEmpty()) {
            data.put("results", new ArrayList<>(benchmarkResults));
            // Pass dataset name if selected
            if (datasetCombo.getValue() != null) {
                data.put("dataset", datasetCombo.getValue().name);
            }
            // Trigger auto-run for renderers that support it (e.g., DecisionBoundary)
            data.put("autoRun", true);
        }
        
        javafx.scene.Node chart = VisualizationManager.getInstance().render(model, type, data);
        
        if (visualizationContainer != null) {
            visualizationContainer.getChildren().clear();
            if (chart != null) visualizationContainer.getChildren().add(chart);
        }
    }

    // addVisualizationButton removed as part of automation cleanup
    
    
    @FXML
    public void onOpenDataAnalysis() {
        try {
            javafx.fxml.FXMLLoader loader = new javafx.fxml.FXMLLoader(getClass().getResource("/data_analysis.fxml"));
            javafx.scene.Parent root = loader.load();
            
            DataAnalysisController controller = loader.getController();
            String datasetName = datasetCombo.getValue() != null ? datasetCombo.getValue().name : "Unknown";
            controller.setData(datasetName, new ArrayList<>(benchmarkResults));
            
            javafx.stage.Stage stage = new javafx.stage.Stage();
            stage.setTitle("Data Analysis Tool");
            stage.setScene(new javafx.scene.Scene(root, 1000, 700));
            stage.show();
        } catch (Exception e) {
            e.printStackTrace();
            showAlert("Error", "Failed to open Data Analysis: " + e.getMessage());
        }
    }
    
    /**
     * Java-JavaScript Bridge for Plotly Click Events
     */
    public class JavaBridge {
        public void onPointClick(int index, String smiles) {
            javafx.application.Platform.runLater(() -> {
                if (index >= 0 && index < benchmarkResults.size()) {
                    CombinedPredictionResponse result = benchmarkResults.get(index);
                    benchmarkTable.getSelectionModel().select(index);
                    showMoleculeDetailPopup(result);
                }
            });
        }
    }
    
    @FXML
    private void onRunBenchmark() {
        DatasetInfo selectedDataset = datasetCombo.getValue();
        Integer sampleSize = sampleSizeCombo.getValue();
        
        if (selectedDataset == null || sampleSize == null) {
            showAlert("Error", "Please select a dataset and sample size.");
            return;
        }
        
        runBenchmarkBtn.setDisable(true);
        statusLabel.textProperty().unbind();
        statusLabel.setText("Running benchmark analysis...");
        
        String selectedModel = modelSelector.getValue();
        if (selectedModel == null) selectedModel = "Random Forest";
        // Map UI name to API key
        String modelKey = "random_forest";
        if (selectedModel.equals("Decision Tree")) modelKey = "decision_tree";
        else if (selectedModel.contains("SDT")) modelKey = "sdt";
        
        final String finalModelKey = modelKey;
        
        // Use DssApiClient to get predictions
        Task<List<CombinedPredictionResponse>> benchmarkTask = new Task<>() {
            @Override
            protected List<CombinedPredictionResponse> call() throws Exception {
                // Load samples from Backend
                List<DatasetSample> samples = apiClient.getDatasetSamples(selectedDataset.name, sampleSize).join();
                
                List<CombinedPredictionResponse> results = new ArrayList<>();
                int total = samples.size();
                int count = 0;
                
                // 1. Get RF Feature Importance (Legacy logic - needs migration to VisualizationManager)
                /*
                try {
                    Map<String, Object> importance = apiClient.getRfFeatureImportance();
                    javafx.application.Platform.runLater(() -> updateRfFeatureImportanceChart(importance));
                } catch (Exception e) {
                    System.err.println("Failed to get RF importance: " + e.getMessage());
                }
                */
                
                for (DatasetSample sample : samples) {
                    try {
                        // 2. Predict
                        MolecularDescriptors descriptors = new MolecularDescriptors();
                        if (count < 3) {
                            System.out.println("Sample " + count + " features keys: " + sample.features.keySet());
                            System.out.println("Sample " + count + " raw MW: " + MainController.this.getFeatureValue(sample.features, "MW", "MolWt", "MolecularWeight", "Molecular Weight"));
                        }
                        descriptors.setMw(MainController.this.getFeatureValue(sample.features, "MW", "MolWt", "MolecularWeight", "Molecular Weight"));
                        descriptors.setLogKow(MainController.this.getFeatureValue(sample.features, "logKow", "LogKow", "LogP", "logP", "MolLogP"));
                        descriptors.setHbd(MainController.this.getFeatureValue(sample.features, "HBD", "NumHDonors", "nHBAcc").intValue());
                        descriptors.setHba(MainController.this.getFeatureValue(sample.features, "HBA", "NumHAcceptors", "nHBDon").intValue());
                        descriptors.setnRotB(MainController.this.getFeatureValue(sample.features, "nRotB", "NumRotatableBonds", "Rotatable Bonds").intValue());
                        descriptors.setTpsa(MainController.this.getFeatureValue(sample.features, "TPSA", "tpsa", "TopoPSA"));
                        descriptors.setLogP(MainController.this.getFeatureValue(sample.features, "logP", "LogP", "MolLogP", "ALogP"));
                        descriptors.setAromaticRings(MainController.this.getFeatureValue(sample.features, "Aromatic_Rings", "NumAromaticRings", "nRing", "Ring Count").intValue());
                        descriptors.setHeteroatomCount(MainController.this.getFeatureValue(sample.features, "Heteroatom_Count", "NumHeteroatoms", "nHetero").intValue());
                        descriptors.setHeavyAtomCount(MainController.this.getFeatureValue(sample.features, "Heavy_Atom_Count", "NumHeavyAtoms", "nHeavy").intValue());
                        
                        if (count < 2) {
                            System.out.println("Processing Sample " + count + ": " + sample.smiles);
                            System.out.println("Computed Descriptors: " + descriptors);
                        }

                        CombinedPredictionResponse result = apiClient.predictCombined(sample.smiles, descriptors, finalModelKey).join();
                        
                        // 3. Get Molecule Name - Use name from dataset sample
                        result.molecule_name = sample.name != null ? sample.name : "Unknown";
                        
                        results.add(result);
                    } catch (Exception e) {
                        System.err.println("Failed to process SMILES: " + sample.smiles + " - " + e.getMessage());
                    }
                    
                    count++;
                    updateProgress(count, total);
                    updateMessage(String.format("Analyzing... (%d/%d)", count, total));
                }
                return results;
            }
        };
        
        benchmarkTask.setOnSucceeded(e -> {
            List<CombinedPredictionResponse> results = benchmarkTask.getValue();
            benchmarkResults.clear();
            benchmarkResults.addAll(results);
            benchmarkTable.setItems(FXCollections.observableArrayList(results));
            statusLabel.textProperty().unbind();
            statusLabel.setText("Benchmark analysis complete. Total: " + results.size());
            runBenchmarkBtn.setDisable(false);
            
            // 회귀 플롯 업데이트
            updateBenchPlot();
            
            if (!results.isEmpty()) {
                benchmarkTable.getSelectionModel().select(0);
            }
        });
        
        benchmarkTask.setOnFailed(e -> {
            showAlert("Error", "Benchmark analysis failed: " + benchmarkTask.getException().getMessage());
            runBenchmarkBtn.setDisable(false);
            statusLabel.textProperty().unbind();
            statusLabel.setText("Analysis failed.");
        });
        
        progressBar.progressProperty().bind(benchmarkTask.progressProperty());
        statusLabel.textProperty().bind(benchmarkTask.messageProperty());
        
        new Thread(benchmarkTask).start();
    }
    
    private void updateVisualization(CombinedPredictionResponse result) {
        // Update Confidence Gauge
        confidenceBar.setProgress(result.confidence);
        confidenceLabel.setText(String.format("%.1f%% (%s)", result.confidence * 100, 
            result.confidence_level != null ? result.confidence_level : "-"));
        
        // Update Agreement Label
        double agreement = 1.0 - Math.abs(result.ml_prediction - result.ontology_score);
        String agreementText;
        if (agreement >= 0.8) {
            agreementText = "High";
            agreementLabel.setStyle("-fx-text-fill: #27ae60; -fx-font-weight: bold;");
        } else if (agreement >= 0.5) {
            agreementText = "Medium";
            agreementLabel.setStyle("-fx-text-fill: #f39c12; -fx-font-weight: bold;");
        } else {
            agreementText = "Low";
            agreementLabel.setStyle("-fx-text-fill: #e74c3c; -fx-font-weight: bold;");
        }
        agreementLabel.setText(agreementText);
        
        // Update Triggered Rules Count and List
        int ruleCount = (result.triggered_rules != null) ? result.triggered_rules.size() : 0;
        triggeredRulesCountLabel.setText(ruleCount + "");
        if (ruleCount > 0) {
            triggeredRulesCountLabel.setStyle("-fx-text-fill: #e74c3c; -fx-font-weight: bold;");
        } else {
            triggeredRulesCountLabel.setStyle("-fx-text-fill: #27ae60;");
        }
        
        // Populate Ontology Rules ListView (Why Explanation)
        triggeredRulesListView.getItems().clear();
        if (result.triggered_rules != null && !result.triggered_rules.isEmpty()) {
            for (TriggeredRule rule : result.triggered_rules) {
                String direction = "positive".equals(rule.toxicity_direction) ? "↑ Incr. Tox" : "↓ Decr. Tox";
                String ruleText = String.format("[%s] %s: %s (Weight: %.2f, %s)",
                    rule.category != null ? rule.category : "Other",
                    rule.name != null ? rule.name : "Rule",
                    rule.detailed_reason != null && !rule.detailed_reason.isEmpty() 
                        ? rule.detailed_reason 
                        : (rule.interpretation != null ? rule.interpretation : "No detail"),
                    rule.weight,
                    direction);
                triggeredRulesListView.getItems().add(ruleText);
            }
        } else {
            triggeredRulesListView.getItems().add("No ontology rules triggered");
        }
    }
    
    // Inner class for descriptor table
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

    /**
     * 분자 상세 정보 팝업 표시
     * 산점도 점 클릭 시 호출됩니다.
     */
    private void showMoleculeDetailPopup(CombinedPredictionResponse result) {
        javafx.stage.Stage popup = new javafx.stage.Stage();
        popup.initModality(javafx.stage.Modality.APPLICATION_MODAL);
        popup.setTitle("Molecule Details - " + result.smiles);
        
        javafx.scene.layout.VBox content = new javafx.scene.layout.VBox(15);
        content.setStyle("-fx-padding: 20; -fx-background-color: white;");
        
        // SMILES Label
        Label smilesLabel = new Label("SMILES: " + result.smiles);
        smilesLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");
        
        // Name Label
        Label nameLabel = new Label("Name: " + (result.molecule_name != null ? result.molecule_name : "Unknown"));
        nameLabel.setStyle("-fx-font-size: 14px; -fx-text-fill: #2980b9;");
        
        // Molecule Image (from backend)
        javafx.scene.image.ImageView imageView = new javafx.scene.image.ImageView();
        imageView.setFitWidth(280);
        imageView.setFitHeight(200);
        imageView.setPreserveRatio(true);
        
        // Load image asynchronously
        javafx.concurrent.Task<javafx.scene.image.Image> imageTask = new javafx.concurrent.Task<>() {
            @Override
            protected javafx.scene.image.Image call() throws Exception {
                java.io.InputStream is = apiClient.getSmilesImage(result.smiles);
                return new javafx.scene.image.Image(is);
            }
        };
        imageTask.setOnSucceeded(e -> imageView.setImage(imageTask.getValue()));
        imageTask.setOnFailed(e -> {
            Label errorLabel = new Label("Image Load Failed");
            errorLabel.setStyle("-fx-text-fill: red;");
        });
        new Thread(imageTask).start();
        
        // Scores Panel
        javafx.scene.layout.GridPane scoresGrid = new javafx.scene.layout.GridPane();
        scoresGrid.setHgap(15);
        scoresGrid.setVgap(8);
        scoresGrid.add(new Label("ML Prediction:"), 0, 0);
        scoresGrid.add(new Label(String.format("%.3f", result.ml_prediction)), 1, 0);
        scoresGrid.add(new Label("Ontology Score:"), 0, 1);
        scoresGrid.add(new Label(String.format("%.3f", result.ontology_score)), 1, 1);
        scoresGrid.add(new Label("Combined Score:"), 0, 2);
        Label combinedLabel = new Label(String.format("%.3f (%s)", result.combined_score, result.risk_level));
        combinedLabel.setStyle(result.combined_score > 0.5 ? "-fx-text-fill: #e74c3c; -fx-font-weight: bold;" : "-fx-text-fill: #27ae60;");
        scoresGrid.add(combinedLabel, 1, 2);
        scoresGrid.add(new Label("Confidence:"), 0, 3);
        scoresGrid.add(new Label(String.format("%.1f%% (%s)", result.confidence * 100, result.confidence_level)), 1, 3);
        
        // Detailed Descriptors Table
        TableView<DescriptorEntry> descriptorTable = new TableView<>();
        descriptorTable.setPrefHeight(300);
        
        TableColumn<DescriptorEntry, String> propCol = new TableColumn<>("Property");
        propCol.setCellValueFactory(new PropertyValueFactory<>("name"));
        propCol.setPrefWidth(180);
        
        TableColumn<DescriptorEntry, String> valCol = new TableColumn<>("Value");
        valCol.setCellValueFactory(new PropertyValueFactory<>("value"));
        valCol.setPrefWidth(150);
        
        descriptorTable.getColumns().addAll(propCol, valCol);
        
        if (result.all_descriptors != null) {
            String[][] fields = {
                {"index", "Index"},
                {"HBA", "H-Ac"},
                {"HBD", "H-Don"},
                {"logP", "logP"},
                {"TPSA", "tPSA"},
                {"nRotB", "n Rot.Bns"},
                {"Heavy_Atom_Count", "n heavy"},
                {"Aromatic_Rings", "n rings"},
                {"hERG", "hERG"},
                {"CACO2", "CACO2"},
                {"CLint_human", "CLint_human"},
                {"HepG2_cytotox", "HepG2_cytotox"},
                {"Solubility", "solubility"},
                {"Fub_human", "Fub_human"},
                {"Source", "Source"},
                {"Lipinski_Score", "Lipinski's score"}
            };
            
            for (String[] field : fields) {
                String key = field[0];
                String displayName = field[1];
                if (result.all_descriptors.containsKey(key)) {
                    Object val = result.all_descriptors.get(key);
                    String valStr = val != null ? val.toString() : "N/A";
                    if (val instanceof Number num) {
                        if (val instanceof Integer) {
                             valStr = String.format("%d", num.intValue());
                        } else {
                             valStr = String.format("%.4f", num.doubleValue());
                        }
                    }
                    descriptorTable.getItems().add(new DescriptorEntry(displayName, valStr));
                }
            }
        }
        
        // Ontology Rules (Why Explanation)
        Label rulesTitle = new Label("Ontology Rules (Why?):");
        rulesTitle.setStyle("-fx-font-weight: bold; -fx-font-size: 13px;");
        
        ListView<String> rulesList = new ListView<>();
        rulesList.setPrefHeight(100);
        if (result.triggered_rules != null && !result.triggered_rules.isEmpty()) {
            for (TriggeredRule rule : result.triggered_rules) {
                String direction = "positive".equals(rule.toxicity_direction) ? "Incr. Tox" : "Decr. Tox";
                rulesList.getItems().add(String.format("[%s] %s - %s (%s)",
                    rule.category != null ? rule.category : "Other",
                    rule.name != null ? rule.name : "Rule",
                    rule.detailed_reason != null ? rule.detailed_reason : rule.interpretation,
                    direction));
            }
        } else {
            rulesList.getItems().add("No rules triggered");
        }
        
        // SHAP Visualization (Local Feature Contribution)
        Label shapTitle = new Label("Feature Contributions (SHAP):");
        shapTitle.setStyle("-fx-font-weight: bold; -fx-font-size: 13px;");
        
        javafx.scene.Node shapChart = null;
        if (result.shap_features != null && !result.shap_features.isEmpty()) {
            // Prepare data for visualization
            Map<String, Object> shapData = new HashMap<>();
            shapData.put("shap_features", result.shap_features);
            
            VisualizationManager vizManager = VisualizationManager.getInstance();
            shapChart = vizManager.render("SHAP", ChartType.SHAP_SUMMARY, shapData);
        } else {
            shapChart = new Label("No SHAP data available (RF model required)");
        }
        
        // Close Button
        Button closeBtn = new Button("Close");
        closeBtn.setOnAction(e -> popup.close());
        closeBtn.setStyle("-fx-background-color: #3498db; -fx-text-fill: white; -fx-padding: 8 20;");
        
        javafx.scene.control.ScrollPane scrollPane = new javafx.scene.control.ScrollPane();
        javafx.scene.layout.VBox scrollContent = new javafx.scene.layout.VBox(15);
        scrollContent.getChildren().addAll(nameLabel, smilesLabel, imageView, scoresGrid, 
            new Label("Molecule Details:"), descriptorTable,
            rulesTitle, rulesList,
            shapTitle, shapChart,
            closeBtn);
        scrollContent.setStyle("-fx-padding: 10; -fx-background-color: white;");
        scrollPane.setContent(scrollContent);
        scrollPane.setFitToWidth(true);
        
        javafx.scene.Scene scene = new javafx.scene.Scene(scrollPane, 450, 700);
        popup.setScene(scene);
        popup.show();
    }

    // Inner class for table display
    public static class AnalysisResult {
        private final String smiles;
        private final String mlPrediction;
        private final String ontoPrediction;
        private final String status;
        
        public AnalysisResult(String smiles, String mlPrediction, String ontoPrediction, String status) {
            this.smiles = smiles;
            this.mlPrediction = mlPrediction;
            this.ontoPrediction = ontoPrediction;
            this.status = status;
        }
        
        public String getSmiles() { return smiles; }
        public String getMlPrediction() { return mlPrediction; }
        public String getOntoPrediction() { return ontoPrediction; }
        public String getStatus() { return status; }
    }
    
    // Full analysis result with all details
    private static class FullAnalysisResult {
        String smiles;
        double mlToxicity;
        String mlRisk;
        String ontoRisk;
        double confidence;
        boolean isConsistent;
        String status;
        String explanation;
        MappingResult mappingResult;
    }

    @FXML
    public void onRefreshDatasets() {
        if (refreshDatasetsBtn != null) {
            refreshDatasetsBtn.setDisable(true);
        }
        loadDatasets();
    }

    private void loadDatasets() {
        apiClient.listDatasets()
            .thenAccept(datasets -> {
                javafx.application.Platform.runLater(() -> {
                     datasetCombo.setItems(FXCollections.observableArrayList(datasets));
                     if (!datasets.isEmpty()) {
                         datasetCombo.setValue(datasets.get(0));
                     } else {
                         showAlert("Debug Info", "Backend returned 0 datasets.\nPlease check the backend console logs for 'Found data directory at:'.");
                     }
                     if (refreshDatasetsBtn != null) {
                         refreshDatasetsBtn.setDisable(false);
                     }
                });
            })
            .exceptionally(e -> {
                System.err.println("Failed to load datasets: " + e.getMessage());
                javafx.application.Platform.runLater(() -> {
                     showAlert("Connection Error", "Failed to load datasets from backend.\nPlease check if the server is running.\n\nError: " + e.getMessage());
                     if (refreshDatasetsBtn != null) {
                         refreshDatasetsBtn.setDisable(false);
                     }
                });
                return null;
            });
    }

    private Double getFeatureValue(Map<String, Double> features, String... keys) {
        for (String key : keys) {
            if (features.containsKey(key)) {
                return features.get(key);
            }
        }
        return 0.0;
    }
}
