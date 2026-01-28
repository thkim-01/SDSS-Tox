package com.example.dto.gui;

import java.io.InputStream;
import java.net.URL;
import java.util.List;
import java.util.Map;
import java.util.ResourceBundle;

import com.example.dto.api.DssApiClient;

import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.concurrent.Task;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Alert;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.VBox;
import javafx.stage.Modality;
import javafx.stage.Stage;

/**
 * 배치 분석 회귀 플롯 컨트롤러
 * - 산점도 (X: 선택한 분자 기술자, Y: 독성 확률)
 * - 점 클릭 시 SMILES 이미지 + 분자 속성 테이블 표시
 */
@SuppressWarnings({"unchecked", "unused"})
public class BatchPlotController implements Initializable {

    @FXML private ComboBox<String> descriptorCombo;
    @FXML private ScatterChart<Number, Number> scatterChart;
    @FXML private NumberAxis xAxis;
    @FXML private NumberAxis yAxis;

    private final DssApiClient apiClient = new DssApiClient();
    private List<Map<String, Object>> plotData;

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        // 기본 분자 기술자 목록
        descriptorCombo.setItems(FXCollections.observableArrayList(
            "logP", "tpsa", "mw", "hbd", "hba", "num_rotatable_bonds", 
            "num_aromatic_rings", "num_heavy_atoms", "num_heteroatoms", "fraction_csp3"
        ));
        descriptorCombo.setValue("logP");

        // 분자 기술자 변경 시 차트 업데이트
        descriptorCombo.setOnAction(e -> updateChart());
    }

    @FXML
    private void loadData() {
        Task<List<Map<String, Object>>> task = new Task<>() {
            @Override
            protected List<Map<String, Object>> call() throws Exception {
                return apiClient.getPlotData().get();
            }
        };

        task.setOnSucceeded(e -> {
            plotData = task.getValue();
            updateChart();
        });

        task.setOnFailed(e -> {
            showAlert("오류", "데이터 로드 실패: " + task.getException().getMessage());
        });

        new Thread(task).start();
    }

    private void updateChart() {
        if (plotData == null || plotData.isEmpty()) {
            return;
        }

        String selectedDescriptor = descriptorCombo.getValue();
        xAxis.setLabel(selectedDescriptor);
        yAxis.setLabel("Toxicity Probability");

        scatterChart.getData().clear();
        XYChart.Series<Number, Number> series = new XYChart.Series<>();
        series.setName("Samples");

        for (int i = 0; i < plotData.size(); i++) {
            Map<String, Object> sample = plotData.get(i);
            Map<String, Object> descriptors = (Map<String, Object>) sample.get("all_descriptors");
            
            Object xValObj = descriptors != null ? descriptors.get(selectedDescriptor) : null;
            Object yValObj = sample.get("probability");
            
            if (xValObj != null && yValObj != null) {
                double xVal = ((Number) xValObj).doubleValue();
                double yVal = ((Number) yValObj).doubleValue();
                
                XYChart.Data<Number, Number> dataPoint = new XYChart.Data<>(xVal, yVal);
                // final int index = i; // Removed unused variable
                
                // 데이터 포인트 추가
                series.getData().add(dataPoint);
            }
        }

        scatterChart.getData().add(series);

        // 데이터 포인트에 클릭 이벤트 추가
        for (int i = 0; i < series.getData().size(); i++) {
            XYChart.Data<Number, Number> dataPoint = series.getData().get(i);
            final int index = i;
            
            Platform.runLater(() -> {
                if (dataPoint.getNode() != null) {
                    dataPoint.getNode().setOnMouseClicked(event -> showSmilesDetail(index));
                    dataPoint.getNode().setStyle("-fx-cursor: hand;");
                }
            });
        }
    }

    private void showSmilesDetail(int index) {
        if (plotData == null || index >= plotData.size()) {
            return;
        }

        Map<String, Object> sample = plotData.get(index);
        String smiles = (String) sample.get("smiles");
        Map<String, Object> descriptors = (Map<String, Object>) sample.get("all_descriptors");

        // 모달 다이얼로그 생성
        Stage dialog = new Stage();
        dialog.initModality(Modality.APPLICATION_MODAL);
        dialog.setTitle("분자 상세 정보 - " + smiles);

        VBox content = new VBox(15);
        content.setPadding(new Insets(20));
        content.setStyle("-fx-background-color: white;");

        // SMILES 라벨
        Label smilesLabel = new Label("SMILES: " + smiles);
        smilesLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");

        // SMILES 이미지 (백엔드에서 가져오기)
        ImageView imageView = new ImageView();
        imageView.setFitWidth(300);
        imageView.setFitHeight(300);
        imageView.setPreserveRatio(true);

        Task<Image> imageTask = new Task<>() {
            @Override
            protected Image call() throws Exception {
                InputStream is = apiClient.getSmilesImage(smiles);
                return new Image(is);
            }
        };

        imageTask.setOnSucceeded(e -> imageView.setImage(imageTask.getValue()));
        imageTask.setOnFailed(e -> {
            Label errorLabel = new Label("이미지 로드 실패");
            errorLabel.setStyle("-fx-text-fill: red;");
        });
        new Thread(imageTask).start();

        // 분자 속성 테이블
        TableView<DescriptorEntry> descriptorTable = new TableView<>();
        descriptorTable.setPrefHeight(400); // Increased height for more data

        TableColumn<DescriptorEntry, String> nameCol = new TableColumn<>("속성");
        nameCol.setCellValueFactory(new PropertyValueFactory<>("name"));
        nameCol.setPrefWidth(200);

        TableColumn<DescriptorEntry, String> valueCol = new TableColumn<>("값");
        valueCol.setCellValueFactory(new PropertyValueFactory<>("value"));
        valueCol.setPrefWidth(150);

        descriptorTable.getColumns().addAll(nameCol, valueCol);

        if (descriptors != null) {
            // Define desired order and display names
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

            // Add Index manually from sample data if available
            if (sample.containsKey("index")) {
                descriptorTable.getItems().add(new DescriptorEntry("Index", String.valueOf(sample.get("index"))));
            }

            for (String[] field : fields) {
                String key = field[0];
                String displayName = field[1];
                
                if (key.equals("index")) continue; // Already handled

                if (descriptors.containsKey(key)) {
                    Object val = descriptors.get(key);
                    String valueStr = val != null ? val.toString() : "N/A";
                    if (val instanceof Number num) {
                        if (val instanceof Integer) {
                             valueStr = String.format("%d", num.intValue());
                        } else {
                             valueStr = String.format("%.4f", num.doubleValue());
                        }
                    }
                    descriptorTable.getItems().add(new DescriptorEntry(displayName, valueStr));
                }
            }
        }

        // 독성 확률 표시
        Double probability = sample.get("probability") != null ? 
            ((Number) sample.get("probability")).doubleValue() : null;
        Label probLabel = new Label("독성 확률: " + 
            (probability != null ? String.format("%.2f%%", probability * 100) : "N/A"));
        probLabel.setStyle("-fx-font-size: 16px; -fx-font-weight: bold; " +
            (probability != null && probability > 0.5 ? "-fx-text-fill: #e74c3c;" : "-fx-text-fill: #27ae60;"));

        content.getChildren().addAll(smilesLabel, imageView, probLabel, 
            new Label("분자 상세 정보:"), descriptorTable);

        ScrollPane scrollPane = new ScrollPane(content);
        scrollPane.setFitToWidth(true);

        Scene scene = new Scene(scrollPane, 400, 600);
        dialog.setScene(scene);
        dialog.show();
    }

    private void showAlert(String title, String message) {
        Platform.runLater(() -> {
            Alert alert = new Alert(Alert.AlertType.ERROR);
            alert.setTitle(title);
            alert.setContentText(message);
            alert.showAndWait();
        });
    }

    // 내부 클래스: 테이블용 DTO
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
