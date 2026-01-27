package com.example.dto.gui;

import java.util.List;
import java.util.Map;

import org.semanticweb.owlapi.model.OWLClass;
import org.semanticweb.owlapi.model.OWLDataFactory;
import org.semanticweb.owlapi.model.OWLObjectProperty;
import org.semanticweb.owlapi.model.OWLOntology;
import org.semanticweb.owlapi.reasoner.NodeSet;

import com.example.dto.DtoLoader;

import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ListView;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.control.TreeItem;
import javafx.scene.control.TreeView;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.stage.Modality;
import javafx.stage.Stage;

/**
 * Window for exploring ontology graph (classes and relationships).
 */
public class OntologyGraphWindow extends Stage {
    private final DtoLoader loader;
    private TreeView<String> classTreeView;
    private ListView<String> propertiesListView;
    private TextArea detailsArea;
    private TextField searchField;
    
    public OntologyGraphWindow(DtoLoader loader) {
        this.loader = loader;
        initializeUI();
    }
    
    private void initializeUI() {
        setTitle("온톨로지 탐색기 - Drug Target Ontology");
        initModality(Modality.NONE);
        
        BorderPane root = new BorderPane();
        root.setStyle("-fx-background-color: #f5f5f5;");
        root.setPadding(new Insets(10));
        
        // Top: Search bar
        HBox searchBox = new HBox(10);
        searchBox.setPadding(new Insets(10));
        searchBox.setStyle("-fx-background-color: white; -fx-border-color: #e0e0e0;");
        
        searchField = new TextField();
        searchField.setPromptText("클래스 검색...");
        searchField.setPrefWidth(300);
        HBox.setHgrow(searchField, Priority.ALWAYS);
        
        Button searchBtn = new Button("검색");
        searchBtn.setOnAction(e -> onSearch());
        
        Button refreshBtn = new Button("새로고침");
        refreshBtn.setOnAction(e -> loadOntologyTree());
        
        searchBox.getChildren().addAll(new Label("검색:"), searchField, searchBtn, refreshBtn);
        root.setTop(searchBox);
        
        // Left: Class tree
        VBox leftPane = new VBox(10);
        leftPane.setPadding(new Insets(10));
        leftPane.setPrefWidth(350);
        
        Label classLabel = new Label("클래스 계층");
        classLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");
        
        classTreeView = new TreeView<>();
        classTreeView.setShowRoot(true);
        VBox.setVgrow(classTreeView, Priority.ALWAYS);
        
        classTreeView.getSelectionModel().selectedItemProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal != null) {
                showClassDetails(newVal.getValue());
            }
        });
        
        leftPane.getChildren().addAll(classLabel, classTreeView);
        root.setLeft(leftPane);
        
        // Center: Properties list
        VBox centerPane = new VBox(10);
        centerPane.setPadding(new Insets(10));
        centerPane.setPrefWidth(250);
        
        Label propsLabel = new Label("관계 (Object Properties)");
        propsLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");
        
        propertiesListView = new ListView<>();
        VBox.setVgrow(propertiesListView, Priority.ALWAYS);
        
        propertiesListView.getSelectionModel().selectedItemProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal != null) {
                showPropertyDetails(newVal);
            }
        });
        
        centerPane.getChildren().addAll(propsLabel, propertiesListView);
        root.setCenter(centerPane);
        
        // Right: Details
        VBox rightPane = new VBox(10);
        rightPane.setPadding(new Insets(10));
        rightPane.setPrefWidth(350);
        
        Label detailsLabel = new Label("상세 정보");
        detailsLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");
        
        detailsArea = new TextArea();
        detailsArea.setEditable(false);
        detailsArea.setWrapText(true);
        VBox.setVgrow(detailsArea, Priority.ALWAYS);
        
        rightPane.getChildren().addAll(detailsLabel, detailsArea);
        root.setRight(rightPane);
        
        // Load data
        loadOntologyTree();
        loadProperties();
        
        Scene scene = new Scene(root, 1000, 700);
        setScene(scene);
    }
    
    private void loadOntologyTree() {
        OWLOntology ontology = loader.getOntology();
        OWLDataFactory df = loader.getDataFactory();
        
        // Root: owl:Thing
        TreeItem<String> rootItem = new TreeItem<>("owl:Thing");
        rootItem.setExpanded(true);
        
        // Get top-level classes
        OWLClass thing = df.getOWLThing();
        NodeSet<OWLClass> subClasses = loader.getReasoner().getSubClasses(thing, true);
        
        int count = 0;
        for (OWLClass cls : subClasses.getFlattened()) {
            if (!cls.isOWLNothing() && count < 50) {  // Limit for performance
                String label = loader.getLabel(cls);
                TreeItem<String> item = new TreeItem<>(label + " (" + cls.getIRI().getShortForm() + ")");
                
                // Add placeholder for lazy loading
                item.getChildren().add(new TreeItem<>("로딩 중..."));
                item.expandedProperty().addListener((obs, wasExpanded, isExpanded) -> {
                    if (isExpanded && item.getChildren().size() == 1 && 
                        item.getChildren().get(0).getValue().equals("로딩 중...")) {
                        loadSubClasses(item, cls);
                    }
                });
                
                rootItem.getChildren().add(item);
                count++;
            }
        }
        
        classTreeView.setRoot(rootItem);
    }
    
    private void loadSubClasses(TreeItem<String> parentItem, OWLClass parentClass) {
        parentItem.getChildren().clear();
        
        NodeSet<OWLClass> subClasses = loader.getReasoner().getSubClasses(parentClass, true);
        int count = 0;
        
        for (OWLClass cls : subClasses.getFlattened()) {
            if (!cls.isOWLNothing() && count < 30) {
                String label = loader.getLabel(cls);
                TreeItem<String> item = new TreeItem<>(label + " (" + cls.getIRI().getShortForm() + ")");
                
                // Check if has subclasses
                if (!loader.getReasoner().getSubClasses(cls, true).isEmpty()) {
                    item.getChildren().add(new TreeItem<>("로딩 중..."));
                    item.expandedProperty().addListener((obs, wasExpanded, isExpanded) -> {
                        if (isExpanded && item.getChildren().size() == 1 && 
                            item.getChildren().get(0).getValue().equals("로딩 중...")) {
                            loadSubClasses(item, cls);
                        }
                    });
                }
                
                parentItem.getChildren().add(item);
                count++;
            }
        }
        
        if (parentItem.getChildren().isEmpty()) {
            parentItem.getChildren().add(new TreeItem<>("(하위 클래스 없음)"));
        }
    }
    
    private void loadProperties() {
        OWLOntology ontology = loader.getOntology();
        ObservableList<String> properties = FXCollections.observableArrayList();
        
        for (OWLObjectProperty prop : ontology.getObjectPropertiesInSignature()) {
            String label = loader.getLabel(prop);
            properties.add(label + " [" + prop.getIRI().getShortForm() + "]");
        }
        
        properties.sort(String::compareTo);
        propertiesListView.setItems(properties);
    }
    
    private void showClassDetails(String classInfo) {
        StringBuilder sb = new StringBuilder();
        sb.append("=== 클래스 정보 ===\n\n");
        sb.append("선택: ").append(classInfo).append("\n\n");
        
        // Extract IRI from display string
        if (classInfo.contains("(") && classInfo.contains(")")) {
            String shortForm = classInfo.substring(classInfo.lastIndexOf("(") + 1, classInfo.lastIndexOf(")"));
            
            // Search for this class in ontology
            List<Map<String, String>> found = loader.searchClassesByLabel(
                    classInfo.substring(0, classInfo.lastIndexOf("(")).trim());
            
            if (!found.isEmpty()) {
                Map<String, String> cls = found.get(0);
                sb.append("IRI: ").append(cls.get("iri")).append("\n\n");
            }
        }
        
        sb.append("이 클래스를 분석에 사용하려면 관련 SMILES 패턴을 입력하세요.");
        
        detailsArea.setText(sb.toString());
    }
    
    private void showPropertyDetails(String propertyInfo) {
        StringBuilder sb = new StringBuilder();
        sb.append("=== 관계 정보 ===\n\n");
        sb.append("선택: ").append(propertyInfo).append("\n\n");
        sb.append("이 관계는 온톨로지에서 클래스 간의 연결을 정의합니다.\n");
        sb.append("예: 'has_associated_disease' 관계는 표적과 질병을 연결합니다.");
        
        detailsArea.setText(sb.toString());
    }
    
    private void onSearch() {
        String query = searchField.getText().trim();
        if (query.isEmpty()) return;
        
        List<Map<String, String>> results = loader.searchClassesByLabel(query);
        
        StringBuilder sb = new StringBuilder();
        sb.append("=== 검색 결과: '").append(query).append("' ===\n\n");
        sb.append("발견된 클래스: ").append(results.size()).append("개\n\n");
        
        int count = 0;
        for (Map<String, String> cls : results) {
            if (count >= 20) {
                sb.append("\n... 외 ").append(results.size() - 20).append("개 더");
                break;
            }
            sb.append("• ").append(cls.get("label")).append("\n");
            sb.append("  IRI: ").append(cls.get("iri")).append("\n\n");
            count++;
        }
        
        detailsArea.setText(sb.toString());
    }
}
