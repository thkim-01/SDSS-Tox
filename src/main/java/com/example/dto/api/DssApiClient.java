package com.example.dto.api;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;

/**
 * FastAPI 백엔드와 통신하는 HTTP 클라이언트.
 * 
 * Phase 5.2: JavaFX GUI 연동
 * - /predictions/randomforest - RF 예측
 * - /explainability/shap - SHAP 설명
 * - /ontology/validate - DTO 규칙 검증
 * - /analysis/ensemble - 앙상블 분석
 */
public class DssApiClient {
    
    private final HttpClient httpClient;
    private final Gson gson;
    private final String baseUrl;
    
    private static final int TIMEOUT_SECONDS = 30;
    
    /**
     * 기본 URL로 클라이언트 생성 (localhost:8000)
     */
    public DssApiClient() {
        this("http://localhost:8000");
    }
    
    /**
     * 지정된 URL로 클라이언트 생성
     * @param baseUrl API 서버 기본 URL
     */
    public DssApiClient(String baseUrl) {
        this.baseUrl = baseUrl;
        this.httpClient = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(Duration.ofSeconds(TIMEOUT_SECONDS))
                .build();
        this.gson = new GsonBuilder()
                .setPrettyPrinting()
                .create();
    }
    
    /**
     * API 서버 헬스체크
     * @return 서버 상태 정보
     */
    public CompletableFuture<HealthResponse> checkHealth() {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/health"))
                .GET()
                .build();
        
        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        return gson.fromJson(response.body(), HealthResponse.class);
                    } else {
                        throw new RuntimeException("Health check failed: " + response.statusCode());
                    }
                });
    }
    
    /**
     * RandomForest 예측 수행
     * @param chemicalId 화학물질 ID
     * @param descriptors 10개 분자 기술자
     * @return 예측 결과
     */
    public CompletableFuture<PredictionResponse> predictToxicity(
            String chemicalId, 
            MolecularDescriptors descriptors) {
        
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("chemical_id", chemicalId);
        requestBody.add("descriptors", gson.toJsonTree(descriptors));
        
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/predictions/randomforest"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(requestBody)))
                .build();
        
        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        return gson.fromJson(response.body(), PredictionResponse.class);
                    } else {
                        throw new RuntimeException("Prediction failed: " + response.body());
                    }
                });
    }
    
    /**
     * SHAP 기반 예측 설명 요청
     * @param chemicalId 화학물질 ID
     * @param descriptors 10개 분자 기술자
     * @param targetClass 타겟 클래스 (0=Safe, 1=Moderate, 2=Toxic)
     * @return SHAP 설명 결과
     */
    public CompletableFuture<ShapResponse> explainPrediction(
            String chemicalId,
            MolecularDescriptors descriptors,
            int targetClass) {
        
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("chemical_id", chemicalId);
        requestBody.add("descriptors", gson.toJsonTree(descriptors));
        requestBody.addProperty("target_class", targetClass);
        
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/explainability/shap"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(requestBody)))
                .build();
        
        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        return gson.fromJson(response.body(), ShapResponse.class);
                    } else {
                        throw new RuntimeException("SHAP explanation failed: " + response.body());
                    }
                });
    }
    
    /**
     * 온톨로지 규칙 기반 검증
     * @param chemicalId 화학물질 ID
     * @param descriptors 10개 분자 기술자
     * @return 검증 결과
     */
    public CompletableFuture<ValidationResponse> validateWithOntology(
            String chemicalId,
            MolecularDescriptors descriptors) {
        
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("chemical_id", chemicalId);
        requestBody.add("descriptors", gson.toJsonTree(descriptors));
        
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/ontology/validate"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(requestBody)))
                .build();
        
        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        return gson.fromJson(response.body(), ValidationResponse.class);
                    } else {
                        throw new RuntimeException("Validation failed: " + response.body());
                    }
                });
    }
    
    /**
     * 앙상블 분석 수행 (4가지 방법 통합)
     * @param chemicalId 화학물질 ID
     * @param descriptors 10개 분자 기술자
     * @return 앙상블 분석 결과
     */
    public CompletableFuture<EnsembleResponse> analyzeEnsemble(
            String chemicalId,
            MolecularDescriptors descriptors) {
        
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("chemical_id", chemicalId);
        requestBody.add("descriptors", gson.toJsonTree(descriptors));
        
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/analysis/ensemble"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(requestBody)))
                .build();
        
        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        return gson.fromJson(response.body(), EnsembleResponse.class);
                    } else {
                        throw new RuntimeException("Ensemble analysis failed: " + response.body());
                    }
                });
    }



    // ==================== Batch Plot API ====================
    public CompletableFuture<java.util.List<Map<String, Object>>> getPlotData() {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/analysis/plot-data"))
                .GET()
                .build();
        
        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        java.lang.reflect.Type listType = new com.google.gson.reflect.TypeToken<java.util.List<Map<String, Object>>>(){}.getType();
                        return gson.fromJson(response.body(), listType);
                    } else {
                        throw new RuntimeException("Get plot data failed: " + response.body());
                    }
                });
    }

    public java.io.InputStream getSmilesImage(String smiles) throws java.io.IOException {
        String encoded = java.net.URLEncoder.encode(smiles, java.nio.charset.StandardCharsets.UTF_8);
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/analysis/smiles-image?smiles=" + encoded))
                .GET()
                .build();
        
        try {
            HttpResponse<java.io.InputStream> response = httpClient.send(request, HttpResponse.BodyHandlers.ofInputStream());
            if (response.statusCode() == 200) {
                return response.body();
            } else {
                throw new java.io.IOException("Failed to get SMILES image, status: " + response.statusCode());
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new java.io.IOException("Interrupted while getting SMILES image", e);
        }
    }

    // ==================== Model Management API ====================
    public CompletableFuture<String> addModel(String type, String path, Map<String, Object> config) {
        JsonObject json = new JsonObject();
        json.addProperty("model_type", type);
        json.addProperty("model_path", path);
        if (config != null) {
            json.add("config", gson.toJsonTree(config));
        }

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/models/add"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(json)))
                .build();

        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        return "Success";
                    } else {
                        throw new RuntimeException("Add model failed: " + response.body());
                    }
                });
    }

    public CompletableFuture<String> switchModel(String type, String path) {
        JsonObject json = new JsonObject();
        json.addProperty("model_type", type);
        json.addProperty("model_path", path);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/models/switch"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(json)))
                .build();

        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        return "Success";
                    } else {
                        throw new RuntimeException("Switch model failed: " + response.body());
                    }
                });
    }

    /**
     * 온톨로지 규칙 목록 조회
     * @return 규칙 목록
     */
    public CompletableFuture<RuleListResponse> listOntologyRules() {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/rules/list"))
                .GET()
                .build();
        
        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        return gson.fromJson(response.body(), RuleListResponse.class);
                    } else {
                        throw new RuntimeException("Failed to list rules: " + response.body());
                    }
                });
    }

    /**
     * ML + Ontology 결합 예측 수행
     * @param smiles SMILES 문자열
     * @param descriptors 분자 기술자
     * @return 결합 예측 결과
     */
    public CompletableFuture<CombinedPredictionResponse> predictCombined(
            String smiles,
            MolecularDescriptors descriptors,
            String modelType) {
        
        System.out.println("=== predictCombined called ===");
        System.out.println("Model: " + modelType);
        
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("smiles", smiles);
        requestBody.addProperty("MW", descriptors.getMw());
        requestBody.addProperty("logKow", descriptors.getLogKow());
        requestBody.addProperty("HBD", descriptors.getHbd());
        requestBody.addProperty("HBA", descriptors.getHba());
        requestBody.addProperty("nRotB", descriptors.getnRotB());
        requestBody.addProperty("TPSA", descriptors.getTpsa());
        requestBody.addProperty("Aromatic_Rings", descriptors.getAromaticRings());
        requestBody.addProperty("Heteroatom_Count", descriptors.getHeteroatomCount());
        requestBody.addProperty("Heavy_Atom_Count", descriptors.getHeavyAtomCount());
        requestBody.addProperty("logP", descriptors.getLogP());
        requestBody.addProperty("model_type", modelType);
        
        String jsonBody = gson.toJson(requestBody);
        
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/predictions/combined"))
                .header("Content-Type", "application/json")
                .timeout(Duration.ofSeconds(30))
                .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                .build();
        
        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        return gson.fromJson(response.body(), CombinedPredictionResponse.class);
                    } else {
                        throw new RuntimeException("Combined prediction failed: " + response.body());
                    }
                });
    }

    /**
     * 데이터셋 배치 분석 요청
     * @param datasetName 데이터셋 이름
     * @param sampleSize 샘플 수
     * @return 분석 결과 목록
     */
    public CompletableFuture<DatasetAnalysisResponse> analyzeDataset(
            String datasetName,
            int sampleSize) {
        
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("dataset_name", datasetName);
        requestBody.addProperty("sample_size", sampleSize);
        
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/datasets/analyze"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(requestBody)))
                .build();
        
        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        return gson.fromJson(response.body(), DatasetAnalysisResponse.class);
                    } else {
                        throw new RuntimeException("Dataset analysis failed: " + response.body());
                    }
                });
    }
    
    // ==================== Analysis API ====================
    @SuppressWarnings("unchecked")
    public Map<String, Object> getRfFeatureImportance() {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/analysis/rf-importance"))
                .GET()
                .build();
        
        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        java.lang.reflect.Type type = new com.google.gson.reflect.TypeToken<Map<String, Object>>(){}.getType();
                        return (Map<String, Object>) gson.fromJson(response.body(), type);
                    } else {
                        throw new RuntimeException("Failed to get RF importance: " + response.body());
                    }
                }).join();
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> getMoleculeName(String smiles) {
        String encoded = java.net.URLEncoder.encode(smiles, java.nio.charset.StandardCharsets.UTF_8);
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/analysis/molecule-name?smiles=" + encoded))
                .GET()
                .build();
        
        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        java.lang.reflect.Type type = new com.google.gson.reflect.TypeToken<Map<String, Object>>(){}.getType();
                        return (Map<String, Object>) gson.fromJson(response.body(), type);
                    } else {
                        throw new RuntimeException("Failed to get molecule name: " + response.body());
                    }
                }).join();
    }
    
    /**
     * SDT 트리 구조 조회
     * @param smiles 선택적 SMILES 문자열 (활성 경로 강조용)
     */
    public CompletableFuture<String> getSdtTree(String smiles) {
        return getSdtTree(smiles, null);
    }
    
    /**
     * SDT 트리 구조 조회 (깊이 제한 포함)
     * @param smiles 선택적 SMILES 문자열 (활성 경로 강조용)
     * @param maxDepth 최대 트리 깊이 (null이면 제한 없음)
     */
    public CompletableFuture<String> getSdtTree(String smiles, Integer maxDepth) {
        String uriStr = baseUrl + "/analysis/sdt-tree";
        boolean hasParams = false;
        
        if (smiles != null && !smiles.isEmpty()) {
            uriStr += "?smiles=" + java.net.URLEncoder.encode(smiles, java.nio.charset.StandardCharsets.UTF_8);
            hasParams = true;
        }
        
        if (maxDepth != null) {
            uriStr += (hasParams ? "&" : "?") + "max_depth=" + maxDepth;
        }
    
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(uriStr))
                .GET()
                .build();
        
        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        return response.body();
                    } else {
                        throw new RuntimeException("Failed to get SDT tree: " + response.body());
                    }
                });
    }

    /**
     * 벤치마크 회귀 플롯 생성 요청
     * @param results 벤치마크 결과 리스트
     * @param xDescriptor X축 분자 기술자
     * @param yDescriptor Y축 분자 기술자
     * @param coloring 색상 기준 분자 기술자
     * @param xLogScale X축 로그 스케일 여부
     * @param yLogScale Y축 로그 스케일 여부
     * @param showLinear Linear Regression 표시 여부
     * @param showRansac RANSAC 표시 여부
     * @return Plotly HTML 문자열
     */
    public CompletableFuture<String> getBenchmarkPlot(
            List<CombinedPredictionResponse> results, 
            String xDescriptor, 
            String yDescriptor,
            String coloring,
            boolean xLogScale,
            boolean yLogScale,
            boolean showLinear,
            boolean showRansac) {
        JsonObject requestBody = new JsonObject();
        requestBody.add("results", gson.toJsonTree(results));
        requestBody.addProperty("x_descriptor", xDescriptor);
        requestBody.addProperty("y_descriptor", yDescriptor);
        requestBody.addProperty("coloring", coloring);
        requestBody.addProperty("x_log_scale", xLogScale);
        requestBody.addProperty("y_log_scale", yLogScale);
        requestBody.addProperty("show_linear", showLinear);
        requestBody.addProperty("show_ransac", showRansac);
        
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/analysis/benchmark-plot"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(requestBody)))
                .build();
        
        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        return response.body();
                    } else {
                        throw new RuntimeException("Failed to generate benchmark plot: " + response.body());
                    }
                });
    }

    /**
     * Decision Tree 시각화 플롯 생성
     * @param maxDepth 트리 최대 깊이
     * @param useSemantic 온톨로지 의미론적 특성 사용 여부
     * @return Plotly HTML 문자열
     */
    public CompletableFuture<String> getDecisionTreePlot(int maxDepth, boolean useSemantic) {
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("max_depth", maxDepth);
        requestBody.addProperty("use_semantic", useSemantic);
        
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/analysis/decision-tree"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(requestBody)))
                .build();
        
        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        return response.body();
                    } else {
                        throw new RuntimeException("Failed to generate decision tree: " + response.body());
                    }
                });
    }

    /**
     * 2D 결정 경계 시각화
     * @param xFeature X축 특성
     * @param yFeature Y축 특성
     * @return Plotly HTML 문자열
     */
    public CompletableFuture<String> getDecisionBoundary2D(
            String xFeature,
            String yFeature,
            String datasetName,
            String modelType,
            Integer sampleSize) {
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("x_feature", xFeature);
        requestBody.addProperty("y_feature", yFeature);
        if (modelType != null) {
            requestBody.addProperty("model_type", modelType);
        }
        if (datasetName != null) {
            requestBody.addProperty("dataset_name", datasetName);
        }
        if (sampleSize != null) {
            requestBody.addProperty("sample_size", sampleSize);
        }
        
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/analysis/decision-boundary-2d"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(requestBody)))
                .build();
        
        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        return response.body();
                    } else {
                        throw new RuntimeException("Failed to generate 2D decision boundary: " + response.body());
                    }
                });
    }

    /**
     * 3D 결정 경계 시각화
     * @param xFeature X축 특성
     * @param yFeature Y축 특성
     * @param zFeature Z축 특성
     * @return Plotly HTML 문자열
     */
    public CompletableFuture<String> getDecisionBoundary3D(String xFeature, String yFeature, String zFeature) {
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("x_feature", xFeature);
        requestBody.addProperty("y_feature", yFeature);
        requestBody.addProperty("z_feature", zFeature);
        
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/analysis/decision-boundary-3d"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(requestBody)))
                .build();
        
        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        return response.body();
                    } else {
                        throw new RuntimeException("Failed to generate 3D decision boundary: " + response.body());
                    }
                });
    }

    // ================= Response DTOs =================
    
    /**
     * 헬스체크 응답
     */
    public static class HealthResponse {
        public String status;
        public String rf_predictor;
        public String shap_explainer;
        public String rule_engine;
        public String simple_qsar;
        public String read_across;
        public String ensemble_dss;
        public String model_path;
        
        public boolean isFullyLoaded() {
            return "loaded".equals(rf_predictor) &&
                   "loaded".equals(rule_engine) &&
                   "loaded".equals(ensemble_dss);
        }
    }
    
    /**
     * RF 예측 응답
     */
    public static class PredictionResponse {
        public String model;
        public String model_version;
        public int prediction_class;
        public String class_name;
        public double confidence;
        public Map<String, Double> probabilities;
        public String chemical_id;
    }
    
    /**
     * SHAP 설명 응답
     */
    public static class ShapResponse {
        public String prediction;
        public String chemical_id;
        public String interpretation_ko;
        public java.util.List<FeatureImportance> feature_importance;
        
        public static class FeatureImportance {
            public String feature;
            public double shap_value;
            public double abs_shap;
            public String direction;
        }
    }
    
    /**
     * 온톨로지 검증 응답
     */
    public static class ValidationResponse {
        public String chemical_id;
        public Map<String, Object> rf_prediction;
        public OntologyValidation ontology_validation;
        public String interpretation;
        
        public static class OntologyValidation {
            public String agreement;
            public int matched_rules_count;
            public double rule_confidence;
            public double combined_confidence;
        }
    }
    
    /**
     * 앙상블 분석 응답
     */
    public static class EnsembleResponse {
        public String chemical_id;
        public EnsembleResults ensemble_results;
        public java.util.List<MethodBreakdown> method_breakdown;
        public String detailed_reasoning;
        public String analysis_timestamp;
        
        public static class EnsembleResults {
            public double score;
            public double confidence;
            public String class_name;
            public double method_agreement;
            public String recommendation;
        }
        
        public static class MethodBreakdown {
            public String method;
            public double score;
            public double confidence;
            public double weight;
            public double weighted_contribution;
            public String prediction;
        }
    }

    /**
     * 데이터셋 목록 응답
     */
    public static class DatasetListResponse {
        public java.util.List<DatasetInfo> datasets;
        public int total;
    }

    public static class DatasetInfo {
        public String name;
        public int size;
        public String smiles_column;
        public java.util.List<String> label_columns;
        public String description;
        
        @Override
        public String toString() {
            return name + " (" + size + " samples)";
        }
    }

    /**
     * 규칙 목록 응답
     */
    public static class RuleListResponse {
        public java.util.List<RuleInfo> rules;
        public Map<String, String> colors;
        public int total;
    }

    public static class RuleInfo {
        public String id;
        public String name;
        public String category;
        public double weight;
        public String interpretation;
    }



    // ==================== Dataset API ====================
    
    /**
     * Get list of available datasets from Backend
     */
    public CompletableFuture<List<com.example.dto.data.DatasetLoader.DatasetInfo>> listDatasets() {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/datasets/list"))
                .GET()
                .build();
                
        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        try {
                             java.lang.reflect.Type listType = new com.google.gson.reflect.TypeToken<List<com.example.dto.data.DatasetLoader.DatasetInfo>>(){}.getType();
                             return gson.fromJson(response.body(), listType);
                        } catch (Exception e) {
                            throw new RuntimeException("Parse error: " + e.getMessage());
                        }
                    } else {
                        throw new RuntimeException("Failed to list datasets: " + response.body());
                    }
                });
    }

    /**
     * Get raw samples from backend
     */
    public CompletableFuture<List<com.example.dto.data.DatasetLoader.DatasetSample>> getDatasetSamples(String datasetName, Integer limit) {
         Map<String, Object> body = new java.util.HashMap<>();
         body.put("dataset_name", datasetName);
         if (limit != null) body.put("limit", limit);
         
         HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/datasets/samples"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(body)))
                .build();
                
        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        try {
                             java.lang.reflect.Type listType = new com.google.gson.reflect.TypeToken<List<Map<String, Object>>>(){}.getType();
                             List<Map<String, Object>> rawList = gson.fromJson(response.body(), listType);
                             
                             List<com.example.dto.data.DatasetLoader.DatasetSample> samples = new java.util.ArrayList<>();
                             for(Map<String, Object> raw : rawList) {
                                 String smiles = (String) raw.get("smiles");
                                 String name = (String) raw.get("molecule_name");
                                 
                                 Map<String, Double> feats = new java.util.HashMap<>();
                                 Object descObj = raw.get("descriptors");
                                 if (descObj instanceof Map<?, ?> rawMap) {
                                     for (Map.Entry<?, ?> entry : rawMap.entrySet()) {
                                         if (entry.getValue() instanceof Number num) {
                                             feats.put(entry.getKey().toString(), num.doubleValue());
                                         }
                                     }
                                 }
                                 samples.add(new com.example.dto.data.DatasetLoader.DatasetSample(smiles, name, null, feats));
                             }
                             return samples;
                        } catch (Exception e) {
                            throw new RuntimeException("Parse error: " + e.getMessage());
                        }
                    } else {
                        throw new RuntimeException("Failed to get samples: " + response.body());
                    }
                });
    }


    /**
     * 결합 예측 응답 (JavaFX PropertyValueFactory용 Getter 포함)
     */
    public static class CombinedPredictionResponse {
        public String smiles;
        public String molecule_name;
        public double ml_prediction;
        public double ontology_score;
        public double combined_score;
        public double confidence;
        public String confidence_level;
        public String confidence_action;
        public String agreement;
        public String risk_level;
        public String explanation;
        public java.util.List<TriggeredRule> triggered_rules;
        public java.util.List<Map<String, Object>> shap_features;
        public Map<String, Object> all_descriptors; // Added for detailed popup

        // Getters for PropertyValueFactory (JavaFX TableView binding)
        public String getSmiles() { return smiles; }
        public String getMolecule_name() { return molecule_name; }
        public double getMl_prediction() { return ml_prediction; }
        public double getOntology_score() { return ontology_score; }
        public double getCombined_score() { return combined_score; }
        public double getConfidence() { return confidence; }
        public String getConfidence_level() { return confidence_level; }
        public String getConfidence_action() { return confidence_action; }
        public String getAgreement() { return agreement; }
        public String getRisk_level() { return risk_level; }
        public String getExplanation() { return explanation; }
        public java.util.List<TriggeredRule> getTriggered_rules() { return triggered_rules; }
        public java.util.List<Map<String, Object>> getShap_features() { return shap_features; }
        public int getBinary_prediction() { return ml_prediction >= 0.5 ? 1 : 0; }
    }

    public static class TriggeredRule {
        public String rule_id;
        public String name;
        public String category;
        public String interpretation;
        public double weight;
        public String detailed_reason;  // 상세 근거 (예: "LogP 4.2 > 3.0 → 높은 지질친화성")
        public double descriptor_value;
        public double threshold_value;
        public String toxicity_direction;

        // Getters for JavaFX PropertyValueFactory
        public String getRule_id() { return rule_id; }
        public String getName() { return name; }
        public String getCategory() { return category; }
        public String getInterpretation() { return interpretation; }
        public double getWeight() { return weight; }
        public String getDetailed_reason() { return detailed_reason; }
        public double getDescriptor_value() { return descriptor_value; }
        public double getThreshold_value() { return threshold_value; }
        public String getToxicity_direction() { return toxicity_direction; }
    }

    /**
     * 데이터셋 분석 응답
     */
    public static class DatasetAnalysisResponse {
        public java.util.List<CombinedPredictionResponse> results;
        public int total;
    }
}

