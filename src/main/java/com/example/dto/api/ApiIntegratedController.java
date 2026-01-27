package com.example.dto.api;

import java.util.concurrent.CompletableFuture;

import com.example.dto.api.DssApiClient.EnsembleResponse;
import com.example.dto.api.DssApiClient.PredictionResponse;
import com.example.dto.api.DssApiClient.ShapResponse;
import com.example.dto.api.DssApiClient.ValidationResponse;

import javafx.application.Platform;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;

/**
 * JavaFX GUI와 FastAPI 백엔드를 연결하는 컨트롤러.
 * 비동기 API 호출을 JavaFX UI 스레드에 안전하게 동기화합니다.
 */
public class ApiIntegratedController {
    
    private final DssApiClient apiClient;
    
    // Observable properties for UI binding
    private final BooleanProperty apiConnected = new SimpleBooleanProperty(false);
    private final StringProperty statusMessage = new SimpleStringProperty("연결 대기 중...");
    
    /**
     * 기본 생성자 (localhost:8000)
     */
    public ApiIntegratedController() {
        this.apiClient = new DssApiClient();
    }
    
    /**
     * 커스텀 URL 생성자
     */
    public ApiIntegratedController(String apiBaseUrl) {
        this.apiClient = new DssApiClient(apiBaseUrl);
    }
    
    /**
     * API 서버 연결 확인
     */
    public void checkConnection() {
        updateStatus("API 서버 연결 확인 중...");
        
        apiClient.checkHealth()
            .thenAcceptAsync(health -> {
                if (health.isFullyLoaded()) {
                    Platform.runLater(() -> {
                        apiConnected.set(true);
                        updateStatus("API 서버 연결됨 (모든 서비스 로드됨)");
                    });
                } else {
                    Platform.runLater(() -> {
                        apiConnected.set(true);
                        updateStatus("API 서버 연결됨 (일부 서비스 미로드)");
                    });
                }
            })
            .exceptionally(ex -> {
                Platform.runLater(() -> {
                    apiConnected.set(false);
                    updateStatus("API 서버 연결 실패: " + ex.getMessage());
                });
                return null;
            });
    }
    
    /**
     * SMILES 기반 앙상블 분석 수행
     * @param smiles SMILES 문자열
     * @param callback 결과 콜백
     */
    public void analyzeSmiles(String smiles, AnalysisCallback callback) {
        // SMILES에서 기술자 계산
        MolecularDescriptors descriptors = DescriptorCalculator.calculateFromSmiles(smiles);
        
        String chemicalId = "smiles_" + smiles.hashCode();
        
        updateStatus("분석 중: " + smiles.substring(0, Math.min(20, smiles.length())) + "...");
        
        // 앙상블 분석 호출
        apiClient.analyzeEnsemble(chemicalId, descriptors)
            .thenAcceptAsync(result -> {
                Platform.runLater(() -> {
                    updateStatus("분석 완료");
                    callback.onSuccess(result, descriptors);
                });
            })
            .exceptionally(ex -> {
                Platform.runLater(() -> {
                    updateStatus("분석 실패: " + ex.getMessage());
                    callback.onError(ex);
                });
                return null;
            });
    }
    
    /**
     * RF 예측만 수행
     */
    public CompletableFuture<PredictionResponse> predictOnly(String smiles) {
        MolecularDescriptors descriptors = DescriptorCalculator.calculateFromSmiles(smiles);
        String chemicalId = "smiles_" + smiles.hashCode();
        return apiClient.predictToxicity(chemicalId, descriptors);
    }
    
    /**
     * SHAP 설명 요청
     */
    public CompletableFuture<ShapResponse> explainOnly(String smiles, int targetClass) {
        MolecularDescriptors descriptors = DescriptorCalculator.calculateFromSmiles(smiles);
        String chemicalId = "smiles_" + smiles.hashCode();
        return apiClient.explainPrediction(chemicalId, descriptors, targetClass);
    }
    
    /**
     * 온톨로지 검증만 수행
     */
    public CompletableFuture<ValidationResponse> validateOnly(String smiles) {
        MolecularDescriptors descriptors = DescriptorCalculator.calculateFromSmiles(smiles);
        String chemicalId = "smiles_" + smiles.hashCode();
        return apiClient.validateWithOntology(chemicalId, descriptors);
    }
    
    /**
     * 분석 결과를 사람이 읽을 수 있는 보고서로 변환
     */
    public String formatEnsembleReport(EnsembleResponse result, MolecularDescriptors desc) {
        StringBuilder sb = new StringBuilder();
        
        sb.append("═══════════════════════════════════════════\n");
        sb.append("        앙상블 DSS 분석 보고서\n");
        sb.append("═══════════════════════════════════════════\n\n");
        
        // 기본 정보
        sb.append("🆔 분석 ID: ").append(result.chemical_id).append("\n");
        sb.append("⏰ 분석 시간: ").append(result.analysis_timestamp).append("\n\n");
        
        // 기술자 정보
        sb.append("── 분자 기술자 ──\n");
        sb.append(String.format("  분자량(MW): %.2f\n", desc.getMw()));
        sb.append(String.format("  logP: %.2f\n", desc.getLogP()));
        sb.append(String.format("  HBD/HBA: %d/%d\n", desc.getHbd(), desc.getHba()));
        sb.append(String.format("  TPSA: %.2f\n", desc.getTpsa()));
        sb.append(String.format("  방향족 고리: %d\n\n", desc.getAromaticRings()));
        
        // 앙상블 결과
        if (result.ensemble_results != null) {
            sb.append("── 앙상블 결과 ──\n");
            sb.append(String.format("  독성 점수: %.1f%%\n", result.ensemble_results.score * 100));
            sb.append(String.format("  신뢰도: %.1f%%\n", result.ensemble_results.confidence * 100));
            sb.append(String.format("  방법 합의도: %.1f%%\n", result.ensemble_results.method_agreement * 100));
            sb.append("\n  추천: ").append(result.ensemble_results.recommendation).append("\n\n");
        }
        
        // 방법별 상세
        if (result.method_breakdown != null) {
            sb.append("── 방법별 분석 ──\n");
            for (var method : result.method_breakdown) {
                String icon = method.score > 0.5 ? "[Toxic]" : "[Safe]";
                sb.append(String.format("  %s %s: %.1f%% (가중치: %.0f%%)\n",
                        icon, method.method, method.score * 100, method.weight * 100));
            }
            sb.append("\n");
        }
        
        // 상세 추론
        if (result.detailed_reasoning != null) {
            sb.append("── 상세 분석 ──\n");
            sb.append(result.detailed_reasoning).append("\n");
        }
        
        return sb.toString();
    }
    
    private void updateStatus(String message) {
        statusMessage.set(message);
    }
    
    // Property getters for UI binding
    public BooleanProperty apiConnectedProperty() { return apiConnected; }
    public StringProperty statusMessageProperty() { return statusMessage; }
    public boolean isApiConnected() { return apiConnected.get(); }
    
    /**
     * 분석 결과 콜백 인터페이스
     */
    public interface AnalysisCallback {
        void onSuccess(EnsembleResponse result, MolecularDescriptors descriptors);
        void onError(Throwable error);
    }
}
