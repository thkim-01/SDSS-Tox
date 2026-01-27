package com.example.dto.api;

import com.google.gson.annotations.SerializedName;

/**
 * 10개 분자 기술자 DTO.
 * FastAPI 요청에 사용되는 분자 기술자 데이터 클래스.
 */
public class MolecularDescriptors {
    
    @SerializedName("MW")
    private double mw;              // 분자량
    
    @SerializedName("logKow")
    private double logKow;          // 옥탄올-물 분배계수
    
    @SerializedName("HBD")
    private int hbd;                // 수소 결합 공여체 수
    
    @SerializedName("HBA")
    private int hba;                // 수소 결합 수용체 수
    
    @SerializedName("nRotB")
    private int nRotB;              // 회전 가능 결합 수
    
    @SerializedName("TPSA")
    private double tpsa;            // 극성 표면적
    
    @SerializedName("Aromatic_Rings")
    private int aromaticRings;      // 방향족 고리 수
    
    @SerializedName("Heteroatom_Count")
    private int heteroatomCount;    // 이종원자 수
    
    @SerializedName("Heavy_Atom_Count")
    private int heavyAtomCount;     // 중원자 수
    
    @SerializedName("logP")
    private double logP;            // 지질친화성

    // 제거된 필드: Python 백엔드와의 호환성을 위해 삭제
    // @SerializedName("FractionCSP3")
    // private double fractionCSP3;    // sp3 탄소 비율
    
    // @SerializedName("MolarRefractivity")  
    // private double molarRefractivity; // 몰 굴절률
    
    /**
     * 기본 생성자
     */
    public MolecularDescriptors() {
    }
    
    /**
     * 전체 파라미터 생성자
     */
    public MolecularDescriptors(double mw, double logKow, int hbd, int hba, 
                                int nRotB, double tpsa, int aromaticRings,
                                int heteroatomCount, int heavyAtomCount, double logP) {
        this.mw = mw;
        this.logKow = logKow;
        this.hbd = hbd;
        this.hba = hba;
        this.nRotB = nRotB;
        this.tpsa = tpsa;
        this.aromaticRings = aromaticRings;
        this.heteroatomCount = heteroatomCount;
        this.heavyAtomCount = heavyAtomCount;
        this.logP = logP;
        // 제거된 필드: fractionCSP3, molarRefractivity
    }
    
    /**
     * Aspirin 샘플 데이터 생성
     */
    public static MolecularDescriptors aspirinSample() {
        return new MolecularDescriptors(
                180.16,  // MW
                1.19,    // logKow
                1,       // HBD
                4,       // HBA
                3,       // nRotB
                63.60,   // TPSA
                1,       // Aromatic_Rings
                4,       // Heteroatom_Count
                13,      // Heavy_Atom_Count
                0.89     // logP
        );
    }
    
    /**
     * 고독성 샘플 데이터 생성
     */
    public static MolecularDescriptors toxicSample() {
        return new MolecularDescriptors(
                450.5,   // MW - 높음
                5.2,     // logKow - 높음 (지질친화성)
                0,       // HBD
                2,       // HBA
                8,       // nRotB
                25.0,    // TPSA - 낮음
                4,       // Aromatic_Rings - 많음
                3,       // Heteroatom_Count
                35,      // Heavy_Atom_Count
                5.0      // logP - 높음
        );
    }
    
    // Getters and Setters
    public double getMw() { return mw; }
    public void setMw(double mw) { this.mw = mw; }
    
    public double getLogKow() { return logKow; }
    public void setLogKow(double logKow) { this.logKow = logKow; }
    
    public int getHbd() { return hbd; }
    public void setHbd(int hbd) { this.hbd = hbd; }
    
    public int getHba() { return hba; }
    public void setHba(int hba) { this.hba = hba; }
    
    public int getnRotB() { return nRotB; }
    public void setnRotB(int nRotB) { this.nRotB = nRotB; }
    
    public double getTpsa() { return tpsa; }
    public void setTpsa(double tpsa) { this.tpsa = tpsa; }
    
    public int getAromaticRings() { return aromaticRings; }
    public void setAromaticRings(int aromaticRings) { this.aromaticRings = aromaticRings; }
    
    public int getHeteroatomCount() { return heteroatomCount; }
    public void setHeteroatomCount(int heteroatomCount) { this.heteroatomCount = heteroatomCount; }
    
    public int getHeavyAtomCount() { return heavyAtomCount; }
    public void setHeavyAtomCount(int heavyAtomCount) { this.heavyAtomCount = heavyAtomCount; }
    
    public double getLogP() { return logP; }
    public void setLogP(double logP) { this.logP = logP; }

    // 제거된 필드: fractionCSP3, molarRefractivity
    // public double getFractionCSP3() { return fractionCSP3; }
    // public void setFractionCSP3(double fractionCSP3) { this.fractionCSP3 = fractionCSP3; }
    // public double getMolarRefractivity() { return molarRefractivity; }
    // public void setMolarRefractivity(double molarRefractivity) { this.molarRefractivity = molarRefractivity; }

    @Override
    public String toString() {
        return String.format(
            "Descriptors[MW=%.2f, logKow=%.2f, HBD=%d, HBA=%d, nRotB=%d, " +
            "TPSA=%.2f, Aromatic=%d, Hetero=%d, Heavy=%d, logP=%.2f]",
            mw, logKow, hbd, hba, nRotB, tpsa, aromaticRings,
            heteroatomCount, heavyAtomCount, logP);
    }
}
