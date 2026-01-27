package com.example.dto.api;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * SMILES 문자열에서 분자 기술자를 계산하는 유틸리티 클래스.
 * 
 * 주의: RDKit 없이 간단한 패턴 매칭 기반으로 근사치를 계산합니다.
 * 정확한 값을 위해서는 RDKit 기반 Python 서비스를 사용해야 합니다.
 */
public class DescriptorCalculator {
    
    // 원소별 대략적인 분자량
    private static final double MW_C = 12.0;
    private static final double MW_H = 1.0;
    private static final double MW_O = 16.0;
    private static final double MW_N = 14.0;
    private static final double MW_S = 32.0;
    private static final double MW_F = 19.0;
    private static final double MW_Cl = 35.5;
    private static final double MW_Br = 80.0;
    
    /**
     * SMILES에서 분자 기술자를 계산합니다.
     * @param smiles SMILES 문자열
     * @return 계산된 분자 기술자
     */
    public static MolecularDescriptors calculateFromSmiles(String smiles) {
        if (smiles == null || smiles.isEmpty()) {
            return MolecularDescriptors.aspirinSample(); // 기본값
        }
        
        MolecularDescriptors desc = new MolecularDescriptors();
        
        // 1. 중원자 수 (Heavy Atom Count) - H를 제외한 원자 수
        int heavyAtomCount = countHeavyAtoms(smiles);
        desc.setHeavyAtomCount(heavyAtomCount);
        
        // 2. 이종원자 수 (Heteroatom Count) - C, H를 제외한 원자 수
        int heteroatomCount = countHeteroatoms(smiles);
        desc.setHeteroatomCount(heteroatomCount);
        
        // 3. 방향족 고리 수
        int aromaticRings = countAromaticRings(smiles);
        desc.setAromaticRings(aromaticRings);
        
        // 4. 수소 결합 공여체 (HBD) - OH, NH
        int hbd = countHBD(smiles);
        desc.setHbd(hbd);
        
        // 5. 수소 결합 수용체 (HBA) - O, N
        int hba = countHBA(smiles);
        desc.setHba(hba);
        
        // 6. 회전 가능 결합 수 (nRotB)
        int nRotB = countRotatableBonds(smiles);
        desc.setnRotB(nRotB);
        
        // 7. 분자량 추정
        double mw = estimateMolecularWeight(smiles);
        desc.setMw(mw);
        
        // 8. logP/logKow 추정 (간단한 규칙 기반)
        double logP = estimateLogP(smiles, aromaticRings, heteroatomCount, heavyAtomCount);
        desc.setLogP(logP);
        desc.setLogKow(logP); // logKow ≈ logP
        
        // 9. TPSA 추정
        double tpsa = estimateTPSA(hbd, hba, heteroatomCount);
        desc.setTpsa(tpsa);
        
        return desc;
    }
    
    private static int countHeavyAtoms(String smiles) {
        // 소문자 방향족 + 대문자 원자 카운트 (괄호, 숫자 제외)
        int count = 0;
        String clean = smiles.replaceAll("[\\[\\]\\(\\)0-9+\\-=#@/\\\\]", "");
        for (char c : clean.toCharArray()) {
            if (Character.isLetter(c) && c != 'H' && c != 'h') {
                count++;
            }
        }
        return Math.max(count, 5); // 최소 5
    }
    
    private static int countHeteroatoms(String smiles) {
        int count = 0;
        String upper = smiles.toUpperCase();
        count += countOccurrences(upper, "O");
        count += countOccurrences(upper, "N");
        count += countOccurrences(upper, "S");
        count += countOccurrences(upper, "F");
        count += countOccurrences(upper, "CL");
        count += countOccurrences(upper, "BR");
        return count;
    }
    
    private static int countAromaticRings(String smiles) {
        // 소문자 c가 연속된 패턴으로 방향족 고리 추정
        int count = 0;
        
        // 벤젠 고리 패턴: c1ccccc1 등
        Pattern benzene = Pattern.compile("c1[cnosc]+1");
        Matcher m = benzene.matcher(smiles.toLowerCase());
        while (m.find()) count++;
        
        // 간단히 소문자 c 개수로 추정
        if (count == 0) {
            int cCount = 0;
            for (char c : smiles.toCharArray()) {
                if (c == 'c' || c == 'n' || c == 'o' || c == 's') cCount++;
            }
            count = cCount / 5; // 5개당 1개 고리로 추정
        }
        
        return count;
    }
    
    private static int countHBD(String smiles) {
        // O-H, N-H 패턴
        int count = 0;
        count += countOccurrences(smiles.toUpperCase(), "O");  // -OH
        count += countOccurrences(smiles.toUpperCase(), "N");  // -NH
        // 에스터/아미드 O는 공여체가 아님 - 간단히 절반으로
        return Math.max(0, count / 2);
    }
    
    private static int countHBA(String smiles) {
        // O, N은 수용체
        int count = 0;
        String upper = smiles.toUpperCase();
        count += countOccurrences(upper, "O");
        count += countOccurrences(upper, "N");
        return count;
    }
    
    private static int countRotatableBonds(String smiles) {
        // 단일 결합 카운트 (대략적)
        int singleBonds = 0;
        String clean = smiles.replaceAll("[\\[\\]\\(\\)0-9+\\-@/\\\\]", "");
        for (int i = 0; i < clean.length() - 1; i++) {
            char c = clean.charAt(i);
            char next = clean.charAt(i + 1);
            if (Character.isLetter(c) && Character.isLetter(next)) {
                // 이중/삼중 결합 표시 없으면 단일 결합
                if (i > 0 && (smiles.charAt(i) == '=' || smiles.charAt(i) == '#')) {
                    continue;
                }
                singleBonds++;
            }
        }
        // 고리 결합 제외 (대략)
        return Math.max(0, singleBonds - countAromaticRings(smiles) * 2);
    }
    
    private static double estimateMolecularWeight(String smiles) {
        double mw = 0;
        String upper = smiles.toUpperCase();
        
        // 원소별 카운트
        int c = countOccurrences(upper.replaceAll("CL", ""), "C");
        int o = countOccurrences(upper, "O");
        int n = countOccurrences(upper, "N");
        int s = countOccurrences(upper, "S");
        int f = countOccurrences(upper, "F");
        int cl = countOccurrences(upper, "CL");
        int br = countOccurrences(upper, "BR");
        
        mw = c * MW_C + o * MW_O + n * MW_N + s * MW_S + 
             f * MW_F + cl * MW_Cl + br * MW_Br;
        
        // H 추가 (대략 C당 1.5개)
        mw += c * 1.5 * MW_H;
        
        return Math.max(mw, 50); // 최소 50
    }
    
    private static double estimateLogP(String smiles, int aromaticRings, 
                                       int heteroatoms, int heavyAtoms) {
        // Wildman-Crippen 방식의 간단한 근사
        double logP = 0;
        
        // 탄소는 친유성 증가
        logP += (heavyAtoms - heteroatoms) * 0.15;
        
        // 방향족 고리는 친유성 증가
        logP += aromaticRings * 0.5;
        
        // 산소/질소는 친수성 증가
        logP -= heteroatoms * 0.3;
        
        // 할로겐은 친유성 증가
        String upper = smiles.toUpperCase();
        logP += countOccurrences(upper, "F") * 0.2;
        logP += countOccurrences(upper, "CL") * 0.5;
        logP += countOccurrences(upper, "BR") * 0.7;
        
        return logP;
    }
    
    private static double estimateTPSA(int hbd, int hba, int heteroatoms) {
        // 극성 표면적 근사
        // O: ~20 Å², N: ~25 Å², H-bond: +5 Å²
        double tpsa = heteroatoms * 20 + (hbd + hba) * 5;
        return Math.max(tpsa, 10);
    }
    
    private static int countOccurrences(String str, String sub) {
        int count = 0;
        int idx = 0;
        while ((idx = str.indexOf(sub, idx)) != -1) {
            count++;
            idx += sub.length();
        }
        return count;
    }
    
    /**
     * 잘 알려진 분자의 기술자 반환 (SMILES 패턴 매칭)
     */
    public static MolecularDescriptors getKnownMolecule(String smiles) {
        // 아스피린
        if (smiles.contains("CC(=O)Oc1ccccc1C(=O)O") || 
            smiles.toLowerCase().contains("aspirin")) {
            return MolecularDescriptors.aspirinSample();
        }
        
        // 그 외는 계산
        return calculateFromSmiles(smiles);
    }
}
