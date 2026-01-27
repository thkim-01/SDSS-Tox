package com.example.dto.data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class DatasetLoader {
    
    public static class DatasetInfo {
        public String name;
        public String type = "unknown"; // Default
        public int size;
        public String description;
        public String smiles_column; // Added
        public List<String> label_columns; // Added
        
        public DatasetInfo() {} // No-args constructor for Gson
        
        public DatasetInfo(String name, String type, int size, String description) {
            this.name = name;
            this.type = type;
            this.size = size;
            this.description = description;
        }
        
        @Override
        public String toString() {
            return String.format("%s (%s) - %d samples", name, type, size);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            DatasetInfo that = (DatasetInfo) o;
            return java.util.Objects.equals(name, that.name);
        }

        @Override
        public int hashCode() {
            return java.util.Objects.hash(name);
        }
    }
    
    public static class DatasetSample {
        public final String smiles;
        public final String name; // Added name field
        public final Double label; // null for unlabeled data
        public final Map<String, Double> features;
        
        public DatasetSample(String smiles, String name, Double label, Map<String, Double> features) {
            this.smiles = smiles;
            this.name = name;
            this.label = label;
            this.features = features;
        }
    }
    
    private static final Map<String, DatasetInfo> BENCHMARK_DATASETS = new HashMap<>();
    
    static {
        // Classification datasets
        BENCHMARK_DATASETS.put("tox21", new DatasetInfo(
            "tox21", "classification", 8014, "Toxicology in the 21st Century - NR-AhR assay"
        ));
        BENCHMARK_DATASETS.put("bbbp", new DatasetInfo(
            "bbbp", "classification", 2039, "Blood-Brain Barrier Penetration"
        ));
        BENCHMARK_DATASETS.put("clintox", new DatasetInfo(
            "clintox", "classification", 1478, "Clinical Toxicity"
        ));
        BENCHMARK_DATASETS.put("hiv", new DatasetInfo(
            "hiv", "classification", 41127, "HIV inhibition"
        ));
        BENCHMARK_DATASETS.put("sider", new DatasetInfo(
            "sider", "classification", 1427, "Side Effect Resource"
        ));
        
        // Regression datasets
        BENCHMARK_DATASETS.put("esol", new DatasetInfo(
            "esol", "regression", 1128, "Aqueous Solubility"
        ));
        BENCHMARK_DATASETS.put("lipophilicity", new DatasetInfo(
            "lipophilicity", "regression", 4200, "Octanol-water partition coefficient"
        ));
        BENCHMARK_DATASETS.put("freesolv", new DatasetInfo(
            "freesolv", "regression", 642, "Hydration free energy"
        ));
    }
    
    public List<DatasetInfo> getAvailableDatasets() {
        return new ArrayList<>(BENCHMARK_DATASETS.values());
    }
    
    public List<DatasetSample> loadDataset(String datasetName, int sampleSize) throws Exception {
        DatasetInfo info = BENCHMARK_DATASETS.get(datasetName);
        if (info == null) {
            throw new IllegalArgumentException("Unknown dataset: " + datasetName);
        }
        
        // For now, return mock data
        // In real implementation, this would load from CSV files
        return generateMockData(datasetName, Math.min(sampleSize, info.size));
    }
    
    private List<DatasetSample> generateMockData(String datasetName, int count) {
        List<DatasetSample> samples = new ArrayList<>();
        Random rand = new Random(42); // Fixed seed for reproducibility
        
        for (int i = 0; i < count; i++) {
            String smiles = generateMockSmiles(i);
            Map<String, Double> features = generateMockFeatures(rand);
            
            Double label = null;
            if (BENCHMARK_DATASETS.get(datasetName).type.equals("classification")) {
                label = rand.nextDouble(); // 0.0 to 1.0 for binary classification
            } else {
                // Regression values with realistic ranges
                switch (datasetName) {
                    case "esol":
                        label = -10.0 + rand.nextGaussian() * 2.0; // logS values
                        break;
                    case "lipophilicity":
                        label = -2.0 + rand.nextGaussian() * 3.0; // logP values
                        break;
                    case "freesolv":
                        label = -15.0 + rand.nextGaussian() * 5.0; // kcal/mol
                        break;
                    default:
                        label = rand.nextGaussian();
                }
            }
            
            samples.add(new DatasetSample(smiles, "Mock Mol " + i, label, features));
        }
        
        return samples;
    }
    
    private String generateMockSmiles(int index) {
        // Generate simple mock SMILES strings
        String[] atoms = {"C", "N", "O", "S", "P", "F", "Cl", "Br"};
        Random rand = new Random(index);
        
        StringBuilder sb = new StringBuilder();
        int length = 5 + rand.nextInt(10);
        
        for (int i = 0; i < length; i++) {
            if (i > 0 && rand.nextDouble() < 0.3) {
                sb.append(rand.nextBoolean() ? "=" : "#");
            }
            sb.append(atoms[rand.nextInt(atoms.length)]);
            if (rand.nextDouble() < 0.2) {
                sb.append("(");
                sb.append(atoms[rand.nextInt(atoms.length)]);
                sb.append(")");
            }
        }
        
        return sb.toString();
    }
    
    private Map<String, Double> generateMockFeatures(Random rand) {
        Map<String, Double> features = new HashMap<>();
        
        // Common molecular descriptors
        features.put("MolWt", 50.0 + rand.nextDouble() * 450.0);
        features.put("LogP", -3.0 + rand.nextDouble() * 8.0);
        features.put("NumHAcceptors", (double) rand.nextInt(10));
        features.put("NumHDonors", (double) rand.nextInt(6));
        features.put("NumRotatableBonds", (double) rand.nextInt(15));
        features.put("TPSA", 0.0 + rand.nextDouble() * 140.0);
        features.put("MolLogP", -3.0 + rand.nextDouble() * 8.0);
        features.put("NumAromaticRings", (double) rand.nextInt(4));
        features.put("NumHeteroatoms", (double) rand.nextInt(10));
        features.put("NumHeavyAtoms", (double) (10 + rand.nextInt(30)));
        
        return features;
    }
}