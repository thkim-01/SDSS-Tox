package com.example.dto.gui;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class DebugLauncher extends Application {
    
    @Override
    public void start(Stage primaryStage) {
        try {
            System.out.println("[DEBUG] Starting application...");
            System.out.println("[DEBUG] Looking for /main.fxml resource...");
            
            var resource = getClass().getResource("/main.fxml");
            System.out.println("[DEBUG] Resource URL: " + resource);
            
            if (resource == null) {
                System.err.println("[ERROR] /main.fxml not found in classpath!");
                return;
            }
            
            System.out.println("[DEBUG] Creating FXMLLoader...");
            FXMLLoader loader = new FXMLLoader(resource);
            
            System.out.println("[DEBUG] Loading FXML...");
            Parent root = loader.load();
            
            System.out.println("[DEBUG] FXML loaded successfully!");
            System.out.println("[DEBUG] Controller: " + loader.getController());
            
            Scene scene = new Scene(root, 1200, 800);
            
            var cssResource = getClass().getResource("/style.css");
            System.out.println("[DEBUG] CSS Resource: " + cssResource);
            if (cssResource != null) {
                scene.getStylesheets().add(cssResource.toExternalForm());
            }
            
            primaryStage.setTitle("SDSS-Tox Debug");
            primaryStage.setScene(scene);
            primaryStage.show();
            
            System.out.println("[DEBUG] Application started successfully!");
            
        } catch (Exception e) {
            System.err.println("[ERROR] Failed to start application:");
            e.printStackTrace();
            
            // Print cause chain
            Throwable cause = e.getCause();
            int depth = 1;
            while (cause != null) {
                System.err.println("[CAUSE " + depth + "] " + cause.getClass().getName() + ": " + cause.getMessage());
                cause = cause.getCause();
                depth++;
            }
        }
    }
    
    public static void main(String[] args) {
        System.out.println("[DEBUG] Launching JavaFX...");
        launch(args);
    }
}
