package com.example.dto.gui;

import javafx.scene.control.SelectionMode;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.input.Clipboard;
import javafx.scene.input.ClipboardContent;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyCodeCombination;
import javafx.scene.input.KeyCombination;

public class TableUtils {

    public static void installCopyPasteHandler(TableView<?> table) {
        // Enable multiple selection
        table.getSelectionModel().setSelectionMode(SelectionMode.MULTIPLE);

        // Install Key Handler (Ctrl+C)
        final KeyCombination copyCombo = new KeyCodeCombination(KeyCode.C, KeyCombination.CONTROL_DOWN);
        
        table.setOnKeyPressed(event -> {
            if (copyCombo.match(event)) {
                copySelectionToClipboard(table);
                event.consume();
            }
        });
    }

    private static void copySelectionToClipboard(TableView<?> table) {
        StringBuilder clipboardString = new StringBuilder();
        
        // Header
        for (TableColumn<?, ?> column : table.getColumns()) {
            clipboardString.append(column.getText()).append("\t");
        }
        clipboardString.append("\n");

        // Data
        for (Integer rowIndex : table.getSelectionModel().getSelectedIndices()) {
            if (rowIndex == -1) continue;
            
            for (TableColumn<?, ?> column : table.getColumns()) {
                Object cellData = column.getCellData(rowIndex);
                clipboardString.append(cellData == null ? "" : cellData.toString()).append("\t");
            }
            clipboardString.append("\n");
        }

        final ClipboardContent content = new ClipboardContent();
        content.putString(clipboardString.toString());
        Clipboard.getSystemClipboard().setContent(content);
    }
}
