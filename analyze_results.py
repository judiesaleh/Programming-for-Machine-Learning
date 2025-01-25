import pandas as pd
import matplotlib.pyplot as plt

# Pfad zur results.csv-Datei
results_file = r"C:\Users\judie\Desktop\Model\yolov5\runs\train\exp8\results.csv"

# Ergebnisse aus der CSV-Datei laden
results = pd.read_csv(results_file)

# Spaltennamen bereinigen (führende Leerzeichen entfernen)
results.columns = results.columns.str.strip()

# Überprüfen der bereinigten Spaltennamen
print(results.columns)

# Diagramme erstellen
plt.figure(figsize=(15, 10))

# Trainingsverlust (Box, Obj, Cls)
plt.subplot(2, 2, 1)
plt.plot(results['epoch'], results['train/box_loss'], label='Train Box Loss')
plt.plot(results['epoch'], results['train/obj_loss'], label='Train Obj Loss')
plt.plot(results['epoch'], results['train/cls_loss'], label='Train Cls Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Components')
plt.legend()

# Validierungsverlust (Box, Obj, Cls)
plt.subplot(2, 2, 2)
plt.plot(results['epoch'], results['val/box_loss'], label='Val Box Loss')
plt.plot(results['epoch'], results['val/obj_loss'], label='Val Obj Loss')
plt.plot(results['epoch'], results['val/cls_loss'], label='Val Cls Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Components')
plt.legend()

# Precision und Recall
plt.subplot(2, 2, 3)
plt.plot(results['epoch'], results['metrics/precision'], label='Precision', color='green')
plt.plot(results['epoch'], results['metrics/recall'], label='Recall', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Precision and Recall')
plt.legend()

# mAP (Mean Average Precision)
plt.subplot(2, 2, 4)
plt.plot(results['epoch'], results['metrics/mAP_0.5'], label='mAP@0.5', color='red')
plt.plot(results['epoch'], results['metrics/mAP_0.5:0.95'], label='mAP@0.5:0.95', color='purple')
plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.title('Mean Average Precision')
plt.legend()

# Diagramme anzeigen
plt.tight_layout()
plt.show()