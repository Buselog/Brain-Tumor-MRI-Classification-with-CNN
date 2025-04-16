import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import tensorflow as tf
import os
from datetime import datetime

# Grafik ve metin çıktısı klasör yolları
analysis_dir = "C:/Users/2022/Desktop/Brain-Tumor-MRI-Classification-with-CNN-main - V2/Brain-Tumor-MRI-Classification-with-CNN-main/BrainTumorClassificationWithCNN/brain_tumor_cnn/results/analysis_history"
text_output_dir = "C:/Users/2022/Desktop/Brain-Tumor-MRI-Classification-with-CNN-main - V2/Brain-Tumor-MRI-Classification-with-CNN-main/BrainTumorClassificationWithCNN/brain_tumor_cnn/results/mtest_history"


from datetime import datetime

def save_analysis_reports(dir_type, file_name, extension="png"):
    """
    Grafiği gösterir, belirtilen dizine kaydeder ve belleği temizler.
    İsteğe bağlı dosya uzantısı kullanılabilir (örneğin: 'png', 'pdf').
    """
    os.makedirs(dir_type, exist_ok=True)
    
    # Tarih damgası ekle (örn: training_history_2025-04-12_18-45-30.png)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_file_name = f"{file_name}_{timestamp}.{extension}"
    save_path = os.path.join(dir_type, full_file_name)
    
    plt.savefig(save_path)   # Sonra kaydet
    plt.show()               # Önce göster
    plt.close()              # Belleği temizle

def save_text_report(text, filename="report", dir_path=text_output_dir):
    """
    Metin çıktısını belirtilen klasöre `.txt` olarak kaydeder.
    """
    os.makedirs(dir_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_path = os.path.join(dir_path, f"{filename}_{timestamp}.txt")
    
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(text)

def plot_training_history(history):
    """Eğitim sürecindeki loss ve accuracy grafiklerini çizer ve kaydeder."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Eğitim Kaybı')
    plt.plot(history['val_loss'], label='Doğrulama Kaybı')
    plt.xlabel('Epochs')
    plt.ylabel('Kayıp (Loss)')
    plt.title('Eğitim ve Doğrulama Kaybı')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.xlabel('Epochs')
    plt.ylabel('Doğruluk (Accuracy)')
    plt.title('Eğitim ve Doğrulama Doğruluğu')
    plt.legend()
    
    save_analysis_reports(analysis_dir, "training_history")
        # Epoch sonuçlarını metin olarak kaydet
    history_lines = ""
    for i in range(len(history["loss"])):
        history_lines += f"Epoch {i+1} - Loss: {history['loss'][i]:.4f}, Val Loss: {history['val_loss'][i]:.4f}, " \
                         f"Accuracy: {history['accuracy'][i]:.4f}, Val Accuracy: {history['val_accuracy'][i]:.4f}\n"
    
    save_text_report(history_lines, "training_history_epoch_details")


# Confusion Matrix Çizme
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Değer')
    plt.title('Confusion Matrix')
    save_analysis_reports(analysis_dir, "confusion_matrix_history")


# ROC Eğrisi Çizme
def plot_roc_curves(y_true, y_probs, class_names):
    plt.figure(figsize=(8, 6))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    save_analysis_reports(analysis_dir, "roc_curves_history")

# Precision-Recall Eğrisi Çizme
def plot_precision_recall(y_true, y_probs, class_names):
    plt.figure(figsize=(8, 6))
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
        plt.plot(recall, precision, label=f'{class_names[i]}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    save_analysis_reports(analysis_dir, "precision_recall_history")

# Model Değerlendirme
def evaluate_model(model, X_val, y_val, class_names):
    y_pred_probs = model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_val, axis=1)

    # Sınıflandırma raporu metin olarak oluştur
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("Sınıflandırma Raporu:\n", report)

    # Sınıflandırma raporunu txt olarak kaydet
    save_text_report(report, "classification_report")

    # Confusion Matrix ve ROC vs.
    plot_confusion_matrix(y_true, y_pred, class_names)
    plot_roc_curves(y_val, y_pred_probs, class_names)
    plot_precision_recall(y_val, y_pred_probs, class_names)

    # Sınıf bazında doğruluklar
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)

    accuracy_lines = ""
    for i, class_name in enumerate(class_names):
        line = f'{class_name} Doğruluk: {class_accuracies[i]:.2f}'
        print(line)
        accuracy_lines += line + "\n"
    
    overall_accuracy = f"Genel Doğruluk: {np.mean(class_accuracies):.4f}"
    print(overall_accuracy)
    accuracy_lines += overall_accuracy + "\n"

    # Doğruluk bilgilerini de txt olarak kaydet
    save_text_report(accuracy_lines, "class_accuracies")
