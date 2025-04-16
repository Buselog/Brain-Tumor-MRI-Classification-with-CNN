import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import tensorflow as tf

def plot_training_history(history):
    """Eğitim sürecindeki loss ve accuracy grafiklerini çizer (kaydetmez)."""
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

    plt.show()
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Değer')
    plt.title('Confusion Matrix')
    plt.show()
    plt.close()


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
    plt.show()
    plt.close()


def plot_precision_recall(y_true, y_probs, class_names):
    plt.figure(figsize=(8, 6))
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
        plt.plot(recall, precision, label=f'{class_names[i]}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()
    plt.close()


def evaluate_model(model, X_val, y_val, class_names):
    y_pred_probs = model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_val, axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names)
    print("Sınıflandırma Raporu:\n", report)

    plot_confusion_matrix(y_true, y_pred, class_names)
    plot_roc_curves(y_val, y_pred_probs, class_names)
    plot_precision_recall(y_val, y_pred_probs, class_names)

    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)

    for i, class_name in enumerate(class_names):
        print(f'{class_name} Doğruluk: {class_accuracies[i]:.2f}')
    
    overall_accuracy = np.mean(class_accuracies)
    print(f"Genel Doğruluk: {overall_accuracy:.4f}")
