#DAHA ÖNCEDEN İNDİRİLEN MODEL VE HİSTORYSİ KULLANILMASI AMAÇLANDI 
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import analyzes.analysis as analysis
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from utils.preprocessv2 import load_data
from models.model import build_model

dataset_path = "C:/Users/2022/Desktop/Brain-Tumor-MRI-Classification-with-CNN-main - V2/Brain-Tumor-MRI-Classification-with-CNN-main/BrainTumorClassificationWithCNN/brain_tumor_cnn/dataset"
print(f"Veri seti yolu: {dataset_path}") 
X_train, y_train = load_data(dataset_path, mode="train")
X_val, y_val = load_data(dataset_path, mode="test")

print(f"✅ Eğitim verisi: {X_train.shape}, Etiketler: {y_train.shape}")
print(f"✅ Test verisi: {X_val.shape}, Etiketler: {y_val.shape}")

# Daha önce eğitilen modeliin geçmişini saklamak için dizinleri ayarla
results_dir ="C:/Users/2022/Desktop/Brain-Tumor-MRI-Classification-with-CNN-main - V2/Brain-Tumor-MRI-Classification-with-CNN-main/BrainTumorClassificationWithCNN/brain_tumor_cnn/results"
models_dir = os.path.join(results_dir, "models")

# Klasörleri oluştur (yoksa)
os.makedirs(models_dir, exist_ok=True)

# Model ve history dosya yolları
model_path = os.path.join(models_dir, "brain_tumor_ann.h5")
history_path = os.path.join(models_dir, "training_history.json")

# Daha önce eğitilmiş model var mı kontrol et
if os.path.exists(model_path):
    use_existing = input("Daha önce eğitilmiş bir model bulundu. Bunu kullanmak ister misiniz? (E/H): ").strip().lower()
    if use_existing == 'e':
        model = load_model(model_path)
        print("Eğitilmiş model yüklendi.")
        train_model = False  # Modeli yeniden eğitmeye gerek yok

        # Eğitim geçmişini yükle
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
            print("Eğitim geçmişi yüklendi.")
        else:
            history = None
            print("Eğitim geçmişi bulunamadı.")
    else:
        model = build_model()
        train_model = True
        history = None
        print("Yeni model oluşturuldu.")
else:
    model = build_model()
    train_model = True
    history = None
    print("Önceden eğitilmiş model bulunamadı, yeni model oluşturuluyor.")

# Eğer yeni model oluşturulmuşsa veya kullanıcı eğitmek istiyorsa, modeli eğit
if train_model:
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy')

    history_obj = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint]
    )

    history = history_obj.history  # Modelin eğitim geçmişini al
    print("Model eğitimi tamamlandı ve kaydedildi.")

    # Eğitim geçmişini JSON dosyasına kaydet
    with open(history_path, "w") as f:
        json.dump(history, f)
    print("Eğitim geçmişi kaydedildi.")

# Modeli değerlendir
test_loss, test_acc = model.evaluate(X_val, y_val)
print(f"Test Doğruluk: {test_acc:.4f}")

# Sınıflandırma raporunu yazdır
y_pred = np.argmax(model.predict(X_val), axis=1)
y_true = np.argmax(y_val, axis=1)

print("Sınıflandırma Raporu:\n", classification_report(y_true, y_pred, target_names=["No Tumor", "Glioma", "Meningioma", "Pituitary"]))

# Eğitim geçmişini varsa görselleştir
if history:
    analysis.plot_training_history(history)
else:
    print("Eğitim geçmişi mevcut değil, grafik çizdirilemiyor.")

# Model değerlendirme
analysis.evaluate_model(model, X_val, y_val, class_names=["No Tumor", "Glioma", "Meningioma", "Pituitary"])

model.summary()

