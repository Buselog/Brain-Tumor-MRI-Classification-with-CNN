import numpy as np
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report
from utils.preprocess import load_data
from models.model import build_model

dataset_path = "dataset/data"
X_train, X_val, y_train, y_val = load_data(dataset_path)

model = build_model()
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_accuracy')

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint]
)

# Modeli değerlendir
test_loss, test_acc = model.evaluate(X_val, y_val)
print(f"Test Doğruluk: {test_acc:.4f}")

# Sınıflandırma raporunu yazdır
y_pred = np.argmax(model.predict(X_val), axis=1)
y_true = np.argmax(y_val, axis=1)

print("Sınıflandırma Raporu:\n", classification_report(y_true, y_pred, target_names=["No Tumor", "Glioma", "Meningioma", "Pituitary"]))

# Eğitilmiş modeli kaydet
model.save("brain_tumor_ann.h5")
print("Model kaydedildi: brain_tumor_ann.h5")