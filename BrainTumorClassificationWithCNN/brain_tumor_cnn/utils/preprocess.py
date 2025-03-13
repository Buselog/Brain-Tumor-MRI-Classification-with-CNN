import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

import tensorflow as ts
to_categorical = ts.keras.utils.to_categorical


def load_data(dataset_path, img_size=128):
    categories = ["no_tumor", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
    data = []
    labels = []

    for i, category in enumerate(categories):
        folder_path = os.path.join(dataset_path, category)
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Görüntüyü griye çevir
            image = cv2.resize(image, (img_size, img_size))  # Boyutu küçült
            data.append(image)  # Görüntüyü düzleştirmeden ekle
            labels.append(i)  # 0: no_tumor, 1: glioma, 2: meningioma, 3: pituitary

    data = np.array(data) / 255.0  # Normalizasyon (0-1 aralığına çekme)
    data = data.reshape(-1, img_size, img_size, 1)  # CNN için 4D şekle dönüştürme (örneğin: (128, 128, 1))
    labels = np.array(labels)

    labels = to_categorical(labels, num_classes=4)  # One-hot encoding

    return train_test_split(data, labels, test_size=0.2, random_state=42)
