import os
import numpy as np
import cv2
import random
from sklearn.model_selection import train_test_split
import tensorflow as ts

to_categorical = ts.keras.utils.to_categorical

# Veri artırma fonksiyonu
def augment_image(image):
    # 1. Döndürme (90, 180, 270 derece)
    if random.random() > 0.5:
        angle = random.choice([90, 180, 270])
        image = cv2.rotate(image, {90: cv2.ROTATE_90_CLOCKWISE,
                                   180: cv2.ROTATE_180,
                                   270: cv2.ROTATE_90_COUNTERCLOCKWISE}[angle])
    
    # 2. Yatay çevirme
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
    
    # 3. Parlaklık değişimi
    if random.random() > 0.5:
        factor = random.uniform(0.7, 1.3)  # %70 ile %130 arasında parlaklık değişimi
        image = np.clip(image * factor, 0, 255).astype(np.uint8)
    
    # 4. Gürültü ekleme
    if random.random() > 0.5:
        noise = np.random.randint(0, 30, image.shape, dtype='uint8')
        image = cv2.add(image, noise)
    
    return image

# Veri yükleme fonksiyonu (veri artırma dahil)
def load_data(dataset_path, img_size=128, augment_factor=2):
    categories = ["no_tumor", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
    data = []
    labels = []

    for i, category in enumerate(categories):
        folder_path = os.path.join(dataset_path, category)
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Griye çevir
            image = cv2.resize(image, (img_size, img_size))  # Boyutu küçült
            
            # Orijinal görüntüyü ekle
            data.append(image.reshape(img_size, img_size, 1))  # 4D şekle dönüştürme
            labels.append(i)

            # Veri artırma yaparak ekstra görüntüler oluştur
            for _ in range(augment_factor):
                augmented_image = augment_image(image)
                data.append(augmented_image.reshape(img_size, img_size, 1))  # 4D şekle dönüştürme
                labels.append(i)

    data = np.array(data) / 255.0  # Normalizasyon (0-1 aralığına çekme)
    labels = np.array(labels)

    labels = to_categorical(labels, num_classes=4)  # One-hot encoding

    return train_test_split(data, labels, test_size=0.2, random_state=42)
