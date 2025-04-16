import os  # Dosya ve dizin işlemleri için
import numpy as np  # Sayısal işlemler yapmak için
import cv2  # Görüntü işleme kütüphanesi (örn: görüntüleri döndürme, parlaklık değiştirme).
import random  # Rastgele işlemler yapmak için
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test olarak ikiye ayırmak için.
import tensorflow as ts  # Derin öğrenme modeli için kullanılır

to_categorical = ts.keras.utils.to_categorical  # Sınıf etiketlerini one-hot encoding formatına çevirmek için


# Veri artırma fonksiyonu
def augment_image(image):  # Rastgele Döndürme
    # 1. Döndürme (90, 180, 270 derece)
    if random.random() > 0.5:  # %50 olasılıkla görüntü döndürülür.
        angle = random.choice([90, 180, 270])  # Döndürme açısı rastgele olarak 90°, 180° veya 270°
        # cv2.rotate fonksiyonu ile görüntü saat yönünde döndürülür:
        image = cv2.rotate(image, {90: cv2.ROTATE_90_CLOCKWISE,
                                   180: cv2.ROTATE_180,
                                   270: cv2.ROTATE_90_COUNTERCLOCKWISE}[angle])

    # 2. Yatay çevirme
    if random.random() > 0.5:  # %50 olasılıkla görüntü yatay olarak ters çevrilir
        image = cv2.flip(image, 1)

    # 3. Parlaklık değişimi
    '''
    image, 128x128 boyutunda bir numpy dizisi ve her piksel 0 ile 255 arasında bir değer içerir
    Factor = 1.2 → Görüntü %20 daha parlak 
    Factor = 0.8 → Görüntü %20 daha karanlık
    '''
    if random.random() > 0.5:
        factor = random.uniform(0.7, 1.3)  # %70 ile %130 arasında parlaklık değişimi
        image = np.clip(image * factor, 0, 255).astype(np.uint8)
        # Görüntülerdeki piksel değerleri 0 ile 255 arasında olmalı

    # 4. Gürültü ekleme
    if random.random() > 0.5:  # %50 olasılıkla rastgele gürültü eklenir
        noise = np.random.randint(0, 30, image.shape, dtype='uint8')
        # ile 0-30 arasında rastgele piksel değerleri eklenir
        image = cv2.add(image, noise)  # Değerler görüntüye eklenir

    return image


# Veri yükleme fonksiyonu (veri artırma dahil)
def load_data(dataset_path, img_size=128, augment_factor=1, mode="train", collect_augmented_examples=True):
    categories = ["notumor", "glioma", "meningioma", "pituitary"]
    data = []
    labels = []
    augmentation_pairs = []  # 👈 Orijinal ve augment edilmiş görüntüleri burada saklayacağız (sadece örnek için)

    dataset_mode_path = os.path.join(dataset_path, mode)

    for i, category in enumerate(categories):
        folder_path = os.path.join(dataset_mode_path, category)

        if not os.path.exists(folder_path):
            continue

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Griye çevirir

            if image is None:
                continue

            image = cv2.resize(image, (img_size, img_size)) # Boyutu küçült (128x128 boyutuna)
            image_reshaped = image.reshape(img_size, img_size, 1) # 4D şekle dönüştürme (128,128,1)
            data.append(image_reshaped)
            labels.append(i) # Etiketler, sınıf numarası olarak eklenir

            if mode == "train": # Sadece eğitim setinde veri artırma yap
            # Veri artırma yaparak ekstra görüntüler oluştur    
                for _ in range(augment_factor): # Her görüntü için augment_factor kadar yeni veri üret
                    augmented_image = augment_image(image) # Veri artırma fonksiyonu (augment_image)
                    augmented_reshaped = augmented_image.reshape(img_size, img_size, 1) # CNN'e uygun hale getirme, 4D şekle dönüştürme
                    data.append(augmented_reshaped) # listeye ekleme işlemi
                    labels.append(i)

                    # 👇 Sadece ilk birkaç örnek için orijinal ve augment edilmiş halini kaydet
                    if collect_augmented_examples and len(augmentation_pairs) < 30:
                        augmentation_pairs.append((image, augmented_image))

    data = np.array(data) / 255.0 # data içindeki sonuç verilerine normalizasyon (0-1 aralığına çekme) uygulama
    labels = to_categorical(np.array(labels), num_classes=4) # numpy dizisine çevirme ve One-hot encoding.
    # Her etiket, uzunluğu num_classes=4 olan bir diziye çevrildi.
    # Etiketlerin sıralama değil, kategorik olduğunu anlatmak için [0,1,1,1], [0,1,0,0] ...
    # sıralı bir değer gibi algılanmaması için one-hot encoding

    if mode == "train":
        return data, labels, augmentation_pairs
    else:
        return data, labels


    # return train_test_split(data, labels, test_size=0.2, random_state=42)
    # random_state = 42 :
    # Verinin bölünmesini rastgele yapmak ama her çalıştırmada aynı sonuç almak için
    # Her çalıştırmada aynı eğitim ve test setleri oluşur
    # Bu, modelin tutarlı şekilde test edilmesini sağlar
    # Eğer random_state verilmezse :
    # train_test_split her çalıştırıldığında farklı şekilde veri böler.
    # Yani bir çalıştırmada image1 test setine giderken, başka bir çalıştırmada eğitim setine gidebilir.
