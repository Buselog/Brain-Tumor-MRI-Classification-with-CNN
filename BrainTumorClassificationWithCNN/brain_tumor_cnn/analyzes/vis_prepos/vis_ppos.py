def save_augmentation_pairs(pairs, dir_type):
    import matplotlib.pyplot as plt
    from datetime import datetime
    import os

    # Tarih damgası ekle
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_file_name = f"augmentation_grid_{timestamp}.png"
    save_path = os.path.join(dir_type, full_file_name)

    fig, axes = plt.subplots(len(pairs), 2, figsize=(6, 3 * len(pairs)))
    fig.suptitle("Veri Artırma Öncesi ve Sonrası", fontsize=16)

    for i, (original, augmented) in enumerate(pairs):
        axes[i, 0].imshow(original, cmap='gray')
        axes[i, 0].set_title("Orijinal")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(augmented, cmap='gray')
        axes[i, 1].set_title("Augment Edilmiş")
        axes[i, 1].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Görsel başarıyla kaydedildi: {save_path}")
