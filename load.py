import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# =============================
# ⚙️ CONFIGURATION GLOBALE
# =============================
BASE_DIR = r"C:\Users\pc\Desktop\i2\deeplearning\projet\FaceMaskDataset\FaceMaskDataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train224")
TEST_DIR  = os.path.join(BASE_DIR, "test224")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


# =============================
# 📦 Fonction principale
# =============================
def load_data(show_examples=False):
    """
    Charge le dataset, applique prétraitement, data augmentation,
    et retourne les datasets et paramètres utiles.
    
    Paramètres :
        show_examples (bool): afficher quelques images avant/après augmentation.
        
    Retourne :
        train_ds, val_ds, test_ds, class_names, data_augmentation, IMG_SIZE, BATCH_SIZE
    """

    print("TRAIN_DIR:", TRAIN_DIR)
    print("TEST_DIR :", TEST_DIR)

    # =============================
    # 📥 Chargement dataset
    # =============================
    train_ds = image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    test_ds = image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_ds.class_names
    print("Classes :", class_names)

    # =============================
    # ⚡ Pipeline optimisé
    # =============================
    train_ds = (
        train_ds
        .shuffle(1000)
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )

    val_ds = (
        val_ds
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )

    test_ds = (
        test_ds
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )

    # =============================
    # 🔄 Data augmentation
    # =============================
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.05, 0.05),
        ],
        name="data_augmentation"
    )

    # =============================
    # 👀 Affichage des images (optionnel)
    # =============================
    if show_examples:
        image_batch, label_batch = next(iter(train_ds))

        plt.figure(figsize=(10, 5))
        for i in range(6):
            ax = plt.subplot(2, 6, i + 1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            plt.title(class_names[label_batch[i]])
            plt.axis("off")

        augmented_images = data_augmentation(image_batch, training=True)

        for i in range(6):
            ax = plt.subplot(2, 6, i + 7)
            plt.imshow(augmented_images[i].numpy().astype("uint8"))
            plt.title("Augmented")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    return train_ds, val_ds, test_ds, class_names, data_augmentation, IMG_SIZE, BATCH_SIZE


def plot_history(history, title_prefix="Modèle 1"):
    hist = history.history

    epochs_range = range(1, len(hist["loss"]) + 1)

    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, hist["loss"], label="Train loss")
    plt.plot(epochs_range, hist["val_loss"], label="Val loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} - Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, hist["accuracy"], label="Train acc")
    plt.plot(epochs_range, hist["val_accuracy"], label="Val acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} - Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


# Si tu exécutes load.py directement (python load.py)
if __name__ == "__main__":
    load_data(show_examples=True)
