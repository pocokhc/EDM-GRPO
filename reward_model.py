import cv2
import numpy as np
import pytesseract
from mnist import IMG_SIZE
from edm import RESULT_DIR
from matplotlib import pyplot as plt
from tensorflow import keras
import mnist

kl = keras.layers


class RewardOCR:
    def ocr(self, img, scale: float = 1) -> str:
        img = cv2.resize(img, (int(28 * scale), int(28 * scale)), interpolation=cv2.INTER_LINEAR)
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
        return str(pytesseract.image_to_string(img, config="--psm 10 -c tessedit_char_whitelist=0123456789")).strip()

    def compute_reward(self, category: int, img: np.ndarray) -> float:
        d = self.ocr(mnist.decode_image(img))
        if d == "":
            return -1
        return 1 if d == str(int(category)) else 0

    def plot_images(self, imgs, pos: int = 0, n_rows: int = 5, n_cols: int = 10) -> None:
        assert len(imgs) >= n_cols * n_rows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
        for i, ax in enumerate(axes.ravel()):
            i += pos
            img = mnist.decode_image(imgs[i])
            ax.imshow(img, cmap="gray")
            ocr_label = self.ocr(img)
            ax.set_title(f"'{ocr_label}'")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "plot_ocr_images.png")


class RewardTF:
    def __init__(self, is_load: bool = True):
        self.model = keras.Sequential(
            [
                kl.Input((IMG_SIZE, IMG_SIZE, 1)),
                kl.Conv2D(32, (3, 3), activation="gelu"),
                kl.MaxPooling2D((2, 2)),
                kl.Conv2D(64, (3, 3), activation="gelu"),
                kl.MaxPooling2D((2, 2)),
                kl.Conv2D(64, (3, 3), activation="gelu"),
                kl.Flatten(),
                kl.Dense(64, activation="gelu"),
                kl.Dense(10, activation="softmax"),
            ]
        )
        if is_load:
            self.model.load_weights(RESULT_DIR / "reward_model.weights.h5")

    def train(self, epochs: int = 1, batch_size: int = 512):
        dataset_imgs, dataset_categories = mnist.load_dataset()

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model.fit(dataset_imgs, dataset_categories, epochs=epochs, batch_size=batch_size)
        self.model.save_weights(RESULT_DIR / "reward_model.weights.h5")
        test_loss, test_acc = self.model.evaluate(dataset_imgs, dataset_categories)
        print(f"Test accuracy: {test_acc:.4f}, loss: {test_loss:.4f}")

    def predict_image(self, img: np.ndarray, encode: bool = True):
        if encode:
            img = mnist.encode_image(img)
        img = img[np.newaxis, ...]
        prediction = self.model(img)[0]
        idx = np.argmax(prediction)
        return int(idx), float(prediction[idx])

    def plot_images(self, imgs, pos: int = 0, n_rows: int = 5, n_cols: int = 10):
        assert len(imgs) >= n_cols * n_rows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
        for i, ax in enumerate(axes.ravel()):
            i += pos
            img = mnist.decode_image(imgs[i])
            ax.imshow(img, cmap="gray")
            tf_label, per = self.predict_image(img)
            ax.set_title(f"'{tf_label}' {per * 100:.0f}%")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "plot_tf_images.png")


if __name__ == "__main__":
    RewardOCR().plot_images(mnist.load_dataset()[0])

    RewardTF(is_load=False).train()
    RewardTF().plot_images(mnist.load_dataset()[0])
