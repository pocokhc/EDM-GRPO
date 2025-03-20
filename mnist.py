import cv2
import numpy as np
from tensorflow import keras

IMG_SIZE = 32
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 1)
CATEGORY_NUM = 10

MNIST_STD = 0.5782144665718079


def load_dataset():
    (x_train, y_train), (_, _) = keras.datasets.mnist.load_data()
    x_train = encode_image(x_train, resize=True)
    return x_train.astype(np.float32), y_train.astype(np.float32)


def print_mnist_std():
    x_train, _ = load_dataset()
    std = float(np.std(x_train))
    print(f"{std=}")
    return std


def encode_image(img, resize: bool = False):
    if resize:
        img = np.array([cv2.resize(x, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR) for x in img])
    if img.shape[-1] != 1:
        img = img[..., np.newaxis]
    img = (img / 255.0) * 2 - 1  # [0,255]->[-1,1]
    return img.astype(np.float32)


def decode_image(x):
    img = np.clip(x, -1.0, 1.0)
    img = (((img + 1) / 2) * 255).astype(np.uint8)  # [-1,1] -> [0,255]
    if img.shape[-1] == 1:
        img = np.squeeze(img, -1)
    return img


def plot_mnist_images(pos: int = 0, n_rows: int = 5, n_cols: int = 10) -> None:
    from matplotlib import pyplot as plt

    imgs, categories = load_dataset()
    imgs = decode_image(imgs)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    for i, ax in enumerate(axes.ravel()):
        i += pos
        img, label = imgs[i], categories[i]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"[{i}] {int(label)}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print_mnist_std()
    plot_mnist_images()
