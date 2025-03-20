import math
import os
from pathlib import Path

import mnist
import numpy as np
import tensorflow as tf
from mnist import CATEGORY_NUM, IMG_SIZE, MNIST_STD
from net import UNet
from matplotlib import pyplot as plt
from tensorflow import keras
from tqdm import tqdm

RESULT_DIR = Path(__file__).parent / "result"
os.makedirs(RESULT_DIR, exist_ok=True)


def log_likelihood_normal(x, mu, sigma):
    return -0.5 * math.log(2 * math.pi) - tf.math.log(sigma) - 0.5 * (((x - mu) / sigma) ** 2)


def denoise(net, noisy_img, sigma, category, sigma_data: float = MNIST_STD, training: bool = False):
    # ノイズ画像の正規化
    c_in = 1 / tf.sqrt(sigma_data**2 + sigma**2)
    scaled_noisy_img = c_in * noisy_img

    # sigmaの正規化
    c_noise = tf.math.log(sigma) / 4

    # 入力画像とネットワーク出力を元にノイズ除去画像を生成
    c_out = sigma * sigma_data / tf.sqrt(sigma**2 + sigma_data**2)
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    network_output = net([scaled_noisy_img, c_noise, category], training=training)
    return c_skip * noisy_img + c_out * network_output


def train(epochs: int = 10, batch_size: int = 128):
    dataset_imgs, dataset_categories = mnist.load_dataset()
    train_dataset = tf.data.Dataset.from_tensor_slices((dataset_imgs, dataset_categories)).shuffle(len(dataset_imgs)).batch(batch_size)

    net = UNet()
    lr_schedule = keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=0.001,
        decay_steps=int(epochs * len(train_dataset) * 0.8),
        end_learning_rate=0.00005,
        power=2.0,
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    loss_history = []
    lr_history = []
    for epoch in range(epochs):
        with tqdm(train_dataset, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for clean_image, category in pbar:
                loss = _edm_train(net, clean_image, category, optimizer)

                # plot用
                lr = float(optimizer.learning_rate)
                loss_history.append(loss)
                lr_history.append(lr)
                pbar.set_postfix(loss=loss, lr=lr)

    net.save_weights(RESULT_DIR / "edm.weights.h5")
    fig, ax1 = plt.subplots()
    ax1.plot(loss_history, label="loss")
    ax1.set_ylim(0, 0.5)
    ax1.legend(loc="upper left")
    ax1.grid()
    ax2 = ax1.twinx()
    ax2.plot(lr_history, color="green", label="lr")
    ax2.legend(loc="upper right")
    plt.savefig(RESULT_DIR / "train_loss.png")


def _edm_train(
    net: UNet,
    clean_img,
    category,
    optimizer,
    p_mean=-1.2,
    p_std=1.2,
    sigma_data: float = MNIST_STD,
):
    batch_size = clean_img.shape[0]

    # loss_fn = keras.losses.MeanSquaredError(reduction="none")
    loss_fn = keras.losses.Huber(reduction="none")

    # ノイズレベルをサンプリング
    sigma = tf.exp(p_mean + p_std * tf.random.normal(shape=(batch_size, 1, 1, 1)))

    # ノイズ入り画像を生成
    noise = tf.random.normal(shape=clean_img.shape)
    noisy_img = clean_img + sigma * noise

    with tf.GradientTape() as tape:
        # ノイズ除去画像を作成
        denoised_image = denoise(net, noisy_img, sigma, category, training=True)

        # 損失を計算
        weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2
        loss = weight * loss_fn(clean_img, denoised_image)[..., tf.newaxis]
        loss = tf.reduce_mean(loss)

    grad = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grad, net.trainable_variables))

    return loss.numpy()


def create_timesptes(N: int, sigma_min=0.002, sigma_max=80, rho=7):
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    t = [(max_inv_rho + i / (N - 1) * (min_inv_rho - max_inv_rho)) ** rho for i in range(N)]
    t += [0]
    return t


def diffuse_generate_eular_deter(net: UNet, batch_size: int, num_steps: int, category: int):
    timesteps = create_timesptes(num_steps)
    x = tf.random.normal((batch_size, IMG_SIZE, IMG_SIZE, 1)) * timesteps[0]

    tf_category = tf.convert_to_tensor([category] * batch_size, dtype=tf.float32)

    step_imgs = []
    for t_curr in tqdm(timesteps[:-1]):
        tf_t_curr = tf.convert_to_tensor([t_curr] * batch_size, dtype=tf.float32)[..., tf.newaxis, tf.newaxis, tf.newaxis]
        x = denoise(net, x, tf_t_curr, tf_category)
        step_imgs.append([x, t_curr])

    return x, step_imgs


def diffuse_generate_eular_stoch(
    net: UNet,
    batch_size: int,
    num_steps: int,
    category: int,
    rl: bool = False,
    ref_net=None,
    progress_leave: bool = True,
):
    # RL用に各ステップの情報を保存

    timesteps = create_timesptes(num_steps)
    x = tf.random.normal((batch_size, IMG_SIZE, IMG_SIZE, 1)) * timesteps[0]

    tf_category = tf.convert_to_tensor([category] * batch_size, dtype=tf.float32)

    trajectory = []
    for t_curr in tqdm(timesteps[:-1], leave=progress_leave):
        tf_t_curr = tf.convert_to_tensor([t_curr] * batch_size, dtype=tf.float32)[..., tf.newaxis, tf.newaxis, tf.newaxis]
        e = tf.random.normal((batch_size, IMG_SIZE, IMG_SIZE, 1)) * t_curr
        denoised_img = denoise(net, x, tf_t_curr, tf_category)
        next_x = denoised_img + e

        if rl:
            # 強化学習用
            state_x = x.numpy()
            state_sigma = tf_t_curr.numpy()
            state_category = tf_category.numpy()
            action = next_x.numpy()
            mean = denoised_img.numpy()
            sigma = float(t_curr)
            logpi = log_likelihood_normal(action, mean, sigma).numpy()
            if ref_net is not None:
                ref_denoised_img = denoise(ref_net, x, tf_t_curr, tf_category)
                ref_logpi = log_likelihood_normal(action, ref_denoised_img, sigma).numpy()
            trajectory.append(
                [
                    {
                        "state_x": state_x[i],
                        "state_sigma": state_sigma[i],
                        "state_category": state_category[i],
                        "action": action[i],
                        "mean": mean[i],
                        "sigma": sigma,
                        "logpi": logpi[i],
                        "ref_logpi": None if ref_net is None else ref_logpi[i],
                    }
                    for i in range(batch_size)
                ]
            )
        else:
            trajectory.append([next_x, denoised_img, t_curr])

        x = next_x
    return x, trajectory


def diffuse_generate_blend(net: UNet, batch_size: int, num_steps: int, category: int):
    timesteps = create_timesptes(num_steps)
    x = tf.random.normal((batch_size, IMG_SIZE, IMG_SIZE, 1)) * timesteps[0]

    tf_category = tf.convert_to_tensor([category] * batch_size, dtype=tf.float32)

    step_imgs = []
    for t_curr, t_next in tqdm(zip(timesteps[:-1], timesteps[1:]), total=len(timesteps) - 1):
        tf_t_curr = tf.convert_to_tensor([t_curr] * batch_size, dtype=tf.float32)[..., tf.newaxis, tf.newaxis, tf.newaxis]

        blend = t_next / t_curr
        denoised_img = denoise(net, x, tf_t_curr, tf_category)
        x = blend * x + (1 - blend) * denoised_img

        step_imgs.append([x, denoised_img, blend, t_curr])

    return x, step_imgs


def diffuse_generate_edm(
    net: UNet,
    batch_size: int,
    num_steps: int,
    category: int,
    s_churn=0,
    s_min=0,
    s_max=float("inf"),
    s_noise=1,
):
    timesteps = create_timesptes(num_steps)
    x = tf.random.normal((batch_size, IMG_SIZE, IMG_SIZE, 1)) * timesteps[0]

    tf_category = tf.convert_to_tensor([category] * batch_size, dtype=tf.float32)

    step_imgs = []
    for t_curr, t_next in tqdm(zip(timesteps[:-1], timesteps[1:]), total=len(timesteps) - 1):
        gamma = min(s_churn / num_steps, np.sqrt(2) - 1) if s_min <= t_curr <= s_max else 0
        t_hat = t_curr + gamma * t_curr
        if gamma > 0:
            e = tf.random.normal((batch_size, IMG_SIZE, IMG_SIZE, 1)) * s_noise
            x = x + ((t_hat**2 - t_curr**2) ** 0.5) * e

        tf_t_hat = tf.convert_to_tensor([t_hat] * batch_size, dtype=tf.float32)[..., tf.newaxis, tf.newaxis, tf.newaxis]
        denoised_img = denoise(net, x, tf_t_hat, tf_category)

        d = (x - denoised_img) / t_hat
        dt = t_next - t_hat

        x_2 = x + dt * d
        if t_next != 0:
            tf_t_next = tf.convert_to_tensor([t_next] * batch_size, dtype=tf.float32)[..., tf.newaxis, tf.newaxis, tf.newaxis]
            denoised_img_2 = denoise(net, x_2, tf_t_next, tf_category)
            d_2 = (x_2 - denoised_img_2) / t_next
            x = x + dt * (d + d_2) / 2
        else:
            x = x_2

        step_imgs.append([x, denoised_img, x_2, denoised_img_2, t_curr])

    return x, step_imgs


def generate(
    net: UNet = None,
    num_steps: int = 100,
    num_samples: int = 8,
    mode: str = "eular_stoch",
    plot_history: bool = True,
    prefix: str = "",
):
    if net is None:
        net = UNet()
        net.load_weights(RESULT_DIR / "edm.weights.h5")

    gen_func = {
        "eular_deter": diffuse_generate_eular_deter,
        "eular_stoch": diffuse_generate_eular_stoch,
        "blend": diffuse_generate_blend,
        "edm": diffuse_generate_edm,
    }[mode]

    # --- generate
    trajectory_list = []
    generated_imgs_list = []
    for category in range(CATEGORY_NUM):
        generated_imgs, trajectory = gen_func(net, num_samples, num_steps, category)
        generated_imgs_list.append(generated_imgs)
        trajectory_list.append(trajectory)

    # --- plot
    plt.figure(figsize=(num_samples, CATEGORY_NUM))
    for category in range(CATEGORY_NUM):
        for i in range(num_samples):
            img = mnist.decode_image(generated_imgs_list[category][i])
            plt.subplot(CATEGORY_NUM, num_samples, category * num_samples + i + 1)
            plt.imshow(img, cmap="gray")
            plt.xticks([])
            plt.yticks([])

            # 左端に行番号を配置
            if i == 0:
                plt.text(-3, img.shape[0] // 2, str(category), va="center", ha="right", fontsize=12)

    plt.savefig(RESULT_DIR / f"plot_{prefix}{mode}.png")

    # --- plot history
    if plot_history:
        target_img_idx = 0

        # 多いので一定間隔で抜き出し
        traj_size = len(trajectory_list[0])
        step_interval = int(traj_size / min(traj_size, 12))
        indexies = list(sorted({0, traj_size - 1} | set(range(0, traj_size, step_interval))))
        w = len(indexies)

        plt.figure(figsize=(w, CATEGORY_NUM))
        plt.xticks([])
        plt.yticks([])
        plt.gca().set_frame_on(False)
        for category in range(0, CATEGORY_NUM):
            for i, step in enumerate(indexies):
                sigma = trajectory_list[category][step][-1]
                img = trajectory_list[category][step][0][target_img_idx]
                img = mnist.decode_image(img)
                plt.subplot(CATEGORY_NUM, w, category * w + i + 1)
                plt.imshow(img, cmap="gray")
                plt.xticks([])
                plt.yticks([])

                # 左側にカテゴリ番号を表示
                if i == 0:
                    plt.text(-3, img.shape[0] // 2, str(category), va="center", ha="right", fontsize=12)

                # 下側にステップ番号を表示
                if category == CATEGORY_NUM - 1:
                    plt.text(img.shape[1] // 2, img.shape[0] + 15, f"Step{step}\n{sigma:.3f}", ha="center", fontsize=10)
        plt.savefig(RESULT_DIR / f"plot_{prefix}{mode}_history.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and generate using SFT.")
    parser.add_argument("--train", action="store_true", help="Run the training process.")
    args = parser.parse_args()

    if args.train:
        train()

    generate(mode="eular_deter")
    generate(mode="eular_stoch")
    generate(mode="blend")
    generate(mode="edm")
