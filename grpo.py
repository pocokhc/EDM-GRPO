import random

import numpy as np
import tensorflow as tf
from net import UNet
from matplotlib import pyplot as plt
from tensorflow import keras
from tqdm import tqdm
import reward_model
import edm
import mnist
from edm import RESULT_DIR


def train_sft(
    epochs: int = 100,
    # --- collect params
    episodes: int = 5,
    num_steps: int = 100,
    collect_batch_size: int = 8,
    warmup_size: int = 1000,
    max_buffer_size: int = 100_000,
    # --- rl params
    train_num: int = 50,
    lr: float = 0.0001,
    batch_size: int = 128,
    clip_range: float = 0.1,  # reference value: 0.2
    gradient_clip_norm: float = 0.5,  # reference value: 0.5
    entropy_weight: float = 0.0,  # reference value: 0.1
    kl_beta: float = 0.2,  # reference value: 0.04
):
    edm_net = UNet()
    edm_net.load_weights(RESULT_DIR / "edm.weights.h5")
    policy_net = UNet()
    policy_net.load_weights(RESULT_DIR / "edm.weights.h5")

    reward_tf = reward_model.RewardTF()
    reward_ocr = reward_model.RewardOCR()

    optimizer = keras.optimizers.Adam(learning_rate=lr)

    buffer = []
    r_mean_list = []
    r_std_list = []
    policy_loss_list = []
    entropy_loss_list = []
    kl_loss_list = []
    for epoch in range(epochs):
        # ---------------------------------
        # collect trajectory
        # ---------------------------------
        trajectory_list = []
        rewards_list = []
        for category in tqdm(
            [
                # 0,
                # 1,
                # 2,
                # 3,
                # 4,
                5,
                # 6,
                # 7,
                # 8,
                # 9,
            ]
        ):
            plots_list = []
            for episode in tqdm(range(episodes), desc="episode", leave=False):
                x, trajectory = edm.diffuse_generate_eular_stoch(
                    policy_net,
                    collect_batch_size,
                    num_steps,
                    category,
                    rl=True,
                    ref_net=edm_net,
                    progress_leave=False,
                )
                trajectory_list.append(trajectory)

                # 出来た画像に対して報酬を計算
                plots = []
                rewards = []
                for i in range(collect_batch_size):
                    img = mnist.decode_image(x.numpy()[i])
                    r = 0

                    tf_val, per = reward_tf.predict_image(img)
                    if category == int(tf_val):
                        if per > 0.99:
                            r += 0.5
                        elif per > 0.95:
                            r += 0.4
                        else:
                            r += 0.1

                    ocr_val = reward_ocr.ocr(img)
                    if ocr_val == str(category):
                        r += 1.0
                    if ocr_val == "":
                        r -= 0.1

                    rewards.append(r)
                    plots.append([r, img, tf_val, per, ocr_val])
                rewards_list.append(rewards)
                plots_list.append(plots)

            # --- save img (debug)
            w = min(collect_batch_size, 16)
            h = min(episodes, 6)
            plt.figure(figsize=(w, h))
            for episode in range(h):
                for batch in range(w):
                    r, img, tf_val, per, ocr_val = plots_list[episode][batch]
                    plt.subplot(h, w, episode * w + batch + 1)
                    plt.imshow(img, cmap="gray")
                    plt.title(f"{r:.3f}\n {tf_val} {per * 100:.0f}% '{ocr_val}'")
                    plt.axis("off")
            plt.ylabel("episode")
            plt.xlabel("batch")
            plt.tight_layout()
            plt.savefig(RESULT_DIR / f"sft_{category}_{epoch}.png")
            plt.clf()

        # --- compute group reward
        r_mean = np.mean(rewards_list)
        r_std = max(np.std(rewards_list), 1e-2)  # あまりにも分散が低いと報酬がinfになるため制限
        for trajectory, rewards in zip(trajectory_list, rewards_list):
            for transition in trajectory:
                for batch, reward in zip(transition, rewards):
                    batch["advantage"] = (reward - r_mean) / r_std
                    buffer.append(batch)
                    if len(buffer) > max_buffer_size:
                        del buffer[0]

        r_mean_list.append(r_mean)
        r_std_list.append(r_std)

        if len(buffer) < warmup_size:
            continue

        # ---------------------------------
        # training
        # ---------------------------------
        with tqdm(range(train_num), desc=f"[{epoch}/{epochs}] Training") as pbar:
            for i in pbar:
                batch = random.sample(buffer, batch_size)
                state_x = np.asarray([e["state_x"] for e in batch])
                state_sigma = np.asarray([e["state_sigma"] for e in batch])
                state_category = np.asarray([e["state_category"] for e in batch])
                action = np.asarray([e["action"] for e in batch])
                old_logpi = np.asarray([e["logpi"] for e in batch])
                advantage = np.asarray([e["advantage"] for e in batch], dtype=np.float32)[..., np.newaxis, np.newaxis, np.newaxis]
                ref_logpi = np.asarray([e["ref_logpi"] for e in batch])

                with tf.GradientTape() as tape:
                    denoised_img = edm.denoise(policy_net, state_x, state_sigma, state_category)
                    new_logpi = edm.log_likelihood_normal(action, denoised_img, state_sigma)

                    # --- PPO
                    ratio = tf.exp(tf.clip_by_value(new_logpi - old_logpi, -10, 10))

                    ratio_clipped = tf.clip_by_value(ratio, 1 - clip_range, 1 + clip_range)
                    loss_unclipped = ratio * advantage
                    loss_clipped = ratio_clipped * advantage

                    # 小さいほうを採用
                    policy_loss = tf.minimum(loss_unclipped, loss_clipped)

                    # H = -π(a|s)lnπ(a|s)
                    entropy_loss = 0.5 * (tf.square(new_logpi) + tf.math.log(2 * tf.constant(np.pi, dtype=tf.float32)) + 1)
                    entropy_loss = tf.reduce_mean(entropy_loss, axis=[1, 2, 3])

                    # --- sft KL, kl = ref_pi/new_pi - log(ref_pi/new_pi) - 1
                    # モデルが離れすぎるとexp(diff)がinfになるので制限
                    diff = tf.clip_by_value(ref_logpi - new_logpi, -10, 10)
                    kl_loss = tf.exp(diff) - diff - 1

                    # policy_loss:最大化、entropy_loss:最大化、kl_loss:最小化
                    loss = tf.reduce_mean(-policy_loss + kl_beta * kl_loss)
                    if entropy_weight != 0:
                        loss -= entropy_weight * tf.reduce_mean(entropy_loss)
                    loss += tf.reduce_sum(policy_net.losses)  # 正則化項

                grads = tape.gradient(loss, policy_net.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, gradient_clip_norm)
                optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))

                pbar.set_postfix(
                    buffer=len(buffer),
                    r_mean=r_mean,
                    r_std=r_std,
                    policy_loss=np.mean(policy_loss),
                    entropy_loss=np.mean(entropy_loss),
                    kl_loss=np.mean(kl_loss),
                )
                policy_loss_list.append(np.mean(policy_loss))
                entropy_loss_list.append(np.mean(entropy_loss))
                kl_loss_list.append(np.mean(kl_loss))

    policy_net.save_weights(RESULT_DIR / "policy.weights.h5")

    fig, ax1 = plt.subplots()
    ax1.plot(r_mean_list, label="reward mean")
    ax1.legend(loc="upper left")
    ax1.grid()
    ax2 = ax1.twinx()
    ax2.plot(r_std_list, color="green", label="reward std")
    ax2.legend(loc="upper right")
    ax2.set_ylim(0, 2)
    plt.savefig(RESULT_DIR / "train_sft_reward.png")

    plt.figure(figsize=(6, 4))
    plt.plot(policy_loss_list, label="policy loss")
    plt.grid()
    plt.title("policy loss")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "train_sft_policy_loss.png")

    plt.figure(figsize=(6, 4))
    plt.plot(entropy_loss_list, label="entropy loss")
    plt.grid()
    plt.title("entropy loss")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "train_sft_entropy_loss.png")

    plt.figure(figsize=(6, 4))
    plt.plot(kl_loss_list, label="kl loss")
    plt.grid()
    plt.title("kl loss")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "train_sft_kl_loss.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and generate using SFT.")
    parser.add_argument("--train", action="store_true", help="Run the training process.")
    args = parser.parse_args()

    if args.train:
        train_sft()

    policy_net = UNet()
    policy_net.load_weights(RESULT_DIR / "policy.weights.h5")
    edm.generate(policy_net, mode="eular_deter", prefix="sft_")
    edm.generate(policy_net, mode="eular_stoch", prefix="sft_")
    edm.generate(policy_net, mode="blend", prefix="sft_")
    edm.generate(policy_net, mode="edm", prefix="sft_")
