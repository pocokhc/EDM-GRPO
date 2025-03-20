import math
from functools import partial

import numpy as np
import tensorflow as tf
from mnist import CATEGORY_NUM, IMG_SHAPE
from matplotlib import pyplot as plt
from tensorflow import keras

kl = keras.layers

# alias
Conv2D1x1 = partial(kl.Conv2D, kernel_size=1, strides=1, padding="valid")
Conv2D3x3 = partial(kl.Conv2D, kernel_size=3, strides=1, padding="same")
IdentityLayer = partial(kl.Lambda, function=lambda x: x)  # 何もしないレイヤー


class FourierFeatures(keras.layers.Layer):
    def __init__(self, dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim

    def build(self, input_shape):
        self.weight = self.add_weight(shape=(1, self.dim // 2), initializer=keras.initializers.RandomNormal(), trainable=False, name="weight")
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if len(inputs.shape) > 1:
            inputs = tf.squeeze(inputs)
        f = 2.0 * math.pi * tf.expand_dims(inputs, axis=1) @ self.weight
        return tf.concat([tf.cos(f), tf.sin(f)], axis=-1)

    @staticmethod
    def plot(dim: int = 128, N: int = 1000):  # for debug
        model = FourierFeatures(dim)
        timestep = np.linspace(0, 80, N)
        emb = model(tf.constant(timestep)).numpy()
        plt.pcolormesh(timestep, np.arange(dim), emb.T, cmap="RdBu")
        plt.ylabel("dimension")
        plt.xlabel("time step")
        plt.colorbar()
        plt.show()


class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def call(self, time: tf.Tensor) -> tf.Tensor:
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = tf.exp(tf.range(half_dim, dtype=tf.float32) * -embeddings)
        embeddings = tf.expand_dims(time, axis=-1) * tf.expand_dims(embeddings, axis=0)
        embeddings = tf.concat([tf.math.sin(embeddings), tf.math.cos(embeddings)], axis=-1)
        return embeddings

    @staticmethod
    def plot(dim: int = 500, N: int = 100):  # for debug
        model = PositionalEmbedding(dim)
        timestep = np.linspace(0, 80, N)
        emb = model(tf.constant(timestep)).numpy()
        plt.pcolormesh(timestep, np.arange(dim), emb.T, cmap="RdBu")
        plt.ylabel("dimension")
        plt.xlabel("time step")
        plt.colorbar()
        plt.show()


class AdaGroupNorm2D(keras.layers.Layer):
    def __init__(self, group_size: int = 32, eps: float = 1e-5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.group_size = group_size
        self.eps = eps

    def build(self, input_shape):
        in_channels = input_shape[-1]

        # group_sizeは割り切れる場合のみ指定、-1: LayerNorm, 1: InstanceNorm
        groups = self.group_size if in_channels % self.group_size == 0 else -1

        self.norm = kl.GroupNormalization(groups=groups, epsilon=self.eps)
        self.gamma = kl.Dense(in_channels, use_bias=False, kernel_initializer="zeros")
        self.beta = kl.Dense(in_channels, use_bias=False, kernel_initializer="zeros")

    def call(self, x, condition, training=False):
        x = self.norm(x, training=training)
        condition = tf.expand_dims(tf.expand_dims(condition, axis=1), axis=2)  # (b,c)->(b,1,1,c)
        gamma = self.gamma(condition, training=training)
        beta = self.beta(condition, training=training)
        return x * (1 + gamma) + beta


class SelfAttention2D(keras.layers.Layer):
    def __init__(self, head_dim: int = 8, **kwargs) -> None:
        super().__init__(**kwargs)
        self.head_dim = head_dim

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.n_head = max(1, in_channels // self.head_dim)
        assert in_channels % self.n_head == 0, f"入力チャンネル数はhead数で分割できる数（head={self.n_head}）"
        self.norm = kl.LayerNormalization()
        self.qkv_proj = Conv2D1x1(in_channels * 3)
        self.out_proj = Conv2D1x1(in_channels, kernel_initializer="zeros", bias_initializer="zeros")
        self.softmax = kl.Softmax(axis=-1)

    def call(self, x, training=False):
        n, h, w, c = x.shape
        x = self.norm(x, training=training)
        qkv = self.qkv_proj(x)
        qkv = tf.reshape(qkv, (n, h * w, c // self.n_head, self.n_head * 3))
        qkv = tf.transpose(qkv, perm=[0, 2, 1, 3])
        q, k, v = tf.split(qkv, num_or_size_splits=3, axis=-1)
        attn = tf.matmul(q, k, transpose_b=True)  # q@k.T
        attn = attn / tf.math.sqrt(tf.cast(k.shape[-1], tf.float32))
        attn = tf.matmul(self.softmax(attn), v)
        y = tf.transpose(attn, perm=[0, 2, 1, 3])
        y = tf.reshape(y, (n, h, w, c))
        return x + self.out_proj(y)


class Downsample(keras.layers.Layer):
    def build(self, input_shape):
        self.conv = kl.Conv2D(
            filters=input_shape[-1],
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer=keras.initializers.Orthogonal(),
        )

    def call(self, x, training=False):
        return self.conv(x, training=training)


class Upsample(keras.layers.Layer):
    def build(self, input_shape):
        self.conv = kl.Conv2D(
            filters=input_shape[-1],
            kernel_size=3,
            strides=1,
            padding="same",
        )

    def call(self, x, training=False):
        input_shape = tf.shape(x)
        x = tf.image.resize(x, size=(input_shape[1] * 2, input_shape[2] * 2), method="nearest")
        return self.conv(x, training=training)


class ResBlock(keras.Model):
    def __init__(self, channels: int, use_attention: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.channels = channels
        self.use_attention = use_attention

    def build(self, input_shape):
        use_projection = input_shape[-1] != self.channels
        self.proj = Conv2D1x1(self.channels) if use_projection else IdentityLayer()
        self.norm1 = AdaGroupNorm2D()
        self.act1 = kl.Activation("silu")
        self.conv1 = Conv2D3x3(self.channels)
        self.norm2 = AdaGroupNorm2D()
        self.act2 = kl.Activation("silu")
        self.conv2 = Conv2D3x3(self.channels)
        self.attn = SelfAttention2D() if self.use_attention else IdentityLayer()

    def call(self, x, condition, training=False):
        r = self.proj(x, training=training)
        x = self.norm1(x, condition, training=training)
        x = self.act1(x, training=training)
        x = self.conv1(x, training=training)
        x = self.norm2(x, condition, training=training)
        x = self.act2(x, training=training)
        x = self.conv2(x, training=training)
        x = x + r
        x = self.attn(x, training=training)
        return x


class ResBlocks(keras.Model):
    def __init__(self, channels_list: list[int], use_attention: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.resblocks = [ResBlock(c, use_attention) for c in channels_list]

    def call(self, x, condition=None, shortcut=None, training=False):
        outputs = []
        for i, resblock in enumerate(self.resblocks):
            if shortcut is not None:
                x = tf.concat([x, shortcut[i]], axis=-1)
            x = resblock(x, condition, training=training)
            outputs.append(x)
        return x, outputs


class UNet(keras.Model):
    def __init__(self, img_shape: tuple = IMG_SHAPE, category_num: int = CATEGORY_NUM, **kwargs) -> None:
        super().__init__(**kwargs)

        # condition
        embedding_dim = 256
        self.time_embedding1 = FourierFeatures(embedding_dim)
        self.time_embedding2 = kl.Dense(64, activation="gelu")
        self.time_embedding3 = kl.Dense(64)
        self.category_embedding1 = kl.Embedding(category_num, embedding_dim)
        self.category_embedding2 = kl.Dense(64, activation="gelu")
        self.category_embedding3 = kl.Dense(64)

        # downsample
        self.down_block11 = ResBlocks([16, 16], use_attention=False)
        self.down_block12 = ResBlocks([16, 16], use_attention=False)
        self.downsample1 = Downsample()  # 32x32 -> 16x16
        self.down_block21 = ResBlocks([16, 16], use_attention=False)
        self.down_block22 = ResBlocks([16, 16], use_attention=False)
        self.downsample2 = Downsample()  # 16x16 -> 8x8
        self.down_block31 = ResBlocks([16, 16], use_attention=False)
        self.down_block32 = ResBlocks([16, 16], use_attention=False)
        self.downsample3 = Downsample()  # 8x8 -> 4x4
        self.down_block41 = ResBlocks([16, 16], use_attention=False)
        self.down_block42 = ResBlocks([16, 16], use_attention=False)
        self.downsample4 = Downsample()  # 4x4 -> 2x2
        self.down_block51 = ResBlocks([16, 16], use_attention=True)
        self.down_block52 = ResBlocks([16, 16], use_attention=False)
        self.downsample5 = Downsample()  # 2x2 -> 1x1

        # middle
        self.middle_block1 = kl.Flatten()
        self.middle_block2 = kl.Dense(128, activation="relu")
        self.middle_block3 = kl.Reshape((1, 1, 128))

        # upsample
        self.upsample5 = Upsample()
        self.up_block52 = ResBlocks([16, 16], use_attention=False)
        self.up_block51 = ResBlocks([16, 16], use_attention=True)
        self.upsample4 = Upsample()
        self.up_block42 = ResBlocks([16, 16], use_attention=False)
        self.up_block41 = ResBlocks([16, 16], use_attention=False)
        self.upsample3 = Upsample()
        self.up_block32 = ResBlocks([16, 16], use_attention=False)
        self.up_block31 = ResBlocks([16, 16], use_attention=False)
        self.upsample2 = Upsample()
        self.up_block22 = ResBlocks([16, 16], use_attention=False)
        self.up_block21 = ResBlocks([16, 16], use_attention=False)
        self.upsample1 = Upsample()
        self.up_block12 = ResBlocks([16, 16], use_attention=False)
        self.up_block11 = ResBlocks([16, 16], use_attention=False)

        self.out_layer = kl.Conv2D(1, (1, 1), padding="same")

        # build & init weight
        self(
            [
                np.zeros((1,) + img_shape),
                np.zeros((1,)),
                np.zeros((1,)),
            ]
        )

    @tf.function
    def call(self, inputs, training=False):
        # 入力: (ノイズ画像、時間ステップ、カテゴリ)
        x, t, category = inputs

        # 時間埋め込み
        t_emb = self.time_embedding1(t, training=training)
        t_emb = self.time_embedding2(t_emb, training=training)
        t_emb = self.time_embedding3(t_emb, training=training)

        # カテゴリの埋め込み
        c_emb = self.category_embedding1(category, training=training)
        c_emb = self.category_embedding2(c_emb, training=training)
        c_emb = self.category_embedding3(c_emb, training=training)

        # 条件付け
        condition = tf.concat([t_emb, c_emb], axis=-1)

        # --- U-Net
        x, o11 = self.down_block11(x, condition, training=training)
        x, o12 = self.down_block12(x, condition, training=training)
        x = self.downsample1(x, training=training)
        x, o21 = self.down_block21(x, condition, training=training)
        x, o22 = self.down_block22(x, condition, training=training)
        x = self.downsample2(x, training=training)
        x, o31 = self.down_block31(x, condition, training=training)
        x, o32 = self.down_block32(x, condition, training=training)
        x = self.downsample3(x, training=training)
        x, o41 = self.down_block41(x, condition, training=training)
        x, o42 = self.down_block42(x, condition, training=training)
        x = self.downsample4(x, training=training)
        x, o51 = self.down_block51(x, condition, training=training)
        x, o52 = self.down_block52(x, condition, training=training)
        x = self.downsample5(x, training=training)

        x = self.middle_block1(x, training=training)
        x = self.middle_block2(x, training=training)
        x = self.middle_block3(x, training=training)

        x = self.upsample5(x, training=training)
        x, _ = self.up_block52(x, condition, o52[::-1], training=training)
        x, _ = self.up_block51(x, condition, o51[::-1], training=training)
        x = self.upsample4(x, training=training)
        x, _ = self.up_block42(x, condition, o42[::-1], training=training)
        x, _ = self.up_block41(x, condition, o41[::-1], training=training)
        x = self.upsample3(x, training=training)
        x, _ = self.up_block32(x, condition, o32[::-1], training=training)
        x, _ = self.up_block31(x, condition, o31[::-1], training=training)
        x = self.upsample2(x, training=training)
        x, _ = self.up_block22(x, condition, o22[::-1], training=training)
        x, _ = self.up_block21(x, condition, o21[::-1], training=training)
        x = self.upsample1(x, training=training)
        x, _ = self.up_block12(x, condition, o12[::-1], training=training)
        x, _ = self.up_block11(x, condition, o11[::-1], training=training)

        x = self.out_layer(x, training=training)
        return x


if __name__ == "__main__":
    PositionalEmbedding.plot()
    FourierFeatures.plot()
    UNet().summary()
