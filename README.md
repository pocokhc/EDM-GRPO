# EDM-GRPO

EDMで学習したMNISTをGRPOで教師ありファインチューニングするコードです。
Qiita用のコードとなります。

[記事]()


# 論文
EDM（Elucidated Diffusion Model）: https://arxiv.org/abs/2206.00364
GRPO（Group Relative Policy Optimization）：https://arxiv.org/abs/2402.03300


# 使用方法

```bash
# 0. requirement
> pip install tensorflow matplotlib opencv-python tqdm

# 1. 学習済みモデルをresultにコピー
> cp weights/edm.weights.h5 result
> cp weights/policy.weights.h5 result

# 2. EDMの画像生成
> python edm.py

# 3. GRPOの画像生成
> python grpo.py
```


## 学習

`result` 配下に結果が出力されます（同名ファイルは上書き）

```bash
# 1. EDM training
> python edm.py --train

# 2. RewardModel training
> python reward_model.py

# 3. GRPO fine-tuning
> python grpo.py --train
```

# Development Version

Python : 3.13.2
Tensorflow : 2.18.0

