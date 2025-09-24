# CPJUMP1 De-leak Split: Embedding Generation and Training Interface

本项目目标：针对为避免数据泄露而重新划分的 CPJUMP1 数据集，提供从数据库到嵌入生成、再到训练对接的完整流程。本流程仅使用 8 通道显微镜张量中的前 5 个荧光通道来生成图像嵌入（Embeddings），并将嵌入与文本提示（prompts）关联。

## 新的数据结构（PostgreSQL）
- 数据库：`cpjump1`
- 表：
  - `cpjump1_dinov3_tensors(tensor_path TEXT, text_id INTEGER)`：每行指向一个 8 通道 `.npy` 图像张量及其文本 ID。
  - `prompts(text_id INTEGER PRIMARY KEY, text TEXT)` 和/或 `texts(text_id INTEGER PRIMARY KEY, text TEXT)`：文本内容。
  - 运行嵌入生成脚本将自动创建：`image_embeddings(id SERIAL PRIMARY KEY, embedding_path TEXT, text_id INTEGER)`。

数据库连接信息（示例）：
```
{
  "host": "172.16.0.217",
  "port": "5432",
  "database": "cpjump1",
  "user": "cpjump1_user",
  "password": "cpjump1_secure_pass_2024"
}
```

## 环境依赖
- Python 3.10+
- PyTorch、torchvision、omegaconf
- psycopg2
- numpy

## 步骤一：生成图像嵌入
脚本：`dinov3/tools/cpjump1_generate_embeddings.py`

功能概述：
- 连接 PostgreSQL，读取 `cpjump1_dinov3_tensors(tensor_path, text_id)`。
- 加载每个 8 通道 `.npy`，仅取前 5 个荧光通道。
- 提供两种 8-bit 量化方法（命令行可选）：
  - 方法A：CellCLIP 风格的百分位裁剪 + 缩放（逐通道独立裁剪 / 缩放）。
  - 方法B：全局均值方差归一化（z-score 截断到 ±clamp_z），线性映射到 [0,255]。
- 使用 DINOv2/v3 教师模型（通过配置文件 + 预训练权重）提取多裁剪特征，并按指定方式聚合为最终嵌入。
- 每个样本输出一个 `.npy` 嵌入文件，并将 `(embedding_path, text_id)` 写入数据库表 `image_embeddings`。

示例命令（方法A：CellCLIP-style 量化）：
```bash
python -m dinov3.tools.cpjump1_generate_embeddings \
  --db-host 172.16.0.217 --db-port 5432 --db-name cpjump1 \
  --db-user cpjump1_user --db-password cpjump1_secure_pass_2024 \
  --output-dir /path/to/emb_out \
  --quant-method cellclip --cellclip-low 1.0 --cellclip-high 99.0 \
  --config-file dinov3/configs/train/cpjump1_vitl16.yaml \
  --pretrained-weights /mnt/deepcad_nfs/xuzijing/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --img-size 224 --n-crops 4 --aggregation avg
```

示例命令（方法B：全局归一化量化，需要 5 通道均值/方差）：
```bash
python -m dinov3.tools.cpjump1_generate_embeddings \
  --db-host 172.16.0.217 --db-port 5432 --db-name cpjump1 \
  --db-user cpjump1_user --db-password cpjump1_secure_pass_2024 \
  --output-dir /path/to/emb_out \
  --quant-method global \
  --global-mean 0.1,0.1,0.1,0.1,0.1 \
  --global-std  0.2,0.2,0.2,0.2,0.2 \
  --global-clamp-z 3.0 \
  --config-file dinov3/configs/train/cpjump1_vitl16.yaml \
  --pretrained-weights /mnt/deepcad_nfs/xuzijing/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --img-size 224 --n-crops 4 --aggregation avg
```

关键参数说明：
- `--quant-method {cellclip, global}`：选择量化方法。
- `--global-mean/--global-std`：当选择 `global` 时，必须提供 5 个浮点数（JSON 文件或逗号分隔字符串）。
- `--config-file/--pretrained-weights`：指定 DINOv2/v3 模型配置与权重；内部使用 `build_model_for_eval` 加载教师模型用于推理。
- `--n-crops/--aggregation`：多视图裁剪聚合（`avg` 或 `max`）。
- 输出：每个样本一个 `.npy`（形如 `(D,)`），并插入 `image_embeddings(embedding_path, text_id)`。

为何采用“每样本一个 .npy 文件”：
- 使数据库记录与磁盘文件一一对应，避免 HDF5 内部索引管理的复杂性；更新与定位问题更清晰。

## 步骤二：训练数据接口
模块：`dinov3/tools/data_loader.py`

提供：
- `EmbeddingTextDataset`（PyTorch Dataset）：从数据库读取 `image_embeddings` 与 `texts/prompts`，返回 `(embedding_tensor, text_string)`。

使用示例：
```python
from dinov3.tools.data_loader import DBConfig, get_db_connection, fetch_embeddings_and_texts, EmbeddingTextDataset
from torch.utils.data import DataLoader

db = DBConfig(host="172.16.0.217", port=5432, database="cpjump1", user="cpjump1_user", password="cpjump1_secure_pass_2024")
conn = get_db_connection(db)
records = fetch_embeddings_and_texts(conn)
conn.close()

dataset = EmbeddingTextDataset(records)
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
for emb, text in loader:
    # 将 (embedding, text) 对接你的 main.py 训练流程
    pass
```

性能建议：
- 多线程数据加载：适度增大 `num_workers`（如 4~8），开启 `pin_memory=True`。
- 磁盘 I/O：将 `emb_out` 放在高吞吐存储（本地 NVMe / 并行文件系统）。

## 故障排查
- 数据库无结果：检查 `cpjump1_dinov3_tensors` 是否已有有效的 `(tensor_path, text_id)` 记录，路径可访问。
- `.npy` 通道数异常：脚本会跳过并提示；确保是 8 通道、可加载为 `float32`。
- 预训练权重加载失败：检查 `--config-file` 与 `--pretrained-weights` 路径是否正确，`.pth` 或 DCP 目录均可。

## 参考
- 嵌入生成脚本逻辑参考了原项目中的 `convert_npz_to_avg_emb.py` 设计思路（多裁剪与聚合），并结合当前仓库的 `build_model_for_eval` 推理入口以兼容 DINOv2/v3 教师模型。
