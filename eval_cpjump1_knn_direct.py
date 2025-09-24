#!/usr/bin/env python3
"""
CPJUMP1 k-NN评估脚本（直接指定配置版）
直接指定rgb_mean和rgb_std，解决配置读取问题
"""

import os
import sys

def run_knn_eval():
    """运行k-NN评估"""
    model_config = "/home/deepcad/xzj/dinov3/outputs/cpjump1_vitl16/config.yaml"
    model_weights = "/home/deepcad/xzj/dinov3/outputs/cpjump1_vitl16/ckpt/39999/teacher_checkpoint.pth"
    output_dir = "/home/deepcad/xzj/dinov3/outputs/cpjump1_vitl16/eval_knn_direct"

    # 数据库配置
    pg_host = "172.16.0.217"
    pg_port = 5432
    pg_db = "cpjump1"
    pg_user = "cpjump1_user"
    pg_password = "cpjump1_secure_pass_2024"

    # 训练和测试数据集配置
    train_dataset = f"CPJump1PostgresNPY:root=/;split=train:pg_host={pg_host}:pg_port={pg_port}:pg_db={pg_db}:pg_user={pg_user}:pg_password={pg_password}"
    test_dataset = f"CPJump1PostgresNPY:root=/;split=val:pg_host={pg_host}:pg_port={pg_port}:pg_db={pg_db}:pg_user={pg_user}:pg_password={pg_password}"

    # 从config文件中提取的rgb_mean和rgb_std
    rgb_mean = [2302.585357163912, 2262.1018075014476, 4284.0141763303445, 2382.809901063049, 1496.6019960251908]
    rgb_std = [2389.2512721729604, 2974.1991759111643, 6384.339457351192, 3640.1488634424036, 3056.291301930611]

    # 构建命令 - 直接指定配置
    cmd = [
        "python", "-m", "dinov3.eval.knn",
        f"model.config_file={model_config}",
        f"model.pretrained_weights={model_weights}",
        f"output_dir={output_dir}",
        f"train.dataset={train_dataset}",
        f"eval.test_dataset={test_dataset}",
        "transform.resize_size=256",
        "transform.crop_size=224",
        f"eval.rgb_mean={rgb_mean}",
        f"eval.rgb_std={rgb_std}"
    ]

    print("运行k-NN评估命令（直接指定配置版）:")
    print(" ".join(cmd))
    print("\n配置信息:")
    print(f"- rgb_mean: {rgb_mean}")
    print(f"- rgb_std: {rgb_std}")
    print("- 训练数据集: train split")
    print("- 测试数据集: val split")
    print("- 模型: ViT-Large patch 16")
    print("- 只处理前5个通道")

    # 运行命令
    os.execvp(cmd[0], cmd)

if __name__ == "__main__":
    run_knn_eval()
