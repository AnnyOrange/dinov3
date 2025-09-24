#!/usr/bin/env python3
"""
CPJUMP1 k-NN评估脚本（简化版）
直接使用config中的rgb_mean和rgb_std
"""

import os
import sys

def run_knn_eval():
    """运行k-NN评估"""
    model_config = "/home/deepcad/xzj/dinov3/outputs/cpjump1_vitl16/config.yaml"
    model_weights = "/home/deepcad/xzj/dinov3/outputs/cpjump1_vitl16/ckpt/39999/teacher_checkpoint.pth"
    output_dir = "/home/deepcad/xzj/dinov3/outputs/cpjump1_vitl16/eval_knn_simple"

    # 数据库配置
    pg_host = "172.16.0.217"
    pg_port = 5432
    pg_db = "cpjump1"
    pg_user = "cpjump1_user"
    pg_password = "cpjump1_secure_pass_2024"

    # 训练和测试数据集配置
    train_dataset = f"CPJump1PostgresNPY:root=/;split=train:pg_host={pg_host}:pg_port={pg_port}:pg_db={pg_db}:pg_user={pg_user}:pg_password={pg_password}"
    test_dataset = f"CPJump1PostgresNPY:root=/;split=val:pg_host={pg_host}:pg_port={pg_port}:pg_db={pg_db}:pg_user={pg_user}:pg_password={pg_password}"

    # 构建命令 - 使用简化的配置
    cmd = [
        "python", "-m", "dinov3.eval.knn",
        f"model.config_file={model_config}",
        f"model.pretrained_weights={model_weights}",
        f"output_dir={output_dir}",
        f"train.dataset={train_dataset}",
        f"eval.test_dataset={test_dataset}",
        "transform.resize_size=256",
        "transform.crop_size=224"
    ]

    print("运行k-NN评估命令（简化版）:")
    print(" ".join(cmd))
    print("\n数据集处理:")
    print("- 只处理前5个通道 (arr[:5])")
    print("- 使用tensor_path字段")
    print("- 返回text_id作为标签")

    # 运行命令
    os.execvp(cmd[0], cmd)

if __name__ == "__main__":
    run_knn_eval()
