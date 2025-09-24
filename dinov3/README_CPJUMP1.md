CPJUMP1 DINOv3 5-Channel 预训练说明

数据
- WebDataset 根目录: `/mnt/deepcad_nfs/CPJUMP1_dataset_dinov3/`
  - 训练: `train/*.tar` (236)
  - 验证: `val/*.tar` (36)
  - 测试: `test/*.tar` (67)
- PostgreSQL 元数据: `cpjump1` 数据库, 表 `cpjump1_dinov3_tensors` (host: 172.16.0.217, port: 5432, user: cpjump1_user)

启动训练
```bash
python -m dinov3.train.train --config-file dinov3/configs/train/cpjump1_vitl16.yaml --output-dir ./outputs/cpjump1_vitl16
python -m dinov3.train.train --config-file dinov3/configs/train/cpjump1_vit7b.yaml  --output-dir ./outputs/cpjump1_vit7b
```

PostgreSQL+NPY 调试
- 将配置中的 dataset_path 设置为: `CPJump1PostgresNPY:root=/;split=train`

关键选项
- student.in_chans: 5
- student.patch_embed_strategy: [inflate|channelvit|dichavit]
- crops.microscopy: true  # 显微增强
- dataset_path: CPJump1WebDataset 或 CPJump1PostgresNPY

备注
- inflate 策略自动将 3->5 通道权重适配
- channelvit/dichavit 使用 per-channel conv 并以均值初始化

