# QWEN2.5-7B GRPO 经济智能体训练 - V5 (350步 checkpoint)

## 目录结构

```
├── RL/
│   ├── reward.py           # 奖励函数（V5版本：双层Barrier + 微观宏观融合）
│   ├── prepare_verl_data.py # 数据预处理脚本
│   └── config/
│       └── econ_grpo_small.yaml  # 训练配置
├── checkpoints_v5/
│   ├── global_step_350/    # 350步 checkpoint
│   │   ├── actor/
│   │   │   ├── lora_adapter/     # LoRA 权重（78MB）
│   │   │   ├── huggingface/      # tokenizer 等配置
│   │   │   ├── extra_state_*.pt  # 训练状态
│   │   │   └── fsdp_config.json
│   │   └── data.pt
│   └── latest_checkpointed_iteration.txt
└── data/
    └── verl_dataset_small/  # 训练/验证数据
        ├── train.parquet
        └── val.parquet
```

## 恢复训练到 700 步

### 1. 准备基础模型
```bash
# 确保有 Qwen2.5-7B-Instruct 基础模型
# 路径: /workspace/models/Qwen2.5-7B-Instruct
```

### 2. 修改配置文件
编辑 `RL/config/econ_grpo_small.yaml`：

```yaml
# 修改路径为你的实际路径
data:
  train_files: /your/path/data/verl_dataset_small/train.parquet
  val_files: /your/path/data/verl_dataset_small/val.parquet

actor_rollout_ref:
  model:
    path: /your/path/models/Qwen2.5-7B-Instruct

custom_reward_function:
  path: /your/path/RL/reward.py

trainer:
  resume_mode: auto  # 改为 auto 以从 checkpoint 恢复
  default_local_dir: /your/path/checkpoints_v5
```

### 3. 从 350 步恢复训练到 700 步
```bash
cd /your/path/RL

python -m verl.trainer.main_ppo \
    --config-path config \
    --config-name econ_grpo_small \
    trainer.total_training_steps=700 \
    trainer.save_freq=350 \
    trainer.test_freq=700 \
    trainer.resume_mode=auto \
    'trainer.default_local_dir=/your/path/checkpoints_v5'
```

## 训练配置摘要

| 参数 | 值 |
|------|-----|
| 基础模型 | Qwen2.5-7B-Instruct |
| 微调方法 | LoRA (rank=8, alpha=16) |
| 总步数 | 700 |
| Checkpoint 保存 | 350, 700 步 |
| 验证 | 700 步 |
| Batch size | 2 |
| Learning rate | 1e-5 |

## Reward 函数特点 (V5)

1. **双层 Barrier**：
   - Layer 1: 跟随 work_target_mix
   - Layer 2: 连续过劳惩罚 (>0.86)

2. **微观宏观融合**：
   - buffer_ratio → work_target
   - regime → work_target_r
   - 动态 alpha 融合

3. **消费目标混合**：
   - micro: buffer_ratio 驱动
   - macro: regime 驱动
   - beta 系数混合

## ⚠️ 重要：恢复训练前的准备

**问题：** 完整模型权重 `model_world_size_1_rank_0.pt` (15GB) 无法上传 GitHub。

**解决方案：** 在新 instance 上，需要先生成这个文件：

```bash
# 1. 克隆此仓库
git clone https://github.com/uqcxu9/qwen2.5_300step.git
cd qwen2.5_300step

# 2. 生成完整模型权重（从基础模型 + LoRA 合并）
python3 << 'EOF'
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "/workspace/models/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)

# 加载 LoRA adapter
model = PeftModel.from_pretrained(
    base_model, 
    "./checkpoints_v5/global_step_350/actor/lora_adapter"
)

# 保存完整状态字典（verl 格式）
state_dict = model.state_dict()
torch.save(
    {"module": state_dict}, 
    "./checkpoints_v5/global_step_350/actor/model_world_size_1_rank_0.pt"
)
print("✅ 模型权重已生成！")
EOF

# 3. 然后恢复训练
```

## 已上传的文件清单

| 文件 | 大小 | 状态 |
|------|------|------|
| `RL/reward.py` | 12KB | ✅ |
| `RL/config/econ_grpo_small.yaml` | 4KB | ✅ |
| `RL/prepare_verl_data.py` | 20KB | ✅ |
| `checkpoints_v5/.../lora_adapter/` | 78MB | ✅ |
| `checkpoints_v5/.../huggingface/` | 16MB | ✅ |
| `checkpoints_v5/.../optim_*.pt` | 155MB | ✅ |
| `checkpoints_v5/.../extra_state_*.pt` | 15KB | ✅ |
| `checkpoints_v5/.../data.pt` | 1.5KB | ✅ |
| `data/verl_dataset_small/` | 10MB | ✅ |
| `model_world_size_1_rank_0.pt` | 15GB | ❌ 需本地生成 |

## 注意事项

- 需要 Qwen2.5-7B-Instruct 基础模型 (路径: `/workspace/models/Qwen2.5-7B-Instruct`)
- 恢复训练前必须先生成 `model_world_size_1_rank_0.pt`
- 恢复训练时使用 `trainer.resume_mode=auto`

