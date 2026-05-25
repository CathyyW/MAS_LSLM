## 切分 MATH 数据

如果要从原始 MATH 数据重新生成 `math_split`，可以使用：

```bash
python ./data/scripts/split_math_non_iid.py \
  --input-dir <MATH_DATA_DIR> \
  --output-dir ./data/math_split \
  --input-split train \
  --public-per-subject 50 \
  --train-ratio 0.8 \
  --seed 42
```

也可以直接从 Hugging Face 数据集读取：

```bash
python ./data/scripts/split_math_non_iid.py \
  --hf-dataset qwedsacf/competition_math \
  --output-dir ./data/math_split \
  --input-split train \
  --public-per-subject 50 \
  --train-ratio 0.8 \
  --seed 42
```

默认 client 学科分布为：

```text
client_0: Algebra, Intermediate Algebra, Prealgebra
client_1: Geometry, Precalculus
client_2: Number Theory, Counting & Probability
```
## 单 MAS 实验 

Client0 模型配置
```python
DEFAULT_REPRO_CONFIG = PaperReproductionConfig(
    leader=ModelConfig(
        role="leader",
        name="DeepSeek-R1-Distill-Qwen-1.5B",
        model_id="models/DeepSeek-R1-Distill-Qwen-1.5B",
    ),
    agents=(
        ModelConfig(
            role="agent",
            name="Gemma-2-2B-it",
            model_id="models/gemma-2-2b-it",
        ),
        ModelConfig(
            role="agent",
            name="Sheared-LLaMA-1.3B-ShareGPT",
            model_id="models/Sheared-LLaMA-1.3B-ShareGPT",
        ),
        ModelConfig(
            role="agent",
            name="Qwen3-1.7B",
            model_id="models/Qwen3-1.7B-FP8",
        ),
    ),
)
```

### 快速推理

```bash
python -m single_mas.cli.infer \
  --question "What is 2+2?" \
  --rounds 1 \
  --team-factory single_mas.experiment_configs:build_reproduction_team \
  --output ./single_mas/outputs/infer_test.jsonl
```

```bash
python -m single_mas.cli.infer \
  --input ./data/math_split/clients/client_0/infer.jsonl \
  --output ./single_mas/outputs/infer_client0.jsonl \
  --limit 5 \
  --rounds 3 \
  --team-factory single_mas.experiment_configs:build_reproduction_team
```


### 生成 SFT 数据


单样本冒烟测试：

```bash
python -m single_mas.cli.smoke_sft \
  --input ./data/math_split/clients/client_0/train.jsonl \
  --output ./single_mas/outputs/sft_smoke_client0.jsonl \
  --team-factory single_mas.experiment_configs:build_reproduction_team
```


完整生成：

```bash
python -m single_mas.cli.prepare_sft_data \
  --input ./data/math_split/clients/client_0/train.jsonl \
  --output ./single_mas/outputs/sft_client0.jsonl \
  --limit 10 \
  --team-factory single_mas.experiment_configs:build_reproduction_team
```

SFT 数据构造逻辑位于 `training/datasets.py`：

1. 每个 Agent 先对题目生成回答；
2. 把题目、参考答案和 Agent 回答拼成 Leader prompt；
3. Leader 对同一个 prompt 采样多条 completion；
4. 根据答案匹配结果筛选可用于监督微调的样本。

默认每个 prompt 采样 16 条 Leader completion，可通过 `--completions-per-prompt` 修改。