# 🚀 快速开始指南 - AI Interview Coach 训练模块

## 📌 项目状态

你现在拥有一个完整的**迁移学习 + 领域适应**训练系统！

---

## 🎯 接下来要做什么

### 立即行动 (Today)

#### 1️⃣ 下载 ASAP 数据集 (10分钟)

```bash
# 访问 Kaggle 下载数据
open https://www.kaggle.com/c/asap-aes/data

# 下载后运行准备脚本
cd ai_interview_coach/scripts
bash prepare_asap_data.sh
```

**需要下载**: `training_set_rel3.tsv`  
**保存到**: `ai_interview_coach/data/training_data/`

#### 2️⃣ 生成面试答案数据 (5分钟)

```bash
cd ai_interview_coach/scripts
python generate_interview_data.py
```

**输出**: 100个待标注的Q&A对  
**位置**: `data/training_data/interview_answers_to_annotate.json`

---

### 本周任务 (This Week)

#### 3️⃣ 人工标注数据 (2-3小时) ⭐ **关键步骤**

**建议分工**:
- 团队成员A: 标注 Q01-Q10 (50条)
- 团队成员B: 标注 Q11-Q20 (50条)

**打开文件**:
```bash
open ai_interview_coach/data/training_data/interview_answers_to_annotate.json
```

**标注要点**:
- 每个答案打4个维度的分数 (1-5)
- 参考答案中的 `[QUALITY_LEVEL]` 标签
- 可以编辑答案文本让其更真实
- 保存为: `interview_answers_annotated.json`

**标注指南已生成**:
```bash
cat ai_interview_coach/data/training_data/annotation_template.txt
```

#### 4️⃣ 训练模型 - 阶段1 (2-3小时 GPU时间)

```bash
cd ai_interview_coach/models

# 安装依赖
pip install torch transformers pandas matplotlib seaborn tqdm

# 开始训练
python train.py \
  --stage 1 \
  --asap_path ../data/training_data/asap_essays.csv \
  --batch_size 16 \
  --epochs 10 \
  --device cuda \
  --save_dir ../checkpoints/stage1
```

**预期结果**: MAE ≈ 0.5, Accuracy ≈ 85%

#### 5️⃣ 训练模型 - 阶段2 (30-60分钟)

```bash
python train.py \
  --stage 2 \
  --interview_path ../data/training_data/interview_answers_annotated.json \
  --load_checkpoint ../checkpoints/stage1/best_model.pt \
  --batch_size 8 \
  --epochs 20 \
  --learning_rate 5e-6 \
  --freeze_bert \
  --save_dir ../checkpoints/stage2
```

**预期结果**: MAE ≈ 0.4, Accuracy ≈ 92%

#### 6️⃣ 评估模型

```bash
python evaluate.py \
  --checkpoint ../checkpoints/stage2/best_model.pt \
  --data_path ../data/training_data/interview_answers_annotated.json \
  --data_type interview \
  --output_dir ../evaluation_results
```

**生成**: 
- 性能指标 (JSON)
- 可视化图表 (PNG)
- 预测样例

---

## 📊 报告需要的内容

### 实验结果部分

运行以下实验并记录结果:

1. **Baseline**: 启发式方法 (现有系统)
2. **Experiment 1**: ASAP预训练 (Stage 1 only)
3. **Experiment 2**: 迁移学习 (Stage 1 + 2) ✓ **最佳**
4. **Ablation**: 无迁移学习 (仅Interview数据)

### 需要报告的指标

- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- Accuracy (±1 point)
- Correlation
- 训练时间
- Per-dimension scores

### 可视化图表

自动生成在 `evaluation_results/`:
- Predicted vs Actual scatter plot
- Error distribution
- Cumulative error curve

---

## 🗂️ 文件结构概览

```
ai_interview_coach/
├── models/                    # ✅ 已创建
│   ├── answer_scorer.py       # BERT评分模型
│   ├── data_loader.py         # 数据加载器
│   ├── train.py               # 训练脚本
│   └── evaluate.py            # 评估脚本
├── scripts/                   # ✅ 已创建
│   ├── generate_interview_data.py   # 生成标注数据
│   └── prepare_asap_data.sh         # ASAP数据准备
├── data/training_data/        # ⏳ 待填充
│   ├── asap_essays.csv               # 从Kaggle下载
│   ├── interview_answers_to_annotate.json    # 生成
│   └── interview_answers_annotated.json      # 人工标注 ⭐
├── checkpoints/               # 🔜 训练后生成
│   ├── stage1/best_model.pt
│   └── stage2/best_model.pt
├── evaluation_results/        # 🔜 评估后生成
│   ├── evaluation_results.json
│   └── evaluation_plots.png
└── TRAINING_README.md         # ✅ 详细文档
```

---

## ⚠️ 常见问题

### Q: 没有GPU怎么办?
**A**: 
- 使用 `--device cpu` (训练会慢很多)
- 或使用 Google Colab 免费GPU
- 或减少epochs: `--epochs 5`

### Q: 标注数据太多了?
**A**: 最少标注50条即可 (10个问题 × 5个答案)

### Q: 如何知道模型训练好了?
**A**: 查看 validation MAE:
- MAE < 0.5 = 很好 ✅
- MAE < 0.7 = 可接受 ⚠️
- MAE > 1.0 = 需要调整 ❌

### Q: 训练中断了怎么办?
**A**: 模型会自动保存checkpoints，使用 `--load_checkpoint` 继续

---

## 📧 需要帮助?

查看详细文档:
```bash
open ai_interview_coach/TRAINING_README.md
```

或检查训练日志:
```bash
cat ai_interview_coach/checkpoints/stage1/training_history.json
```

---

## ✅ 检查清单

**数据准备**:
- [ ] 下载 ASAP 数据集
- [ ] 运行 `prepare_asap_data.sh`
- [ ] 运行 `generate_interview_data.py`
- [ ] 完成人工标注 (重要! ⭐)

**模型训练**:
- [ ] 安装训练依赖包
- [ ] Stage 1: ASAP预训练
- [ ] Stage 2: 面试数据微调
- [ ] 保存最佳模型

**实验评估**:
- [ ] 运行 evaluate.py
- [ ] 记录所有指标
- [ ] 保存可视化图表
- [ ] 对比baseline

**报告撰写**:
- [ ] 方法部分 (描述迁移学习方法)
- [ ] 数据部分 (ASAP + 标注数据)
- [ ] 实验部分 (指标和图表)
- [ ] 讨论部分 (优势和局限性)

---

## 🎓 引用

记得在报告中引用 ASAP 数据集:

```
ASAP Automated Essay Scoring Competition (2012). 
Kaggle. https://www.kaggle.com/c/asap-aes
```

---

## 🚀 开始吧!

**第一步**: 下载 ASAP 数据集
```bash
cd ai_interview_coach/scripts
bash prepare_asap_data.sh
```

**第二步**: 生成标注数据
```bash
python generate_interview_data.py
```

**第三步**: 标注数据 (团队协作) 💪

祝你训练顺利! 🎉

