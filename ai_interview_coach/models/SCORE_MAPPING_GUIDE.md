# 分数映射指南

## 📊 分数区间设计

### 训练时: **0-5分区间**

**原因**:
1. **更好地利用ASAP数据**: 
   - Set 2 (1-6) → 映射为 0-5
   - Set 3-4 (0-3) → 映射为 0-5
   - Set 5-6 (0-4) → 映射为 0-5

2. **包含极值**: 0分可以表示"完全错误"的答案

3. **训练稳定**: 0-5范围比1-5更宽，梯度更稳定

### 预测时: **可选择输出范围**

模型支持三种输出格式：

#### 1. 0-5分 (默认，训练范围)
```python
result = model.predict(question, answer, output_range='0-5')
# 输出: 0.0 - 5.0
```

#### 2. 1-6分 (简单+1)
```python
result = model.predict(question, answer, output_range='1-6')
# 输出: 1.0 - 6.0 (0-5 + 1)
```

#### 3. 1-5分 (缩放)
```python
result = model.predict(question, answer, output_range='1-5')
# 输出: 1.0 - 5.0 (0-5 * 4/5 + 1)
```

## 🔄 分数转换公式

### ASAP数据归一化 (训练时)

| Essay Set | 原始范围 | 归一化公式 | 结果范围 |
|-----------|---------|-----------|---------|
| Set 2 | 1-6 | `(score - 1) * 5/5` | 0-5 |
| Set 3-4 | 0-3 | `score * 5/3` | 0-5 |
| Set 5-6 | 0-4 | `score * 5/4` | 0-5 |

### 模型输出转换 (预测时)

| 目标范围 | 转换公式 | 示例 |
|---------|---------|------|
| 0-5 | `score` (不变) | 2.5 → 2.5 |
| 1-6 | `score + 1` | 2.5 → 3.5 |
| 1-5 | `score * 4/5 + 1` | 2.5 → 3.0 |

## 📝 使用建议

### 对于面试答案评分

**推荐使用 1-5分区间**:
- 更符合常见的评分习惯
- 1分 = 很差，5分 = 优秀
- 与现有启发式评分系统兼容

```python
# 在 model_eval.py 中
result = model.predict(question, answer, output_range='1-5')
```

### 对于实验报告

**可以使用 0-5分区间**:
- 更精确的分数表示
- 可以区分"极差"(0分)和"很差"(1分)
- 在报告中说明分数范围即可

## ⚠️ 注意事项

1. **标注数据**: 如果手动标注的面试数据使用1-5分，代码会自动转换到0-5进行训练

2. **评估指标**: MAE/RMSE等指标会基于训练时的0-5范围计算

3. **最终输出**: 根据应用场景选择合适的输出范围

## 🔍 验证分数映射

运行以下代码验证映射是否正确：

```python
from models.answer_scorer import BERTAnswerScorer

model = BERTAnswerScorer()

# 测试不同输出范围
question = "What is backpropagation?"
answer = "Backpropagation is a method for training neural networks."

result_05 = model.predict(question, answer, output_range='0-5')
result_16 = model.predict(question, answer, output_range='1-6')
result_15 = model.predict(question, answer, output_range='1-5')

print(f"0-5: {result_05['overall_score']}")
print(f"1-6: {result_16['overall_score']}")
print(f"1-5: {result_15['overall_score']}")
```

## 📊 分数分布示例

假设模型预测原始分数为 2.5 (0-5范围):

| 输出格式 | 分数值 | 含义 |
|---------|-------|------|
| 0-5 | 2.5 | 中等偏下 |
| 1-6 | 3.5 | 中等偏下 |
| 1-5 | 3.0 | 中等 |

## 🎯 总结

- **训练**: 统一使用 0-5 分区间
- **预测**: 根据需求选择输出范围
- **推荐**: 面试评分使用 1-5 分区间
- **优势**: 更好地利用ASAP数据，训练更稳定

