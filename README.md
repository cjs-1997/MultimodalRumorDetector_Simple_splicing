# 多模态谣言检测系统

## 项目简介
本项目是一个基于深度学习的多模态谣言检测系统，结合文本和图像特征进行谣言识别。系统使用ResNet18处理图像特征，DistilBERT处理文本特征，并通过融合层进行联合预测。

## 环境要求
- Python 3.7+
- PyTorch 1.8+
- Transformers 4.0+
- Pandas, NumPy, Matplotlib
- CUDA 11.0+ (如需GPU加速)

安装依赖：
```bash
pip install torch transformers pandas numpy matplotlib seaborn tqdm
```

## 数据准备
1. 数据格式要求：
   - 制表符分隔的文本文件
   - 必须包含列：post_id, post_text, image_id(s), label
   - label取值：real/fake 或 true/false

2. 图像要求：
   - 存放在指定目录下
   - 支持格式：jpg, png, gif, bmp, webp

示例数据结构：
```
post_id    post_text    image_id(s)    label
123    "Breaking news..."    img1.jpg,img2.jpg    fake
```

## 训练流程
1. 修改ResNet18+8.py中的配置参数：
   - DATA_PATH: 训练数据路径
   - IMAGE_DIR: 图像目录
   - LOCAL_BERT_PATH: BERT模型路径
   - 其他训练参数

2. 运行训练脚本：
```bash
python ResNet18+8.py
```

3. 训练输出：
   - 模型保存在outputs/models目录
   - 训练日志保存在outputs/logs
   - 训练曲线保存在outputs/results

## 评估流程
1. 修改eval_test.py中的配置参数：
   - MODEL_PATH: 要加载的模型路径
   - DATA_PATH: 测试数据路径

2. 运行评估脚本：
```bash
python eval_test.py
```

3. 评估输出：
   - 预测结果保存在outputs/results/inference
   - 包含CSV结果文件和可视化图表

## 模型文件说明
- best_model_epoch{epoch}_acc{accuracy}.pth: 各epoch最佳模型
- final_model.pth: 最终训练完成的模型

## 结果解读
1. 评估指标：
   - accuracy: 整体准确率
   - confusion_matrix: 混淆矩阵
   - classification_report: 分类报告

2. 可视化结果：
   - confusion_matrix.png: 混淆矩阵图
   - prediction_correctness.png: 预测正确性分布
   - probability_distribution.png: 预测概率分布

## 示例用法
1. 批量预测：
```python
from eval_test import InferenceEngine
engine = InferenceEngine("models/final_model.pth")
results = engine.predict("test_data.txt")
```

2. 单样本预测：
```python
result = engine.predict_single(
    "Breaking news: UFO spotted!", 
    ["ufo1.jpg", "ufo2.jpg"]
)
print(result)
```

## 注意事项
1. 首次运行会自动下载BERT模型，请确保网络连接
2. 图像处理需要Pillow库支持
3. 推荐使用GPU进行训练
