# 多模态谣言检测系统

## 项目简介
本项目是一个基于深度学习的多模态谣言检测系统，结合文本、图像、情感分析特征进行谣言识别。系统使用ResNet18处理图像特征，DistilBERT处理文本特征，并通过融合层进行联合预测。

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
# 数据参数
TEXT_MAX_LENGTH = 128       # 文本截断长度
IMAGE_SIZE = (224, 224)     # 图像统一尺寸
MAX_IMAGES_PER_POST = 5     # 最大图像处理数

# 模型参数
TEXT_FEAT_DIM = 128         # 文本特征维度
IMG_FEAT_DIM = 256          # 图像特征维度
FUSION_DIM = 124            # 图文融合维度
FINAL_DIM = 128             # 最终特征维度
DROPOUT_RATE = 0.5          # 防止过拟合

# 训练参数
BATCH_SIZE = 动态调整        # GPU内存感知
EPOCHS = 50                 # 训练轮次
LEARNING_RATE = 1e-4        # 学习率
TRAIN/VAL/TEST_SPLIT = 0.7/0.15/0.15 # 数据集划分


## 项目目录结构
```
multimodal-rumor-detection/
├── local_bert_models/            # 本地BERT模型文件
│   └── distilbert-base-uncased/  # DistilBERT模型文件
├── outputs/                      # 输出目录
│   ├── logs/                     # 训练日志
│   ├── models/                   # 训练好的模型文件
│   └── results/                  # 评估结果和可视化
│       └── inference/            # 推理结果
├── twitter_dataset/              # 推特数据集
│   ├── devset/                   # 开发集
│   │   ├── posts.txt             # 帖子文本数据
│   │   └── images/               # 对应图片(1000+张)
│   └── testset/                  # 测试集
│       ├── posts_groundtruth.txt # 测试集文本数据
│       └── images/               # 测试集图片
├── ResNet18+8.py                 # 主训练脚本
├── eval_test.py                  # 评估脚本
└── README.md                     # 项目说明文件
```

## 数据准备
1. 数据格式要求：
   - 制表符分隔的文本文件
   - 必须包含列：post_id, post_text, image_id(s), label
   - label取值：real/fake 或 true/false

2. 数据集说明：
   - 开发集(twitter_dataset/devset): 包含posts.txt和images目录
   - 测试集(twitter_dataset/testset): 包含posts_groundtruth.txt和images目录
   - 图片格式支持：jpg, png, gif等常见格式

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
模型保存在outputs/models目录下：
- best_model_epoch{1-49}_acc{0.94-0.99}.pth: 各epoch最佳模型
- final_model.pth: 最终训练完成的模型
- 模型命名格式：best_model_epoch{epoch}_acc{accuracy}.pth

## BERT模型说明
本地BERT模型存放在local_bert_models/distilbert-base-uncased/目录下，包含：
- config.json: 模型配置文件
- pytorch_model.bin: PyTorch模型权重
- tokenizer相关文件

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
