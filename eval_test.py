import os
import re
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
from PIL import Image
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


# ===================== 配置参数 =====================
class Config:
    # 数据配置 (修改这些路径以匹配您的数据位置)
    DATA_PATH = "twitter_dataset/testset/posts_groundtruth(some_out).txt"  # 修改为您的数据文件路径
    IMAGE_DIR = "twitter_dataset/testset/images"         # 修改为您的图像目录
    LOCAL_BERT_PATH = "local_bert_models/distilbert-base-uncased"  # 本地BERT模型路径
    
    # 模型配置
    TEXT_MAX_LENGTH = 128                               # 文本最大长度
    TEXT_FEAT_DIM = 128                                 # 文本特征维度
    IMG_FEAT_DIM = 256                                  # 图像特征维度
    FUSION_DIM = 124                                    # 图文融合特征维度
    FINAL_DIM = 128                                     # 最终融合特征维度
    DROPOUT_RATE = 0.5                                  # Dropout率
    MAX_IMAGES_PER_POST = 5                             # 每个帖子最多处理的图像数量
    
    # 推理配置
    BATCH_SIZE = 16                                     # 批次大小
    MODEL_PATH = "outputs/models/best_model_epoch28_acc0.9923.pth"  # 训练好的模型路径
    
    # 图像处理配置
    IMAGE_SIZE = (224, 224)                             # 图像尺寸
    IMAGE_MEAN = (0.485, 0.456, 0.406)                  # 归一化均值
    IMAGE_STD = (0.229, 0.224, 0.225)                   # 归一化标准差
    
    # 输出目录
    OUTPUT_DIR = "outputs"                              # 输出根目录
    RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")   # 结果目录
    
    # 支持的图像扩展名
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    
    # 确保目录存在
    @staticmethod
    def setup_directories():
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)

# ===================== NLTK资源处理 =====================
def setup_nltk():
    """设置NLTK资源路径，优先使用本地缓存"""
    try:
        # 尝试设置本地缓存路径
        nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
        if os.path.exists(nltk_data_path):
            nltk.data.path.append(nltk_data_path)
            print(f"使用本地NLTK资源: {nltk_data_path}")
        
        # 检查是否已下载vader_lexicon
        nltk.data.find("sentiment/vader_lexicon.zip")
        print("NLTK资源已存在")
    except LookupError:
        print("下载NLTK资源...")
        try:
            nltk.download('vader_lexicon', quiet=True)
            print("NLTK资源下载成功")
        except Exception as e:
            print(f"无法下载NLTK资源: {e}")
            print("请手动下载vader_lexicon.zip并放置在nltk_data/sentiment目录下")
            print("或者使用以下命令: python -m nltk.downloader vader_lexicon")
            
# 初始化情感分析器
try:
    sia = SentimentIntensityAnalyzer()
    print("情感分析器初始化成功")
except Exception as e:
    print(f"情感分析器初始化失败: {e}")
    sia = None

def analyze_sentiment(text):
    """
    使用NLTK的VADER进行情感分析
    
    返回包含四个情感维度的字典:
    - 'neg': 负面情感强度 (0.0-1.0)
    - 'neu': 中性情感强度 (0.0-1.0)
    - 'pos': 正面情感强度 (0.0-1.0)
    - 'compound': 综合情感分数 (-1.0到1.0)
    """
    if not sia:
        # 情感分析器不可用，返回默认值
        return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    
    try:
        return sia.polarity_scores(text)
    except Exception as e:
        print(f"情感分析出错: {e}")
        return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}

# ===================== 数据处理函数 =====================
def preprocess_text(text):
    """清理文本数据"""
    if not isinstance(text, str):
        return ""
    
    # 移除URL
    text = re.sub(r'http\S+', '', text)
    # 移除用户提及
    text = re.sub(r'@\w+', '', text)
    # 移除特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    # 替换多个空格
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def find_image_file(image_id):
    """查找图像文件（支持多种扩展名）"""
    for ext in Config.IMAGE_EXTENSIONS:
        img_path = os.path.join(Config.IMAGE_DIR, f"{image_id}{ext}")
        if os.path.exists(img_path):
            return img_path
    return None

def load_and_preprocess_data(data):
    """加载和预处理推理数据"""
    if isinstance(data, str) and os.path.exists(data):
        print(f"从文件加载数据: {data}")
        df = pd.read_csv(data, sep='\t')
    elif isinstance(data, pd.DataFrame):
        print("使用提供的DataFrame数据")
        df = data
    else:
        raise ValueError("不支持的输入类型，应为文件路径或DataFrame")
    
    # 验证必要列
    required_columns = ['post_id', 'post_text', 'image_id(s)']
    for col in required_columns:
        if col not in df.columns:
            # 尝试使用可能的变体
            if col == 'image_id(s)' and 'image_id' in df.columns:
                print("检测到'image_id'列而非'image_id(s)'列，将使用'image_id'")
                df['image_id(s)'] = df['image_id']
            else:
                raise ValueError(f"数据中缺少'{col}'列")
    
    print(f"成功加载 {len(df)} 条记录")
    print("预处理数据...")
    
    # 清理文本
    df['cleaned_text'] = df['post_text'].apply(preprocess_text)
    
    # 处理图像ID
    df['image_ids'] = df['image_id(s)'].apply(
        lambda x: [img.strip() for img in str(x).split(',') if img.strip()]
    )
    
    # 添加情感特征
    df['text_data'] = df['cleaned_text'].apply(
        lambda text: {
            'cleaned_text': text,
            'sentiment_neg': analyze_sentiment(text)['neg'],
            'sentiment_neu': analyze_sentiment(text)['neu'],
            'sentiment_pos': analyze_sentiment(text)['pos'],
            'sentiment_compound': analyze_sentiment(text)['compound']
        }
    )
    
    # 如果有标签，处理标签
    if 'label' in df.columns:
        # 检查标签类型
        if df['label'].dtype == 'object':
            # 如果是字符串类型，进行映射
            label_map = {'real': 0, 'fake': 1, 'true': 0, 'false': 1}
            df['label'] = df['label'].str.lower().map(label_map)
        else:
            # 如果是数值类型，直接使用
            pass
        
        df['label'] = df['label'].fillna(0)
        has_labels = True
        
        # 添加真实标签的字符串表示
        df['true_label'] = df['label'].map({0: 'real', 1: 'fake'})
    else:
        has_labels = False
    
    return df, has_labels

# ===================== 数据集类 =====================
class MultimodalRumorDataset(Dataset):
    """多模态谣言检测数据集（支持多图像处理）"""
    
    def __init__(self, df, tokenizer=None):
        self.df = df
        self.tokenizer = tokenizer
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=Config.IMAGE_MEAN,
                std=Config.IMAGE_STD
            )
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 获取预处理结果
        text_data = row['text_data']
        if isinstance(text_data, dict):
            cleaned_text = text_data['cleaned_text']
            sentiment_features = [
                text_data['sentiment_neg'],
                text_data['sentiment_neu'],
                text_data['sentiment_pos'],
                text_data['sentiment_compound']
            ]
        else:
            cleaned_text = text_data
            sentiment_features = [0.0, 1.0, 0.0, 0.0]  # 默认中性情感
        
        # 将情感特征转换为张量
        sentiment_tensor = torch.tensor(sentiment_features, dtype=torch.float)
        
        # Tokenize文本（如果有分词器）
        if self.tokenizer:
            tokenized = self.tokenizer(
                cleaned_text,
                padding='max_length',
                max_length=Config.TEXT_MAX_LENGTH,
                truncation=True,
                return_tensors='pt'
            )
            input_ids = tokenized['input_ids'].squeeze(0)
            attention_mask = tokenized['attention_mask'].squeeze(0)
        else:
            input_ids = torch.zeros(Config.TEXT_MAX_LENGTH, dtype=torch.long)
            attention_mask = torch.zeros(Config.TEXT_MAX_LENGTH, dtype=torch.long)
        
        # 图像处理（处理多张图像）
        image_ids = row['image_ids']
        images = []
        
        # 最多处理MAX_IMAGES_PER_POST张图像
        for i, img_id in enumerate(image_ids[:Config.MAX_IMAGES_PER_POST]):
            img_path = find_image_file(img_id)
            if img_path:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image = self.transform(image)
                    images.append(image)
                except Exception as e:
                    print(f"无法加载图像 {img_path}: {e}")
                    # 创建空图像占位符
                    image = Image.new('RGB', Config.IMAGE_SIZE, (0, 0, 0))
                    image = self.transform(image)
                    images.append(image)
            else:
                # 创建空图像占位符
                image = Image.new('RGB', Config.IMAGE_SIZE, (0, 0, 0))
                image = self.transform(image)
                images.append(image)
        
        # 如果图像数量不足，填充空图像
        while len(images) < Config.MAX_IMAGES_PER_POST:
            image = Image.new('RGB', Config.IMAGE_SIZE, (0, 0, 0))
            image = self.transform(image)
            images.append(image)
        
        # 堆叠图像张量
        images = torch.stack(images)
        
        # 返回数据
        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'images': images,
            'sentiment': sentiment_tensor,  # 添加情感特征
            'post_id': row['post_id'],
            'post_text': row['post_text'],
            'image_id': row['image_id(s)']
        }
        
        # 如果有标签，添加标签
        if 'label' in row:
            item['label'] = torch.tensor(row['label'], dtype=torch.long)
        
        return item

# ===================== 模型定义 =====================
class MultimodalRumorDetector(nn.Module):
    """多模态谣言检测模型（支持多图像处理）"""
    
    def __init__(self):
        super(MultimodalRumorDetector, self).__init__()
        
        # 文本分支 (使用本地BERT模型)
        try:
            self.bert = BertModel.from_pretrained(Config.LOCAL_BERT_PATH)
            print(f"成功加载本地BERT模型: {Config.LOCAL_BERT_PATH}")
        except Exception as e:
            print(f"无法加载本地BERT模型: {e}")
            print("将使用默认的BERT模型")
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        # 冻结BERT参数
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # BiLSTM层 (128维隐藏层 × 双向 = 256维输出)
        self.bilstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 文本特征处理层 (降维到128维)
        self.text_fc = nn.Sequential(
            nn.Linear(256, Config.TEXT_FEAT_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE)
        )
        
        # 图像分支 (ResNet18更轻量)
        self.resnet = models.resnet18(pretrained=True)
        # 移除最后的分类层
        self.resnet.fc = nn.Identity()
        
        # 图像特征处理层 (降维到256维)
        self.img_fc = nn.Sequential(
            nn.Linear(512, Config.IMG_FEAT_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE)
        )
        
        # 图文融合层 (128 + 256 = 384 → 124维)
        self.fusion_fc = nn.Sequential(
            nn.Linear(Config.TEXT_FEAT_DIM + Config.IMG_FEAT_DIM, Config.FUSION_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE)
        )
        
        # 最终特征融合层 (124维图文融合特征 + 4维情感特征 = 128维)
        self.final_fusion = nn.Sequential(
            nn.Linear(Config.FUSION_DIM + 4, Config.FINAL_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE)
        )
        
        # 分类层 (128维输入 → 2类输出)
        self.classifier = nn.Sequential(
            nn.Linear(Config.FINAL_DIM, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, input_ids, attention_mask, images, sentiment):
        """
        前向传播
        
        输入:
        - input_ids: BERT输入token IDs [batch_size, seq_len]
        - attention_mask: BERT注意力掩码 [batch_size, seq_len]
        - images: 输入图像 [batch_size, MAX_IMAGES_PER_POST, 3, 224, 224]
        - sentiment: 情感特征 [batch_size, 4]
        
        输出:
        - 分类概率 [batch_size, 2]
        """
        batch_size = input_ids.size(0)
        
        # === 文本特征提取 ===
        with torch.no_grad():
            bert_outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        bert_embeddings = bert_outputs.last_hidden_state
        
        # 获取序列实际长度
        seq_lengths = attention_mask.sum(dim=1).cpu()
        
        # BiLSTM处理序列
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            bert_embeddings, 
            seq_lengths, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        packed_output, (hidden, _) = self.bilstm(packed_embeddings)
        
        # 获取BiLSTM的最后隐藏状态（前向和后向拼接）
        text_features = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        text_features = self.text_fc(text_features)  # [batch_size, TEXT_FEAT_DIM]
        
        # === 图像特征提取 ===
        # 重塑图像张量: [batch_size * MAX_IMAGES_PER_POST, 3, H, W]
        images = images.view(-1, *images.shape[2:])
        
        # 提取图像特征
        img_features = self.resnet(images)
        img_features = self.img_fc(img_features)  # [batch_size*MAX_IMAGES_PER_POST, IMG_FEAT_DIM]
        
        # 重塑回原始形状: [batch_size, MAX_IMAGES_PER_POST, IMG_FEAT_DIM]
        img_features = img_features.view(batch_size, Config.MAX_IMAGES_PER_POST, -1)
        
        # 对每个帖子的多张图像取平均特征
        img_features = torch.mean(img_features, dim=1)  # [batch_size, IMG_FEAT_DIM]
        
        # === 文本+图像融合 ===
        # 拼接文本和图像特征
        text_img_fused = torch.cat([text_features, img_features], dim=1)  # [batch_size, TEXT_FEAT_DIM + IMG_FEAT_DIM]
        
        # 降维到124维
        fused_features = self.fusion_fc(text_img_fused)  # [batch_size, FUSION_DIM]
        
        # === 情感特征融合 ===
        # 与情感特征拼接 (124 + 4 = 128维)
        final_features = torch.cat([fused_features, sentiment], dim=1)  # [batch_size, FUSION_DIM + 4]
        
        # 最终融合处理
        final_features = self.final_fusion(final_features)  # [batch_size, FINAL_DIM]
        
        # === 分类 ===
        return self.classifier(final_features)

# ===================== 推理引擎 =====================
class InferenceEngine:
    """多模态谣言检测推理引擎"""
    
    def __init__(self, model_path=None):
        """
        参数:
        - model_path: 训练好的模型路径 (可选)
        """
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 设置NLTK资源
        setup_nltk()
        
        # 加载配置
        Config.setup_directories()
        
        # 加载分词器
        self.tokenizer = self.load_tokenizer()
        
        # 加载模型
        self.model = self.load_model(model_path or Config.MODEL_PATH)
    
    def load_tokenizer(self):
        """加载BERT分词器"""
        print("加载BERT分词器...")
        try:
            # 首先尝试加载本地分词器
            tokenizer = BertTokenizer.from_pretrained(Config.LOCAL_BERT_PATH)
            print(f"成功加载本地BERT分词器: {Config.LOCAL_BERT_PATH}")
        except Exception as e:
            # 如果本地分词器加载失败，使用默认分词器
            print(f"无法加载本地BERT分词器: {e}")
            print("使用默认BERT分词器")
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        return tokenizer
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        print(f"加载模型: {model_path}")
        model = MultimodalRumorDetector().to(self.device)
        
        # 加载模型权重
        if os.path.exists(model_path):
            # 打印可用的CUDA设备内存
            if self.device.type == 'cuda':
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"GPU 内存总量: {total_mem:.1f} GB")
            
            # 加载模型
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"模型权重加载成功")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model.eval()  # 设置为评估模式
        return model
    
    def predict(self, data):
        """执行批量预测"""
        # 加载和预处理数据
        df, has_labels = load_and_preprocess_data(data)
        
        # 创建数据集
        dataset = MultimodalRumorDataset(df, self.tokenizer)
        
        # 创建数据加载器
        data_loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=2
        )
        
        # 执行预测
        print("执行预测...")
        all_preds = []
        all_probs = []
        all_labels = [] if has_labels else None
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="预测批次"):
                # 移到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['images'].to(self.device)
                sentiment = batch['sentiment'].to(self.device)  # 添加情感特征
                
                # 如果有标签，收集标签
                if has_labels and 'label' in batch:
                    labels = batch['label'].to(self.device)
                    all_labels.extend(labels.cpu().numpy())
                
                # 前向传播（添加情感特征输入）
                outputs = self.model(input_ids, attention_mask, images, sentiment)
                probs, preds = torch.max(outputs, 1)
                
                # 收集结果
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # 创建结果DataFrame
        results_df = df.copy()
        results_df['prediction'] = all_preds
        results_df['probability'] = all_probs
        results_df['prediction_label'] = results_df['prediction'].map({0: 'real', 1: 'fake'})
        
        # 如果有标签，计算指标
        if has_labels and all_labels is not None:
            # 添加真实标签列（字符串表示）
            results_df['true_label'] = results_df['label'].map({0: 'real', 1: 'fake'})
            
            # 添加预测是否正确列
            results_df['correct'] = (results_df['prediction'] == results_df['label'])
            
            accuracy = accuracy_score(all_labels, all_preds)
            conf_matrix = confusion_matrix(all_labels, all_preds)
            
            # 计算分类报告时检查类别数量
            unique_labels = np.unique(all_labels)
            if len(unique_labels) == 2:
                class_report = classification_report(all_labels, all_preds, target_names=['real', 'fake'])
            else:
                # 如果只有一个类别，创建自定义报告
                if unique_labels[0] == 0:
                    class_report = "所有样本都被预测为真实"
                else:
                    class_report = "所有样本都被预测为虚假"
                
                # 添加基本指标
                class_report += f"\n准确率: {accuracy:.4f}"
            
            print("\n推理结果:")
            print(f"准确率: {accuracy:.4f}")
            print("\n混淆矩阵:")
            print(conf_matrix)
            print("\n分类报告:")
            print(class_report)
            
            # 添加可视化
            self.visualize_results(all_labels, all_preds, results_df)
            
            return results_df, {
                'accuracy': accuracy,
                'confusion_matrix': conf_matrix.tolist(),
                'classification_report': class_report
            }
        else:
            return results_df, None
    
    def predict_single(self, text, image_ids):
        """预测单个样本"""
        # 创建单样本DataFrame
        data = pd.DataFrame({
            'post_id': ['single_sample'],
            'post_text': [text],
            'image_id(s)': [','.join(image_ids)],
            'label': 0  # 添加虚拟标签（整数）
        })
        
        # 执行预测
        results_df, _ = self.predict(data)
        
        # 提取结果
        prediction = results_df['prediction_label'].iloc[0]
        probability = results_df['probability'].iloc[0]
        
        return {
            'text': text,
            'image_ids': image_ids,
            'prediction': prediction,
            'probability': probability,
            'explanation': f"模型预测为'{prediction}'，置信度{probability:.4f}"
        }
    
    def visualize_results(self, true_labels, pred_labels, results_df):
        """可视化推理结果"""
        # 创建输出目录
        inference_dir = os.path.join(Config.RESULTS_DIR, "inference")
        os.makedirs(inference_dir, exist_ok=True)
        
        # 保存结果到CSV
        results_file = os.path.join(inference_dir, "inference_results.csv")
        results_df.to_csv(results_file, index=False)
        print(f"预测结果已保存到: {results_file}")
        
        # 混淆矩阵可视化
        conf_matrix = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['real', 'fake'], 
                    yticklabels=['real', 'fake'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        conf_file = os.path.join(inference_dir, "confusion_matrix.png")
        plt.savefig(conf_file)
        plt.close()
        print(f"混淆矩阵已保存到: {conf_file}")
        
        # 概率分布可视化
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df['probability'], bins=20, kde=True)
        plt.xlabel('Prediction Probability')
        plt.ylabel('Count')
        plt.title('Prediction Probability Distribution')
        prob_file = os.path.join(inference_dir, "probability_distribution.png")
        plt.savefig(prob_file)
        plt.close()
        print(f"概率分布图已保存到: {prob_file}")
        
        # 标签分布可视化
        plt.figure(figsize=(8, 5))
        if 'true_label' in results_df.columns:
            # 真实标签分布
            sns.countplot(x='true_label', data=results_df)
            plt.title('True Label Distribution')
        else:
            # 预测标签分布
            sns.countplot(x='prediction_label', data=results_df)
            plt.title('Predicted Label Distribution')
        label_file = os.path.join(inference_dir, "label_distribution.png")
        plt.savefig(label_file)
        plt.close()
        print(f"标签分布图已保存到: {label_file}")
        
        # 预测正确率可视化
        if 'correct' in results_df.columns:
            plt.figure(figsize=(8, 5))
            sns.countplot(x='correct', data=results_df)
            plt.title('Prediction Correctness')
            plt.xlabel('Correct Prediction')
            plt.ylabel('Count')
            correct_file = os.path.join(inference_dir, "prediction_correctness.png")
            plt.savefig(correct_file)
            plt.close()
            print(f"预测正确率图已保存到: {correct_file}")
        
        # 保存指标到JSON
        metrics = {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'precision': classification_report(true_labels, pred_labels, output_dict=True)['weighted avg']['precision'] if len(np.unique(true_labels)) > 1 else accuracy_score(true_labels, pred_labels),
            'recall': classification_report(true_labels, pred_labels, output_dict=True)['weighted avg']['recall'] if len(np.unique(true_labels)) > 1 else accuracy_score(true_labels, pred_labels),
            'f1_score': classification_report(true_labels, pred_labels, output_dict=True)['weighted avg']['f1-score'] if len(np.unique(true_labels)) > 1 else accuracy_score(true_labels, pred_labels),
        }
        metrics_file = os.path.join(inference_dir, "inference_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"推理指标已保存到: {metrics_file}")

# ===================== 主函数 =====================
def main():
    # 配置参数
    MODEL_PATH = "outputs/models/best_model_epoch28_acc0.9923.pth"  # 修改为您的模型路径
    DATA_PATH = "twitter_dataset/testset/posts_groundtruth(some_out).txt"  # 修改为您的数据路径
    
    # 初始化推理引擎
    engine = InferenceEngine(MODEL_PATH)
    
    # 执行批量预测
    results_df, metrics = engine.predict(DATA_PATH)
    
    # 打印部分结果
    if results_df is not None:
        print("\n前5条预测结果:")
        # 包含真实标签和预测标签
        if 'true_label' in results_df.columns:
            print(results_df[['post_id', 'post_text', 'true_label', 'prediction_label', 'probability']].head())
        else:
            print(results_df[['post_id', 'post_text', 'prediction_label', 'probability']].head())
    
    # 执行单样本预测
    # sample_text = "Breaking news: UFO spotted over New York City!"
    # sample_images = ["ufo_123", "ufo_456"]
    # sample_result = engine.predict_single(sample_text, sample_images)
    # print("\n单样本预测结果:")
    # print(sample_result)

if __name__ == "__main__":
    # 检查并安装缺失的库
    try:
        import torch
    except ImportError:
        print("安装必要的库...")
        import subprocess
        subprocess.run(["pip", "install", "torch", "torchvision", "transformers", 
                        "pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", 
                        "tqdm", "Pillow", "nltk"])
    
    main()