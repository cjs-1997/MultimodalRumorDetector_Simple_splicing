import os
import re
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import BertModel, BertTokenizer
from PIL import Image
import torch.cuda as cuda

# ===================== 配置参数 =====================
class Config:
    # 数据配置 (修改这些路径以匹配您的数据位置)
    DATA_PATH = "twitter_dataset/devset/posts(some_out).txt"      # 修改为您的数据文件路径
    IMAGE_DIR = "twitter_dataset/devset/images"         # 修改为您的图像目录
    LOCAL_BERT_PATH = "local_bert_models/distilbert-base-uncased"  # 本地BERT模型路径
    
    # 模型配置
    TEXT_MAX_LENGTH = 128                               # 文本最大长度
    TEXT_FEAT_DIM = 256                                 # 文本特征维度 (128 * 2)
    IMG_FEAT_DIM = 512                                  # 图像特征维度
    FUSION_DIM = 128                                    # 融合特征维度
    DROPOUT_RATE = 0.5                                  # Dropout率
    MAX_IMAGES_PER_POST = 5                             # 每个帖子最多处理的图像数量
    
    # 训练配置
    BATCH_SIZE = 8 if cuda.is_available() and cuda.get_device_properties(0).total_memory < 8e9 else 16
    EPOCHS = 30                                         # 训练轮数
    LEARNING_RATE = 1e-4                                # 学习率
    
    # 数据集划分
    TRAIN_SPLIT = 0.7                                   # 训练集比例
    VAL_SPLIT = 0.15                                    # 验证集比例
    TEST_SPLIT = 0.15                                   # 测试集比例
    
    # 图像处理配置
    IMAGE_SIZE = (224, 224)                             # 图像尺寸
    IMAGE_MEAN = (0.485, 0.456, 0.406)                  # 归一化均值
    IMAGE_STD = (0.229, 0.224, 0.225)                   # 归一化标准差
    
    # 输出目录
    OUTPUT_DIR = "outputs"                              # 输出根目录
    MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "models")  # 模型保存目录
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")           # 日志目录
    RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")     # 结果目录
    
    # 支持的图像扩展名
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    
    # 确保目录存在
    @staticmethod
    def setup_directories():
        os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)

# ===================== 数据处理函数 =====================
def find_image_file(image_id):
    """查找图像文件（支持多种扩展名）"""
    for ext in Config.IMAGE_EXTENSIONS:
        img_path = os.path.join(Config.IMAGE_DIR, f"{image_id}{ext}")
        if os.path.exists(img_path):
            return img_path
    return None

def load_and_process_data():
    """加载和预处理数据"""
    print(f"加载数据文件: {Config.DATA_PATH}")
    
    # 加载数据 (假设是制表符分隔的文本文件)
    df = pd.read_csv(Config.DATA_PATH, sep='\t')
    
    # 验证必要列
    required_columns = ['post_id', 'post_text', 'image_id(s)', 'label']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"数据文件中缺少'{col}'列")
    
    print(f"成功加载 {len(df)} 条记录")
    print("预处理数据...")
    
    # 清理文本
    df['cleaned_text'] = df['post_text'].apply(preprocess_text)
    
    # 处理图像ID
    df['image_ids'] = df['image_id(s)'].apply(
        lambda x: [img.strip() for img in str(x).split(',') if img.strip()]
    )
    
    # 映射标签
    label_map = {'real': 0, 'fake': 1, 'true': 0, 'false': 1}
    df['label'] = df['label'].str.lower().map(label_map)
    
    # 处理缺失值
    df['label'] = df['label'].fillna(0)
    
    # 划分数据集 (确保同一个帖子的所有数据在同一集合)
    print("划分数据集...")
    unique_posts = df['post_id'].unique()
    test_val_size = Config.VAL_SPLIT + Config.TEST_SPLIT
    train_ids, test_val_ids = train_test_split(
        unique_posts, 
        test_size=test_val_size,
        random_state=42
    )
    
    test_size = Config.TEST_SPLIT / test_val_size
    val_ids, test_ids = train_test_split(
        test_val_ids,
        test_size=test_size,
        random_state=42
    )
    
    # 划分数据集
    train_df = df[df['post_id'].isin(train_ids)]
    val_df = df[df['post_id'].isin(val_ids)]
    test_df = df[df['post_id'].isin(test_ids)]
    
    print(f"训练集: {len(train_df)} 条 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"验证集: {len(val_df)} 条 ({len(val_df)/len(df)*100:.1f}%)")
    print(f"测试集: {len(test_df)} 条 ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df

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
        
        # 文本处理
        text = row['cleaned_text']
        
        # Tokenize文本（如果有分词器）
        if self.tokenizer:
            tokenized = self.tokenizer(
                text,
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
        
        # 标签
        label = row['label']
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'images': images,  # 形状为 [MAX_IMAGES_PER_POST, 3, H, W]
            'label': torch.tensor(label, dtype=torch.long)
        }

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
        
        # 文本特征处理层
        self.text_fc = nn.Sequential(
            nn.Linear(256, Config.FUSION_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE)
        )
        
        # 图像分支 (ResNet18更轻量)
        self.resnet = models.resnet18(pretrained=True)
        # 移除最后的分类层
        self.resnet.fc = nn.Identity()
        
        # 图像特征处理层
        self.img_fc = nn.Sequential(
            nn.Linear(512, Config.IMG_FEAT_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE)
        )
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 多模态融合
        self.fusion = nn.Sequential(
            nn.Linear(Config.FUSION_DIM + Config.IMG_FEAT_DIM, Config.FUSION_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE)
        )
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(Config.FUSION_DIM, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, input_ids, attention_mask, images):
        """
        前向传播
        
        输入:
        - input_ids: BERT输入token IDs [batch_size, seq_len]
        - attention_mask: BERT注意力掩码 [batch_size, seq_len]
        - images: 输入图像 [batch_size, MAX_IMAGES_PER_POST, 3, 224, 224]
        
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
        text_features = self.text_fc(text_features)
        
        # === 图像特征提取 ===
        # 重塑图像张量: [batch_size * MAX_IMAGES_PER_POST, 3, H, W]
        images = images.view(-1, *images.shape[2:])
        
        # 提取图像特征
        img_features = self.resnet(images)
        img_features = self.img_fc(img_features)
        
        # 重塑回原始形状: [batch_size, MAX_IMAGES_PER_POST, IMG_FEAT_DIM]
        img_features = img_features.view(batch_size, Config.MAX_IMAGES_PER_POST, -1)
        
        # 对每个帖子的多张图像取平均特征
        img_features = torch.mean(img_features, dim=1)
        
        # === 多模态融合 ===
        fused = torch.cat([text_features, img_features], dim=1)
        fused = self.fusion(fused)
        
        # === 分类 ===
        return self.classifier(fused)

# ===================== 训练函数 =====================
def train_model(model, train_loader, val_loader, device):
    """训练模型"""
    # 准备训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # 创建日志文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(Config.LOG_DIR, f"train_{timestamp}.log")
    
    # 训练记录
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    
    print(f"开始训练 {Config.EPOCHS} 个epochs")
    print(f"使用设备: {device}")
    print(f"批次大小: {Config.BATCH_SIZE}")
    print(f"学习率: {Config.LEARNING_RATE}")
    
    # 清除GPU缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    for epoch in range(1, Config.EPOCHS + 1):
        epoch_start = time.time()
        
        # === 训练阶段 ===
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"训练 Epoch {epoch}/{Config.EPOCHS}"):
            # 移到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['images'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 清除中间变量以节省内存
            del outputs, loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # === 验证阶段 ===
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        
        # 更新历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # 更新最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(Config.MODEL_SAVE_DIR, f"best_model_epoch{epoch}_acc{val_acc:.4f}.pth")
            torch.save(model.state_dict(), model_path)
            save_msg = f" | 保存最佳模型: {model_path}"
        else:
            save_msg = ""
        
        # 计算时间并打印日志
        epoch_time = time.time() - epoch_start
        log_msg = (f"Epoch {epoch}/{Config.EPOCHS} | "
                   f"time: {epoch_time:.1f}s | "
                   f"Train_Loss: {train_loss:.4f} | Train_Acc: {train_acc:.4f} | "
                   f"Val_Loss: {val_loss:.4f} | Val_Acc: {val_acc:.4f}{save_msg}")
        print(log_msg)
        
        # 保存日志
        with open(log_file, "a") as f:
            f.write(log_msg + "\n")
    
    # 保存最终模型
    final_model_path = os.path.join(Config.MODEL_SAVE_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"训练完成! 最终模型已保存到: {final_model_path}")
    
    return history

def validate_model(model, data_loader, criterion, device):
    """验证模型"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="验证"):
            # 移到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['images'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)
            
            # 统计
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 清除中间变量以节省内存
            del outputs, loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    avg_loss = val_loss / len(data_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

# ===================== 评估函数 =====================
def evaluate_model(model, test_loader, device):
    """评估模型"""
    print("在测试集上评估模型...")
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    # 创建结果文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(Config.RESULTS_DIR, f"results_{timestamp}.csv")
    metrics_file = os.path.join(Config.RESULTS_DIR, "evaluation_metrics.txt")
    confusion_file = os.path.join(Config.RESULTS_DIR, "confusion_matrix.png")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="测试"):
            # 移到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['images'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask, images)
            probs, preds = torch.max(outputs, 1)
            
            # 收集结果
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # 清除中间变量以节省内存
            del outputs
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=['真实', '虚假'])
    
    # 保存结果
    results_df = pd.DataFrame({
        'label': all_labels,
        'prediction': all_preds,
        'probability': all_probs
    })
    results_df.to_csv(results_file, index=False)
    
    # 保存评估指标
    with open(metrics_file, 'w') as f:
        f.write("multimodal rumor detection results\n")
        f.write("==============================\n\n")
        f.write(f"dataset size: {len(all_labels)}\n")
        f.write(f"accuracy: {accuracy:.4f}\n\n")
        f.write("confusion matrix:\n")
        f.write(np.array2string(conf_matrix))
        f.write("\n\nclassification report:\n")
        f.write(class_report)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['real', 'fake'], 
                yticklabels=['real', 'fake'])
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.title('confusion matrix')
    plt.savefig(confusion_file)
    plt.close()
    
    print(f"评估结果已保存到 {Config.RESULTS_DIR}")
    
    # 返回准确率
    return accuracy

# ===================== 辅助函数 =====================
def plot_training_history(history):
    """绘制训练历史图表"""
    plt.figure(figsize=(12, 10))
    
    # 损失图
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='training loss')
    plt.plot(history['val_loss'], label='validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率图
    plt.subplot(2, 1, 2)
    plt.plot(history['train_acc'], label='Train_Acc')
    plt.plot(history['val_acc'], label='Val_Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 保存图表
    plot_path = os.path.join(Config.RESULTS_DIR, "training_history.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"训练历史图表已保存到: {plot_path}")

# ===================== 主函数 =====================
def main():
    # 设置输出目录
    Config.setup_directories()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载和预处理数据
    train_df, val_df, test_df = load_and_process_data()
    
    # 加载BERT分词器
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
    
    # 创建数据集
    print("创建数据集...")
    train_dataset = MultimodalRumorDataset(train_df, tokenizer)
    val_dataset = MultimodalRumorDataset(val_df, tokenizer)
    test_dataset = MultimodalRumorDataset(test_df, tokenizer)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        num_workers=2
    )
    
    # 初始化模型
    print("初始化模型...")
    model = MultimodalRumorDetector().to(device)
    print("模型架构:")
    print(model)
    
    # 训练模型
    print("开始训练模型...")
    history = train_model(model, train_loader, val_loader, device)
    
    # 可视化训练历史
    plot_training_history(history)
    
    # 评估模型
    print("评估模型...")
    accuracy = evaluate_model(model, test_loader, device)
    
    print("\n===== 处理完成! =====")
    print(f"最终测试准确率: {accuracy:.4f}")

if __name__ == "__main__":
    # 检查并安装缺失的库
    try:
        import torch
    except ImportError:
        print("安装必要的库...")
        import subprocess
        subprocess.run(["pip", "install", "torch", "torchvision", "transformers", 
                        "pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", 
                        "tqdm", "Pillow"])
    
    main()