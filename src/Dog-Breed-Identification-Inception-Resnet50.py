import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score, precision_recall_curve, average_precision_score

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# PyTorch
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
print(f'Train_data shape：{df.shape}')
df.head()

dog_breeds = sorted(df['breed'].unique())
print(f'Number of unique breed：{len(dog_breeds)}\n')
print(dog_breeds)
breed_classes = df.breed.value_counts().reset_index()
breed_classes['count'].describe()

def barw(ax): 
    for p in ax.patches:
        val = p.get_width() #height of the bar
        x = p.get_x()+ p.get_width() # x-position 
        y = p.get_y() + p.get_height()/2 #y-position
        ax.annotate(round(val,2),(x,y))
        
#finding top dog brands
plt.figure(figsize = (15,30))
ax0 = sns.countplot(data=df, y='breed', hue='breed', order=df['breed'].value_counts().index, palette='husl', legend=False)
# 'rainbow', 'Set3', 'husl', 'Paired', 'Spectral', 'RdYlBu', 'viridis' 
barw(ax0)
plt.show()

# 创建一个LabelEncoder()类
le = LabelEncoder() 
# .loc()使用标签进行选择；fit_transform学习（fit）所有不同的狗品种标签，并将每个品种转换为唯一的数字（transfrom）
df.loc[:, 'breed'] = le.fit_transform(df.loc[:, 'breed']) 
df.head()

class Dog_Breed_Dataset(Dataset):
    
    def __init__(self, df:pd.DataFrame, img_base_path:str, split:str, transforms=None):
        self.df = df
        self.ima_base_path = img_base_path
        self.split = split
        self.transforms = transforms
        
    def __getitem__(self, index):
        img_path = os.path.join(self.img_base_path + self.df.loc[index, 'id'], '.jpg')
        img = Image.open(img_path)
        if self.transforms:
            img = self.transforms(img)
        if self.split != "test":
            y = self.df.loc[index, 'breed']
            return img, y
        else:
            return img
    
    def __len__(self):
        return len(self.df)                
    
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.1 ,contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
simple_transforms = transforms.Compose([
    transforms.Resize((224,224)),    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train, val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['breed']) # stratify确保训练集和验证集中各类别的比例保持一致
train = train.reset_index(drop=True) # 丢弃原来的索引，创建新的从0开始的连续索引
val   = val.reset_index(drop=True)

train_dataloader = Dog_Breed_Dataset(train, img_base_path='/kaggle/input/dog-breed-identification/train', split='train', transforms=train_transforms)
val_dataloader = Dog_Breed_Dataset(val, img_base_path='/kaggle/input/dog-breed-identification/train', split='val', transforms=simple_transforms)

train_set = DataLoader(train_dataloader, batch_size=64, shuffle=True, num_workers=4)
val_set = DataLoader(val_dataloader, batch_size=64, shuffle=False, num_workers=4)

print(f'train_set length：{len(train_set.dataset)}\nval_set length：{len(val_set.dataset)}')

from tqdm import tqdm

def train_model(train_set, val_set, model, epochs=20):    
    
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []
    # Top-3和Top-5准确率历史记录
    val_top3_acc_history = []
    val_top5_acc_history = []
    # 验证集上的最佳准确率
    best_val_loss = 1_000_000.0    
    # 获取初始权重
    weights = model.get_weights()
    
    for epoch in tqdm(range(epochs)):
        print("="*20, "Epoch: ", str(epoch), "="*20)
        
        train_correct_pred = 0
        val_correct_pred = 0
        val_top3_correct = 0
        val_top5_correct = 0
        train_acc = 0
        val_acc = 0
        train_loss = 0
        val_loss = 0
        
        # Training
        model.train()
        
        for x, y in train_set: # 一个批次                      
            x = x.clone().detach().to(device).requires_grad_(True)
            y = y.clone().detach().long().to(device)
            
            model.optim.zero_grad()
            
            preds = model(x)            
                    
            loss = model.criterion(preds,y) # criterion模型中定义的交叉熵损失            
                      
            loss.backward()
            model.optim.step()

            preds = torch.argmax(preds, dim=1) # 找出预测结果中概率最高的类别索引           
            train_correct_pred += (preds.long().unsqueeze(1) == y.unsqueeze(1)).sum().item() # .unsqueeze(1)在第1维添加一个维度，使形状匹配
            
            train_loss += loss.item() # .item()将Pytorch张量转换为Python标量           
        
        train_acc = train_correct_pred / len(train_set.dataset)
        
        train_acc_history.append(train_acc)
               
        train_loss_history.append(train_loss)
        
        # Valitation
        model.eval()
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for x, y in val_set:                            
                x = x.clone().detach().to(device)
                y = y.clone().detach().long().to(device)    

                preds = model(x)                

                loss = model.criterion(preds,y)                                         
                
                val_loss += loss.item()

                # 保存预测和标签用于后续评估
                all_val_preds.append(preds.cpu())
                all_val_labels.append(y.cpu())

                # Top-1准确率
                top1_preds = torch.argmax(preds, dim=1)
                val_correct_pred += (top1_preds.long().unsqueeze(1) == y.unsqueeze(1)).sum().item()
                
                # 计算Top-3和Top-5准确率
                _, top3_indices = torch.topk(preds, 3, dim=1)
                _, top5_indices = torch.topk(preds, 5, dim=1)
                
                # 检查真实标签是否在Top-3预测中
                for i, label in enumerate(y):
                    if label in top3_indices[i]:
                        val_top3_correct += 1
                    if label in top5_indices[i]:
                        val_top5_correct += 1
                
        model.scheduler.step() # 学习率调度器更新，在每个epoch后使用       
        
        val_acc = val_correct_pred / len(val_set.dataset)
        val_top3_acc = val_top3_correct / len(val_set.dataset)
        val_top5_acc = val_top5_correct / len(val_set.dataset)
        
        val_acc_history.append(val_acc)
        val_loss_history.append(val_loss)
        val_top3_acc_history.append(val_top3_acc)
        val_top5_acc_history.append(val_top5_acc)
        
        # 保存最佳模型的权重
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            weights = model.get_weights()
            
        print("Train acc: {:.4f} | Train Loss: {:.4f} | Validation acc: {:.4f} | Validation Loss: {:.4f}".format(train_acc, train_loss, val_acc, val_loss))
        print("Top-3 acc: {:.4f} | Top-5 acc: {:.4f}".format(val_top3_acc, val_top5_acc))
    
    # 加载最佳模型
    model.load_weights(weights)
    
    return [train_acc_history, train_loss_history, val_acc_history, val_loss_history, val_top3_acc_history, val_top5_acc_history], model, all_val_preds, all_val_labels

from torchvision.models import Inception_V3_Weights
inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
inception.AuxLogits # 访问Inception V3模型中的辅助分类器
"""
辅助分类器在训练早期阶段帮助加速网络收敛
梯度传播：为深层网络提供额外的梯度流，减轻梯度消失问题
正则化：起到一定的正则化作用，防止过拟合
"""

for name, module in inception.named_children():
    print(name)
    
inception_model = nn.Sequential(
    inception.Conv2d_1a_3x3,
    inception.Conv2d_2a_3x3,
    inception.Conv2d_2b_3x3,
    inception.maxpool1,
    inception.Conv2d_3b_1x1,
    inception.Conv2d_4a_3x3,
    inception.maxpool2,
    inception.Mixed_5b, # 混合了多个不同类型的卷积操作，形成并行结构
    inception.Mixed_5c,
    inception.Mixed_5d,
    inception.Mixed_6a,
    inception.Mixed_6b,
    inception.Mixed_6c,
    inception.Mixed_6d,
    inception.Mixed_6e,
    # 跳过辅助层AuxLogits（在特征提取或迁移学习中，通常只需要网络的特征提取部分）
    inception.Mixed_7a,
    inception.Mixed_7b,
    inception.Mixed_7c,
    inception.avgpool
    # 跳过dropout层、全连接层fc（迁移学习时，目标任务的类别数量通常与预训练任务不同，需要替换分类层）
)
"""
Dropout层是一种正则化技术，在神经网络训练过程中随机暂时关闭一部分神经元。
具体来说，在每次训练迭代中，按照预设概率随机选择一些神经元，将它们的输出暂时设为零，使这些神经元在当前批次的前向和反向传播中不参与计算。
通过迫使网络不依赖于任何特定神经元，减少复杂的共适应关系，防止过拟合。
"""

from torchvision.models import ResNet50_Weights
resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)

for name, module in resnet50.named_children():
    print(name)
    
resnet50_model = nn.Sequential(
    resnet50.conv1,
    resnet50.bn1,
    resnet50.relu,
    resnet50.maxpool,
    resnet50.layer1,
    resnet50.layer2,
    resnet50.layer3,
    resnet50.layer4,
    resnet50.avgpool
    # 跳过全连接层fc
)

for param in resnet50_model.parameters():    
    param.requires_grad = False # 在反向传播过程中，这些参数不需要计算梯度，也就不会被优化器更新
    
for param in inception_model.parameters():    
    param.requires_grad = False
    
class Model(nn.Module):
    
    def __init__(self, inception_model, resnet50_model, epochs=20):
        super(Model,self).__init__() # 初始化父类
        
        self.inception_model = inception_model
        self.resnet50_model = resnet50_model        
        
        self.output = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(4096,120) # 输入维度为4096，输出维度为120（breeds），将提取的特征映射到最终的类别概率            
        ) # simple fc layer
        
        self.to(device)
        # Optimizer 优化器: 改为 AdamW
        self.optim = torch.optim.AdamW(self.output.parameters(), lr=1e-4, weight_decay=1e-4) 
        # Loss 使用交叉损失
        self.criterion = torch.nn.CrossEntropyLoss()
        # Scheduler: 改为 CosineAnnealingLR
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=epochs, eta_min=1e-6)
        
    def forward(self, x):
        X1 = self.inception_model(x)
        X2 = self.resnet50_model(x)
        
        # 将多维特征张量 reshape 为二维张量，第一维保持不变，其余维度合并，-1自动计算这一维的大小
        # 例如，如果X1形状是[64, 2048, 1, 1](批次大小, 通道数, 高, 宽)，调用后变为[64, 2048]
        X1 = X1.view(X1.size(0), -1) # resize to (batchsize, -1)
        X2 = X2.view(X2.size(0), -1) # resize to (batchsize, -1)

        # 特征融合feature fusion/集成学习ensemble
        X = torch.cat([X1, X2], dim=1) # 在特征维度(第二维)上连接，保持批次维度不变
        # 例X1形状为[64, 2048](InceptionV3特征），如果X2形状为[64, 2048](ResNet50特征)，连接后X形状为[64, 4096]，前2048个特征来自InceptionV3，后2048个来自ResNet50
         
        P = self.output(X)        
        
        return P
    
    def get_weights(self):
        return self.output.state_dict() # 保存模型的权重参数
    
    def load_weights(self, weights):
        self.output.load_state_dict(weights) # 加载保存的权重参数到模型的分类器部分
        
print(torch.__version__)  # 查看当前版本
print(torch.version.cuda) # 查看CUDA版本
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name()}")

model = Model(inception_model, resnet50_model, epochs=20)

history, model, all_val_preds, all_val_labels = train_model(train_set, val_set, model)

# 处理验证集的预测结果以进行详细评估
all_val_preds = torch.cat(all_val_preds)
all_val_labels = torch.cat(all_val_labels)

# 获取最终预测
final_preds = torch.argmax(all_val_preds, dim=1).numpy()
true_labels = all_val_labels.numpy()

# 计算混淆矩阵
conf_matrix = confusion_matrix(true_labels, final_preds)

# 计算每个类别的精确率、召回率和F1分数
precision, recall, f1, support = precision_recall_fscore_support(true_labels, final_preds, average=None)
weighted_f1 = f1_score(true_labels, final_preds, average='weighted')

# 将类别从数字转回名称
breed_names = le.inverse_transform(np.unique(true_labels))

# 可视化结果
# 1. Training and Validation Results (Accuracy & Loss)
fig, axs = plt.subplots(2, 2, figsize=(18, 12))

# Accuracy
axs[0, 0].plot(range(20), history[0], label="Training accuracy")
axs[0, 0].plot(range(20), history[2], label="Validation accuracy")
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].set_title('Training & Validation Accuracy')
axs[0, 0].grid(True)
axs[0, 0].legend()

# Loss
axs[0, 1].plot(range(20), history[1], label="Training Loss")
axs[0, 1].plot(range(20), history[3], label="Validation Loss")
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].set_title('Training & Validation Loss')
axs[0, 1].grid(True)
axs[0, 1].legend()

# Top-3 & Top-5 Accuracy
axs[1, 0].plot(range(20), history[2], label="Top-1 accuracy")
axs[1, 0].plot(range(20), history[4], label="Top-3 accuracy")
axs[1, 0].plot(range(20), history[5], label="Top-5 accuracy")
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Accuracy')
axs[1, 0].set_title('Top-N Accuracy')
axs[1, 0].grid(True)
axs[1, 0].legend()

# 综合评估指标
metrics_summary = {
    'Metric': ['Top-1 Accuracy', 'Top-3 Accuracy', 'Top-5 Accuracy', 'Weighted F1'],
    'Value': [history[2][-1], history[4][-1], history[5][-1], weighted_f1]
}
axs[1, 1].axis('off')
axs[1, 1].table(cellText=[[f"{val:.4f}" for val in metrics_summary['Value']]],
                rowLabels=metrics_summary['Metric'],
                loc='center',
                cellLoc='center')
axs[1, 1].set_title('Summary Metrics')

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.show()

# 2. 绘制混淆矩阵热力图
# 如果类别太多，只选择前20个类别
if len(breed_names) > 20:
    # 通过支持度(support)排序，选择样本最多的20个类别
    top_classes_idx = np.argsort(support)[-20:]
    selected_conf_matrix = conf_matrix[top_classes_idx][:, top_classes_idx]
    selected_breed_names = breed_names[top_classes_idx]
else:
    selected_conf_matrix = conf_matrix
    selected_breed_names = breed_names

plt.figure(figsize=(16, 14))
sns.heatmap(selected_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=selected_breed_names,
            yticklabels=selected_breed_names)
plt.title('Confusion Matrix (Top 20 Classes)')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# 3. 每个类别的精确率和召回率
# 同样，如果类别太多，只选择前20个类别
if len(breed_names) > 20:
    top_classes_idx = np.argsort(support)[-20:]
    selected_precision = precision[top_classes_idx]
    selected_recall = recall[top_classes_idx]
    selected_f1 = f1[top_classes_idx]
    selected_breed_names = breed_names[top_classes_idx]
else:
    selected_precision = precision
    selected_recall = recall
    selected_f1 = f1
    selected_breed_names = breed_names

# 创建精确率和召回率的对比图
plt.figure(figsize=(16, 10))
x = np.arange(len(selected_breed_names))
width = 0.25

plt.bar(x - width, selected_precision, width, label='Precision')
plt.bar(x, selected_recall, width, label='Recall')
plt.bar(x + width, selected_f1, width, label='F1 Score')

plt.xlabel('Dog Breeds')
plt.ylabel('Scores')
plt.title('Precision, Recall & F1 Score by Class')
plt.xticks(x, selected_breed_names, rotation=90)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('precision_recall_f1.png')
plt.show()

# 4. 绘制前五个类别（按字母排序）的 PR 曲线
# 获取概率输出
val_probs = torch.softmax(all_val_preds, dim=1).numpy()
# 将真实标签进行 one-hot 编码
n_classes = len(dog_breeds) # 使用已排序的 dog_breeds
classes_indices = np.arange(n_classes)
y_true_bin = label_binarize(true_labels, classes=classes_indices)

# 获取前五个类别名称及其对应的索引
top_5_breed_names = dog_breeds[:5]
top_5_indices = le.transform(top_5_breed_names) # 获取这些名称对应的数字标签

plt.figure(figsize=(10, 8))

for i, breed_name in zip(top_5_indices, top_5_breed_names):
    # 计算该类别的精确率、召回率和阈值 (One-vs-Rest)
    precision_class, recall_class, _ = precision_recall_curve(y_true_bin[:, i], val_probs[:, i])
    # 计算该类别的 AP
    ap_class = average_precision_score(y_true_bin[:, i], val_probs[:, i])
    # 绘制 PR 曲线
    plt.plot(recall_class, precision_class, marker='.', label=f'{breed_name} (AP = {ap_class:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Top 5 Dog Breeds (Alphabetical)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('top5_precision_recall_curve.png')
plt.show()

# 输出整体指标
print(f"Overall weighted F1 score: {weighted_f1:.4f}")
print(f"Final validation accuracy: {history[2][-1]:.4f}")
print(f"Top-3 accuracy: {history[4][-1]:.4f}")
print(f"Top-5 accuracy: {history[5][-1]:.4f}")

test_data = pd.DataFrame([])
for dirname, _, filename in os.walk('/kaggle/input/dog-breed-identification/test/'):
    filename = pd.Series(filename)
    test_data = pd.concat([test_data, filename], axis=0)
test_data.columns = ['id']
test_data['id'] = test_data['id'].str.replace(".jpg","")

# Dataset shape
print(f"Test dataset shape: {test_data.shape}")
# Sample of the train_data DataFrame
test_data.head()

test_dataset = Dog_Breed_Dataset(
    df=test_data,
    img_base_path='/kaggle/input/dog-breed-identification/test/',
    split='test',
    transforms=simple_transforms
)

test_set = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

def test_model(test_set, model):       
              
        # Predictions DataFrame
        prob_preds = pd.DataFrame([])
        
        # 使用 tqdm 显示测试进度
        for x in tqdm(test_set, desc="Testing"):
            # Convert data to Tensor            
            x = x.clone().detach().to(device)            
            # Predict
            pred = model(x)  
            prob_pred = torch.nn.functional.softmax(pred, dim=1)
            
            prob_pred = prob_pred.detach().cpu().numpy()             
            prob_pred = pd.DataFrame(prob_pred)
            
            prob_preds = pd.concat([prob_preds, prob_pred], axis=0)            
            
        return prob_preds       

test_preds = test_model(test_set, model)
test_preds.shape
test_preds.head()

# Set columns to breed names
num_classes = []
for num_class in test_preds.columns:
    num_classes.append(num_class)

num_classes = np.array(num_classes)
num_classes = le.inverse_transform(num_classes)
test_preds.columns = list(num_classes)

test_preds.head()

# Set id column
test_preds = test_preds.reset_index(drop=True)
ids = test_data.loc[:,'id']
test_preds = pd.concat([ids, test_preds], axis=1)
test_preds.head()

test_preds.to_csv('submission.csv', index=None)