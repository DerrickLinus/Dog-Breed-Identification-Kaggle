# 项目简介
- 本项目基于Kaggle平台的“Dog Breed Identification”比赛，任务是**对120个类别的犬种图像进行识别分类**。本实验主要采用迁移学习的方法，利用预训练模型来实现对图像的多分类任务。  
- 具体而言，本实验采用两种方法（包含三个模型），一种是基于**InceptionV3和Resnet50**预训练模型的**集成学习**，另一种是基于**ConvNeXt**预训练模型的**微调**。
# 数据介绍
数据集下载：https://www.kaggle.com/competitions/dog-breed-identification/data

该比赛提供的数据集包含四个文件：  
📦 dog-breed-identification  
 ┣ 📂 train            # 训练数据集  
 ┃ ┗ 🖼️ 10222张图像    # 120个犬种类别  
 ┃  
 ┣ 📂 test             # 测试数据集  
 ┃ ┗ 🖼️ 10357张图像    # 待分类的犬种图像  
 ┃  
 ┣ 📄 label.csv        # 训练集标签文件  
 ┃ ┗ 📋 对应train文件夹中犬种图像的分类标签  
 ┃  
 ┗ 📄 submission.csv   # 提交格式示例  
   ┗ 📋 需提交至Kaggle平台的文件格式模板  
# 项目文件
- src
    - 基于ConvNeXt的微调模型
        - `Dog-Breed-Identification-ConvNeXt.ipynb`
        - `Dog-Breed-Identification-ConvNeXt.py`
    - 基于InceptionV3和Resnet50的融合模型
        - `Dog-Breed-Identification-Inception-Resnet50.ipynb`
        - `Dog-Breed-Identification-Inception-Resnet50.py`
- `Result-ConvNeXt`：ConvNeXt微调模型对应的实验结果  
- `Result-InceptionV3-Resnet50`：InceptionV3和Resnet50的融合模型对应的实验结果  
# 实验结果
| Methods | Accuracy | AP | Kaggle |
|:-------:|:--------:|:--:|:------:|
| InceptionV3+Resnet50 | 0.8362 | - | 1.08857 |
| ConvNeXt | 0.8837 | 0.8923 | 0.42089 |
> 注：Kaggle分数越低越好；其他更多评价指标详见代码
# 环境配置
## 我的环境
本地：Cursor IDE
云端服务器：Kaggle
Pytorch版本：2.6.0+cu126
Python版本：3.10.16
GPU：NVIDIA GeForce RTX 4070 Laptop GPU

## 创建虚拟环境（推荐使用虚拟环境）
### 直接创建（使用本地默认python解释器）
python -m venv venv
- 第一个 venv 是Python的模块名（固定的）`
- 第二个 venv 是想要创建的虚拟环境文件夹的名称（可以自定义）

PS：
- 不同项目下的虚拟环境是完全独立的，不会发生冲突；
- 因为每个虚拟环境都在其项目目录下，有自己独立的路径。

### 指定python解释器
"C:\Path\To\Python3.13.1\python.exe" -m venv venv
- 在Windows搜索栏中搜索"Python 3.13.1"
- 右键点击Python 3.13.1，选择"打开文件位置"
- 右键点击快捷方式，选择"属性"
- 在"目标"字段中可以看到Python的安装路径

PS：如果遇到权限问题
- 按 Windows + X
- 选择"Windows PowerShell (管理员)"或"命令提示符 (管理员)"
- 导航到项目目录：cd D:/cucdlh/postgraduate_1_1/ML
- 创建虚拟环境：& "C:\Users\27728\AppData\Local\Programs\Python\Python313\python.exe" -m venv venv

## 在虚拟环境下工作
### 激活虚拟环境
.\venv\Scripts\activate
激活成功后，命令行前面会出现`(venv)`标识
### 在虚拟环境中进行操作
- 安装所需的包（不同解释器下的环境依赖不同）
- 运行python代码
- 执行其他项目相关的命令

## 退出虚拟环境
deactivate

## 其他
- 在IDE（如Cursor）中，确保选择虚拟环境中的Python解释器
- 在运行代码时，确保终端中已经激活了虚拟环境（命令行前面有(venv)标识）
- 如果使用多个终端窗口，每个窗口都需要单独激活虚拟环境

## 建议
1. 在项目根目录创建一个 `requirements.txt` 文件，记录所有依赖包
2. 使用 `pip freeze > requirements.txt` 导出当前环境的依赖
3. 在新环境中使用 `pip install -r requirements.txt` 安装所有依赖