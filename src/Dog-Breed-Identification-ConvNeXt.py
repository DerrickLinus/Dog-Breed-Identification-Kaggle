## Deep-Leaning framework
import torch                            
import torch.nn as nn                   
import torchvision                      
import torchvision.transforms as tf     
from torch.utils.data import Dataset    
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAUPRC, MulticlassAccuracy 
import timm                 

## Auxiliary libraries related to deep learning
from torchinfo import summary 
import transformers           
from sklearn.model_selection import train_test_split              

## Plot
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid      
import pandas as pd              
import seaborn as sns         
from PIL import Image   

## Others
import numpy as np
from tqdm import tqdm         
import random                 
import re                     
import time                  
import math                  
from pathlib import Path     

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 

df = pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv')
print('Shape of df:',df.shape)
df.head()

dog_breeds = sorted(df['breed'].unique())
# dog_breeds = df['breed'].sort_values().drop_duplicates().tolist()
print('Number of unique breeds:', len(dog_breeds))
print(dog_breeds)

X_train, X_test, y_train, y_test = train_test_split(df['id'], df['breed'], test_size=0.1, random_state=seed)

pd.concat([X_train, y_train], axis=1).to_csv('train.csv', index = False) # 9199 x 2
pd.concat([X_test,  y_test],  axis=1).to_csv('val.csv',  index = False) # 1023 x 2
df_train = pd.read_csv('train.csv')
df_val   = pd.read_csv('val.csv')
print(f'train_csv size：{df_train.shape}\nval_csv size  ：{df_val.shape}')

class DogBreedDataset(Dataset):
    def __init__(self, img_folder:str, all_labels:list, csv_file=None, transform=None):
        self.root = Path(img_folder)
        self.transform = transform
        
        if csv_file is not None:
            data = pd.read_csv(csv_file)
            img_names, self.labels = data['id'], data['breed']
            self.img_paths = [self.root / f'{i}.jpg' for i in img_names]       
        else:
            self.img_paths = list(self.root.glob('*.jpg'))
            self.labels = None
            
        self.id2label = dict(enumerate(all_labels)) # self.id2label = dict(zip(all_labels,range(len(all_labels))))
        self.label2id = {v:k for k,v in self.id2label.items()}
        
    def __getitem__(self, index):
        image = Image.open(self.img_paths[index]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            label = self.label2id[self.labels[index]]
            label = torch.tensor(label)
            return image, label # (image, label) a turple
        else:
            return (image, )
    
    def __len__(self):
        return len(self.img_paths)
    
train_set = DogBreedDataset('/kaggle/input/dog-breed-identification/train', all_labels=dog_breeds, csv_file='train.csv', transform=tf.Resize([224,224]))
val_set   = DogBreedDataset('/kaggle/input/dog-breed-identification/train', all_labels=dog_breeds, csv_file='val.csv'  , transform=tf.Resize([224,224]))
test_set  = DogBreedDataset('/kaggle/input/dog-breed-identification/test' , all_labels=dog_breeds, transform=tf.Resize([224,224]))
print(f'train set size：{len(train_set)}\nval set size：{len(val_set)}\ntest set size：{len(test_set)}')
# print(train_set.labels)
# print(val_set.labels)

# function to show bar length
def barw(ax): 
    for p in ax.patches:
        val = p.get_width() #height of the bar
        x = p.get_x()+ p.get_width() # x- position 
        y = p.get_y() + p.get_height()/2 #y-position
        ax.annotate(round(val,2),(x,y))
        
# inding top dog brands
plt.figure(figsize = (15,30))
ax0 = sns.countplot(data=df, y='breed', hue='breed', order=df['breed'].value_counts().index, palette='husl', legend=False)
# 'rainbow', 'Set3', 'husl', 'Paired', 'Spectral', 'RdYlBu', 'viridis' 
barw(ax0)
plt.show()

print(f'train set size：{len(train_set)}\nval set size  ：{len(val_set)}\ntest set size ：{len(test_set)}')

def sample_imgs_in_dataset(data, num):
    indices = random.sample(list(range(len(data))), num)
    samples = [data[i] for i in indices] # (image, label)
    return samples

s_sum = 6
image_wall = [sample_imgs_in_dataset(train_set, s_sum),
              sample_imgs_in_dataset(val_set  , s_sum),
              sample_imgs_in_dataset(test_set , s_sum)]

fig = plt.figure(figsize=(14,14.))
grid = ImageGrid(
    fig, 111,
    nrows_ncols = (3, s_sum),
    axes_pad=0.3 # pad between axes in inch.
) 
for i in range(3):
    for j in range(s_sum):
        ax = grid[i*s_sum+j] # calculate index of this image
        ax.imshow(image_wall[i][j][0]) # image_wall[i][j][0] -> image, image_wall[i][j][0] -> label
        if len(image_wall[i][j]) > 1:  # label is not None
            ax.set_title(train_set.id2label[image_wall[i][j][1].item()]) # .item()：tensor -> standard 
        if j == 0:
            ax.set_ylabel(['train', 'val', 'test'][i])
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
plt.show()

# Using normalization parameters pre-trained on ImageNet
norm_params = {
    'mean':[0.485, 0.456, 0.406],
    'std' :[0.229, 0.224, 0.225]
}

simple_tf = tf.Compose([
    tf.ToTensor(),
    tf.Resize((224,224), antialias=True),
    tf.Normalize(**norm_params) # **：unpacking operator
])

train_tf = tf.Compose([
    tf.PILToTensor(),
    tf.ConvertImageDtype(torch.float),
    # tf.AutoAugment(tf.AutoAugmentPolicy.IMAGENET),
    tf.ColorJitter(brightness=0.1 ,contrast=0.1, saturation=0.1),
    tf.RandomChoice([
        tf.Compose([
            tf.RandomRotation(10),
            tf.RandomResizedCrop((224,224), scale=(0.8, 1.0), antialias=True)
        ]),
        tf.Compose([
            tf.RandomAffine(10, translate=(0.0, 0.05), scale=(0.9, 1.0)),
            tf.Resize((224,224), antialias=True)
        ]),
    ]),
    tf.RandomHorizontalFlip(),
    tf.Normalize(**norm_params)
])

train_set = DogBreedDataset('/kaggle/input/dog-breed-identification/train', all_labels=dog_breeds, csv_file='train.csv', transform=train_tf)
val_set   = DogBreedDataset('/kaggle/input/dog-breed-identification/train', all_labels=dog_breeds, csv_file='val.csv', transform=simple_tf)
test_set  = DogBreedDataset('/kaggle/input/dog-breed-identification/test',  all_labels=dog_breeds, transform=simple_tf)
print(train_set.labels)
print(val_set.labels)

def unnormalize(norm_img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    for i in range(3): # R,G,B
        norm_img[i] = norm_img[i]*std[i] + mean[i]
    return norm_img

image_wall = [sample_imgs_in_dataset(train_set, s_sum),
              sample_imgs_in_dataset(train_set, s_sum),
              sample_imgs_in_dataset(train_set, s_sum),
              sample_imgs_in_dataset(val_set,   s_sum),
              sample_imgs_in_dataset(test_set,  s_sum)]

fig = plt.figure(figsize=(14,14.))
grid = ImageGrid(
    fig, 111,
    nrows_ncols=(5,s_sum),
    axes_pad = 0.3
)

for i in range(5):
    for j in range(s_sum):
        ax = grid[i*s_sum+j]
        img = image_wall[i][j][0]
        img = unnormalize(img).permute(1, 2, 0).numpy() # tensor:(C,H,W) -> matplotlib needs (H,W,C)
        ax.imshow(img)
        if len(image_wall[i][j]) > 1:
            ax.set_title(train_set.id2label[image_wall[i][j][1].item()])
        if j == 0:
            ax.set_ylabel(['train', 'train', 'train', 'val', 'test'][i])
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
plt.show()

"""
ConvNeXt Small Model (convnext_small.in12k_ft_in1k):
- Architecture: ConvNeXt small variant
- Pre-training: First trained on ImageNet-12K (12,000 classes)
- Fine-tuning: Then fine-tuned on ImageNet-1K (1,000 classes)
- Model Size: Smaller than base version, balanced between efficiency and performance
"""
def build_convnext_model():
    model = timm.create_model('convnext_small.in12k_ft_in1k', pretrained=True, num_classes=120)
    return model

model = build_convnext_model()
summary(model, depth=4)

for name, param in model.named_parameters():
    print(f'{name}\t{param.shape}\t{param.requires_grad}') 

def freeze_backbone(model):
    for name, param in model.named_parameters():
        if name.startswith('head'):
            param.requires_grad = True
        elif name.startswith('stages.3.'):
            param.requires_grad = True
        elif name.startswith('stages.2.blocks.2'):
            param.requires_grad = True
        else:
            param.requires_grad = False

"""
- Normalization layer parameters (.norm.)
- Bias parameters (.bias)
"""
def _get_no_decay_params(model):
    no_decay_params = []
    for name, param in model.named_parameters():
        if ".norm." in name:
            no_decay_params.append(param)
        elif name.endswith('.bias'):
            no_decay_params.append(param)
    return set(no_decay_params)

def get_optim(pt_model, lr, weight_decay, lr_decay):
    param_groups = []
    no_decay_params = _get_no_decay_params(pt_model)
    for name, param in pt_model.named_parameters():
        if param.requires_grad is True:
            if name.startswith('stages.3.'):
                scale = lr_decay
            elif name.startswith('stages.2.'):
                scale = lr_decay ** 2
            else:
                scale = 1.0
            param_group = {'params':param, 'lr':lr * scale}
            if name not in no_decay_params:
                param_group['weight_decay'] = weight_decay
            
            param_groups.append(param_group)
    optimizer = torch.optim.AdamW(param_groups, lr=lr)
    return optimizer

def get_scheduler(optim, num_warmup_steps, num_training_steps):
    return transformers.get_linear_schedule_with_warmup(optim, num_warmup_steps, num_training_steps)

def train(model, dataloaders, optimizer, scheduler, **kwargs):
    # logger
    logger = {
        'train/loss':[],
        'train/ap':[],
        'train/acc':[],
        'val/loss':[],
        'val/ap':[],
        'val/acc':[],
        'lr':[]
    }
    
    train_loader, val_loader = dataloaders # DataLoader(dataset, batch_size, shuffle, num_workers, pin_memory, drop_last...)
    device = kwargs['device']
    loss = nn.CrossEntropyLoss().to(device)
    model = model.to(device)
    
    for epoch in range(kwargs['max_epochs']):
        # Training
        loss_list = []
        model.train()
        ap_metric = MulticlassAUPRC(num_classes=120) # Average Precision
        acc_metric = MulticlassAccuracy(num_classes=120) # Accuracy
        last_time = time.time()
        for local_step, (img,label) in enumerate(train_loader): # local_step start from 0, automatically +1 
            step = epoch * kwargs['steps_per_epoch'] + local_step # local_step：num of batch，step：total step
            img, label = img.to(device), label.to(device)
            
            optimizer.zero_grad()
            logits = model(img)
            l = loss(logits, label)
            ap_metric.update(logits, label)
            acc_metric.update(logits, label)
            l.backward()
            optimizer.step() # update parameters
            scheduler.step() 
    
            # log 
            loss_list.append(l.detach().cpu().item())
            logger['train/loss'].append(l.detach().cpu().item())
            logger['lr'].append(optimizer.state_dict()['param_groups'][0]['lr'])
            if (local_step % 5 == 0 and local_step != 0) or local_step == kwargs['steps_per_epoch'] - 1:
                print('Epoch {}/{} | Step {}/{} | loss: {:.5f} Accuracy: {:.4f} AP: {:.4f} time: {:.1f}s'.format(
                    epoch, kwargs['max_epochs'],
                    local_step, kwargs['steps_per_epoch'],
                    sum(loss_list)/len(loss_list),
                    acc_metric.compute().cpu().item(),
                    ap_metric.compute().cpu().item(),
                    time.time()-last_time
                ))
                last_time = time.time()
        logger['train/acc'].append(acc_metric.compute().cpu().item())
        logger['train/ap'].append(ap_metric.compute().cpu().item())
        
        # Validation
        if True:
            loss_list = []
            ap_metric = MulticlassAUPRC(num_classes=120)
            acc_metric = MulticlassAccuracy(num_classes=120)
            print('-'*20 + '  Validating  '+'-'*20)
            model.eval()
            with torch.no_grad():
                for local_step, (img,label) in enumerate(val_loader):
                    img, label = img.to(device), label.to(device)
                    logits = model(img)
                    l = loss(logits, label)
                    ap_metric.update(logits, label)
                    acc_metric.update(logits, label)
                    loss_list.append(l.cpu().item())
                    
                avg_val_loss = sum(loss_list)/len(loss_list)
                
                # log
                logger['val/acc'].append(acc_metric.compute().cpu().item())
                logger['val/ap'].append(ap_metric.compute().cpu().item())
                logger['val/loss'].append(avg_val_loss)
                print('Epoch {}/{} | loss:{:.5f} Accuracy:{:.4f} AP:{:.4f}'.format(
                    epoch, kwargs['max_epochs'],
                    avg_val_loss,
                    acc_metric.compute().cpu().item(),
                    ap_metric.compute().cpu().item()
                ))
        print('='*53)
    return logger

batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_epochs = 30
lr = 1e-5
weight_decay = 0.01
num_warmup_steps = 0.5*math.ceil(len(train_set)/batch_size)
num_training_steps = max_epochs * math.ceil(len(train_set)/batch_size)
lr_decay = 0.1

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_dataloader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
model = build_convnext_model()
optimizer = get_optim(model, lr, weight_decay, lr_decay=lr_decay)
scheduler = get_scheduler(optimizer, num_warmup_steps, num_training_steps)
freeze_backbone(model)

print(summary(model, depth=2))

logger = train(
    model, (train_dataloader, val_dataloader), optimizer, scheduler,
    device = device,
    max_epochs = max_epochs,
    steps_per_epoch = len(train_dataloader)
)

print(torch.__version__)
print(torch.version.cuda)
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name()}")

torch.save(model.state_dict(), 'convnext_model.pth')
from IPython.display import FileLink
FileLink(r'convnext_model.pth')

fig, axs = plt.subplots(4, 1, figsize=(10,15))

train_steps = len(logger['train/loss'])
val_steps = len(logger['val/loss'])
train_indices = np.linspace(0, train_steps-1, val_steps, dtype=int)

sns.lineplot(
    x = range(val_steps),
    y = [logger['train/loss'][i] for i in train_indices], 
    ax = axs[0]
)
sns.lineplot(
    x = range(val_steps),
    y = logger['val/loss'],
    ax = axs[0],
    color = 'orange'
)
axs[0].set_title('Loss')

sns.lineplot(
    x = range(len(logger['train/ap'])),
    y = logger['train/ap'],
    ax = axs[1]
)
sns.lineplot(
    x = range(len(logger['val/ap'])),
    y = logger['val/ap'],
    ax = axs[1]
)
axs[1].set_title('Average Precision')

sns.lineplot(
    x = range(len(logger['train/acc'])),
    y = logger['train/acc'],
    ax = axs[2]
)
sns.lineplot(
    x = range(len(logger['val/acc'])),
    y = logger['val/acc'],
    ax = axs[2]
)
axs[2].set_title('Accuracy')

sns.lineplot(
    x = range(len(logger['lr'])),
    y = logger['lr'],
    ax = axs[3]
)
axs[3].set_title('Learning Rate')

plt.tight_layout()
plt.show()

import warnings
warnings.filterwarnings("ignore", message="The reduce argument of torch.scatter*")
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
ap_metric = MulticlassAUPRC(num_classes=120, average=None).to(device)
acc_metric = MulticlassAccuracy(num_classes=120, average=None).to(device)
print('-'*20 + '  validation  ' + '-'*20)
model.eval()
with torch.no_grad():
    for x, y in tqdm(val_dataloader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        ap_metric.update(logits, y)
        acc_metric.update(logits, y)
ap = [round(i, 4) for i in ap_metric.compute().cpu().tolist()]
acc = [round(i, 4) for i in acc_metric.compute().cpu().tolist()]
print(f'AP：{ap}')
print(f'ACC：{acc}')

plt.figure(figsize=(20,20))
plt.subplot(2,1,1)
plt.bar(test_set.id2label.values(), ap)
plt.xticks(rotation = 90)
plt.title('Average Precision')
plt.subplot(2,1,2)
plt.bar(test_set.id2label.values(), acc)
plt.xticks(rotation = 90)
plt.title('Accuracy')
plt.show()

test_dataloader = DataLoader(test_set, batch_size = batch_size, shuffle=False)
print('-'*20 + '  test  ' + '-'*20)
model.eval()
all_probs = []
with torch.no_grad():
    for x in tqdm(test_dataloader):
        x = x[0].to(device)
        probs = torch.softmax(model(x), dim = -1)
        all_probs.append(probs)
all_probs = torch.cat(all_probs, dim = 0)

with open('submission.csv', 'w+') as f:
    f.write('id,' + ','.join(test_set.id2label.values()) + '\n')
    for path, prob in zip(test_set.img_paths, all_probs):
        f.write(path.stem + ',' + ','.join([str(i) for i in prob.tolist()]) + '\n')

FileLink('submission.csv')