import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

import yaml
import torch
import os
import argparse
import json

def parse_args():
    
    parser = argparse.ArgumentParser(description='ClassNet', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--configpath",
        type=str,
        default='/MyClassNet/config.yaml',
    )
    
    args = parser.parse_args()
    return args

args = parse_args()
configpath = args.configpath
with open(configpath, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

model_name = config['model_name']

name = config['dataset_name']
number = ['3+1','6+2','9+3','18+6','27+9'] #'3+1','6+2','9+3','18+6','27+9'

test_dir = "/MyClassNet/origin_mstar/dataset/test" # elev15-full

# 目标类型
classes = ["2S1", "BMP-2", "BRDM-2", "BTR-60", "BTR-70", "D7", "T-62", "T-72", "ZIL-131", "ZSU-234"]
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

# 数据变换
transform = {
    "train": transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]),
    "test": transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
}

# 模型定义
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        elapsed_time = time.time() - start_time

        # 只打印第50个信息
        if epoch == 49:
            print(f"Epoch [{epoch + 1}/{epochs}] completed in {elapsed_time:.2f} seconds, Loss: {running_loss / len(train_loader):.4f}")

# 测试函数
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    accuracy = correct / total
    return accuracy, all_labels, all_preds

# 保存模型函数
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Best model saved to {path}")

# 加载模型函数
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    print(f"Model weights loaded from {path}")
    return model

# 蒙特卡洛实验
num_experiments = config['num_experiments']
accuracies = []
best_accuracy = 0.0
best_confusion_matrix = None


#for name in dataset_name:
for num in number:
    print(f"********Dataset name: {name}, number: {num}*********")
    train_dir = f"/MyClassNet/augmentation/train/{name}/{num}/"
    output_path= f"/MyClassNet/augmentation/result/result-t15/{name}/{num}/{model_name}/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    best_model_path = os.path.join(output_path,'best_model.pth')
    best_img_path = os.path.join(output_path,'best_confusion_matrix.png')
    acc_path = os.path.join(output_path,'best_acc.txt')
    
    # 数据加载
    train_dataset = datasets.ImageFolder(train_dir, transform=transform["train"])
    test_dataset = datasets.ImageFolder(test_dir, transform=transform["test"])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 检查数据集类别是否正确
    assert train_dataset.class_to_idx == class_to_idx, "类别与标签映射不一致！"
    
    for experiment in tqdm(range(num_experiments)):
        print(f"Running Experiment {experiment + 1}/{num_experiments}...")
        # 初始化模型
        if model_name =='ResNet50':
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif model_name =='ShuffleNet':
            model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        model = model.to(device)

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练和测试
        train_model(model, train_loader, criterion, optimizer, epochs=100)
        accuracy, true_labels, predicted_labels = evaluate_model(model, test_loader)
        accuracies.append(accuracy)

        # 更新最优结果
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_confusion_matrix = confusion_matrix(true_labels, predicted_labels)
            save_model(model, best_model_path)  # 仅保存最优模型权重

        print(f"Experiment {experiment + 1} Accuracy: {accuracy * 100:.2f}%")
    # 加载最佳模型进行复现
    print(f"\nLoading the best model for evaluation...")
    if model_name == 'ResNet50':
        best_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif model_name == 'ShuffleNet':
        best_model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
    best_model.fc = nn.Linear(best_model.fc.in_features, len(classes))
    best_model = load_model(best_model, best_model_path)

    accuracy, true_labels, predicted_labels = evaluate_model(best_model, test_loader)

    with open(acc_path, "w") as file:
        file.write("Monte Carlo Experiment Results:\n")
        file.write(f"{'Average Accuracy:':<20} {np.mean(accuracies) * 100:.2f}%\n")
        file.write(f"{'Best Accuracy:':<20} {best_accuracy * 100:.2f}%\n")

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(best_confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Best Accuracy: {best_accuracy * 100:.2f}%)")
    plt.savefig(best_img_path)