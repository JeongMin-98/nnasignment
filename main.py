import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
from tqdm import tqdm
import copy
import wandb
import argparse
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Train ResNet models')
parser.add_argument('--model', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'all'],
                    help='Model architecture to use')
parser.add_argument('--optimizer', type=str, default='sgd',
                    choices=['sgd', 'adam'], help='Optimizer type')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--use_wandb', action='store_true')
parser.add_argument('--no_display', action='store_true')
args = parser.parse_args()

if args.use_wandb:
    wandb.init(project="resnet-comparison",
               name=f"{args.model}-{args.optimizer}-lr{args.lr}-bs{args.batch_size}",
               config=vars(args))

dataset_path = 'dataset'
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'valid')
test_path = os.path.join(dataset_path, 'test')

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_path, train_transform)
val_dataset = datasets.ImageFolder(val_path, val_test_transform)
test_dataset = datasets.ImageFolder(test_path, val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

classes = train_dataset.classes

def initialize_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def evaluate_model(model, loader, criterion, device, desc='Eval'):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc=desc):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            _, pred = torch.max(output, 1)
            total_loss += loss.item() * x.size(0)
            correct += (pred == y).sum().item()
            total += x.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    return total_loss / total, correct / total, all_preds, all_labels

def plot_confusion_matrix(all_preds, all_labels, classes, name):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(name)
    fname = f"{name.lower().replace(' ', '_')}.png"
    plt.savefig(fname)
    if args.use_wandb:
        wandb.log({name: wandb.Image(fname)})

def train_model(model_name: str):
    print(f"\n🚀 Training {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model(model_name, len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            correct += (preds == labels).sum().item()
            total += inputs.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"[{epoch+1}/{args.epochs}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, f"{model_name}_best.pth")

        scheduler.step(val_loss)
        if args.use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "epoch": epoch
            })
    model.load_state_dict(best_wts)
    torch.save(model.state_dict(), f"{model_name}_final.pth")
    return model, history

def test_model(model_name: str):
    print(f"\n🧪 Testing {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model(model_name, len(classes)).to(device)
    model.load_state_dict(torch.load(f"{model_name}_best.pth"))
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, preds, labels = evaluate_model(model, test_loader, criterion, device, 'Test')
    print(f"Test Accuracy for {model_name}: {test_acc:.4f}")
    if not args.no_display:
        plot_confusion_matrix(preds, labels, classes, f"{model_name} Confusion Matrix")
    if args.use_wandb:
        wandb.log({"test_acc": test_acc})
    return test_acc

# Main
results = {}
model_list = ['resnet18', 'resnet34', 'resnet50'] if args.model == 'all' else [args.model]

for mname in model_list:
    model, hist = train_model(mname)
    acc = test_model(mname)
    results[mname] = {'history': hist, 'test_acc': acc}

if args.model == 'all' and not args.no_display:
    try:
        plt.figure(figsize=(12, 5))
        for name in model_list:
            plt.subplot(1, 2, 1)
            plt.plot(results[name]['history']['val_acc'], label=name)
            plt.title("Validation Accuracy")
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(results[name]['history']['val_loss'], label=name)
            plt.title("Validation Loss")
            plt.legend()
        plt.tight_layout()
        plt.savefig('resnet_comparison.png')
        if args.use_wandb:
            wandb.log({"resnet_comparison": wandb.Image('resnet_comparison.png')})
    except Exception as e:
        print(f"[Plot Error] {e}")

if args.use_wandb:
    wandb.finish()

print("✅ All experiments completed.")
