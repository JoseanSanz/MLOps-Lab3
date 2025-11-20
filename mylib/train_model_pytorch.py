import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# Load pre-trained MobileNetV2 model
model = models.mobilenet_v2(pretrained=True)

# freeze base model parameters
for param in model.parameters():
    param.requires_grad = False

# Modify the classifier to match the number of classes (e.g., 2 classes)
# Obtener el número de características de entrada de la última capa
num_ftrs = model.classifier[1].in_features
# Reemplazar la capa final por una nueva con tus clases
num_classes = 3
model.classifier[1] = nn.Linear(num_ftrs, num_classes)

# we load the beans dataset from torchvision
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
# load the dataset
train_dataset = datasets.ImageFolder(root="data/beans/train", transform=transform)
val_dataset = datasets.ImageFolder(root="data/beans/validation", transform=transform)
test_dataset = datasets.ImageFolder(root="data/beans/test", transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Move model to GPU if available
if torch.cuda.is_available():
    model.to("cuda")


# Training loop (simplified)
def train_model(model, train_loader, criterion, optimizer, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            if torch.cuda.is_available():
                inputs, labels = inputs.to("cuda"), labels.to("cuda")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}"
        )


train_model(model, train_loader, criterion, optimizer, num_epochs=2)


# Evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            if torch.cuda.is_available():
                inputs, labels = inputs.to("cuda"), labels.to("cuda")
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")


evaluate_model(model, test_loader)

# save the trained model to onnx format
model.eval()
model.cpu()

# obtain de size of the input
size = transform.transforms[0].size
if isinstance(size, dict):
    height = size.get("height", size.get("shortest_edge", 224))
    width = size.get("width", height)
elif isinstance(size, int):
    height = width = size
else:
    height = width = 224  # valor por defecto seguro

dummy_input = torch.randn(1, 3, height, width, dtype=torch.float32)

# Exportar a ONNX
output_path = "./mylib/model.onnx"
print(f"Exportando modelo a {output_path}...")
torch.onnx.export(
    model,
    input_example,
    output_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=12,
)

# obtain the labels mapping and save it in a json file

class_names = train_dataset.classes
label_map = {i: class_name for i, class_name in enumerate(class_names)}
with open("./mylib/labels.json", "w") as f:
    json.dump(label_map, f)

print(
    f"Modelo exportado correctamente a {output_path} y etiquetas de clases a {labels_file}"
)
