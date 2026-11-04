import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets,transforms

train_dataset = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
# 读取一张照片
# image, label = train_dataset[0]
#取出来的是[1,28,28]的tensor，想要变成图片需要用imshow但是要去掉一个维度用squeeze
#plt.imshow(image.squeeze(),cmap="gray")
#plt.title(f"Label:{label}")
#plt.show()

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)
# images,labels = next(iter(train_loader))
# print(images.shape)
# print(labels.shape)

test_dataset = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

#构建模型
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784,128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        logist = self.fc2(x)
        return logist

model = MNISTModel()
# print(model)
#images,labels = next(iter(train_loader))
#outputs = model(images)

# print("输入形状：",images.shape)
# print("输出形状：",outputs.shape)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

loss_history = []
epochs = 5
for epoch in range(epochs):
    total_loss = 0
    for images, labels in train_loader:
        outputs = model(images)
        loss = loss_fn(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    n = len(train_loader)
    avg_loss = total_loss / n
    loss_history.append(avg_loss)
    print(f"Epoch[{epoch+1}/{epochs}],Loss:{avg_loss:.4f}")

correct = 0
total = 0
model.eval()
with torch.no_grad():

    for images,labels in test_loader:

        outputs = model(images)
        pred = torch.argmax(outputs,dim=1)

        correct_mask = pred == labels
        correct += correct_mask.sum().item()
        total += labels.size(0)
    accuracy = correct / total * 100
    print(f"Test Accuracy:{accuracy:.2f}%")

x = range(1, epochs+1)

y = loss_history
plt.plot(x,y)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.savefig("loss.png")
plt.show()

torch.save(model.state_dict(),"mnist_model.pth")


