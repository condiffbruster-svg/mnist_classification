import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets,transforms

#重建模型
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
        logits = self.fc2(x)
        return logits

model = MNISTModel()
model.load_state_dict(torch.load("mnist_model.pth",weights_only = True))
model.eval()

test_datasets = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

image, label = test_datasets[10]
#在最前面加一个batch维度
image = image.unsqueeze(0)

#不建立梯度记录
with torch.no_grad():
    outputs = model(image)
    pred = torch.argmax(outputs,dim=1)#输出的形状是[1,10]现在在第二个维度中找最大的
print("输入：",label)
print("模型输出：",pred.item())



