import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor


# ----------加载模型----------

# 定义模型
class NeuralNetwork(nn.Module):
    def __init__(self):     # 初始化，实例化模型的时候就会调用
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()     # [64, 1, 28, 28] -> [64, 1*28*28]
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),      # [64, 1*28*28] -> [64, 512]
            nn.ReLU(),
            nn.Linear(512, 512),        # [64, 512] -> [64, 512]
            nn.ReLU(),
            nn.Linear(512, 10)          # [64, 512] -> [64, 10]
        )

    def forward(self, x):   # 前向传播，输入数据进网络的时候才会调用
        x = self.flatten(x)                     # [64, 1*28*28]
        logits = self.linear_relu_stack(x)      # [64, 10]
        return logits

if __name__ == '__main__':
    # 加载MNIST数据集的测试集
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # 加载模型
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))

    # 种类名称
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    # 切换到测试模式
    model.eval()

    # 取第一个图像和对应的标签
    image, label = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        pred = model(image)     # 输入进网络得到输出
        predicted, actual = classes[pred[0].argmax(0)], classes[label]      # 得到预测值和标签值
        print(f'Predicted: "{predicted}", Actual: "{actual}"')