from matplotlib import figure
import torch
from torch import tensor
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from matplotlib import font_manager, rc

font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
plt.rc('figure', titlesize=30)  # figure title 폰트 크기

device = 'cpu'  # 'cuda' if torch.cuda.is_available() else
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
print(device + " is available")

training_data = datasets.MNIST("data", train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST("data", train=False, download=True, transform=ToTensor())

trainLoader = torch.utils.data.DataLoader(training_data, batch_size=100, shuffle=False, drop_last=False)
testLoader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False, drop_last=False)


class network(nn.Module):  # 네트워크 모델 정의
    def __init__(self):
        super(network, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, (3, 3), padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.conv1_k = self.conv1.weight.detach()
        self.conv2_k = self.conv2.weight.detach()

        self.input_data = 0
        self.after_conv1 = 0
        self.after_pooling1 = 0
        self.after_conv2 = 0
        self.after_pooling2 = 0

    def forward(self, x):
        self.conv1_k = self.conv1.weight.detach()  # 업데이트된 커널을 다시 저장
        self.conv2_k = self.conv2.weight.detach()  # 업데이트된 커널을 다시 저장

        # index = 0(숫자 4), 7(숫자 8), 10(숫자 9)
        self.input_data = x[0]

        x = self.conv1(x)  # ((28 + 0 - 5)/1)+1    output = 24 * 24 * 6
        self.after_conv1 = x[0]

        x = self.pool(F.relu(x))
        self.after_pooling1 = x[0]

        x = self.conv2(x)  # ((12 + 0 - 5)/1)+1    output =  8 *  8 * 16
        self.after_conv2 = x[0]

        x = self.pool(F.relu(x))
        self.after_pooling2 = x[0]

        x = x.view(-1, 16 * 7 * 7)  # 1차원으로 변경  (nn.flatten)
        x = self.fc1(x)  # 1층 레이어
        x = self.fc2(x)  # 2층 레이어
        x = self.fc3(x)  # 3층 레이어

        return x


def train():
    for epoch in range(Epochs):
        loss_sum = 0
        visualizing_kernel(epoch)
        for data, target in trainLoader:
            X, y = data.to(device), target.to(device)  # cross
            optimizer.zero_grad()
            prediction = model(X)  # 결과 출력
            loss = criterion(prediction, y)  # cross 로스 계산
            loss.backward()  # 로스 역전파
            optimizer.step()  # 실질적 웨이트 수정
            loss_sum += loss.item()

        draw(epoch)
        print("epoch = %d   loss = %f" % (epoch + 1, round(loss_sum / batch_count, 3)))
        test()


def draw(epoch):
    plt.figure(figsize=(15, 10))  # conv1, conv2의 첫번째 커널에 대한 필터결과
    plt.suptitle("Feature Map after each layer\nepoch = " + str(epoch + 1))
    plt.subplot(1, 5, 1)
    plt.title('Input Data')
    plt.imshow(model.input_data[0, :, :].detach(), cmap=plt.cm.gray_r)
    plt.subplot(1, 5, 2)
    plt.title('First Conv')
    plt.imshow(model.after_conv1[0, :, :].detach(), cmap=plt.cm.gray_r)
    plt.subplot(1, 5, 3)
    plt.title('First Pooling')
    plt.imshow(model.after_pooling1[0, :, :].detach(), cmap=plt.cm.gray_r)
    plt.subplot(1, 5, 4)
    plt.title('Second Conv')
    plt.imshow(model.after_conv2[0, :, :].detach(), cmap=plt.cm.gray_r)
    plt.subplot(1, 5, 5)
    plt.title('Second Pooling')
    plt.imshow(model.after_pooling2[0, :, :].detach(), cmap=plt.cm.gray_r)
    plt.show()

    plt.figure(figsize=(15, 10))  # 같은 input 데이터에 대한 각 필터로 필터링 후 차이
    plt.suptitle("Visualizing Feature Map each Filter\nepoch = " + str(epoch + 1))
    plt.subplot(1, 4, 1)
    plt.title('Input Data')
    plt.imshow(model.input_data[0, :, :].detach(), cmap=plt.cm.gray_r)
    plt.subplot(1, 4, 2)
    plt.title('어정쩡 필터')
    plt.imshow(model.after_conv1[0, :, :].detach(), cmap=plt.cm.gray_r)
    plt.subplot(1, 4, 3)
    plt.title('Conv1 3번 필터(가로)')
    plt.imshow(model.after_conv1[3, :, :].detach(), cmap=plt.cm.gray_r)
    plt.subplot(1, 4, 4)
    plt.title('Conv1 6번 필터(세로)')
    plt.imshow(model.after_conv1[5, :, :].detach(), cmap=plt.cm.gray_r)


def visualizing_kernel(epoch):
    plt.figure(figsize=(15, 10))  # 1층 컨볼루션 kernel 출력
    plt.suptitle("Visualizing Conv1 kernel\nepoch = " + str(epoch + 1))
    for i, kernel in enumerate(model.conv1_k):
        plt.subplot(1, 6, i + 1)
        plt.imshow(kernel[0, :, :].detach(), cmap='gray')
        plt.axis('off')
    plt.show()

    plt.figure(figsize=(15, 10))  # 2층 컨볼루션 kernel 출력
    plt.suptitle("Visualizing Conv2 kernel\nepoch = " + str(epoch + 1))
    for i, kernel in enumerate(model.conv2_k):
        plt.subplot(3, 6, i + 1)
        plt.imshow(kernel[0, :, :].detach(), cmap='gray')
        plt.axis('off')
    plt.show()


def test():
    correct = 0

    with torch.no_grad():
        for data, target in testLoader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)

            _, predicted = torch.max(outputs, 1)  # 가장 큰 인덱스 위치를 리턴함  @ return value, index
            correct += predicted.eq(target).sum()  # 정답과 일치한 경우 정답 카운트를 증가

    data_num = len(test_data)  # 데이터 총 건수
    print('{:.3f}\n'.format(correct))


model = network().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

Epochs = 10
batch_count = len(trainLoader)

train()
