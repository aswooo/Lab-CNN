import torch
from torch import tensor
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
print(device + " is using")

training_data = datasets.MNIST("data", train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST("data", train=False, download=True, transform=ToTensor())

trainLoader = torch.utils.data.DataLoader(training_data, batch_size=100, shuffle=True, drop_last=False)
testLoader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, drop_last=False)


class network(nn.Module):  # 네트워크 모델 정의
    def __init__(self):
        super(network, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, (3, 3), padding=1),   # in_channels = 1, out_channels = 6, kernel_size = 3       output = 28 * 24 * 6
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                   # max 풀링  kernel_size = 2, stride = 2                     output = 14 * 14 * 6
            nn.Conv2d(6, 16, (3, 3), padding=1),  # in_channels = 6, out_channels = 16, kernel_size = 3      output = 14 * 14 * 16
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                    # max 풀링  kernel_size = 2, stride = 2                     output =  7 *  7 * 16
        )
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        # torch.nn.MaxPool2d( kernel_size , stride = None , padding = 0 , dilation = 1 , return_indices = False , ceil_mode = False )

        self.fully_connected = nn.Sequential(
            nn.Linear(16 * 7 * 7, 120),  # 1층 레이어
            nn.Linear(120, 84),          # 2층 레이어
            nn.Linear(84, 10)            # 3층 레이어
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 16 * 7 * 7)  # 1차원 전환  (nn.flatten)
        x = self.fully_connected(x)
        return x


def train():
    for epoch in range(Epochs):
        loss_sum = 0
        for data, target in trainLoader:
            X, y = data.to(device), target.to(device)  # cross
            optimizer.zero_grad()
            prediction = model(X)  # 결과 출력
            loss = criterion(prediction, y)  # cross 로스 계산
            loss.backward()  # 로스 역전파
            optimizer.step()  # 실질적 웨이트 수정
            loss_sum += loss.item()
        print("epoch = %d   loss = %f" % (epoch + 1, round(loss_sum / batch_count, 3)))
        test()


def test():
    correct = 0
    with torch.no_grad():
        for data, target in testLoader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)  # 출력 계산

            # 추론 계산
            _, predicted = torch.max(outputs, 1)  # 가장 큰 인덱스 위치를 리턴함  @ return value, index
            correct += predicted.eq(target).sum()  # 정답과 일치한 경우 정답 카운트를 증가

    data_num = len(test_data)  # 데이터 총 건수
    print("accuracy = {}/10000\n".format(correct))


model = network().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

Epochs = 15
batch_count = len(trainLoader)

train()
