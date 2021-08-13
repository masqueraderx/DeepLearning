import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(          # (B, 1, 28, 28)
                in_channels=1,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2       # if stride=1, then padding=(kernel_size - 1) / 2
            ),                  # (B, 32, 28, 28)
            nn.ReLU(),          # (B, 32, 28, 28)
            nn.MaxPool2d(kernel_size=2)         # (B, 32, 14, 14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),         # (B, 64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2)                     # (B, 64, 7, 7)
        )
        self.out = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)               # (B, 64, 7, 7) -> (B, 64 * 7 * 7)
        return self.out(x)


if __name__ == '__main__':
    train_data = torchvision.datasets.MNIST(
        root='./MNIST',
        train=True,
        transform=torchvision.transforms.ToTensor(),    # 0-1
        download=DOWNLOAD_MNIST
    )
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1
    )
    print(train_data.data.size())
    print(train_data.targets.size())
    plt.figure()
    plt.imshow(train_data.data[0].numpy(), cmap='gray')
    plt.title('%i' % train_data.targets[0])
    plt.show()

    test_data = torchvision.datasets.MNIST(
        root='./MNIST',
        train=DOWNLOAD_MNIST
    )
    test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255
    test_y = test_data.targets.type(torch.LongTensor)[:2000]

    net = CNN()
    print(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    loss_func = torch.nn.CrossEntropyLoss()

    # train
    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            prediction = net(batch_x)
            loss = loss_func(prediction, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                test_out = net(test_x)
                pred_y = torch.max(test_out, 1)[1].data.numpy().squeeze()

                accuracy = sum(pred_y == test_y.data.numpy()) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(),
                      '| test accuracy: %.2f' % accuracy)

    test_out = net(test_x[:10])
    pred_y = torch.max(test_out, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction_number')
    print(test_y[:10], 'true label')
