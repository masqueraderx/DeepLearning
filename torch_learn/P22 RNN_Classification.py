import torch
import torchvision
import torch.nn as nn
from torch.autograd import variable
import torch.utils.data as Data

EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = False

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h, c) = self.rnn(x, None)       #(batch_size, time_step, input_size)
        return self.out(r_out[:, -1, :])


if __name__ == '__main__':
    train_data = torchvision.datasets.MNIST(
        root='./MNIST',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST
    )
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1
    )

    test_data = torchvision.datasets.MNIST(
        root='./MNIST',
        train=False,
        download=DOWNLOAD_MNIST
    )

    test_x = test_data.data.type(torch.FloatTensor)[:2000] / 255
    test_y = test_data.targets.type(torch.LongTensor)[:2000]

    rnn = RNN()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            prediction = rnn(batch_x.view(-1, 28, 28))      # reshape x to (batch, time_step, input_size)
            loss = loss_func(prediction, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                test_out = rnn(test_x)
                pred_y = torch.max(test_out, 1)[1].data.numpy().squeeze()
                accuracy = sum(pred_y == test_y.numpy()) / len(pred_y)

                print('Epoch: ', epoch, '| Accuracy: %.4f' % accuracy, '| Loss: %.4f' % loss.data.numpy())
