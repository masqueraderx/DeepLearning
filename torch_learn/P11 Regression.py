import torch
from torch.autograd import variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x

if __name__ == '__main__':
    x = torch.unsqueeze(torch.linspace(-5, 5, 100), dim=1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())

    x, y = variable(x), variable(y)

    net = Net(1, 10, 1)

    print(net)
    plt.ion()
    plt.show()

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.02)

    for i in range(100):
        y_pred = net(x)
        loss = loss_func(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-')
            plt.text(0.5, 0, 'loss=%.4f' % loss.data.numpy())
            plt.pause(0.1)
    plt.ioff()
    plt.show()


