import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        return self.output(x)

if __name__ == '__main__':
    n_data = torch.ones(100, 2)
    x0 = torch.normal(2 * n_data, 1)
    y0 = torch.zeros(100)
    x1 = torch.normal(-2 * n_data, 1)
    y1 = torch.ones(100)
    x = torch.cat((x0, x1), 0).type(torch.FloatTensor) # FloatTensor = 32-bit floating
    y = torch.cat((y0, y1), 0).type(torch.LongTensor) # LongTensor = 64-bit Integer
    x, y = Variable(x), Variable(y)

    plt.figure()
    plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1])
    plt.title('origin data distribution')
    plt.show()

    net = Net(2, 10, 2)
    net2 = torch.nn.Sequential(
        torch.nn.Linear(2, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 2),
    )
    print(net2)
    print(net)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    loss_func = torch.nn.CrossEntropyLoss()

    plt.ion()
    for i in range(100):
        out = net(x)
        loss = loss_func(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            plt.cla()
            prediction = torch.max(F.softmax(out), 1)[1] # need indices
            pred_y = prediction.data.numpy().squeeze()

            target_y = y.data.numpy()
            plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y)
            accuracy = sum(pred_y == target_y) / len(x.data.numpy())

            plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy)
            plt.pause(0.1)
    plt.ioff()
    plt.show()

