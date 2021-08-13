import torch
from torch.autograd import variable
import torch.functional as F
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

def save(x, y):
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )

    optimizer = torch.optim.SGD(net1.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()

    for i in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if not os.path.exists('./model'):
        os.mkdir('./model')
    torch.save(net1, './model/net1.pkl')
    torch.save(net1.state_dict(), './model/net1_params.pkl')

    plt.figure()
    plt.scatter(x.data.numpy(), y.data.numpy(), label='origin data')
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', label='predict')
    plt.text(1, -1, 'Loss=.%4f' % loss.data.numpy())
    plt.title('orgin train')
    plt.legend()
    plt.show()

def load_net(x, y):
    net2 = torch.load('./model/net1.pkl')
    prediction = net2(x)

    plt.figure()
    plt.scatter(x.data.numpy(), y.data.numpy(), label='origin data')
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', label='predict')
    plt.title('Restore net')
    plt.legend()
    plt.show()

def load_net_params(x, y):
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    net3.load_state_dict(torch.load('./model/net1_params.pkl'))
    prediction = net3(x)

    plt.figure()
    plt.scatter(x.data.numpy(), y.data.numpy(), label='origin data')
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', label='predict')
    plt.title('Restore parameters')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #fake data
    x = torch.unsqueeze(torch.linspace(-5, 5, 100), dim=1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())
    x, y = variable(x), variable(y)

    plt.figure()
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.show()

    save(x, y)
    load_net(x, y)
    load_net_params(x, y)
