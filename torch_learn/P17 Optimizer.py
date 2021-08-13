import torch
from torch.autograd import variable
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12


class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        return self.output(x)


if __name__ == '__main__':
    x = torch.unsqueeze(torch.linspace(-5, 5, 1000), dim=1)
    y = x.pow(2) + 2 * torch.rand(x.size())

    dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(dataset=dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=1)

    plt.figure()
    plt.scatter(x.numpy(), y.numpy())
    plt.title('Origin data')
    plt.show()

    SGD_net = Net(1, 20, 1)
    Adam_net = Net(1, 20, 1)
    RMSprop_net = Net(1, 20, 1)
    Momentum_net = Net(1, 20, 1)
    nets = [SGD_net, Adam_net, RMSprop_net, Momentum_net]

    SGD_opt = torch.optim.SGD(SGD_net.parameters(), lr=LR)
    Adam_opt = torch.optim.Adam(Adam_net.parameters(), lr=LR, betas=(0.9, 0.999))
    RMSprop_opt = torch.optim.RMSprop(RMSprop_net.parameters(), lr=LR, alpha=0.9)
    Momentum_opt = torch.optim.SGD(Momentum_net.parameters(), lr=LR, momentum=0.8)
    optimizers = [SGD_opt, Adam_opt, RMSprop_opt, Momentum_opt]

    loss_func = torch.nn.MSELoss()
    loss_his = [[], [], [], []]
    labels = ['SGD', 'Adam', 'RMSprop', 'Momentum']

    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader):
            b_x, b_y = variable(batch_x), variable(batch_y)
            for cur_net, cur_opt, cur_loss in zip(nets, optimizers, loss_his):
                prediction = cur_net(b_x)
                loss = loss_func(prediction, b_y)
                cur_opt.zero_grad()
                loss.backward()
                cur_opt.step()
                cur_loss.append(loss.data.numpy())
                if step % 10 == 0:
                    print('Epoch: ', epoch, '| Step: ', step, '| Loss: ', loss.data.numpy())

    plt.figure()
    for i, opt_loss in enumerate(loss_his):
        plt.plot(range(len(opt_loss)), opt_loss, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.show()




