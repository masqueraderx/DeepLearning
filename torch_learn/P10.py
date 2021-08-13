import torch
from torch.autograd import Variable
import torch.nn.functional as F
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    x = torch.linspace(-5, 5, 100)
    x = Variable(x)
    x_np = x.data.numpy()

    y_relu = F.relu(x).data.numpy()
    y_sigmoid = F.sigmoid(x).data.numpy()
    y_tanh = F.tanh(x).data.numpy()
    y_softplus = F.softplus(x).data.numpy()

    plt.figure(figsize=(10,5))
    plt.subplot(2,2,1)
    plt.plot(x_np, y_relu, label='relu')
    plt.legend(loc='best')

    plt.subplot(2,2,2)
    plt.plot(x_np, y_sigmoid, label='sigmoid')
    plt.legend(loc='best')

    plt.subplot(2,2,3)
    plt.plot(x_np, y_tanh, label='tanh')
    plt.legend(loc='best')

    plt.subplot(2,2,4)
    plt.plot(x_np, y_softplus, label='softplus')
    plt.legend(loc='best')

    plt.show()