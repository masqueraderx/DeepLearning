import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

EPOCH = 10
BATCH_SIZE = 64
LR = 0.01
DOWNLOAD_MNIST = False
N_TEST_IMG = 5

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


if __name__ == '__main__':
    train_data = torchvision.datasets.MNIST(
        root='./MNIST',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST
    )
    print(train_data.data == train_data.train_data)

    plt.figure()
    plt.imshow(train_data.data[0].numpy(), cmap='gray')
    plt.title('%i' % train_data.targets[0].numpy())
    plt.show()

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
    test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / float(255)
    test_y = torch.unsqueeze(test_data.targets, dim=1).type(torch.LongTensor)[:2000]

    autoencoder = AutoEncoder()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    loss_func = nn.MSELoss()

    # initialize figure
    f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
    plt.ion()  # continuously plot

    # original data (first row) for viewing
    view_data = train_data.data[:N_TEST_IMG].view(-1, 28 * 28).type(torch.FloatTensor) / 255.
    for i in range(N_TEST_IMG):
        a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())

    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            b_x = batch_x.view(-1, 28 * 28)
            b_y = batch_x.view(-1, 28 * 28)
            encoded, decoded = autoencoder(b_x)
            loss = loss_func(decoded, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 500 == 0 and epoch in [0, 5, EPOCH-1]:
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

                # plotting decoded image (second row)
                _, decoded_data = autoencoder(view_data)
                for i in range(N_TEST_IMG):
                    a[1][i].clear()
                    a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                    a[1][i].set_xticks(())
                    a[1][i].set_yticks(())
                plt.draw()
                plt.pause(0.05)
    plt.ioff()
    plt.show()


    # visualize in 3D plot
    view_data = train_data.data[:200].view(-1, 28 * 28).type(torch.FloatTensor) / 255.
    encoded_data, _ = autoencoder(view_data)
    fig = plt.figure(2)
    ax = Axes3D(fig)
    X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
    values = train_data.targets[:200].numpy()
    for x, y, z, s in zip(X, Y, Z, values):
        c = cm.rainbow(int(255 * s / 9))
        ax.text(x, y, z, s, backgroundcolor=c)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())
    plt.show()





