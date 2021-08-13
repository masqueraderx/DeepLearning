import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

INPUT_SIZE = 1
TIME_STEP = 10
LR = 0.01

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        # x: (batch_size, time_step, input_size)
        # h_state: (n_layers, batch_size, hidden_size)
        # r_out: (batch_size, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        # or
        # outs = self.out(r_out.view(-1, 32))
        # outs = outs.view(-1, TIME_STEP, 1)
        outs = []
        for step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, step, :]))
        return torch.stack(outs, dim=1), h_state


if __name__ == '__main__':

    rnn = RNN()
    print(rnn)

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)

    h_state = None
    plt.figure(1, figsize=(12, 5))
    plt.ion()

    for step in range(100):
        start, end = step * np.pi, (step + 1) * np.pi  # time range
        steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)

        x_np = np.sin(steps)
        y_np = np.cos(steps)
        x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
        y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

        r_out, h_state = rnn(x, h_state)
        h_state = h_state.data

        loss = loss_func(r_out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        plt.plot(steps, y_np.flatten(), 'r-', label='true label')
        plt.plot(steps, r_out.data.numpy().flatten(), 'b-', label='prediction')
        plt.legend(loc='best')
        plt.draw()
        plt.pause(0.05)

    plt.ioff()
    plt.show()






