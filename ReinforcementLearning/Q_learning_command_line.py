import numpy as np
import time
import pandas as pd

N_STATES = 6
ACTIONS = ['left', 'right']
EPSILON = 0.9   # Greedy rate
ALPHA = 0.1     # learning rate
GAMMA = 0.9     # decay parameter
MAX_EPISODES = 10
FRESH_TIME = 0.3

np.random.seed(2021)

def create_q_table(n_states, actions):
    return pd.DataFrame(np.zeros((n_states, len(actions))), columns=actions)

def choose_action(state, q_table):
    state_action = q_table.iloc[state, :]
    if np.random.uniform() > EPSILON or (state_action == 0).all():
        action = np.random.choice(ACTIONS)
    else:
        action = state_action.idxmax()
    return action

def move(S, A):
    if A == 'right':
        if S == N_STATES-1:
            R = 1
            S_ = 'terminal'
        else:
            R = 0
            S_ = S + 1
    else:
        R = 0
        if S == 0:
            S_ = 0
        else:
            S_ = S - 1
    return S_, R

def env_update(S, round, step_count):
    env_list = ['-'] * N_STATES + ['T']
    if S == 'terminal':
        print(' Round %s takes %s steps' % (round, step_count))
        time.sleep(2)
    else:
        env_list[S] = 'o'
        print('\r{}'.format(''.join(env_list)), end='')
        time.sleep(FRESH_TIME)

def rl():
    q_table = create_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        is_terminal = False
        S = 0
        step_count = 0
        env_update(S, episode, step_count)
        while not is_terminal:
            A = choose_action(S, q_table)
            S_, R = move(S, A)
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminal = True
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            S = S_
            env_update(S, episode, step_count+1)
            step_count += 1
    return q_table

if __name__ == '__main__':
    q_table = rl()
    print('q_table is: \n')
    print(q_table)







