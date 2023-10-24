import pandas as pd
import matplotlib.pyplot as plt

types = [
    'Acrobot-v1',
    'BipedalWaler-v3',
    'CartPole-v1',
    'LunarLander-v2',
    'LunarLanderContinuous-v2',
    'MountainCarContinuous-v0',
    'Pendulum-v1'
    ]

for type in types:

    df = pd.read_csv('train_history - {0}/loss_logs.csv'.format(type))

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['actor loss'], label='Actor Loss')
    plt.plot(df.index, df['critic loss'], label='Critic Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('{0} loss_logs'.format(type))
    plt.legend()
    plt.show()

for type in types:

    df = pd.read_csv('train_history - {0}/score_logs.csv'.format(type))

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['scores'], label='Scores')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('{0} score_logs'.format(type))
    plt.legend()
    plt.show()