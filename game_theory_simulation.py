import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


class Player:
    def __init__(self, name, initial_reward=0):
        self.name = name
        self.reward = initial_reward
        self.history = {}

    def update_history(self, other_player, other_move):
        if 'their_move' not in self.history[other_player.name].keys():
            self.history[other_player.name]['their_move'] = [other_move]
        else:
            self.history[other_player.name]['their_move'].append(other_move)

    def evaluate(self, other_player):
        self_move = self.history[other_player.name][self.name][-1]
        other_move = self.history[other_player.name]['their_move'][-1]

        if self_move == other_move == 'cooperate':
            self.reward += 2
        elif self_move == 'cooperate' and other_move == 'defect':
            self.reward -= 3
        elif self_move == 'defect' and other_move == 'cooperate':
            self.reward += 3
        elif self_move == other_move == 'defect':
            self.reward -= 2


class TitForTat(Player):
    def move(self, other_player):
        if other_player.name not in self.history.keys():
            self.history[other_player.name] = {}
            self.history[other_player.name][self.name] = ['cooperate']
            return 'cooperate'
        else:
            previous_other_move = self.history[other_player.name]['their_move'][-1]
            self.history[other_player.name][self.name].append(previous_other_move)
            return previous_other_move


class GrimTrigger(Player):
    def __init__(self, name, initial_reward=0):
        super().__init__(name, initial_reward)
        self.triggers = {}

    def move(self, other_player):
        if other_player.name not in self.history.keys():
            self.history[other_player.name] = {}
            self.history[other_player.name][self.name] = ['cooperate']
            self.triggers[other_player.name] = False
            return 'cooperate'
        else:
            previous_other_move = self.history[other_player.name]['their_move'][-1]
            if previous_other_move == 'defect':
                self.triggers[other_player.name] = True
            if self.triggers[other_player.name]:
                self.history[other_player.name][self.name].append('defect')
                return 'defect'
            self.history[other_player.name][self.name].append('cooperate')
            return 'cooperate'


class Holy(Player):
    def move(self, other_player):
        if other_player.name not in self.history.keys():
            self.history[other_player.name] = {}
            self.history[other_player.name][self.name] = ['cooperate']
        else:
            self.history[other_player.name][self.name].append('cooperate')
        return 'cooperate'


class Evil(Player):
    def move(self, other_player):
        if other_player.name not in self.history.keys():
            self.history[other_player.name] = {}
            self.history[other_player.name][self.name] = ['defect']
        else:
            self.history[other_player.name][self.name].append('defect')
        return 'defect'


class Chaotic(Player):
    def move(self, other_player, prob=0.5):
        choice = np.random.choice(['cooperate', 'defect'], p=[prob, 1 - prob])
        if other_player.name not in self.history.keys():
            self.history[other_player.name] = {}
            self.history[other_player.name][self.name] = [choice]
        else:
            self.history[other_player.name][self.name].append(choice)
        return choice


class Tribal(Player):
    def move(self, other_player):
        if other_player.name not in self.history.keys():
            self.history[other_player.name] = {}
            if type(other_player) == type(self):
                self.history[other_player.name][self.name] = ['cooperate']
                return 'cooperate'
            else:
                self.history[other_player.name][self.name] = ['defect']
                return 'defect'
        else:
            if type(other_player) == type(self):
                self.history[other_player.name][self.name].append('cooperate')
                return 'cooperate'
            else:
                self.history[other_player.name][self.name].append('defect')
                return 'defect'


class Pavlov(Player):
    def move(self, other_player):
        if other_player.name not in self.history.keys():
            self.history[other_player.name] = {}
            self.history[other_player.name][self.name] = ['cooperate']
            return 'cooperate'
        else:
            previous_other_move = self.history[other_player.name]['their_move'][-1]
            previous_self_move = self.history[other_player.name][self.name][-1]
            if previous_other_move == previous_self_move:
                self.history[other_player.name][self.name].append('cooperate')
                return 'cooperate'
            self.history[other_player.name][self.name].append('defect')
            return 'defect'


def play_game(players_):
    p1 = players_[0]
    p2 = players_[1]

    move1 = p1.move(p2)
    move2 = p2.move(p1)

    p1.update_history(p2, move2)
    p2.update_history(p1, move1)

    p1.evaluate(p2)
    p2.evaluate(p1)


def select_game_players(players_):
    pairs_ = []
    selected_individuals = []
    while True:
        possible_partners = list(set(range(len(players_))) - set(selected_individuals))
        if len(possible_partners) == 0:
            break
        j, k = np.random.choice(possible_partners, 2, replace=False)
        pairs_.append((players_[j], players_[k]))
        selected_individuals.extend([j, k])

    return pairs_


if __name__ == '__main__':
    n_games = 1000
    n_players = 10000

    players = []
    strategy_groups = {'TitForTat': [],
                       'GrimTrigger': [],
                       'Holy': [],
                       'Evil': [],
                       'Chaotic': [],
                       'Tribal': [],
                       'Pavlov': []}
    probabilities = [0.5,
                     0.2,
                     0.03,
                     0.01,
                     0.1,
                     0.01,
                     0.15]
    indices_chosen = []
    for i in range(n_players):
        strategy_options = [TitForTat('TitForTat_{}'.format(i)),
                            GrimTrigger('GrimTrigger_{}'.format(i)),
                            Holy('Holy_{}'.format(i)),
                            Evil('Evil_{}'.format(i)),
                            Chaotic('Chaotic_{}'.format(i)),
                            Tribal('Tribal_{}'.format(i)),
                            Pavlov('Pavlov_{}'.format(i))]

        index = np.random.choice(list(range(len(strategy_options))), p=probabilities)

        players.append(strategy_options[index])
        indices_chosen.append(index)
        strategy_groups[list(strategy_groups.keys())[index]].append(players[-1])

    indices_chosen = set(indices_chosen)

    for i in range(n_games):
        print("PLAYING GAME {}/{}".format(i + 1, n_games))
        t1 = time.time()
        pairs = select_game_players(players)
        for pair in pairs:
            play_game(pair)
        t2 = time.time()
        print("TIME: {} seconds".format(t2 - t1))

    print()
    print("PLAYER NAMES: {}".format([player.name for player in players]))
    print("FINAL REWARDS: {}".format([player.reward for player in players]))
    print()

    plt.figure(figsize=(15, 10))
    plt.title('Distributions')
    plt.xlabel('Reward')
    plt.ylabel('Count')

    rewards_dict = {}
    for key, value in strategy_groups.items():
        if len(value) == 0:
            continue
        rewards = []
        for player in value:
            rewards.append(player.reward)

        rewards_dict[key] = rewards
        print("{} Reward: Average: {}, Std.: {}".format(key, np.mean(rewards), np.std(rewards)))

        plt.hist(rewards)

    types = list(strategy_groups.keys())
    plt.legend([types[i] for i in indices_chosen])

    p_df = pd.DataFrame()
    t_df = pd.DataFrame()
    mask = np.zeros((len(indices_chosen), len(indices_chosen)), dtype=bool)
    for i, column0 in enumerate(rewards_dict.keys()):
        for ii, column1 in enumerate(list(rewards_dict.keys())[:i]):
            mask[ii, i] = True
            t, p = ttest_ind(rewards_dict[column0], rewards_dict[column1], axis=0, equal_var=False,
                             nan_policy='propagate', alternative='two-sided')

            p_df.loc[column0, column1] = p_df.loc[column1, column0] = p
            t_df.loc[column0, column1] = t_df.loc[column1, column0] = t

    p_df = p_df[rewards_dict.keys()].reindex(rewards_dict.keys())
    t_df = t_df[rewards_dict.keys()].reindex(rewards_dict.keys())

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    sns.heatmap(p_df, ax=ax1, annot=True, mask=mask, cmap='binary')
    ax1.set_title('p-values')
    sns.heatmap(t_df, ax=ax2, annot=True, mask=mask, cmap='binary')
    ax2.set_title('t-values')

    plt.show()
