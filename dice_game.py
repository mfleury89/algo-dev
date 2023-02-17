from itertools import combinations


def count_wins(dice1, dice2):
    assert len(dice1) == 6 and len(dice2) == 6
    dice1_wins, dice2_wins = 0, 0

    for i in dice1:
        for j in dice2:
            if i > j:
                dice1_wins += 1
            if j > i:
                dice2_wins += 1

    return dice1_wins, dice2_wins


def find_the_best_dice(dices, loses_to):
    assert all(len(dice) == 6 for dice in dices)

    for i in range(len(dices)):
        loses_to[i] = []
    pairs = combinations(range(len(dices)), 2)
    for pair in pairs:
        wins = count_wins(dices[pair[0]], dices[pair[1]])
        if wins[0] < wins[1]:
            loses_to[pair[0]].append((pair[1], wins[1]))
        if wins[1] < wins[0]:
            loses_to[pair[1]].append((pair[0], wins[0]))

    for die, losses in loses_to.items():
        if len(losses) == 0:
            return die
    return -1


def compute_strategy(dices):
    assert all(len(dice) == 6 for dice in dices)

    strategy = dict()
    loses_to = {}
    best_die = find_the_best_dice(dices, loses_to)
    if best_die != -1:
        strategy["choose_first"] = True
        strategy["first_dice"] = best_die
    else:
        strategy["choose_first"] = False
        for i in range(len(dices)):
            betters = loses_to[i]
            best_die = -1
            best_wins = 0
            for better in betters:
                if better[1] > best_wins:
                    best_die = better[0]
                    best_wins = better[1]
            strategy[i] = best_die

    return strategy

if __name__ == '__main__':
    print(count_wins([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]))  # (15, 15)
    print(count_wins([1, 1, 6, 6, 8, 8], [2, 2, 4, 4, 9, 9]))  # (16, 20)

    print(find_the_best_dice([[1, 1, 6, 6, 8, 8], [2, 2, 4, 4, 9, 9], [3, 3, 5, 5, 7, 7]], {}))  # -1
    print(find_the_best_dice([[1, 1, 2, 4, 5, 7], [1, 2, 2, 3, 4, 7], [1, 2, 3, 4, 5, 6]], {}))  # 2
    print(find_the_best_dice([[3, 3, 3, 3, 3, 3], [6, 6, 2, 2, 2, 2], [4, 4, 4, 4, 0, 0], [5, 5, 5, 1, 1, 1]], {}))  # -1

    print(compute_strategy([[1, 1, 4, 6, 7, 8], [2, 2, 2, 6, 7, 7], [3, 3, 3, 5, 5, 8]]))  # {'choose_first': False, 0: 1, 1: 2, 2: 0}
    print(compute_strategy([[4, 4, 4, 4, 0, 0], [7, 7, 3, 3, 3, 3], [6, 6, 2, 2, 2, 2], [5, 5, 5, 1, 1, 1]]))  # {'choose_first': True, 'first_dice': 1}