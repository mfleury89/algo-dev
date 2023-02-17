from random import randint, shuffle


def is_even(p):
    p[p.index(0)] = 16

    n = len(p)
    sign = 0
    s = -1
    while s < n - 1:
        u = s + 1
        t = u
        while u < n - 1:
            u += 1
            if p[u] < p[t]:
                t = u
        temp = p[s + 1]
        p[s + 1] = p[t]
        p[t] = temp
        if t != s + 1:
            sign = 1 - sign
        s += 1

    if sign == 0:
        return True
    else:
        return False


def get_2d_index(list_2d, element):
    for row in range(len(list_2d)):
        if element in list_2d[row]:
            return tuple([row, list_2d[row].index(element)])


def find_numbers_to_move(list_2d):
    flattened_position = list_2d[0] + list_2d[1] + list_2d[2] + list_2d[3]

    change_mask = []
    for i, num in enumerate(flattened_position):
        if num == i + 1 or num == 0:
            change_mask.append(0)
        else:
            change_mask.append(1)

    if sum(change_mask) == 0:
        return []

    change_mask = [change_mask[:4], change_mask[4:8], change_mask[8:12], change_mask[12:]]

    numbers_to_move = []
    for i, row in enumerate(change_mask):
        if 1 in row:
            for j, num in enumerate(row):
                if num == 1:
                    numbers_to_move.append(list_2d[i][j])

    return numbers_to_move


def show_position(list_2d):
    for row in list_2d:
        print(row)
    print()

    for i in range(50000000):
        pass


def move_to_position(list_2d, moves, element, location, hold, reverse_locations, watch_solution):
    index = get_2d_index(list_2d, element)

    while index != location:
        moved = False
        i = randint(1, 2)  # add random order of movement dimensions to avoid getting stuck in endless loop
        if i == 1:
            while index[0] < location[0] and (index[0] + 1, index[1]) not in hold:
                reverse_locations.append((index[0], index[1]))
                list_2d[index[0]][index[1]] = list_2d[index[0] + 1][index[1]]
                list_2d[index[0] + 1][index[1]] = element
                moves.append(list_2d[index[0]][index[1]])
                index = get_2d_index(list_2d, element)
                if watch_solution:
                    show_position(list_2d)
                moved = True
            while index[0] > location[0] and (index[0] - 1, index[1]) not in hold:
                reverse_locations.append((index[0], index[1]))
                list_2d[index[0]][index[1]] = list_2d[index[0] - 1][index[1]]
                list_2d[index[0] - 1][index[1]] = element
                moves.append(list_2d[index[0]][index[1]])
                index = get_2d_index(list_2d, element)
                if watch_solution:
                    show_position(list_2d)
                moved = True
            while index[1] < location[1] and (index[0], index[1] + 1) not in hold:
                reverse_locations.append((index[0], index[1]))
                list_2d[index[0]][index[1]] = list_2d[index[0]][index[1] + 1]
                list_2d[index[0]][index[1] + 1] = element
                moves.append(list_2d[index[0]][index[1]])
                index = get_2d_index(list_2d, element)
                if watch_solution:
                    show_position(list_2d)
                moved = True
            while index[1] > location[1] and (index[0], index[1] - 1) not in hold:
                reverse_locations.append((index[0], index[1]))
                list_2d[index[0]][index[1]] = list_2d[index[0]][index[1] - 1]
                list_2d[index[0]][index[1] - 1] = element
                moves.append(list_2d[index[0]][index[1]])
                index = get_2d_index(list_2d, element)
                if watch_solution:
                    show_position(list_2d)
                moved = True
        if i == 2:
            while index[1] < location[1] and (index[0], index[1] + 1) not in hold:
                reverse_locations.append((index[0], index[1]))
                list_2d[index[0]][index[1]] = list_2d[index[0]][index[1] + 1]
                list_2d[index[0]][index[1] + 1] = element
                moves.append(list_2d[index[0]][index[1]])
                index = get_2d_index(list_2d, element)
                if watch_solution:
                    show_position(list_2d)
                moved = True
            while index[1] > location[1] and (index[0], index[1] - 1) not in hold:
                reverse_locations.append((index[0], index[1]))
                list_2d[index[0]][index[1]] = list_2d[index[0]][index[1] - 1]
                list_2d[index[0]][index[1] - 1] = element
                moves.append(list_2d[index[0]][index[1]])
                index = get_2d_index(list_2d, element)
                if watch_solution:
                    show_position(list_2d)
                moved = True
            while index[0] < location[0] and (index[0] + 1, index[1]) not in hold:
                reverse_locations.append((index[0], index[1]))
                list_2d[index[0]][index[1]] = list_2d[index[0] + 1][index[1]]
                list_2d[index[0] + 1][index[1]] = element
                moves.append(list_2d[index[0]][index[1]])
                index = get_2d_index(list_2d, element)
                if watch_solution:
                    show_position(list_2d)
                moved = True
            while index[0] > location[0] and (index[0] - 1, index[1]) not in hold:
                reverse_locations.append((index[0], index[1]))
                list_2d[index[0]][index[1]] = list_2d[index[0] - 1][index[1]]
                list_2d[index[0] - 1][index[1]] = element
                moves.append(list_2d[index[0]][index[1]])
                index = get_2d_index(list_2d, element)
                if watch_solution:
                    show_position(list_2d)
                moved = True

        if not moved:
            if index[0] != location[0]:
                i = randint(0, 1)  # add random order of movement dimensions to avoid getting stuck in endless loop
                if i == 0:
                    if index[1] - 1 >= 0 and (index[0], index[1] - 1) not in hold:
                        reverse_locations.append((index[0], index[1]))
                        list_2d[index[0]][index[1]] = list_2d[index[0]][index[1] - 1]
                        list_2d[index[0]][index[1] - 1] = element
                        moves.append(list_2d[index[0]][index[1]])
                        index = get_2d_index(list_2d, element)
                        if watch_solution:
                            show_position(list_2d)
                else:
                    if index[1] + 1 <= 3 and (index[0], index[1] + 1) not in hold:
                        reverse_locations.append((index[0], index[1]))
                        list_2d[index[0]][index[1]] = list_2d[index[0]][index[1] + 1]
                        list_2d[index[0]][index[1] + 1] = element
                        moves.append(list_2d[index[0]][index[1]])
                        index = get_2d_index(list_2d, element)
                        if watch_solution:
                            show_position(list_2d)

            if index[1] != location[1]:
                i = randint(0, 1)
                if i == 0:
                    if index[0] - 1 >= 0 and (index[0] - 1, index[1]) not in hold:
                        reverse_locations.append((index[0], index[1]))
                        list_2d[index[0]][index[1]] = list_2d[index[0] - 1][index[1]]
                        list_2d[index[0] - 1][index[1]] = element
                        moves.append(list_2d[index[0]][index[1]])
                        index = get_2d_index(list_2d, element)
                        if watch_solution:
                            show_position(list_2d)
                else:
                    if index[0] + 1 <= 3 and (index[0] + 1, index[1]) not in hold:
                        reverse_locations.append((index[0], index[1]))
                        list_2d[index[0]][index[1]] = list_2d[index[0] + 1][index[1]]
                        list_2d[index[0] + 1][index[1]] = element
                        moves.append(list_2d[index[0]][index[1]])
                        index = get_2d_index(list_2d, element)
                        if watch_solution:
                            show_position(list_2d)


def shift_non_zero_element(list_2d, moves, element, location, hold, reverse_locations, watch_solution):
    index = get_2d_index(list_2d, element)
    while index != location:
        move_to_position(list_2d, moves, 0, location, hold + [index], reverse_locations, watch_solution)  # move 0 to location to approach index from location
        index = get_2d_index(list_2d, element)
        move_to_position(list_2d, moves, 0, index, hold, reverse_locations, watch_solution) # move 0 to index to move element in direction of location
        index = get_2d_index(list_2d, element)

        if index == location:
            break


def three_cycle(list_2d, moves, hold, reverse_locations, watch_solution):
    move_to_position(list_2d, moves, 0, (1, 1), hold, reverse_locations, watch_solution)
    move_to_position(list_2d, moves, 0, (0, 1), [], [], watch_solution)
    move_to_position(list_2d, moves, 0, (0, 0), [], [], watch_solution)
    move_to_position(list_2d, moves, 0, (1, 0), [], [], watch_solution)
    move_to_position(list_2d, moves, 0, (1, 1), [], [], watch_solution)


def solution(position, watch_solution=False):
    moves = []
    hold = []
    position = [position[:4], position[4:8], position[8:12], position[12:]]
    show_position(position)
    move_to_position(position, moves, 0, (3, 3), hold, [], watch_solution)  # move zero to its final position,
                                                            # so that if numbers_to_move == numbers_to_move_sorted
                                                            # then the puzzle is already solved

    flattened_position = position[0] + position[1] + position[2] + position[3]
    if not is_even(flattened_position.copy()): # now that all remaining moves are essentially three-cycles and a certain number
                                     # of transpositions have occurred to move 0, check if permutation is even or not
                                     # (i.e., if it can be solved from here using three-cycles);
                                     # if it took an odd number of transpositions to move the 0, then the whole
                                     # permutation is odd and the puzzle is not solvable, but if it took an even
                                     # number of transpositions to move the 0 then the whole permutation will be even
        print("Permutation is odd, thus the puzzle is not solvable.")
        return None

    reverse_locations = []
    while True:
        numbers_to_move = find_numbers_to_move(position)

        if not numbers_to_move:
            show_position(position)
            return moves

        numbers_to_move_sorted = sorted(numbers_to_move)  # Sort numbers_to_move, and figure out from this sorting
                                                          # which three-cycle to do next
        first_number = numbers_to_move_sorted[0]
        second_number = numbers_to_move[0]
        for k in range(len(numbers_to_move) - 1, 0, -1):
            if k != numbers_to_move.index(first_number):
                third_number = numbers_to_move[k]
                break

        triple = [first_number, second_number, third_number]

        for j, m in enumerate(triple):
            if j % 3 == 0:
                new_position = (0, 0)
            elif j % 3 == 1:
                new_position = (0, 1)
            else:  # j % 3 == 2
                new_position = (1, 0)

            shift_non_zero_element(position, moves, m, new_position, hold, reverse_locations, watch_solution)
            hold.append(new_position)

            if j % 3 == 2:
                three_cycle(position, moves, hold, reverse_locations, watch_solution)
                hold.clear()
                reverse_locations.reverse()
                for location in reverse_locations:
                    move_to_position(position, moves, 0, location, hold, [], watch_solution)
                reverse_locations.clear()


if __name__ == '__main__':
    for _ in range(1):
        board = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
        shuffle(board)
        moves_ = solution(board, watch_solution=False)
        if moves_ is not None:
            print("Moves:", moves_)
            print("Solution took {} moves.".format(len(moves_)))
            print()
