#!/usr/bin/env python3
import random

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def search_best_next_move(self, initial_tree_node):
        """
        Use minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        DEPTH = 3

        best_value, best_move = float("-inf"), 0
        for child in initial_tree_node.compute_and_get_children():
            v = self.minimax(child, DEPTH - 1)
            if v > best_value:
                best_value, best_move = v, child.move
        return ACTION_TO_STR[best_move]

    def minimax(self, node, depth):
        if depth == 0:
            return self.heuristic(node)
        children = node.compute_and_get_children()
        if not children:                      # terminal state
            return self.heuristic(node)
        if node.state.get_player() == 0:      # green = MAX
            return max(self.minimax(c, depth - 1) for c in children)
        else:                                 # red = MIN
            return min(self.minimax(c, depth - 1) for c in children)

    def heuristic(self, node):
        state = node.state
        green, red = state.get_player_scores()
        value = 10 * (green - red)

        hooks = state.get_hook_positions()
        fish_pos = state.get_fish_positions()
        fish_scores = state.get_fish_scores()
        caught = state.get_caught()

        # Fish already on a hook almost counts as points
        if caught[0] is not None:
            value += fish_scores[caught[0]]
        if caught[1] is not None:
            value -= fish_scores[caught[1]]

        # Reward being close to valuable fish (x wraps around, board is 20 wide)
        def dist(a, b):
            dx = abs(a[0] - b[0])
            dx = min(dx, 20 - dx)
            return dx + abs(a[1] - b[1])

        best = 0
        for f, pos in fish_pos.items():
            if fish_scores[f] > 0:
                best = max(best, fish_scores[f] / (dist(hooks[0], pos) + 1))
        value += best

        return value

        
        
        
        
        
        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING MINIMAX ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!
