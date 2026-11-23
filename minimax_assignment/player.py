#!/usr/bin/env python3
import gc
import time

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR

TIME_LIMIT = 0.045 

# Transposition-table flags: is the stored value exact, or only a bound?
EXACT, LOWER, UPPER = 0, 1, 2


class TimeUp(Exception):
    """Raised when the time budget for this move is used up."""


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        while True:
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()
        self.start_time = 0
        self.table = {}        # transposition table: state -> (depth, value, flag, best move)
        self.last_depth = 0    # deepest finished search last move (useful for testing)

    def player_loop(self):
        # The game tree is full of parent<->child reference cycles, so Python's
        # garbage collector can pause in the middle of a search. We switch it off
        # and clean up ourselves after each move, during the opponent's turn.
        gc.disable()

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()
            node = Node(message=msg, player=0)
            best_move = self.search_best_next_move(initial_tree_node=node)
            self.sender({"action": best_move, "search_time": None})
            gc.collect()

    
    def search_best_next_move(self, initial_tree_node):
        self.start_time = time.time()
        self.table = {}   # node.depth restarts at 0 every move, so old entries are invalid

        moves = initial_tree_node.compute_and_get_children()
        if not moves:
            return "stay"

        best_move = moves[0].move
        depth = 1
        try:
            while depth <= len(initial_tree_node.observations):
                best_move, scores = self.search_root(moves, depth)
                self.last_depth = depth
                # Move ordering: next depth starts with the moves that scored best
                moves = [m for _, m in sorted(zip(scores, moves),
                                              key=lambda pair: -pair[0])]
                depth += 1
        except TimeUp:
            pass   # use the best move from the last depth that finished

        return ACTION_TO_STR[best_move]

    def search_root(self, moves, depth):
        """Search every green move to the given depth; return the best one and all scores."""
        alpha, beta = float("-inf"), float("inf")
        best_value, best_move = float("-inf"), moves[0].move
        scores = []
        for child in moves:
            value = self.alphabeta(child, depth - 1, alpha, beta)
            scores.append(value)
            if value > best_value:
                best_value, best_move = value, child.move
            alpha = max(alpha, best_value)
        return best_move, scores

    def alphabeta(self, node, depth, alpha, beta):
        if time.time() - self.start_time > TIME_LIMIT:
            raise TimeUp()

        # Have we already searched this exact state at least this deep?
        key = self.state_key(node)
        stored = self.table.get(key)
        stored_move = None
        if stored is not None:
            stored_depth, stored_value, flag, stored_move = stored
            if stored_depth >= depth:
                if flag == EXACT:
                    return stored_value
                if flag == LOWER:
                    alpha = max(alpha, stored_value)
                if flag == UPPER:
                    beta = min(beta, stored_value)
                if alpha >= beta:
                    return stored_value

        # Cut-off depth reached or game over: estimate with the heuristic
        children = [] if depth == 0 else node.compute_and_get_children()
        if not children:
            value = self.heuristic(node)
            self.table[key] = (depth, value, EXACT, None)
            return value

        # Move ordering: try the best move found for this state earlier first
        if stored_move is not None:
            children = sorted(children, key=lambda c: c.move != stored_move)

        original_alpha, original_beta = alpha, beta
        best_move = None

        if node.state.get_player() == 0:          # green = MAX
            value = float("-inf")
            for child in children:
                child_value = self.alphabeta(child, depth - 1, alpha, beta)
                if child_value > value:
                    value, best_move = child_value, child.move
                alpha = max(alpha, value)
                if alpha >= beta:
                    break                         # red would never allow this branch
        else:                                     # red = MIN
            value = float("inf")
            for child in children:
                child_value = self.alphabeta(child, depth - 1, alpha, beta)
                if child_value < value:
                    value, best_move = child_value, child.move
                beta = min(beta, value)
                if alpha >= beta:
                    break                         # green would never allow this branch

        # After a cut-off the value is only a bound, not the exact value
        if value <= original_alpha:
            flag = UPPER
        elif value >= original_beta:
            flag = LOWER
        else:
            flag = EXACT
        self.table[key] = (depth, value, flag, best_move)
        return value

    @staticmethod
    def state_key(node):
        """Everything that decides a node's future, so equal keys mean equal states."""
        state = node.state
        green, red = state.get_player_scores()
        fish = frozenset((f, tuple(pos)) for f, pos in state.get_fish_positions().items())
        hooks = state.get_hook_positions()
        return (state.get_player(),      # who moves next
                node.depth,              # position in the fish observation sequence
                tuple(hooks[0]), tuple(hooks[1]),
                fish,                    # remaining fish and where they are
                state.get_caught(),      # fish on each line
                green - red)             # score difference

   
    def heuristic(self, node):
        state = node.state
        fish_scores = state.get_fish_scores()
        green, red = state.get_player_scores()

        # Points already scored matter most
        value = 10 * (green - red)

        # A fish on the line is almost a point
        green_caught, red_caught = state.get_caught()
        if green_caught is not None:
            value += fish_scores[green_caught]
        if red_caught is not None:
            value -= fish_scores[red_caught]

        # Bonus for being close to a valuable fish
        green_hook = state.get_hook_positions()[0]
        best = 0
        for f, pos in state.get_fish_positions().items():
            if fish_scores[f] > 0:
                best = max(best, fish_scores[f] / (self.distance(green_hook, pos) + 1))
        return value + best

    @staticmethod
    def distance(a, b):
        """Steps between two positions; the x-axis wraps around (board is 20 wide)."""
        dx = abs(a[0] - b[0])
        dx = min(dx, 20 - dx)
        return dx + abs(a[1] - b[1])
