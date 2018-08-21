
from sample_players import DataPlayer
from mcts import MctsNode
from isolation.isolation import _WIDTH, _HEIGHT, _SIZE
import math
import time
import random


_CORNERS = [0, 10, 104, 114]
_WALLS = list(range(1, 10)) + list(range(105, 114)) + [i * (_WIDTH + 2) for i in range(1, _HEIGHT - 1)] + [i * (_WIDTH + 2) + (_WIDTH - 1) for i in range(1, _HEIGHT - 1)]
_CENTER = 57

class CustomPlayer_Minimax(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        if state.ply_count == 0:
            # center preferred
            self.queue.put(_CENTER)
        elif state.ply_count == 1:
            # open wide preferred
            opens = [i for i in state.actions() if i not in _CORNERS and i not in _WALLS and i != 57]
            self.queue.put(random.choice(opens))
        else:
            move, depth = self.iterative_deepening(state, depth_limit=6)
            self.queue.put(move)

    def iterative_deepening(self, state, depth_limit):
        start = int(time.time() * 1000)
        best_move = None
        depth_iter = 0
        for depth in range(1, depth_limit+1):
            best_move, score = self.alpha_beta_search(state, depth, start)
            depth_iter = depth
            if self.time_exceeded(start): break
        return best_move, depth_iter

    def alpha_beta_search(self, state, depth, start):
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        for a in state.actions():
            v = self.min_value(state.result(a), alpha, beta, depth - 1, start)
            alpha = max(alpha, v)
            if v > best_score:
                best_score = v
                best_move = a
            if self.time_exceeded(start): break
        if not best_move:
            best_move = random.choice(state.actions())
        return best_move, best_score
    
    def min_value(self, state, alpha, beta, depth, start):
        if state.terminal_test(): return state.utility(self.player_id)
        if depth <= 0: return self.h_score_mixture(state)
        v = float("inf")
        for a in state.actions():
            v = min(v, self.max_value(state.result(a), alpha, beta, depth - 1, start))
            if v <= alpha:
                return v
            beta = min(beta, v)
            if self.time_exceeded(start): break
        return v

    def max_value(self, state, alpha, beta, depth, start):
        if state.terminal_test(): return state.utility(self.player_id)
        if depth <= 0: return self.h_score_mixture(state)
        v = float("-inf")
        for a in state.actions():
            v = max(v, self.min_value(state.result(a), alpha, beta, depth - 1, start))
            if v >= beta:
                return v
            alpha = max(alpha, v)
            if self.time_exceeded(start): break
        return v
    
    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)
    
    # center
    def h_score_close_to_center(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        dis = self.distance(own_loc, _CENTER)
        if dis < 2:
            return 2 * len(own_liberties) - len(opp_liberties)
        elif dis == 2:
            return 1.5 * len(own_liberties) - len(opp_liberties)
        else:
            return len(own_liberties) - len(opp_liberties)
    
    def h_score_away_from_center(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        dis = self.distance(own_loc, _CENTER)
        if dis < 2:
            return len(own_liberties) - len(opp_liberties)
        elif dis == 2:
            return 1.5 * len(own_liberties) - len(opp_liberties)
        else:
            return 2 * len(own_liberties) - len(opp_liberties)
    
    # walls
    def h_score_close_to_walls(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        dis = self.distance_to_walls(state)
        if dis < 2:
            return 2 * len(own_liberties) - len(opp_liberties)
        elif dis == 2:
            return 1.5 * len(own_liberties) - len(opp_liberties)
        else:
            return len(own_liberties) - len(opp_liberties)
    
    def h_score_away_from_walls(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        dis = self.distance_to_walls(state)
        if dis < 2:
            return len(own_liberties) - len(opp_liberties)
        elif dis == 2:
            return 1.5 * len(own_liberties) - len(opp_liberties)
        else:
            return 2 * len(own_liberties) - len(opp_liberties)
    
    # corners
    def h_score_close_to_corners(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        dis = self.distance_to_corners(state)
        if dis < 2:
            return 2 * len(own_liberties) - len(opp_liberties)
        elif dis == 2:
            return 1.5 * len(own_liberties) - len(opp_liberties)
        else:
            return len(own_liberties) - len(opp_liberties)
        
    def h_score_away_from_corners(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        dis = self.distance_to_corners(state)
        if dis < 2:
            return len(own_liberties) - len(opp_liberties)
        elif dis == 2:
            return 1.5 * len(own_liberties) - len(opp_liberties)
        else:
            return 2 * len(own_liberties) - len(opp_liberties)
    
    # open wide
    def h_score_in_open_wide(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        dis = self.distance(own_loc, _CENTER) * self.distance_to_corners(state) * self.distance_to_walls(state)
        if dis < 8:
            return 2 * len(own_liberties) - len(opp_liberties)
        elif dis >= 8 and dis < 27:
            return 1.5 * len(own_liberties) - len(opp_liberties)
        else:
            return len(own_liberties) - len(opp_liberties)
    
    def h_score_out_open_wide(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        dis = self.distance(own_loc, _CENTER) * self.distance_to_corners(state) * self.distance_to_walls(state)
        if dis < 8:
            return len(own_liberties) - len(opp_liberties)
        elif dis >= 8 and dis < 27:
            return 1.5 * len(own_liberties) - len(opp_liberties)
        else:
            return 2 * len(own_liberties) - len(opp_liberties)
    
    # opponent
    def h_score_close_to_opponent(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        dis = self.distance(own_loc, opp_loc)
        if dis < 2:
            return 2 * len(own_liberties) - len(opp_liberties)
        elif dis == 2:
            return 1.5 * len(own_liberties) - len(opp_liberties)
        else:
            return len(own_liberties) - len(opp_liberties)
    
    def h_score_away_from_opponent(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        dis = self.distance(own_loc, opp_loc)
        if dis < 2:
            return len(own_liberties) - len(opp_liberties)
        elif dis == 2:
            return 1.5 * len(own_liberties) - len(opp_liberties)
        else:
            return 2 * len(own_liberties) - len(opp_liberties)
    
    # mixture
    def h_score_mixture(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        dis_opp = self.distance(own_loc, opp_loc)
        dis_center = self.distance(own_loc, _CENTER)
        dis_walls = self.distance_to_walls(state)
        dis_corners = self.distance_to_corners(state)
        k = 4
        k = self.h_score_mixture_pos_helper(dis_opp, k)
        k = self.h_score_mixture_pos_helper(dis_center, k)
        k = self.h_score_mixture_pos_helper(dis_walls, k)
        k = self.h_score_mixture_neg_helper(dis_corners, k)
        return k * len(own_liberties) - 4 * len(opp_liberties)
    
    def h_score_mixture_1(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        dis_opp = self.distance(own_loc, opp_loc)
        dis_center = self.distance(own_loc, _CENTER)
        dis_walls = self.distance_to_walls(state)
        dis_corners = self.distance_to_corners(state)
        if dis_opp < 2 and dis_center < 2 and dis_walls < 2 and dis_corners >= 4:
            return 2 * len(own_liberties) - len(opp_liberties)
        elif dis_opp <= 3 and dis_center <= 3 and dis_walls <= 3 and dis_corners >= 2:
            return 1.5 * len(own_liberties) - len(opp_liberties)
        else:
            return len(own_liberties) - len(opp_liberties)
    
    def h_score_mixture_2(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        dis_opp = self.distance(own_loc, opp_loc)
        dis_center = self.distance(own_loc, _CENTER)
        dis_walls = self.distance_to_walls(state)
        dis_corners = self.distance_to_corners(state)
        if dis_opp < 2 and dis_center < 2 and dis_walls < 2 and dis_corners > 2:
            return 2 * len(own_liberties) - len(opp_liberties)
        else:
            return len(own_liberties) - len(opp_liberties)
    
    def h_score_mixture_3(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        dis_opp = self.distance(own_loc, opp_loc)
        dis_center = self.distance(own_loc, _CENTER)
        dis_walls = self.distance_to_walls(state)
        dis_corners = self.distance_to_corners(state)
        return len(own_liberties) - len(opp_liberties) + 0.5 * dis_corners - 0.5 * (dis_opp + dis_center + dis_walls)
    
    # look ahead
    def look_ahead_score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        own_next = sum([len(state.liberties(move)) for move in own_liberties])
        opp_next = sum([len(state.liberties(move)) for move in opp_liberties])
        return len(own_liberties) * own_next - len(opp_liberties) * opp_next
    
    # utility
    def distance_to_walls(self, state):
        own_loc = state.locs[self.player_id]
        x_player, y_player = own_loc // (_WIDTH + 2), own_loc % (_WIDTH + 2)
        return min(x_player, _WIDTH + 1 - x_player, y_player, _HEIGHT - 1 - y_player)
    
    def distance_to_corners(self, state):
        own_loc = state.locs[self.player_id]
        min_dist = float('inf')
        for corner in _CORNERS:
            d = self.distance(corner, own_loc)
            if d < min_dist:
                min_dist = d
        return min_dist
    
    def distance(self, node1, node2):
        dx = int(abs(node1 % (_WIDTH + 2) - node2 % (_WIDTH + 2)))
        dy = int(abs(node1 // (_WIDTH + 2) - node2 // (_WIDTH + 2)))
        d = math.sqrt(dx**2 + dy**2)
        return d
    
    def h_score_mixture_pos_helper(self, value, k):
        if value < 2:
            k = k + 1
        elif value <= 3:
            k = k + 0.5
        return k
    
    def h_score_mixture_neg_helper(self, value, k):
        if value >= 4:
            k = k + 1
        elif value >= 2:
            k = k + 0.5
        return k
    
    def time_exceeded(self, start):
        end = int(time.time() * 1000)
        if end - start >= 140:
            return True
        else:
            return False


class CustomPlayer_MCTS(DataPlayer):
    """ Implement your own agent to play knight's Isolation with Monte Carlo Tree Search
    """
    def get_action(self, state):
        import random
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            self.queue.put(self.monte_carlo_tree_search(state, simulations_number=80))
    
    def monte_carlo_tree_search(self, state, simulations_number):
        root = MctsNode(state)
        for _ in range(0, simulations_number):
            leaf = self.traverse(root)
            reward = leaf.rollout()
            leaf.backpropagate(reward)
        bc = root.best_child()
        return root.best_child_action(bc)
    
    def traverse(self, root):
        current_node = root
        while not current_node.state.terminal_test():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

CustomPlayer = CustomPlayer_Minimax
