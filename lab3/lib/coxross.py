import matplotlib.pyplot as plt
from typing import List
from scipy.special import comb


def binomial_option_price(S, K, r, u, d, T):
    """
    Calculates the price of a European call option using the binomial model.

    :param S: Initial price of the underlying asset
    :param K: Strike price of the option
    :param r: Risk-free interest rate
    :param u: Upward price movement factor
    :param d: Downward price movement factor
    :param T: Number of periods (binomial tree)
    :return: Option value at the initial time
    """
    p_star = (1 + r - d) / (u - d)  # Risk-neutral probability
    option_value = 0

    for j in range(T + 1):
        S_T = S * (u ** j) * (d ** (T - j))
        payoff = max(S_T - K, 0)
        prob = comb(T, j) * (p_star ** j) * ((1 - p_star) ** (T - j))
        option_value += prob * payoff

    return option_value / ((1 + r) ** T)


class RecombiningTree:
    def __init__(self, S0, u, d, steps, r=0.1):
        """
        Create a recombining tree with given parameters
        :param S0: value of the asset at time 0
        :param u: up factor
        :param d: down factor
        :param steps: number of steps in the market (tree) - how many times the asset price can change and how deep the tree is
        :param r: risk free rate
        :var p: risk neutral probability
        """
        self.S0 = S0
        self.u = u
        self.d = d
        self.steps = steps
        self.r = r
        self.p = (1 + r - d) / (u - d)  # risk neutral probability
        self.tree = self.build_tree()

    def build_tree(self):
        tree = []
        for n in range(self.steps + 1):
            level = [self.S0 * (self.u ** (n - j)) * (self.d ** j) for j in range(n + 1)]
            tree.append(level)
        return tree

    def display_tree(self):
        for level in self.tree:
            print(level)

    def visualize_tree(self):
        fig, ax = plt.subplots(figsize=(10, 6))

        for n in range(self.steps + 1):
            for j in range(n + 1):
                x = n
                y = -j + n / 2
                ax.scatter(x, y, color='blue')
                ax.text(x, y + 0.2, f"{self.tree[n][j]:.2f}", ha='center', va='center', fontsize=10, color='black')

                if n > 0:
                    if j < n:
                        ax.plot([n - 1, x], [-j + (n - 1) / 2, y], 'k-')
                    if j > 0:
                        ax.plot([n - 1, x], [-j + 1 + (n - 1) / 2, y], 'k-')

        ax.set_xticks(range(self.steps + 1))
        ax.set_yticks([])
        ax.set_title("Market Tree Visualization")
        plt.show()

    def get_left_parent(self, level: int, node_index: int) -> (float, int, int):
        """
        Get the left parent of a node in the tree
        :param level: level of the node
        :param node_index: index of the node in the level
        :return: tuple:
            - value of the left parent
            - level of the left parent
            - index of the left parent in the level
        """
        if level > self.steps or node_index > level:
            raise Exception("Invalid level or node index")
        if node_index == 0:
            if level == 0:
                return None
            return self.tree[level - 1][0], level - 1, 0
        if node_index == level:
            return None
        return self.tree[level - 1][node_index], level - 1, node_index

    def get_right_parent(self, level: int, node_index: int) -> (float, int, int):
        """
        Get the right parent of a node in the tree
        :param level: level of the node
        :param node_index: index of the node in the level
        :return: tuple:
            - value of the right parent
            - level of the right parent
            - index of the right parent in the level
        """
        if level > self.steps or node_index > level:
            raise Exception("Invalid level or node index")
        if node_index == 0:
            return None
        return self.tree[level - 1][node_index - 1], level - 1, node_index - 1

    def get_both_parents(self, level: int, node_index: int) -> [(float, int, int), (float, int, int)]:
        """
        Get both parents of a node in the tree
        :param level: level of the node
        :param node_index: index of the node in the level
        :return: tuple of tuples:
            - first tuple: left parent
            - second tuple: right parent
            in each tuple:
                - value of the parent
                - level of the parent
                - index of the parent in the level
        """
        return self.get_left_parent(level, node_index), self.get_right_parent(level, node_index)

    def get_right_child(self, level: int, node_index: int) -> (float, int, int):
        """
        Get the right child of a node in the tree
        :param level: level of the node
        :param node_index: index of the node in the level
        :return: tuple:
            - value of the right child
            - level of the right child
            - index of the right child in the level
        """
        if level > self.steps or node_index > level:
            raise Exception("Invalid level or node index")
        return self.tree[level + 1][node_index], level + 1, node_index

    def get_left_child(self, level: int, node_index: int) -> (float, int, int):
        """
        Get the left child of a node in the tree
        :param level: level of the node
        :param node_index: index of the node in the level
        :return: tuple:
            - value of the left child
            - level of the left child
            - index of the left child in the level
        """
        if level > self.steps or node_index > level:
            raise Exception("Invalid level or node index")
        return self.tree[level + 1][node_index + 1], level + 1, node_index + 1

    def get_both_children(self, level: int, node_index: int) -> [(float, int, int), (float, int, int)]:
        """
        Get both children of a node in the tree
        :param level: level of the node
        :param node_index: index of the node in the level
        :return: tuple of tuples:
            - first tuple: left child
            - second tuple: right child
            in each tuple:
                - value of the child
                - level of the child
                - index of the child in the level
        """
        return self.get_left_child(level, node_index), self.get_right_child(level, node_index)

    def set_last_level(self, values: List[int]) -> None:
        """
        Set the last level of the tree
        :param values: list of values for the last level
        """
        if len(values) != self.steps + 1:
            raise Exception("Invalid number of values")
        self.tree[-1] = values

    def get_price_of_asset(self, func, trajectory=None, t=0):
        """
        Get the price of the asset at time t
        :param t: time
        :param func: function to calculate the price using whole trajectory (!)
        :param trajectory: trajectory of the asset
        :return: price of the asset
        """
        if trajectory is None:
            trajectory = [self.S0]
        if t == self.steps:
            return func(trajectory)
        return (self.p * self.get_price_of_asset(func, trajectory + [self.u * trajectory[-1]], t + 1) + (
                1 - self.p) * self.get_price_of_asset(func, trajectory + [self.d * trajectory[-1]], t + 1)) / (
                    1 + self.r)


S0 = 100
u = 1.3
d = 0.8
steps = 10

tree = RecombiningTree(S0, u, d, steps)
print("Visualization of the tree:")
tree.display_tree()
tree.visualize_tree()


# take 0,0 and then 1,0 1,1 etc
# print("Children of 0,0", tree.get_both_children(0, 0))
# print("Children of 1,0", tree.get_both_children(1, 0))
# print("Children of 1,1", tree.get_both_children(1, 1))
# print("Children of 2,0", tree.get_both_children(2, 0))
# print("Children of 2,1", tree.get_both_children(2, 1))
#
# print("Parents of 0,0", tree.get_both_parents(0, 0))
# print("Parents of 1,0", tree.get_both_parents(1, 0))
# print("Parents of 1,1", tree.get_both_parents(1, 1))
# print("Parents of 2,0", tree.get_both_parents(2, 0))
# print("Parents of 2,1", tree.get_both_parents(2, 1))


def call_option(trajectory, k=90):
    """
    Calculate the price of a call option using the trajectory
    :param k: strike price
    :param trajectory: trajectory of the asset
    :return: price of the call option
    """
    return max(0, trajectory[-1] - k)

def put_option(trajectory, k=90):
    """
    Calculate the price of a put option using the trajectory
    :param k: strike price
    :param trajectory: trajectory of the asset
    :return: price of the put option
    """
    return max(0, k - trajectory[-1])


def max_trajectory(trajectory):
    """
    Calculate the maximum value of the trajectory
    :param trajectory: trajectory of the asset
    :return: maximum value of the trajectory
    """
    return max(trajectory)


# this Answers for exercises:
print("------------------")
print()
print("Exercises")
price_of_call_option = tree.get_price_of_asset(call_option)
theoretical_price = binomial_option_price(S0, 90, 0.1, u, d, steps)
print("I want to calculate the price of a call option with strike price 90")
print("Price of the asset", price_of_call_option)  # ESSA!
print("Price of the asset (theoretical value) ", theoretical_price)
print('SAME?: ', abs(price_of_call_option - theoretical_price) < 0.01)
print("--I want to calculate the price of max of trajectory now: ")
print("Price of the asset", tree.get_price_of_asset(max_trajectory))

print("\n---ADDITIONAL EXAMPLE---")
print("--I want to calculate the price of a put option with strike price 90")
print("Price of the asset", tree.get_price_of_asset(put_option))
