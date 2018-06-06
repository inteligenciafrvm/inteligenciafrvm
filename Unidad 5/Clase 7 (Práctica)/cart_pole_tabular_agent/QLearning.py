import random

"""
Q-learning approach for different RL problems
as part of the basic series on reinforcement learning @
https://github.com/vmayoral/basic_reinforcement_learning

Inspired by https://gym.openai.com/evaluations/eval_kWknKOkPQ7izrixdhriurA

        @author: Victor Mayoral Vilches <victor@erlerobotics.com>
"""


class QLearning:

    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def get_q(self, state, action):
        """
        Gets the tabular Q-value for the specified state and action pair. Returns 0 if there is no value for such pair
        """
        return self.q.get((state, action), 0.0)

    def learn(self, state1, action1, reward, state2):
        """
        Performs a Q-learning update for a given state transition
        """
        new_max_q = max([self.get_q(state2, a) for a in self.actions])
        self.learn_q(state1, action1, reward, reward + self.gamma * new_max_q)

    def learn_q(self, state, action, reward, value):
        """
        Internal method where the Q-learning update is performed

        Q-learning update:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        """
        old_v = self.q.get((state, action), None)
        if old_v is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = old_v + self.alpha * (value - old_v)

    def choose_action(self, state, return_q=False):
        """
        Chooses an action according to the learning previously performed
        """
        q = [self.get_q(state, a) for a in self.actions]
        max_q = max(q)

        if random.random() < self.epsilon:
            min_q = min(q)
            max_min_q_distance = max(abs(min_q), abs(max_q))

            # add random values to all the actions, recalculate max_q
            q = [q[i] + random.random() * max_min_q_distance
                 - .5 * max_min_q_distance for i in range(len(self.actions))]
            max_q = max(q)

        count = q.count(max_q)

        # In case there're several state-action max values
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == max_q]
            i = random.choice(best)
        else:
            i = q.index(max_q)

        action = self.actions[i]

        if return_q:  # if they want it, give it!
            return action, q
        return action
