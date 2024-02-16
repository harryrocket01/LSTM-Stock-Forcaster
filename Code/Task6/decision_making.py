"""
DecisionMaking.py
Forcasting and Predicting Stock
Author - Harry Softley-Graham
Written - Nov 2023 - Jan 2024
"""
from typing import Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


class DecisionMaking:
    """
    Class: DecisionMaking

    This class contains all of the functions needed to suggest
    buy-sell-hold actions for task 6.

    Attributes:
        dataframe (): containing stock data and infomation
        learning_rate (): rate q learning values should update
        discount_factor (): weight of future rewards
        epsilon (): rate updates should decrease
        Q (): q table of buy sell hold function
        budget (): budget the agent can use
        current_budget (): current budget the agent has
        existing_portfolio (): existing stocks held
        portfolio (): stocks owned by agent
        portfolio_history (): price history of portfolio

    Methods:
        calculate_reward() : calulates reward function to update q table
        take_action() : takes action on a day, either buy, sell or hold
        train() : trains deep reinforcement learning model
        update_q() : updates q learning table
        buy_sell_reccomendation() : reccomends what to do, buy/sell/hold
        plot() : plots results and graphics

    Args:
        dataframe (Any): Inputted Dataframe of Stock Data
        budget (int): budget the algorithm can use
        reward_vector (list): Reward Vector To Bias behavior
        existing_portfolio (list): Array of existing assests

    Example:
        agent = DecisionMaking(dataframe)
    """

    def __init__(
        self,
        dataframe: Any,
        budget: int = 10000,
        reward_vector: list = [0, 0, 0],
        existing_portfolio: list = [],
        fin_indicators: Any = None,
    ):
        self.dataframe = dataframe
        self.dataframe = self.dataframe.reset_index()
        self.learning_rate = 0.1
        self.discount_factor = 0.8
        self.epsilon = 1
        self.q_table = np.zeros((len(self.dataframe), 3))

        self.q_table[-1, :] = reward_vector

        self.budget = budget
        self.current_budget = budget
        self.existing_portfolio = existing_portfolio
        self.portfolio = existing_portfolio
        self.portfolio_history = []
        self.fin_indicators = fin_indicators

    def calculate_reward(self, state: int, action: int, weight: float):
        """
        Function: calculate_reward

        Calculates rewards at a given state given the current state
        and the given action. Reward is calulated buy taking the
        action, then calulating the compound return of that action.

        Args:
            state (int): current state
            action (int): action taken in current state
            weight (float): multiplier for reward

        Returns:
            compounded_return(float): compound return of chosen action

        Example:
                agent = DecisionMaking(dataframe).calculate_reward(state, action, weight)
        """
        buy_price = self.dataframe["high"].iloc[state]
        open_price = self.dataframe["open"].iloc[state]
        close_price = self.dataframe["close"].iloc[state]

        q_value = self.q_table[state][action]
        if q_value < 1:
            value = 1
        else:
            value = q_value

        initial_value = self.current_budget + sum(self.portfolio) * open_price

        if action == 2:  # sell
            if len(self.portfolio) > 0:
                sold_price = self.portfolio.pop(0)
                self.current_budget += sold_price

        elif action == 0:  # buy
            if self.current_budget > buy_price:
                self.portfolio.append(buy_price)
                self.current_budget -= buy_price

        new_value = self.current_budget + sum(self.portfolio) * close_price

        self.portfolio_history.append(new_value)
        # Compounded excess return
        period_return = (new_value - initial_value) / initial_value
        compounded_return = (1 + period_return) * value - 1

        return compounded_return * weight

    def take_action(self, state: int, weight: float):
        """
        Function: take_action

        Either calculates random action or takes optimal action. This
        is dependant on given epsion value. take note, epsilon dacays
        over time.

        Args:
            state (int): current state
            action (int): action taken in past state
            weight (float): multiplier for reward

        Returns:
            action (int): future action
            reward (float): given reward for given action

        Example:
                agent = DecisionMaking(dataframe).take_action(state, weight)
        """
        if np.random.rand() > self.epsilon:
            action = int(np.random.randint(0, 3))

        else:
            action = int(np.argmax(self.q_table[state, :]))

        reward = self.calculate_reward(state, action, weight)
        return action, reward

    def train(self, episodes: int = 100, decay: float = 0.95, weight: float = 1):
        """
        Function: train

        Trains reinforcement learning algorithm over number of episodes for a
        given decay through each episode.

        Args:
            episodes (int): number of episodes to iterates through
            decay (float): rate at whitch epsilon decays at
            weight (float): multiplier for reward

        Returns:
            None

        Example:
            agent = DecisionMaking(dataframe).train()
        """
        print("Buy Sell Agent - Set")
        for _ in range(0, episodes):
            self.current_budget = self.budget  # reset current budget
            self.portfolio_history = []  # reset history
            self.portfolio = self.existing_portfolio
            state = 0
            prev_state = None
            prev_action = None
            for _ in range(0, len(self.dataframe)):
                action, reward = self.take_action(state, weight)
                self.update_q(reward, prev_state, prev_action, state)
                prev_state = state
                prev_action = action
                state += 1
            self.epsilon *= decay
        print("Buy Sell Agent - Trained")

    def update_q(
        self,
        reward: float,
        prev_state: Optional[int],
        prev_action: Optional[int],
        new_state: int,
    ):
        """
        Function: update_q

        Updates Q table for the given state-action pair. This
        is done through the Q learning equation.

        Args:
            reward (float): given reward from the state-action pair
            prev_state (int): previous state value
            prev_action (int): previous action taken
            new_state (int): new state value
            action (int): current action taken

        Returns:
            None

        Example:
            agent = DecisionMaking(dataframe).update_q(state,reward,prev_state,
                                                    prev_action,new_state,action)
        """
        if prev_state is not None and prev_state is not None:
            c_q = self.q_table[prev_state, prev_action]
            m_f_q = np.max(self.q_table[new_state, :])
            n_q = c_q + self.learning_rate * (
                reward + np.multiply(self.discount_factor, m_f_q) - c_q
            )
            self.q_table[prev_state, prev_action] = n_q

    def buy_sell_reccomendation(self):
        """
        Function:buy_sell_reccomendation

        Generates a buy-sell-hold reccomendation for the provided input.
        This is done by fetching the last Q value, and scaling the
        number of stocks to buy.

        Args:
            None

        Example:
            agent = DecisionMaking(dataframe).buy_sell_reccomendation()
        """
        self.plot()

        last_state_q_values = np.argmax(self.q_table[-2, :])
        last_date = pd.to_datetime(self.dataframe["date"].iloc[-1])
        row = self.fin_indicators.loc[last_date]
        max_stock = float(self.budget / self.dataframe["close"].iloc[-1])
        strength = float((((100 - row["rsi"]) / 100) * (row["adx"] / 100)))
        to_buy = int(max_stock * strength)

        if last_state_q_values == 0:  # Buy
            print(f"Buy Sell Agent - Buy: {to_buy} shares")

            return last_state_q_values
        if last_state_q_values == 2:  # Sell
            print(f"Buy Sell Agent - Sell: {to_buy} shares")
            return last_state_q_values

        print("Buy Sell Agent - Hold")
        return last_state_q_values

    def plot(self):
        """
        Function: plot

        Creates a plot for the given results. Plots the
        optimal choice for each day of stock movements.

        Args:
            None

        Returns:
            None

        Example:
            agent = DecisionMaking(dataframe).plot()
        """
        actions = np.argmax(self.q_table, axis=1)

        # Plot the scatter plot
        fig, _ = plt.subplots()

        plt.plot(self.dataframe["date"], self.dataframe["close"])
        scatter = plt.scatter(
            self.dataframe["date"],
            self.dataframe["close"],
            c=actions,
            cmap="viridis",
            marker="o",
        )

        # Label the axes
        plt.ylabel("Close Prices")
        plt.title("Results of Buy-Hold-Sell Agent")

        # Add colorbar
        cbar = plt.colorbar(scatter, ticks=[0, 1, 2])
        cbar.set_label("Action (0: Buy, 1: Hold, 2: Sell)")
        fig.set_size_inches(5, 3)
        plt.gcf().autofmt_xdate()
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

        plt.savefig("DAPs_Code/Task6/Graphics/BuySellHold.pdf")
        plt.close()
