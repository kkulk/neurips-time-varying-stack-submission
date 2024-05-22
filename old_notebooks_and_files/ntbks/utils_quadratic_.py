import numpy as np
import pandas as pd
from numpy import linalg as la
import argparse
import scipy.linalg  as sla
import random
from scipy.optimize import fsolve
from scipy.sparse import random as scirand
from scipy.optimize import minimize
from scipy.linalg import block_diag

def sample_from_d_sphere(d):
    """
    Samples a vector uniformly at random from the d-dimensional unit sphere.
    :param d: Dimension of the sphere.
    """
    vector = np.random.normal(0, 1, d)
    norm = np.linalg.norm(vector)
    unit_vector = vector / norm
    
    return unit_vector

class incent:
    def __init__(self, x_d, u_d, delta,u_init, eta_init, gamma_init, MAXINNER, MAXOUTER):
        self.delta = delta
        self.x_d = x_d
        self.u_d = u_d
        self.u = u_init
        
class quadratic:
    def __init__(self, Q1, Q2, A1, A2, eta_init, MAXINNER, u, d):
        """
        Initializes the Bertrand game model.
        :param theta: 2x3 matrix of theta constants where theta[i, :] corresponds to theta_i constants.
        :param eta: Step size.
        :param MAXINNER: Number of iterations of inner loop. 
        :param u: Control input provided by a planner.
        """
        self.num_players = 2
        self.d = d
        self.Q1 = Q1
        self.Q2 = Q2
        self.A1 = A1
        self.A2 = A2
        self.eta_init = eta_init
        self.MAXINNER = MAXINNER
        self.u = u
        self.x1 = np.ones(d)
        self.x2 = np.ones(d) 
        self.upper = 10*np.ones(d)
        self.lower = -10*np.ones(d)
        self.eta = eta_init

    def proj(self, x):
        """
        Projects the price x onto the interval [lower, upper].
        :param x: Price.
        :return: Projected price.
        """
        return np.clip(x, self.lower, self.upper)

    
    def gradPlayer(self, i):
        """
        Computes the gradient of cost function
        :param i: Player index (0 or 1).
        :param x: Current prices of both players.
        :return: Value of the marginal demand function.
        """
        x1 = self.x1
        x2 = self.x2
        if i == 0:
            return 2*self.Q1 @ x1 + self.A1 @ x2 + self.u 
        else: 
            return 2*self.Q2 @ x2 + self.A2.T @ x1 + self.u 
        
    def runGradPlay(self):
        """
        Runs gradient play for MAXINNER number of iterations. 
        """
        history_0 = []
        history_1 = []
        player_costs_0 = []
        player_costs_1 = []
        for k in range(self.MAXINNER):
            # Toggle this on and off for decaying stepsizes at the inner loop.
            #self.eta = self.eta_init/(k+1)
            x_next1 = np.copy(self.x1)
            x_next2 = np.copy(self.x2)
            x_next1 = self.proj(self.x1 - self.eta * self.gradPlayer(0))
            x_next2 = self.proj(self.x2 - self.eta * self.gradPlayer(1))
            self.x1 = x_next1
            self.x2 = x_next2
            print(self.gradPlayer(0))
            history_0.append(np.copy(self.x1))
            history_1.append(np.copy(self.x2))
            print(self.x1)
        player_costs_0.append(self.player_cost(0))
        player_costs_1.append(self.player_cost(1))
        return history_0, history_1, player_costs_0, player_costs_1
    
    def runSGDPlay(self, var):
        """
        Runs stochastic gradient play for MAXINNER number of iterations. 
        """
        history_0 = []
        history_1 = []
        player_costs_0 = []
        player_costs_1 = []
        for k in range(self.MAXINNER):
            # Toggle this on and off for decaying stepsizes at the inner loop.
            #self.eta = self.eta_init/(k+1)
            x_next1 = np.copy(self.x1)
            x_next2 = np.copy(self.x2)
            noise_1 = np.random.normal(0,var,self.d)
            noise_2 = np.random.normal(0,var,self.d)
            x_next1 = self.proj(self.x1 - self.eta * self.gradPlayer(0) + noise_1)
            x_next2 = self.proj(self.x2 - self.eta * self.gradPlayer(1) + noise_2)
            self.x1 = x_next1
            self.x2 = x_next2
            print(self.gradPlayer(0))
            history_0.append(np.copy(self.x1))
            history_1.append(np.copy(self.x2))
            print(self.x1)
        player_costs_0.append(self.player_cost(0))
        player_costs_1.append(self.player_cost(1))
        return history_0, history_1, player_costs_0, player_costs_1
    
    def solve_nash(self, x0):
        """
        Solves a system of nonlinear equations to check Nash equilibrium.

        :param x0: Initial guess for the solution (2d-dimensional vector).
        :return: Solution of the system of equations.
        """

        def first_order_conditions(vars):
            x1, x2 = vars[:len(vars)//2], vars[len(vars)//2:]
            return np.array([
                2 * self.Q1 @ x1 + self.A1 @ x2 + self.u,
                2 * self.Q2 @ x2 + self.A2.T @ x1 + self.u
            ]).flatten()

        solution = fsolve(first_order_conditions, x0)
        return solution
    def player_cost(self, i):
        """
        Returns player costs.
        """
        cost = 0
        x1 = self.x1
        x2 = self.x2
        if i == 0:
            cost = x1.T @ self.Q1 @ x1+ x1.T @ self.A1 @ x2 + self.u.T @ x1    
        else:
            cost = x2.T @ self.Q2 @ x2+ x1.T @ self.A2 @ x2 + self.u.T @ x2    
        return cost
                
    def check_second_order(self):
        """
        Check if game Hessian is positive definite. 
        :param x: A vector [x1, x2].
        :return: True if the Hessian is positive definite, False otherwise.
        """
        matrix = block_diag(self.Q1, self.Q2)
        eigenvalues = np.linalg.eigvals(matrix)
        return "Game Hessian is positive definite: " + str(np.all(eigenvalues > 0))
    
    def check_jacobian(self,u):
        matrix = np.array([[2*self.Q1 + np.diag(u), self.A1], [self.A2.T, 2*self.Q2 + np.diag(u)]])
        eigenvalues = np.linalg.eigvals(matrix)
        return np.all(eigenvalues > 0)