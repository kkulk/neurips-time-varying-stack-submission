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
    def __init__(self, Q1, Q2, A1, A2, x0_d, x1_d, u_d, d, 
                delta,u_init, eta_init, gamma_init, MAXINNER, MAXOUTER, burn_in,
                lam=0
                ):
        self.d = d
        self.delta = delta
        self.x0_d = x0_d
        self.x1_d = x1_d
        self.u_d = u_d
        self.ud=np.hstack((u_d,u_d))
        self.u = u_init
        self.eta = eta_init
        self.game = quadratic(Q1, Q2, A1, A2, self.eta, MAXINNER, u_init, d)
        self.Q1 = Q1
        self.Q2 = Q2
        self.A1 = A1
        self.A2 = A2
        self.lower_u = -5*np.ones(self.d)
        self.upper_u = 5*np.ones(self.d)
        self.burn_in = burn_in
        self.MAXINNER = MAXINNER
        self.MAXOUTER = MAXOUTER
        self.gamma = gamma_init
        self.gamma_init = gamma_init
        self.lam=lam
        self.G = np.vstack((np.hstack((self.Q1 , self.A1)), np.hstack((self.A2.T, self.Q2))))
        
    def loss(self, x0, x1, u):
        return np.linalg.norm(x0 - self.x0_d)**2 +np.linalg.norm(x1 - self.x1_d)**2 + np.linalg.norm(u - self.u_d)**2
    
    def proj_u(self, u):
        """
        Projects the incentive onto the interval [lower_u, upper_u].
        :param u: Incentive.
        :return: Projected incentive.
        """
        return np.clip(u, self.lower_u, self.upper_u)
    
    def getGradDFO(self, x0, x1, u, v):
        """
        Computes the gradient for the DFO step.
        :param last_prices: Last prices from running gradient play.
        :param u: Current value of control input.
        :return: Gradient for updating u.
        """
        #v = sample_from_d_sphere(self.d)
        tilde_u = u+self.delta*v
        gradient = (self.d/self.delta)*self.loss(x0, x1,tilde_u)*v
        return gradient
    
    def runDFOSGD(self,x_init, var):
        """
        Updates u according to a derivative free gradient step for MAXOUTER number of iterations. 
        """
        x1=x_init[0:self.d]
        x2=x_init[self.d:2*self.d]
        u_history = [self.u]
        x_history=[x_init]
        loss_history = [self.loss(x1, x2, u_history[-1])]
        play_loss=[self.get_player_loss(x1,x2,u_history[-1])]
        self.game = quadratic(self.Q1, self.Q2, self.A1, self.A2, self.eta, self.MAXINNER, self.u, self.d)
        for k in range(self.burn_in + self.MAXOUTER):
            v = sample_from_d_sphere(self.d) 
            tilde_u=u_history[-1]+self.delta*v
            history_0, history_1 = self.game.runSGDPlay(x_history[-1],tilde_u, var=var) # Run firm gradient play
            
            x_history.append(np.hstack((history_0[-1],history_1[-1])))
            
            grad_dfo = self.getGradDFO(history_0[-1], history_1[-1], u_history[-1],v)# Compute gradient for DFO
            u_history.append(self.proj_u(u_history[-1] - self.gamma * grad_dfo))

            loss_history.append(self.loss(history_0[-1], history_1[-1], u_history[-1]))
            self.gamma = self.gamma_init/(5*(k+1))
            play_loss.append(self.get_player_loss(history_0[-1],history_1[-1],u_history[-1]))
        return loss_history, u_history,x_history, play_loss
    
    
    
    def runDFO(self,x_init):
        """
        Updates u according to a derivative free gradient step for MAXOUTER number of iterations. 
        """
        x1=x_init[0:self.d]
        x2=x_init[self.d:2*self.d]
        u_history = [self.u]
        x_history=[x_init]
        loss_history = [self.loss(x1, x2, u_history[-1])]
        play_loss=[self.get_player_loss(x1,x2,u_history[-1])]
        self.game = quadratic(self.Q1, self.Q2, self.A1, self.A2, self.eta, self.MAXINNER, self.u, self.d)
        for k in range(self.burn_in + self.MAXOUTER):
            v = sample_from_d_sphere(self.d) 
            tilde_u=u_history[-1]+self.delta*v
            history_0, history_1 = self.game.runGradPlay(x_history[-1],tilde_u) # Run firm gradient play
            
            x_history.append(np.hstack((history_0[-1],history_1[-1])))
            
            grad_dfo = self.getGradDFO(history_0[-1], history_1[-1], u_history[-1],v)# Compute gradient for DFO
            u_history.append(self.proj_u(u_history[-1] - self.gamma * grad_dfo))

            loss_history.append(self.loss(history_0[-1], history_1[-1], u_history[-1]))
            self.gamma = self.gamma_init/(5*(k+1))
            play_loss.append(self.get_player_loss(history_0[-1],history_1[-1],u_history[-1]))
        return loss_history, u_history, x_history, play_loss

    def runDFOBR(self, x_init):
        """
        Updates u according to a derivative free gradient step for MAXOUTER number of iterations. 
        """
        x1=x_init[0:self.d]
        x2=x_init[self.d:2*self.d]
        u_history = [self.u]
        x_history=[x_init]
        loss_history = [self.loss(x1, x2, u_history[-1])]
        play_loss=[self.get_player_loss(x1,x2,u_history[-1])]
        self.game = quadratic(self.Q1, self.Q2, self.A1, self.A2, self.eta, self.MAXINNER, self.u, self.d)
        for k in range(self.burn_in + self.MAXOUTER):
            v = sample_from_d_sphere(self.d) 
            tilde_u=u_history[-1]+self.delta*v
            history_0, history_1 = self.game.get_xd_constrained(tilde_u)
            
            x_history.append(np.hstack((history_0[-1],history_1[-1])))
            
            grad_dfo = self.getGradDFO(history_0[-1], history_1[-1], u_history[-1],v)# Compute gradient for DFO
            u_history.append(self.proj_u(u_history[-1] - self.gamma * grad_dfo))

            loss_history.append(self.loss(history_0[-1], history_1[-1], u_history[-1]))
            self.gamma = self.gamma_init/(5*(k+1))
            play_loss.append(self.get_player_loss(history_0[-1],history_1[-1],u_history[-1]))
        return loss_history, u_history, x_history, play_loss

    def get_player_opt(self, x1,x2,u, i):
        if i==0:
            xi_star=-1/2*la.inv(self.game.Q1)@(self.game.A1@x2+u)
        else:
            xi_star=-1/2*la.inv(self.game.Q2)@(self.game.A2@x1+u)
        return xi_star 
    
    def get_player_loss(self,x1,x2,u):
        loss=0

        xi_star=self.get_player_opt(x1,x2,u,0)
        loss+=self.game.get_player_cost(x1,x2,u,0)-self.game.get_player_cost(xi_star,x2,u,0)
        xi_star=self.get_player_opt(x1,x2,u,1)
        loss+=self.game.get_player_cost(x1,x2,u,1)-self.game.get_player_cost(x1,xi_star,u,1)
        return loss

    def _gradFRGM(self,u,x):
        #print(x)
        #print(self.lam*(u-self.ud))
        return x+self.lam*(u-self.ud)

    def runIncentRGM(self, u0,x0, eta=0.005, var=0.001, MAXOUTER=5000,MAXINNER=500,tag='BR', BRFLAG=False, lam=1/2):
        us=[u0]
        #print(us[-1])
        if lam!=None:
            self.lam=lam
        self.ud=np.hstack((self.u_d,self.u_d))
        ups=self.get_ps()
        xps1,xps2=self.game.get_br(ups)
        xps=np.hstack((xps1,xps2))
        err=[la.norm(u0-ups)**2]
        x0s=[x0]
        #print(x0s[-1])
        self.xd=self.game.get_xd(self.ud)
        
        for i in range(MAXOUTER):
            if tag=='BR':
                x1,x2=self.game.get_br(us[-1])
                x0_new=np.hstack((x1,x2))
            elif tag=='GP':

                x1,x2 = self.game.runGradPlay_alt(x0s[-1],us[-1])
                x0_new=np.hstack((x1[-1],x2[-1]))
            else:
                x1,x2 = self.game.runSGDPlay_alt(x0s[-1],us[-1], var=var)
                x0_new=np.hstack((x1[-1],x2[-1]))
            
            x0s.append(x0_new)
            g_hat=self._gradFRGM(us[-1],x0s[-1])

            us.append(us[-1]-eta*g_hat) 
            err.append(la.norm(us[-1]-ups)**2+la.norm(x0s[-1]-xps)**2)
            
        err=np.asarray(err)
        us=np.asarray(us)
        x0s=np.asarray(x0s)
        data={
            'err': err,
            'us' : us,
            'x0s': x0s
        }
        return data
    
    def get_ps(self):
        ud=self.ud
        Qinv= la.inv(np.vstack((np.hstack((2*self.Q1 , self.A1)), np.hstack((self.A2.T, 2*self.Q2)))))
        mat=la.inv(-Qinv+self.lam*np.eye(2*self.d))
        return self.lam*mat@ud

    def get_sopt(self):
        ud=self.ud
        #G=np.vstack((np.hstack((self.Q1 , self.A1)), np.hstack((self.A2.T, self.Q2))))
        #Ginv= la.inv(G)
        #I=np.eye(2*self.d)
        #M= Ginv.T@G@
        return self.ud

    
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
        self.eta_bound = self.get_eta_bound()

    def proj(self, x):
        """
        Projects the price x onto the interval [lower, upper].
        :param x: Price.
        :return: Projected price.
        """
        return np.clip(x, self.lower, self.upper)

    
    def gradPlayer_old(self, i):
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
        
    def gradPlayer(self,x1,x2,u, i):
        """
        Computes the gradient of cost function
        :param i: Player index (0 or 1).
        :param x: Current prices of both players.
        :return: Value of the marginal demand function.
        """
        #x1 = self.x1
        #x2 = self.x2
        #u1=u[0:self.d]
        #u2=u[self.d:2*self.d]
        if i == 0:
            return 2*self.Q1 @ x1 + self.A1 @ x2 + u
        else: 
            return 2*self.Q2 @ x2 + self.A2.T @ x1 + u
        
    def runGradPlay(self, x0,u, eta=-10., M=None):
        """
        Runs gradient play for MAXINNER number of iterations. 
        """
        if eta==-10.:
            eta=self.eta

        x1s=[x0[0:self.d]]
        x2s=[x0[self.d:2*self.d]]
        if M==None:
            M=self.MAXINNER

        for k in range(M):
            # Toggle this on and off for decaying stepsizes at the inner loop.
            #self.eta = self.eta_init/(k+1)
            #print(x1s[-1],x2s[-1],u)
            #print(x1s[-1],self.gradPlayer(x1s[-1],x2s[-1],u,0),self.gradPlayer(x1s[-1],x2s[-1],u,1))
            x1s.append(self.proj(x1s[-1] - eta * self.gradPlayer(x1s[-1],x2s[-1],u,0)))
            x2s.append(self.proj(x2s[-1] - eta * self.gradPlayer(x1s[-1],x2s[-1],u,1)))

        x1=np.asarray(x1s)
        x2=np.asarray(x2s)
        return x1,x2
    
    def runSGDPlay(self,x0,u, var=0.001, eta=0.001, MAXINNER=None):
        """
        Runs stochastic gradient play for MAXINNER number of iterations. 
        """

        x1s=[x0[0:self.d]]
        x2s=[x0[self.d:self.d*2]]

        if MAXINNER==None:
            MAXINNER=self.MAXINNER

        for k in range(MAXINNER):
            noise_1 = np.random.normal(0,var,self.d)
            noise_2 = np.random.normal(0,var,self.d)
            x1s.append(self.proj(x1s[-1] - eta * self.gradPlayer(x1s[-1],x2s[-1],u, 0) + noise_1))
            x2s.append(self.proj(x2s[-1] - eta * self.gradPlayer(x1s[-1],x2s[-1],u, 1) + noise_2))

        x1=np.asarray(x1s)
        x2=np.asarray(x2s)

        return x1,x2
    
    def runGradPlay_alt(self, x0,u, eta=-10., M=None):
        """
        Runs gradient play for MAXINNER number of iterations. 
        """

        u1=u[0:self.d]
        u2=u[self.d:2*self.d]
        #print(u1,u2)
        if eta==-10.:
            eta=self.eta

        x1s=[x0[0:self.d]]
        x2s=[x0[self.d:2*self.d]]
        if M==None:
            M=self.MAXINNER

        for k in range(M):
            # Toggle this on and off for decaying stepsizes at the inner loop.

            x1s.append(self.proj(x1s[-1] - eta * self.gradPlayer(x1s[-1],x2s[-1],u1,0)))
            x2s.append(self.proj(x2s[-1] - eta * self.gradPlayer(x1s[-1],x2s[-1],u2,1)))

        x1=np.asarray(x1s)
        x2=np.asarray(x2s)
        return x1,x2 
    
    def runSGDPlay_alt(self,x0,u, var=0.001, eta=0.001, MAXINNER=None):
        """
        Runs stochastic gradient play for MAXINNER number of iterations. 
        """
        u1=u[0:self.d]
        u2=u[self.d:2*self.d]
        x1s=[x0[0:self.d]]
        x2s=[x0[self.d:self.d*2]]

        if MAXINNER==None:
            MAXINNER=self.MAXINNER

        for k in range(MAXINNER):
            noise_1 = np.random.normal(0,var,self.d)
            noise_2 = np.random.normal(0,var,self.d)
            x1s.append(self.proj(x1s[-1] - eta * self.gradPlayer(x1s[-1],x2s[-1],u1, 0) + noise_1))
            x2s.append(self.proj(x2s[-1] - eta * self.gradPlayer(x1s[-1],x2s[-1],u2, 1) + noise_2))

        x1=np.asarray(x1s)
        x2=np.asarray(x2s)

        return x1,x2 
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
    def get_player_cost(self,x1,x2,u, i):
        """
        Returns player costs.
        """
        cost = 0
        if i == 0:
            cost = x1.T @ self.Q1 @ x1+ x1.T @ self.A1 @ x2 + u.T @ x1    
        else:
            cost = x2.T @ self.Q2 @ x2+ x1.T @ self.A2 @ x2 + u.T @ x2    
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
    
    def get_alpha_L(self,u):
        matrix = np.vstack((np.hstack((2*self.Q1 , self.A1)), np.hstack((self.A2.T, 2*self.Q2))))
        #print("shape of matrix: ", np.shape(matrix))
        matrix=matrix+matrix.T
        eigenvalues = np.linalg.eigvals(matrix)
        return eigenvalues
    
    def get_eta_bound(self):
        eigs = self.get_alpha_L(self.u)
        alpha = min(eigs)
        L = max(eigs)
        eta_bound = alpha/(2*L**2)
        return eta_bound 
    
    def check_jacobian(self,u):
        matrix = np.vstack((np.hstack((2*self.Q1 , self.A1)), np.hstack((self.A2.T, 2*self.Q2))))
        matrix=matrix+matrix.T
        eigenvalues = np.linalg.eigvals(matrix)
        return np.all(eigenvalues > 0)
    
    def get_xd(self,u):
        matrix = np.vstack((np.hstack((2*self.Q1 , self.A1)), np.hstack((self.A2.T, 2*self.Q2))))
        x=-np.linalg.inv(matrix)@u
        x1=x[0:self.d]
        x2=x[self.d:2*self.d]
        history_0=[x1]
        history_1=[x2]
        player_costs_0=[x1.T @ self.Q1 @ x1+ x1.T @ self.A1 @ x2 + self.u.T @ x1] 
        player_costs_1=[x2.T @ self.Q2 @ x2+ x1.T @ self.A2 @ x2 + self.u.T @ x2]
        return history_0, history_1, player_costs_0, player_costs_1
    
    def get_xd_constrained(self,u):
        u=np.hstack((u,u))
        matrix = np.vstack((np.hstack((2*self.Q1 , self.A1)), np.hstack((self.A2.T, 2*self.Q2))))
        x=-np.linalg.inv(matrix)@u
        x1=x[0:self.d]
        x2=x[self.d:2*self.d]
        history_0=[x1]
        history_1=[x2]
        return history_0, history_1
    
    def get_br(self,u):
        #u=np.hstack((u,u))
        matrix = np.vstack((np.hstack((2*self.Q1 , self.A1)), np.hstack((self.A2.T, 2*self.Q2))))
        x=-np.linalg.inv(matrix)@u
        x1=x[0:self.d]
        x2=x[self.d:2*self.d]
        return x1,x2
    
    
        
