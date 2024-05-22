import numpy as np
import pandas as pd
from numpy import linalg as la
import scipy.linalg  as sla
       
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

    
    def gradPlayer(self,x1,x2,u, i):
        """
        Computes the gradient of cost function
        :param i: Player index (0 or 1).
        :param x: Current prices of both players.
        :return: Value of the marginal demand function.
        """
        #x1 = self.x1
        #x2 = self.x2
        u1=u[0:self.d]
        u2=u[self.d:2*self.d]
        if i == 0:
            return 2*self.Q1 @ x1 + self.A1 @ x2 + u1
        else: 
            return 2*self.Q2 @ x2 + self.A2.T @ x1 + u2


    def runGradPlay(self, x0,u, eta=0.01, M=None):
        """
        Runs gradient play for MAXINNER number of iterations. 
        """
        x1s=[x0[0:self.d]]
        x2s=[x0[self.d:2*self.d]]
        if M==None:
            M=self.MAXINNER

        for k in range(M):
            # Toggle this on and off for decaying stepsizes at the inner loop.
            #self.eta = self.eta_init/(k+1)
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
        self.nash=self.get_nash()
        self.x1n=self.nash[0:self.d]
        self.x2n=self.nash[self.d:self.d*2]
        error={
            'p1':[la.norm(x1s[-1]-self.x1n)**2], 
            'p2': [la.norm(x2s[-1]-self.x2n)**2]
                }
        if MAXINNER==None:
            MAXINNER=self.MAXINNER

        for k in range(MAXINNER):
            # Toggle this on and off for decaying stepsizes at the inner loop.
            #self.eta = self.eta_init/(k+1)
            #x_next1 = np.copy(self.x1)
            #x_next2 = np.copy(self.x2)
            noise_1 = np.random.normal(0,var,self.d)
            noise_2 = np.random.normal(0,var,self.d)
            x1s.append(self.proj(x1s[-1] - eta * self.gradPlayer(x1s[-1],x2s[-1],u, 0) + noise_1))
            x2s.append(self.proj(x2s[-1] - eta * self.gradPlayer(x1s[-1],x2s[-1],u, 1) + noise_2))
            error['p1'].append(la.norm(x1s[-1]-self.x1n)**2)
            error['p2'].append(la.norm(x2s[-1]-self.x2n)**2)
        x1=np.asarray(x1s)
        x2=np.asarray(x2s)
        error['p1']=np.asarray(error['p1'])
        error['p2']=np.asarray(error['p2'])
        data={
            'x1': x1,
            'x2': x2,
            'err': error
        }
        return data
    
    
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
    
    def get_nash(self):
        matrix = np.vstack((np.hstack((2*self.Q1 , self.A1)), np.hstack((self.A2.T, 2*self.Q2))))
        u=np.hstack((self.u, self.u))
        return -np.linalg.inv(matrix)@u

    def get_xd(self,u):
        matrix = np.vstack((np.hstack((2*self.Q1 , self.A1)), np.hstack((self.A2.T, 2*self.Q2))))
        return -np.linalg.inv(matrix)@u
        
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
                
    
    def check_jacobian(self,u):
        matrix = np.vstack((np.hstack((2*self.Q1 , self.A1)), np.hstack((self.A2.T, 2*self.Q2))))
        matrix=matrix+matrix.T
        eigenvalues = np.linalg.eigvals(matrix)
        return np.all(eigenvalues > 0)
    
    def get_alpha_L(self,u):
        matrix = np.vstack((np.hstack((2*self.Q1 , self.A1)), np.hstack((self.A2.T, 2*self.Q2))))
        #print("shape of matrix: ", np.shape(matrix))
        matrix=matrix+matrix.T
        eigenvalues = np.linalg.eigvals(matrix)

        return eigenvalues
    

class Incent():
    def __init__(self, ud, agent_dim, game, seed, UDCOST=False) -> None:
        self.ud = ud
        self.agent_dim = agent_dim
        self.u_dim = np.shape(ud)[0]
        self.game=game
        self.seed=seed
        np.random.seed(seed)
        self.UDCOST=UDCOST
        self.G=np.vstack((np.hstack((2*self.game.Q1 , self.game.A1)), 
                     np.hstack((self.game.A2.T, 2*self.game.Q2))))
        self.alpha=min(la.eigvals(la.inv(self.G).T@la.inv(self.G)))

    def _gradF(self, u):
        G=np.vstack((np.hstack((2*self.game.Q1 , self.game.A1)), 
                     np.hstack((self.game.A2.T, 2*self.game.Q2))))
        xd=self.game.get_xd(self.ud)
        if self.UDCOST:
            return la.inv(G).T@(xd+la.inv(G)@u)+(u-self.ud) 
        else:
            return la.inv(G).T@(xd+la.inv(G)@u)
    
    def runIncentBR(self, u0, eta=0.01, M=1000,ud=np.array([1,1,1,1]), var=0.0):
        us=[u0]
        d=self.agent_dim
        err=[la.norm(u0-ud)**2]
        for i in range(M):
            us.append(us[-1]-eta*self._gradF(us[-1])+np.random.normal(0,var,2*d))
            err.append(la.norm(us[-1]-ud)**2)
        err=np.asarray(err)
        us=np.asarray(us)
        data={
            'err': err,
            'us' : us
        }
        return data
    
    def _gradFDFOBR(self, u,x0=[], delta=1,MAXINNER=0, var=0.001):
        d=self.agent_dim*2
        v = np.random.normal(0,1, size=(d,))
        v=v/la.norm(v)
        if self.UDCOST:
            l = 0.5*la.norm(self.xd-self.game.get_xd(u+delta*v))**2+0.5*la.norm(u+delta*v-self.ud)**2
        else: 
            l = 0.5*la.norm(self.xd-self.game.get_xd(u+delta*v))**2
        return d*l*v/delta, x0
    
    def _gradFDFO(self, u,x0,delta=1, MAXINNER=1000, var=0.001):
        d=2*self.agent_dim
        v = np.random.normal(0,1, size=(d,))
        v=v/la.norm(v)
        xd=self.game.get_xd(self.ud)
        data=self.game.runSGDPlay(x0,u+delta*v, var=var, MAXINNER=MAXINNER)
        x1=data['x1']
        x2=data['x2']
        x0_new=np.hstack((x1[-1],x2[-1]))
        br=self.game.get_xd(u+delta*v)
        if self.UDCOST:
            l = 0.5*la.norm(xd-x0_new)**2+0.5*la.norm(u+delta*v-self.ud)**2
        else:
            l = 0.5*la.norm(xd-x0_new)**2
        return d*l*v/delta, x0_new
    
    def runIncentDFO(self, u0,x0, eta=0.005, MAXOUTER=5000,var=0.001, MAXINNER=1000, BRFLAG=False, DECAY=False, delta=1):
        us=[u0]
        err=[la.norm(u0-self.ud)**2]
        x0s=[x0]
        self.xd=self.game.get_xd(self.ud)
        if BRFLAG:
            grad_fn = self._gradFDFOBR
        else:
            grad_fn = self._gradFDFO
        if DECAY:
            eta0=eta
            alpha=1 #self.alpha
        for i in range(MAXOUTER):
            if DECAY and i>=1:
                eta=eta0/(alpha*(i+1))
            g_hat, x0=grad_fn(us[-1],x0s[-1], MAXINNER=MAXINNER,var=var, delta=delta)
            x0s.append(x0)
            us.append(us[-1]-eta*g_hat) 
            err.append(la.norm(us[-1]-self.ud)**2)
        err=np.asarray(err)
        us=np.asarray(us)
        x0s=np.asarray(x0s)
        data={
            'err': err,
            'us' : us,
            'x0s': x0s
        }
        return data
    
    def _gradFRGM(self, u):
        return u-self.ud

    def runIncentRGM(self, u0,x0, eta=0.005, MAXOUTER=5000,MAXINNER=500, BRFLAG=False):
        us=[u0]
        err=[la.norm(u0-self.ud)**2]
        x0s=[x0]
        self.xd=self.game.get_xd(self.ud)
        for i in range(MAXOUTER):
            g_hat=self._gradFRGM(us[-1])
            data=self.game.runSGDPlay(x0,us[-1], var=0.001, MAXINNER=MAXINNER)
            x1=data['x1']
            x2=data['x2']
            x0_new=np.hstack((x1[-1],x2[-1])) 
            us.append(us[-1]-eta*g_hat) 
            err.append(la.norm(us[-1]-self.ud)**2)
            x0s.append(x0_new)
        err=np.asarray(err)
        us=np.asarray(us)
        x0s=np.asarray(x0s)
        data={
            'err': err,
            'us' : us,
            'x0s': x0s
        }
        return data