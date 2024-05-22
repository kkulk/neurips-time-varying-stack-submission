import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import scipy.linalg  as sla
import pickle 
from numpy import linalg as la
from tqdm import tqdm, trange
import os

### Utility Functions

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def get_data(loc_lst,price_lst_id, filename='../data/alldata.npy'):
    '''
    Load all data
    q: companies x dates x locations (sources) x prices
    '''
    q=np.load(filename)
    qp2=q[1]; q_p2=[] # Uber
    qp1=q[0]; q_p1=[] # Lyft
    for i in loc_lst:
        for p in price_lst_id:
            q_p2.append(qp2[:,i,p])
            q_p1.append(qp1[:,i,p])
    return np.asarray(q_p1),np.asarray(q_p2)

def reload_all_data(df, GET_PERFORMATIVE_PARAMS=False, STORE=False):
    '''
    Loads all data from rides data frame df
    '''
    companies_ = df.cab_type.unique()
    sources_ = np.delete(df.source.unique(),9)
    dates_ = df.date.unique()[0:14]
    prices_ = np.array([5+5*(i+1) for i in range(5)])
    q = np.zeros((len(companies_),len(dates_),len(sources_),len(prices_)))

    for i in range(len(companies_)):
        for j in trange(len(dates_)):
            for k in range(len(sources_)):
                for l in range(len(prices_)):
                    q[i,j,k,l] = len(df[(df['cab_type']==companies_[i])&(df['date']==dates_[j])
                                 &(df['source']==sources_[k])&(df['price']==prices_[l])])
    if GET_PERFORMATIVE_PARAMS:
        A=[]
        A_=[]
        q_bar = np.mean(q,axis=1)
        q_bar_p2=q_bar[1]
        q_bar_p1=q_bar[0]
        for source_idx in range(len(sources_)):
            A.append([np.diagflat((-3*q_bar_p2[source_idx])/(2*np.array(prices_))),
                     np.diagflat((-3*q_bar_p1[source_idx])/(2*np.array(prices_)))])
            A_.append([np.diagflat((3*q_bar_p1[source_idx])/(4*np.array(prices_))),
                        np.diagflat((3*q_bar_p2[source_idx])/(4*np.array(prices_)))])
        if STORE:
            np.save('own_performative_effects_A.npy', A)
            np.save('comp_performative_effects_A_.npy', A_)
        return q, A, A_
    else:
        return q
    
class ddrideshare:
    def __init__(self, loc_lst,price_lst,seed=2,lam=[0.0,0.0], base=True, params={'A1':[],'A2':[],'Ac1':[],'Ac2':[]},maxx=10,tot_rev=0, filename='../data/datadic.p'):
        self.loc_lst = loc_lst
        self.price_lst=price_lst
        self.n=2
        self.d=len(loc_lst)
        self.m=1
        self.lam1=lam[0]; 
        self.lam2=lam[1]
        self.l=[-maxx for i in range(self.d)]
        self.u=[maxx for i in range(self.d)]
        self.tot_rev=tot_rev
        if base:
            self.params=getparams(loc_lst)
            self.A1=self.params['A1']
            self.A2=self.params['A2']
            self.Ac1=self.params['Ac1']
            self.Ac2=self.params['Ac2']
        else:
            self.params=params
            self.A1=self.params['A1']
            self.A2=self.params['A2']
            self.Ac1=self.params['Ac1']
            self.Ac2=self.params['Ac2']
        self.I=np.eye(self.d)
        self.seed=seed
        np.random.seed(self.seed)
        self.q_dic=get_data_dic(filename)
        self.locations_ = self.q_dic['locations']
        self.dates_ = self.q_dic['dates']
        self.prices_=self.q_dic['prices']
        self.qlbar=self.q_dic['lyft_mean']
        self.qubar=self.q_dic['uber_mean']
        self.verbose=False

        
    def setup_distribution(self,centered=False):
        if centered:
            self.qu=self.q_dic['uber centered']
            self.ql=self.q_dic['lyft centered']
        else:
            self.qu=self.q_dic['uber']
            self.ql=self.q_dic['lyft']

        self.qlbar_=np.zeros((len(self.loc_lst),len(self.price_lst)))
        self.qubar_=np.zeros((len(self.loc_lst),len(self.price_lst)))
        self.ql_=np.zeros((len(self.dates_),len(self.loc_lst),len(self.price_lst)))
        self.qu_=np.zeros((len(self.dates_),len(self.loc_lst),len(self.price_lst)))
        for loc in self.loc_lst:
            for p in self.price_lst:
                self.qlbar_[loc,p]=self.qlbar[loc,p]
                self.qubar_[loc,p]=self.qubar[loc,p]
                self.ql_[:,loc,p]=self.ql[:,loc,p]
                self.qu_[:,loc,p]=self.qu[:,loc,p]
                
        self.locations = {'Haymarket Square':(42.3628, -71.0583), 'Back Bay':(42.3503, -71.0810),
                     'North End':(42.3647, -71.0542), 'North Station':(42.3661, -71.0631),
                     'Beacon Hill':(42.3588, -71.0707), 'Boston University':(42.3505, -71.1054),
                     'Fenway':(42.3467, -71.0972), 'South Station':(42.3519, -71.0552),
                     'Theatre District':(42.3519, -71.0643),# 'West End':(42.3644, -71.0661),
                     'Financial District':(42.3559, -71.0550), 'Northeastern University':(42.3398, -71.0892)}

        self.sources = []
        self.lats = []
        self.lons = []
        self.locs_ids={}
        for source in self.locations_:
            self.locs_ids[source]=self.locations[source]
        for source, coord in self.locs_ids.items():
            self.sources.append(source)
            self.lats.append(coord[0])
            self.lons.append(coord[1])
                
    def proj(self,x):
        y=np.zeros(np.shape(x))
        for i in range(self.n):
            for j in range(self.d):
                if x[i][j]<=self.l[j]:
                    y[i][j]=self.l[j]
                elif self.l[j]<x[i][j] and x[i][j]<self.u[j]:
                    y[i][j]=x[i][j]
                else:
                    y[i][j]=self.u[j]
        return y

    def getgrad_incentive(self,x,z1_,z2_,u):

        p1=-(self.A1-self.lam1*self.I).T@x[0]-1/2*(z1_+u[0]+self.Ac1@x[1])
        p2=-(self.A2-self.lam2*self.I).T@x[1]-1/2*(z2_+u[1]+self.Ac2@x[0])

        return np.vstack((p1,p2))

    def getgrad(self,x,z1_,z2_, perform=[True,True], MYOPIC=False):
        if np.all(perform):
            p1=-(self.A1-self.lam1*self.I).T@x[0]-1/2*(z1_+self.Ac1@x[1])
            p2=-(self.A2-self.lam2*self.I).T@x[1]-1/2*(z2_+self.Ac2@x[0])
        else:
            if perform[0]:
                p1=-(self.A1-self.lam1*self.I).T@x[0]-1/2*(z1_+self.Ac1@x[1])
            else:
                if MYOPIC and not(perform[0]):
                    p1=(self.lam1*self.I).T@x[0]-1/2*(z1_)
                else:
                    p1=-(self.A1-self.lam1*self.I).T@x[0]-1/2*(z1_)
                
            if perform[1]:
                p2=-(self.A2-self.lam2*self.I).T@x[1]-1/2*(z2_+self.Ac2@x[0])
            else:
                #
                if MYOPIC and not(perform[1]):
                    p2=(self.lam2*self.I).T@x[1]-1/2*(z2_)
                else:
                    p2=-(self.A2-self.lam2*self.I).T@x[1]-1/2*(z2_)
        return np.vstack((p1,p2))
    
    def getgrad_nash(self,x,z1_,z2_):
        p1=-(self.A1-self.lam1*self.I).T@x[0]-1/2*(z1_+self.Ac1@x[1])
        p2=-(self.A2-self.lam2*self.I).T@x[1]-1/2*(z2_+self.Ac2@x[0])
        return np.vstack((p1,p2))
     
    def getgrad_so(self,x,z1_,z2_, ):

        p1=-(self.A1-self.lam1*self.I).T@x[0]-1/2*(z1_+self.Ac1@x[1])-1/2*(self.Ac2@x[0])
        p2=-(self.A2-self.lam2*self.I).T@x[1]-1/2*(z2_+self.Ac2@x[0])-1/2*(self.Ac1@x[1])

        return np.vstack((p1,p2))

    

    def getHess(self,x):
        H1=-(self.A1.T-self.lam1*self.I)
        H2=-(self.A2.T-self.lam2*self.self.I)
        return H1,H2
    
    def sample_base_demand(self,q_,n=1, batch=1):
        vals=np.random.choice(q_,size=batch)
        return np.mean(vals)

    def std_base_demand(self, q_, n=1, batch=1):
        vals=np.random.choice(q_,size=batch)
        return np.var(vals)/batch

    def D_z_std(self, q_,locs=None, batch=1):
        z0=[]
        if locs==None:
            for loc_id in self.loc_lst:
                z0.append(self.std_base_demand(q_[loc_id], batch=batch))
        else:
            for loc_id in locs:
                z0.append(self.std_base_demand(q_[loc_id], batch=batch))
        return np.asarray(z0).flatten() 
    
    def D_z(self, q_, locs=None,batch=1):
        z0=[]
        if locs==None:
            for loc_id in self.loc_lst:
                z0.append(self.sample_base_demand(q_[loc_id], batch=batch))
        else:
            for loc_id in locs:
                z0.append(self.sample_base_demand(q_[loc_id], batch=batch))
        return np.asarray(z0).flatten()
    

    def query_env_player(self, x,player, q_, locs=None, batch=1):
        '''
           if perform=true:
              return environment query with the performative effects
        '''
        z_=self.D_z(q_,locs=locs, batch=batch)
        #print(z_)
        #print(np.shape(z_))
        if player==0:
            z=z_+self.A1@x[0]+self.Ac1@x[1]
        if player==1:
            z=z_+self.A2@x[1]+self.Ac2@x[0]
        return z

    def loss(self,x,player,q_,locs=None,batch=1):
        z=self.query_env_player(x,player,q_,locs=locs,batch=batch)
        if player==0:
            return -0.5*z@x[player]+self.lam1*la.norm(x[player])
        else:
            return -0.5*z@x[player]+self.lam2*la.norm(x[player])
       

        
    def social_opt_cost(self,x,q1,q2,locs=None,batch=1):
        z1=self.query_env_player(x,0, q1,locs=locs,batch=batch)
        z2=self.query_env_player(x,1, q2,locs=locs,batch=batch)

        return -0.5*z1@x[0]+self.lam1*la.norm(x[0])-0.5*z2@x[1]+self.lam2*la.norm(x[1])

    def revenue(self,x,player,q_,batch=1, locs=None, price_index=0):
        z=self.query_env_player(x,player,q_,locs=locs, batch=batch)
        return 0.5*z@(x[player]+self.tot_rev*self.prices_[price_index])
    
    def revenue_so(self,x,q1,q2,batch=1, locs=None, price_index=0):
        z1=self.query_env_player(x,0, q1,locs=locs,batch=batch)
        z2=self.query_env_player(x,1, q2,locs=locs,batch=batch)
        return 0.5*z1@(x[0]+self.tot_rev*self.prices_[price_index])+0.5*z2@(x[1]+self.tot_rev*self.prices_[price_index])
    
    def revenue_loc(self,x,player,q_,locs=None, batch=1, price_index=0):
        z=self.query_env_player(x,player, q_,locs=locs,batch=batch)
        return 0.5*np.multiply(z,x[player]+self.tot_rev*self.prices_[price_index])
    
    def revenue_loc_so(self,x,q1,q2,locs=None, batch=1, price_index=0):
        z1=self.query_env_player(x,0, q1,locs=locs,batch=batch)
        z2=self.query_env_player(x,1, q2,locs=locs,batch=batch)
        return 0.5*np.multiply(z1,x[0]+self.tot_rev*self.prices_[price_index])+0.5*np.multiply(z2,x[1]+self.tot_rev*self.prices_[price_index])

    def demand(self,x,player,q_,locs=None,batch=1):
        return self.query_env_player(x,player, q_,locs=locs,batch=batch)    
    
    def runSGD(self,x0,price_index=0,eta=0.001,BATCH=10,MAXITER=1000, verbose=False, perform_sgd=[True,True], RETURN=True, MYOPIC=False, tot_rev=1):
        '''
        Runs for one price bin and all locations
        '''
        self.stepsize_sgd=eta
        self.batch_sgd=BATCH
        self.maxiter_sgd=MAXITER
        self.perform_comp_sgd=perform_sgd
        self.tot_rev=tot_rev
        self.price_index_sgd=price_index
        print("Price we are running at : ", self.prices_[self.price_index_sgd])
        q_lyft_=self.ql_[:,:,self.price_index_sgd].T
        q_uber_=self.qu_[:,:,self.price_index_sgd].T
        #print(np.shape(q_lyft_))
        
        self.xo_sgd=x0
        self.x_sgd=[x0]; 

        self.rev_sgd_p1=[self.revenue(self.x_sgd[-1],0,q_lyft_,price_index=self.price_index_sgd)]
        self.rev_sgd_p2=[self.revenue(self.x_sgd[-1],1,q_uber_,price_index=self.price_index_sgd)]
        self.rev_sgd_p1_loc=[self.revenue_loc(self.x_sgd[-1],0,q_lyft_,price_index=self.price_index_sgd)]
        self.rev_sgd_p2_loc=[self.revenue_loc(self.x_sgd[-1],1,q_uber_,price_index=self.price_index_sgd)]
        self.demand_sgd_p1=[self.demand(self.x_sgd[-1],0,q_lyft_)]
        self.demand_sgd_p2=[self.demand(self.x_sgd[-1],1,q_uber_)]
        self.loss_sgd_p1=[self.loss(self.x_sgd[-1],0,q_lyft_)]
        self.loss_sgd_p2=[self.loss(self.x_sgd[-1],1,q_uber_)]
        
        for i in range(self.maxiter_sgd):
            z1_=self.D_z(q_lyft_, batch=BATCH)
            z2_=self.D_z(q_uber_, batch=BATCH)
            self.x_sgd.append(self.proj(self.x_sgd[-1]-eta*self.getgrad(self.x_sgd[-1],z1_,z2_,perform=perform_sgd,MYOPIC=MYOPIC)))
            self.rev_sgd_p1.append(self.revenue(self.x_sgd[-1],0,q_lyft_,batch=BATCH,price_index=self.price_index_sgd))
            self.rev_sgd_p2.append(self.revenue(self.x_sgd[-1],1,q_uber_,batch=BATCH,price_index=self.price_index_sgd))
            self.rev_sgd_p1_loc.append(self.revenue_loc(self.x_sgd[-1],0,q_lyft_,batch=BATCH,price_index=self.price_index_sgd))
            self.rev_sgd_p2_loc.append(self.revenue_loc(self.x_sgd[-1],1,q_uber_,batch=BATCH,price_index=self.price_index_sgd))
            self.demand_sgd_p1.append(self.demand(self.x_sgd[-1],0,q_lyft_))
            self.demand_sgd_p2.append(self.demand(self.x_sgd[-1],1,q_uber_))
            self.loss_sgd_p1.append(self.loss(self.x_sgd[-1],0,q_lyft_))
            self.loss_sgd_p2.append(self.loss(self.x_sgd[-1],1,q_uber_))
        if RETURN:
            dic={}
            dic['x']=self.x_sgd
            dic['loss_p1']=np.asarray(self.loss_sgd_p1)
            dic['loss_p2']=np.asarray(self.loss_sgd_p2)
            dic['revenue_total_p1']=np.asarray(self.rev_sgd_p1)
            dic['revenue_total_p2']=np.asarray(self.rev_sgd_p2)
            dic['revenue_by_loc_p1']=np.asarray(self.rev_sgd_p1_loc)
            dic['revenue_by_loc_p2']=np.asarray(self.rev_sgd_p2_loc)
            dic['demand_p1']=np.asarray(self.demand_sgd_p1)
            dic['demand_p2']=np.asarray(self.demand_sgd_p2)
            return dic
    
    def runSGDAsync(self,x0,price_index=0,eta=0.001,BATCH=10,MAXITER=1000, 
                    verbose=False, perform_sgd=[True,True], RETURN=True, MYOPIC=False, 
                    tot_rev=1, ps=[1,1]):
        '''
        Runs for one price bin and all locations
        '''
        self.stepsize_sgd=eta
        self.batch_sgd=BATCH
        self.maxiter_sgd=MAXITER
        self.perform_comp_sgd=perform_sgd
        self.tot_rev=tot_rev
        self.price_index_sgd=price_index
        print("Price we are running at : ", self.prices_[self.price_index_sgd])
        q_lyft_=self.ql_[:,:,self.price_index_sgd].T
        q_uber_=self.qu_[:,:,self.price_index_sgd].T
        #print(np.shape(q_lyft_))
        self.sgd_ps=ps 
        self.sgd_pvals=[]
        self.xo_sgd=x0
        self.x_sgd=[x0]; 

        self.rev_sgd_p1=[self.revenue(self.x_sgd[-1],0,q_lyft_,price_index=self.price_index_sgd)]
        self.rev_sgd_p2=[self.revenue(self.x_sgd[-1],1,q_uber_,price_index=self.price_index_sgd)]
        self.rev_sgd_p1_loc=[self.revenue_loc(self.x_sgd[-1],0,q_lyft_,price_index=self.price_index_sgd)]
        self.rev_sgd_p2_loc=[self.revenue_loc(self.x_sgd[-1],1,q_uber_,price_index=self.price_index_sgd)]
        self.demand_sgd_p1=[self.demand(self.x_sgd[-1],0,q_lyft_)]
        self.demand_sgd_p2=[self.demand(self.x_sgd[-1],1,q_uber_)]
        self.loss_sgd_p1=[self.loss(self.x_sgd[-1],0,q_lyft_)]
        self.loss_sgd_p2=[self.loss(self.x_sgd[-1],1,q_uber_)]
        
        for i in range(self.maxiter_sgd):
            z1_=self.D_z(q_lyft_, batch=BATCH)
            z2_=self.D_z(q_uber_, batch=BATCH)
            p1update=np.random.binomial(1,ps[0])
            p2update=np.random.binomial(1,ps[1])
            self.sgd_pvals.append([p1update,p2update])
            update=self.proj(self.x_sgd[-1]-eta*self.getgrad(self.x_sgd[-1],z1_,z2_,perform=perform_sgd,MYOPIC=MYOPIC))
            
            if p1update and p2update:
                self.x_sgd.append(update)
            elif p1update and not(p2update):
                self.x_sgd.append(np.vstack((update[0],self.x_sgd[-1][1])))
            elif not(p1update) and p2update: 
                self.x_sgd.append(np.vstack((self.x_sgd[-1][0],update[1])))
            else:
                self.x_sgd.append(np.vstack((self.x_sgd[-1][0],self.x_sgd[-1][1])))

            self.rev_sgd_p1.append(self.revenue(self.x_sgd[-1],0,q_lyft_,batch=BATCH,price_index=self.price_index_sgd))
            self.rev_sgd_p2.append(self.revenue(self.x_sgd[-1],1,q_uber_,batch=BATCH,price_index=self.price_index_sgd))
            self.rev_sgd_p1_loc.append(self.revenue_loc(self.x_sgd[-1],0,q_lyft_,batch=BATCH,price_index=self.price_index_sgd))
            self.rev_sgd_p2_loc.append(self.revenue_loc(self.x_sgd[-1],1,q_uber_,batch=BATCH,price_index=self.price_index_sgd))
            self.demand_sgd_p1.append(self.demand(self.x_sgd[-1],0,q_lyft_))
            self.demand_sgd_p2.append(self.demand(self.x_sgd[-1],1,q_uber_))
            self.loss_sgd_p1.append(self.loss(self.x_sgd[-1],0,q_lyft_))
            self.loss_sgd_p2.append(self.loss(self.x_sgd[-1],1,q_uber_))
        if RETURN:
            dic={}
            dic['x']=self.x_sgd
            dic['loss_p1']=np.asarray(self.loss_sgd_p1)
            dic['loss_p2']=np.asarray(self.loss_sgd_p2)
            dic['revenue_total_p1']=np.asarray(self.rev_sgd_p1)
            dic['revenue_total_p2']=np.asarray(self.rev_sgd_p2)
            dic['revenue_by_loc_p1']=np.asarray(self.rev_sgd_p1_loc)
            dic['revenue_by_loc_p2']=np.asarray(self.rev_sgd_p2_loc)
            dic['demand_p1']=np.asarray(self.demand_sgd_p1)
            dic['demand_p2']=np.asarray(self.demand_sgd_p2)
            return dic
        
    def runNash(self,x0,price_index=0,eta=0.001,MAXITER=1000, RETURN=True):
        '''
        Runs for one price bin and all locations
        
        '''
        self.stepsize_nash=eta
        self.maxiter_nash=MAXITER
        self.price_index_nash=price_index
        print("Price we are running at : ", self.prices_[self.price_index_nash])
        q_lyft_=self.ql_[:,:,self.price_index_nash].T
        q_uber_=self.qu_[:,:,self.price_index_nash].T
        
        self.xo_nash=x0
        self.x_nash=[x0]; 
        z1_=np.mean(q_lyft_,axis=1)
        z2_=np.mean(q_uber_,axis=1)
        for i in range(self.maxiter_nash):
            self.x_nash.append(self.proj(self.x_nash[-1]-eta*self.getgrad_nash(self.x_nash[-1],z1_,z2_)))
            
        if RETURN:
            dic={}
            dic['nash']=self.x_nash
            return dic
        
    def runSO(self,x0,price_index=0,eta=0.001,BATCH=10,MAXITER=1000, verbose=False, RETURN=True,tot_rev=1):
        '''
        Runs for one price bin and all locations
        '''
        self.stepsize_so=eta
        self.batch_so=BATCH
        self.maxiter_so=MAXITER
        self.tot_rev=tot_rev
        self.price_index_so=price_index
        print("Price we are running at : ", self.prices_[self.price_index_so])
        q_lyft_=self.ql_[:,:,self.price_index_so].T
        q_uber_=self.qu_[:,:,self.price_index_so].T
        #print(np.shape(q_lyft_))
        
        self.xo_so=x0
        self.x_so=[x0]; 

        self.rev_so=[self.revenue_so(self.x_so[-1],q_lyft_,q_uber_,price_index=self.price_index_so)]
        self.rev_so_loc=[self.revenue_loc_so(self.x_so[-1],q_lyft_,q_uber_,price_index=self.price_index_so)]
        
        self.rev_so_p1=[self.revenue(self.x_so[-1],0,q_lyft_,price_index=self.price_index_so)]
        self.rev_so_p2=[self.revenue(self.x_so[-1],1,q_uber_,price_index=self.price_index_so)]
        self.rev_so_p1_loc=[self.revenue_loc(self.x_so[-1],0,q_lyft_,price_index=self.price_index_so)]
        self.rev_so_p2_loc=[self.revenue_loc(self.x_so[-1],1,q_uber_,price_index=self.price_index_so)]

        self.demand_so_p1=[self.demand(self.x_so[-1],0,q_lyft_)]
        self.demand_so_p2=[self.demand(self.x_so[-1],1,q_uber_)]
        self.loss_so_p1=[self.loss(self.x_so[-1],0,q_lyft_)]
        self.loss_so_p2=[self.loss(self.x_so[-1],1,q_uber_)]
        
        self.loss_so=[self.social_opt_cost(self.x_so[-1],q_lyft_,q_uber_)]
        
        for i in range(self.maxiter_sgd):
            z1_=self.D_z(q_lyft_, batch=BATCH)
            z2_=self.D_z(q_uber_, batch=BATCH)
            self.x_so.append(self.proj(self.x_so[-1]-eta*self.getgrad_so(self.x_so[-1],z1_,z2_)))
            
            self.rev_so_p1.append(self.revenue(self.x_so[-1],0,q_lyft_,batch=BATCH,price_index=self.price_index_so))
            self.rev_so_p2.append(self.revenue(self.x_so[-1],1,q_uber_,batch=BATCH,price_index=self.price_index_so))
            self.rev_so_p1_loc.append(self.revenue_loc(self.x_so[-1],0,q_lyft_,batch=BATCH,price_index=self.price_index_so))
            self.rev_so_p2_loc.append(self.revenue_loc(self.x_so[-1],1,q_uber_,batch=BATCH,price_index=self.price_index_so))
            self.demand_so_p1.append(self.demand(self.x_so[-1],0,q_lyft_))
            self.demand_so_p2.append(self.demand(self.x_so[-1],1,q_uber_))
            self.loss_so_p1.append(self.loss(self.x_so[-1],0,q_lyft_))
            self.loss_so_p2.append(self.loss(self.x_so[-1],1,q_uber_))
            self.rev_so.append(self.revenue_so(self.x_so[-1],q_lyft_,q_uber_,price_index=self.price_index_so))
            self.rev_so_loc.append(self.revenue_loc_so(self.x_so[-1],q_lyft_,q_uber_,price_index=self.price_index_so))
            self.loss_so.append(self.social_opt_cost(self.x_so[-1],q_lyft_,q_uber_))
        if RETURN:
            dic={}
            dic['x']=self.x_so
            dic['loss_p1']=np.asarray(self.loss_so_p1)
            dic['loss_p2']=np.asarray(self.loss_so_p2)
            dic['loss_so']=np.asarray(self.loss_so)
            dic['revenue_so_total']=np.asarray(self.rev_so)
            dic['revenue_so_loc']=np.asarray(self.rev_so_loc)
            dic['revenue_total_p1']=np.asarray(self.rev_so_p1)
            dic['revenue_total_p2']=np.asarray(self.rev_so_p2)
            dic['revenue_by_loc_p1']=np.asarray(self.rev_so_p1_loc)
            dic['revenue_by_loc_p2']=np.asarray(self.rev_so_p2_loc)
            dic['demand_p1']=np.asarray(self.demand_so_p1)
            dic['demand_p2']=np.asarray(self.demand_so_p2)
            return dic
    
    def update_estimate(self, x,z1_,z2_, B=1, UNCORR=False, RETURN_U=False):
        '''
        least squares update
        '''
        # query environment
        u1 = np.random.normal(0,B,size=(self.d,))
        u2 = np.random.normal(0,B,size=(self.d,))
        v = np.vstack((u1,u2))  
        q1=self.query_env_player(x+v,0, self.z1_base, batch=self.batch_agd)
        q2=self.query_env_player(x+v,1, self.z2_base, batch=self.batch_agd)
        z1=z1_+self.A1@x[0]+self.Ac1@x[1]
        z2=z2_+self.A2@x[1]+self.Ac2@x[0]
        
        # Update estimates
        barA1_hat = np.hstack((self.A1_hat[-1],self.Ac1_hat[-1]))
        #print("shape barA1_hat : ", np.shape(barA1_hat))
        if UNCORR:
            dA1=np.diag(np.diagonal(self.A1_hat[-1]))
            dAc1=np.diag(np.diagonal(self.Ac1_hat[-1]))
            barA1_hat = np.hstack((dA1,dAc1))

            #print("shape barA1_hat redesign : ", np.shape(barA1_hat))
        #print("shape     A1 hat: ", np.shape(self.A1_hat))
        #print("shape     A1    : ", np.shape(self.A1))
        #print("shape    Ac1 hat: ", np.shape(self.Ac1_hat))
        #print("shape    Ac1    : ", np.shape(self.Ac1))
        #print("shape bar A1 hat: ", np.shape(barA1_hat))
        v_temp=v.reshape(self.n*self.d,1)
        barA1_hat_new = barA1_hat + self.nu*((q1.reshape(self.d,1)-z1.reshape(self.d,1)-barA1_hat@v_temp)@(v_temp.T))

        barA2_hat = np.hstack((self.A2_hat[-1],self.Ac2_hat[-1]))
        if UNCORR:
            dA2=np.diag(np.diagonal(self.A2_hat[-1]))
            dAc2=np.diag(np.diagonal(self.Ac2_hat[-1]))
            barA2_hat = np.hstack((dA2,dAc2))
        v_ = np.vstack((u2,u1))
        v_temp=v_.reshape(self.n*self.d,1)
        barA2_hat_new = barA2_hat + self.nu*((q2.reshape(self.d,1)-z2.reshape(self.d,1)-barA2_hat@v_temp)@(v_temp.T))
        
        if UNCORR:
            A1_hat=np.diag(np.diagonal(barA1_hat_new[:,:self.d]))
            Ac1_hat=np.diag(np.diagonal(barA1_hat_new[:,self.d:]))
            A2_hat=np.diag(np.diagonal(barA2_hat_new[:,:self.d]))
            Ac2_hat=np.diag(np.diagonal(barA2_hat_new[:,self.d:]))
        else:
            A1_hat  = barA1_hat_new[:,:self.d]
            Ac1_hat = barA1_hat_new[:,self.d:]
            A2_hat = barA2_hat_new[:,:self.d]
            Ac2_hat = barA2_hat_new[:,self.d:]

        if RETURN_U:
            return A1_hat, Ac1_hat, A2_hat, Ac2_hat, v
        else:
            return A1_hat, Ac1_hat, A2_hat, Ac2_hat
    
    def getgrad_adaptive(self,x,z1_,z2_):
        '''
        '''
        if np.all(self.perform_agd):
            p1=-(self.A1_hat[-1]-self.lam1*self.I).T@x[0]-1/2*(z1_+self.Ac1_hat[-1]@x[1])
            p2=-(self.A2_hat[-1]-self.lam2*self.I).T@x[1]-1/2*(z2_+self.Ac2_hat[-1]@x[0])
        else:
            if self.perform_agd[0]:
                p1=-(self.A1_hat[-1]-self.lam1*self.I).T@x[0]-1/2*(z1_+self.Ac1_hat[-1]@x[1])
            else: 
                p1=-(self.A1_hat[-1]-self.lam1*self.I).T@x[0]-1/2*(z1_)
            if self.perform_agd[1]:
                p2=-(self.A2_hat[-1]-self.lam2*self.I).T@x[1]-1/2*(z2_+self.Ac2_hat[-1]@x[0])
            else:
                p2=-(self.A2_hat[-1]-self.lam2*self.I).T@x[1]-1/2*(z2_)
        return np.vstack((p1,p2))

    def runAGDAsync(self,x0,A_dic, price_index=0,eta=0.001,nu=0.01, 
                    BATCH=10,MAXITER=1000, verbose=False, perform_agd=[True,True], 
                    RETURN=True, INNERITER=1, B=1, UNCORR=False,tot_rev=1, ps=[1,1]):
        '''
        Runs for one price bin and all locations
        '''
        self.stepsize_agd=eta
        self.nu=nu
        self.batch_agd=BATCH
        self.maxiter_agd=MAXITER
        self.perform_agd=perform_agd
        self.A1_hat=[A_dic['A1_hat']]
        self.Ac1_hat=[A_dic['Ac1_hat']]
        self.A2_hat=[A_dic['A2_hat']]
        self.Ac2_hat=[A_dic['Ac2_hat']]
        self.tot_rev=tot_rev
        self.price_index_agd=price_index
        print("Price we are running at : ", self.prices_[self.price_index_agd])
        self.z1_base=self.ql_[:,:,self.price_index_agd].T
        self.z2_base=self.qu_[:,:,self.price_index_agd].T
        #print(np.shape(q_lyft_))
        self.ps=ps
        self.pvals=[]
        self.xo_agd=x0
        self.x_agd=[x0]; 
        eta=eta/(np.log(len(self.x_agd))+1)
        nu=nu/len(self.x_agd) #(np.log(len(self.x_agd))+1)
        #if len(self.x_agd)>100:
        #    nu=0.005
        #elif len(self.x_agd)>300:
        #    nu=0.005
        self.rev_agd_p1=[self.revenue(self.x_agd[-1],0,self.z1_base,price_index=self.price_index_agd)]
        self.rev_agd_p2=[self.revenue(self.x_agd[-1],1,self.z2_base,price_index=self.price_index_agd)]
        self.rev_agd_p1_loc=[self.revenue_loc(self.x_agd[-1],0,self.z1_base,price_index=self.price_index_agd)]
        self.rev_agd_p2_loc=[self.revenue_loc(self.x_agd[-1],1,self.z2_base,price_index=self.price_index_agd)]
        self.demand_agd_p1=[self.demand(self.x_agd[-1],0,self.z1_base)]
        self.demand_agd_p2=[self.demand(self.x_agd[-1],1,self.z2_base)]
        self.loss_agd_p1=[self.loss(self.x_agd[-1],0,self.z1_base)]
        self.loss_agd_p2=[self.loss(self.x_agd[-1],1,self.z2_base)]
        
        for i in range(self.maxiter_agd):
            z1=self.D_z(self.z1_base, batch=self.batch_agd)
            z2=self.D_z(self.z2_base,batch=self.batch_agd)

            p1update=np.random.binomial(1,ps[0])
            p2update=np.random.binomial(1,ps[1])
            self.pvals.append([p1update,p2update])
            update=self.proj(self.x_agd[-1]-eta*self.getgrad_adaptive(self.x_agd[-1], z1, z2))
            if p1update and p2update:
                self.x_agd.append(update)
            elif p1update and not(p2update):
                self.x_agd.append(np.vstack((update[0],self.x_agd[-1][1])))
            elif not(p1update) and p2update: 
                self.x_agd.append(np.vstack((self.x_agd[-1][0],update[1])))
            else:
                self.x_agd.append(np.vstack((self.x_agd[-1][0],self.x_agd[-1][1])))

            for i in range(INNERITER):
                A1_hat,Ac1_hat,A2_hat,Ac2_hat = self.update_estimate(self.x_agd[-1], z1, z2, B=B, UNCORR=UNCORR)
            self.A1_hat.append(A1_hat)
            self.Ac1_hat.append(Ac1_hat)
            self.A2_hat.append(A2_hat)
            self.Ac2_hat.append(Ac2_hat)
            self.rev_agd_p1.append(self.revenue(self.x_agd[-1],0, self.z1_base))
            self.rev_agd_p2.append(self.revenue(self.x_agd[-1],1, self.z2_base))

            self.rev_agd_p1_loc.append(self.revenue_loc(self.x_agd[-1],0,self.z1_base,batch=self.batch_agd,price_index=self.price_index_agd))
            self.rev_agd_p2_loc.append(self.revenue_loc(self.x_agd[-1],1,self.z2_base,batch=self.batch_agd,price_index=self.price_index_agd))
            self.demand_agd_p1.append(self.demand(self.x_agd[-1],0,self.z1_base))
            self.demand_agd_p2.append(self.demand(self.x_agd[-1],1,self.z2_base))
            self.loss_agd_p1.append(self.loss(self.x_agd[-1],0,self.z1_base))
            self.loss_agd_p2.append(self.loss(self.x_agd[-1],1,self.z2_base))
        if RETURN:
            dic={}
            dic['x']=self.x_agd
            dic['A1_hat']=self.A1_hat
            dic['Ac1_hat']=self.Ac1_hat
            dic['A2_hat']=self.A2_hat
            dic['Ac2_hat']=self.Ac2_hat
            dic['loss_p1']=np.asarray(self.loss_agd_p1)
            dic['loss_p2']=np.asarray(self.loss_agd_p2)
            dic['revenue_total_p1']=np.asarray(self.rev_agd_p1)
            dic['revenue_total_p2']=np.asarray(self.rev_agd_p2)
            dic['revenue_by_loc_p1']=np.asarray(self.rev_agd_p1_loc)
            dic['revenue_by_loc_p2']=np.asarray(self.rev_agd_p2_loc)
            dic['demand_p1']=np.asarray(self.demand_agd_p1)
            dic['demand_p2']=np.asarray(self.demand_agd_p2)
            return dic
        
    def zograd(self,x, z1_, z2_, A_1=[], A_1_=[], A_2=[], A_2_=[], delta=0.001, BATCH=1):
        p1 = np.zeros(self.d)
        p2 = np.zeros(self.d)
        for sample in range(BATCH):
            v1_ = np.random.normal(0,1,size=(self.d,))
            v1 = v1_/la.norm(v1_)
            v2_ = np.random.normal(0,1,size=(self.d,))
            v2 = v2_/la.norm(v2_)

            z1 = z1_+A_1@(x[0]+(delta*v1))+A_1_@(x[1]+(delta*v2))
            l1 = -0.5*z1@(x[0]+(delta*v1))+self.lam1*la.norm(x[0]+(delta*v1))
            p1 += (self.d/delta)*l1*v1

            z2 = z2_+A_2@(x[1]+(delta*v2))+A_2_@(x[0]+(delta*v1))
            l2 = -0.5*z2@(x[1]+(delta*v2))+self.lam2*la.norm(x[1]+(delta*v2))
            p2 += (self.d/delta)*l2*v2
            
        p1=p1/BATCH
        p2=p2/BATCH
        return np.vstack((p1,p2))

    def runAGD(self,x0,A_dic, price_index=0,eta=0.001,nu=0.01, BATCH=10,MAXITER=1000, verbose=False, perform_agd=[True,True], RETURN=True, INNERITER=1, B=1, UNCORR=False,tot_rev=1):
        '''
        Runs for one price bin and all locations
        '''
        self.stepsize_agd=eta
        self.nu=nu
        self.batch_agd=BATCH
        self.maxiter_agd=MAXITER
        self.perform_agd=perform_agd
        self.A1_hat=[A_dic['A1_hat']]
        self.Ac1_hat=[A_dic['Ac1_hat']]
        self.A2_hat=[A_dic['A2_hat']]
        self.Ac2_hat=[A_dic['Ac2_hat']]
        self.tot_rev=tot_rev
        self.price_index_agd=price_index

        print("Price we are running at : ", self.prices_[self.price_index_agd])
        self.z1_base=self.ql_[:,:,self.price_index_agd].T
        self.z2_base=self.qu_[:,:,self.price_index_agd].T
        #print(np.shape(q_lyft_))
        
        self.xo_agd=x0
        self.x_agd=[x0]; 
        eta=eta/(np.log(len(self.x_agd))+1)
        nu=nu/len(self.x_agd) #(np.log(len(self.x_agd))+1)
        #if len(self.x_agd)>100:
        #    nu=0.005
        #elif len(self.x_agd)>300:
        #    nu=0.005
        self.rev_agd_p1=[self.revenue(self.x_agd[-1],0,self.z1_base,price_index=self.price_index_agd)]
        self.rev_agd_p2=[self.revenue(self.x_agd[-1],1,self.z2_base,price_index=self.price_index_agd)]
        self.rev_agd_p1_loc=[self.revenue_loc(self.x_agd[-1],0,self.z1_base,price_index=self.price_index_agd)]
        self.rev_agd_p2_loc=[self.revenue_loc(self.x_agd[-1],1,self.z2_base,price_index=self.price_index_agd)]
        self.demand_agd_p1=[self.demand(self.x_agd[-1],0,self.z1_base)]
        self.demand_agd_p2=[self.demand(self.x_agd[-1],1,self.z2_base)]
        self.loss_agd_p1=[self.loss(self.x_agd[-1],0,self.z1_base)]
        self.loss_agd_p2=[self.loss(self.x_agd[-1],1,self.z2_base)]
        
        for i in range(self.maxiter_agd):
            z1=self.D_z(self.z1_base, batch=self.batch_agd)
            z2=self.D_z(self.z2_base,batch=self.batch_agd)

            self.x_agd.append(self.proj(self.x_agd[-1]-eta*self.getgrad_adaptive(self.x_agd[-1], z1, z2)))
            for i in range(INNERITER):
                A1_hat,Ac1_hat,A2_hat,Ac2_hat = self.update_estimate(self.x_agd[-1], z1, z2, B=B, UNCORR=UNCORR)
            self.u_vals.append(v)
            self.A1_hat.append(A1_hat)
            self.Ac1_hat.append(Ac1_hat)
            self.A2_hat.append(A2_hat)
            self.Ac2_hat.append(Ac2_hat)
            self.rev_agd_p1.append(self.revenue(self.x_agd[-1],0, self.z1_base))
            self.rev_agd_p2.append(self.revenue(self.x_agd[-1],1, self.z2_base))

            self.rev_agd_p1_loc.append(self.revenue_loc(self.x_agd[-1],0,self.z1_base,batch=self.batch_agd,price_index=self.price_index_agd))
            self.rev_agd_p2_loc.append(self.revenue_loc(self.x_agd[-1],1,self.z2_base,batch=self.batch_agd,price_index=self.price_index_agd))
            self.demand_agd_p1.append(self.demand(self.x_agd[-1],0,self.z1_base))
            self.demand_agd_p2.append(self.demand(self.x_agd[-1],1,self.z2_base))
            self.loss_agd_p1.append(self.loss(self.x_agd[-1],0,self.z1_base))
            self.loss_agd_p2.append(self.loss(self.x_agd[-1],1,self.z2_base))
        if RETURN:
            dic={}
            dic['x']=self.x_agd
            dic['A1_hat']=self.A1_hat
            dic['Ac1_hat']=self.Ac1_hat
            dic['A2_hat']=self.A2_hat
            dic['Ac2_hat']=self.Ac2_hat
            dic['loss_p1']=np.asarray(self.loss_agd_p1)
            dic['loss_p2']=np.asarray(self.loss_agd_p2)
            dic['revenue_total_p1']=np.asarray(self.rev_agd_p1)
            dic['revenue_total_p2']=np.asarray(self.rev_agd_p2)
            dic['revenue_by_loc_p1']=np.asarray(self.rev_agd_p1_loc)
            dic['revenue_by_loc_p2']=np.asarray(self.rev_agd_p2_loc)
            dic['demand_p1']=np.asarray(self.demand_agd_p1)
            dic['demand_p2']=np.asarray(self.demand_agd_p2)
            
            return dic
        
    def gradRGD(self,x,z1_,z2_):
        p1=(self.lam1*self.I).T@x[0]-1/2*(z1_)
        p2=(self.lam2*self.I).T@x[1]-1/2*(z2_)
        return np.vstack((p1,p2))

    def runRGD(self,x0,price_index=0,eta=0.001,BATCH=10,MAXITER=1000, verbose=False, RETURN=True,tot_rev=1):
        '''
        Runs for one price bin and all locations
        '''
        self.stepsize_rgd=eta
        self.batch_rgd=BATCH
        self.maxiter_rgd=MAXITER
        self.tot_rev=tot_rev
        self.price_index_rgd=price_index
        print("Price we are running at : ", self.prices_[self.price_index_rgd])
        print('maxiter : ', MAXITER)
        print('maxiter rgd: ', self.maxiter_rgd)
        q_lyft_=self.ql_[:,:,self.price_index_rgd].T
        q_uber_=self.qu_[:,:,self.price_index_rgd].T
        #print(np.shape(q_lyft_))
        
        self.xo_rgd=x0
        self.x_rgd=[x0]; 

        self.rev_rgd_p1=[self.revenue(self.x_rgd[-1],0,q_lyft_,price_index=self.price_index_rgd)]
        self.rev_rgd_p2=[self.revenue(self.x_rgd[-1],1,q_uber_,price_index=self.price_index_rgd)]
        self.rev_rgd_p1_loc=[self.revenue_loc(self.x_rgd[-1],0,q_lyft_,price_index=self.price_index_rgd)]
        self.rev_rgd_p2_loc=[self.revenue_loc(self.x_rgd[-1],1,q_uber_,price_index=self.price_index_rgd)]
        self.demand_rgd_p1=[self.demand(self.x_rgd[-1],0,q_lyft_)]
        self.demand_rgd_p2=[self.demand(self.x_rgd[-1],1,q_uber_)]
        self.loss_rgd_p1=[self.loss(self.x_rgd[-1],0,q_lyft_)]
        self.loss_rgd_p2=[self.loss(self.x_rgd[-1],1,q_uber_)]
        
        for i in range(self.maxiter_rgd):
            z1_=self.query_env_player(self.x_rgd[-1], 0,q_lyft_)
            z2_=self.query_env_player(self.x_rgd[-1], 1,q_uber_)
            self.x_rgd.append(self.proj(self.x_rgd[-1]-eta*self.gradRGD(self.x_rgd[-1],z1_,z2_)))
            
            self.rev_rgd_p1.append(self.revenue(self.x_rgd[-1],0,q_lyft_,batch=BATCH,price_index=self.price_index_rgd))
            self.rev_rgd_p2.append(self.revenue(self.x_rgd[-1],1,q_uber_,batch=BATCH,price_index=self.price_index_rgd))
            self.rev_rgd_p1_loc.append(self.revenue_loc(self.x_rgd[-1],0,q_lyft_,batch=BATCH,price_index=self.price_index_rgd))
            self.rev_rgd_p2_loc.append(self.revenue_loc(self.x_rgd[-1],1,q_uber_,batch=BATCH,price_index=self.price_index_rgd))
            self.demand_rgd_p1.append(self.demand(self.x_rgd[-1],0,q_lyft_))
            self.demand_rgd_p2.append(self.demand(self.x_rgd[-1],1,q_uber_))
            self.loss_rgd_p1.append(self.loss(self.x_rgd[-1],0,q_lyft_))
            self.loss_rgd_p2.append(self.loss(self.x_rgd[-1],1,q_uber_))
            
        if RETURN:
            dic={}
            dic['x']=self.x_rgd
            dic['loss_p1']=np.asarray(self.loss_rgd_p1)
            dic['loss_p2']=np.asarray(self.loss_rgd_p2)
            dic['revenue_total_p1']=np.asarray(self.rev_rgd_p1)
            dic['revenue_total_p2']=np.asarray(self.rev_rgd_p2)
            dic['revenue_by_loc_p1']=np.asarray(self.rev_rgd_p1_loc)
            dic['revenue_by_loc_p2']=np.asarray(self.rev_rgd_p2_loc)
            dic['demand_p1']=np.asarray(self.demand_rgd_p1)
            dic['demand_p2']=np.asarray(self.demand_rgd_p2)
            return dic
    
    def get_dataframe_for_plot(self,rev_ig_p1, rev_ig_p2, demand_ig_p1, demand_ig_p2, rev_ig_p1_loc, rev_ig_p2_loc,
                               rev_comp_p1, rev_comp_p2, demand_comp_p1, demand_comp_p2, rev_comp_p1_loc, rev_comp_p2_loc,x_comp, x_ig,
                               shift=4900, shift_amt=0.001, mean_back=100, scale=1):
        locations = {'Haymarket Square':(42.3628, -71.0583), 'Back Bay':(42.3503, -71.0810),
                     'North End':(42.3647, -71.0542), 'North Station':(42.3661, -71.0631),
                     'Beacon Hill':(42.3588, -71.0707), 'Boston University':(42.3505, -71.1054),
                     'Fenway':(42.3467, -71.0972), 'South Station':(42.3519, -71.0552),
                     'Theatre District':(42.3519, -71.0643),# 'West End':(42.3644, -71.0661),
                     'Financial District':(42.3559, -71.0550), 'Northeastern University':(42.3398, -71.0892)}

        sources = []
        lats = []
        lons = []
        locs_={}
        for source in self.locations_:
            locs_[source]=locations[source]
        for source, coord in locs_.items():
            sources.append(source)
            lats.append(coord[0])
            lons.append(coord[1])

        df_dic_all_a={}
        df_dic_all_a['centroid_lat']=[]
        df_dic_all_a['centroid_lon']=[]
        df_dic_all_a['demand']=[]
        df_dic_all_a['demand_change']=[]
        df_dic_all_a['company']=[]
        #df_dic_all_a['revenue']=[]
        df_dic_all_a['revenue_change']=[]
        df_dic_all_a['revenue_sign']=[]
        df_dic_all_a['demand_sign']=[]
        df_dic_all_b={}
        df_dic_all_b['centroid_lat']=[]
        df_dic_all_b['centroid_lon']=[]
        df_dic_all_b['demand']=[]
        df_dic_all_b['demand_change']=[]
        #df_dic_all_b['revenue']=[]
        df_dic_all_b['revenue_change']=[]
        df_dic_all_b['company']=[]
        df_dic_all_b['revenue_sign']=[]
        df_dic_all_b['demand_sign']=[]
        df_dic_all_a['price_change']=[]
        df_dic_all_b['price_change']=[]
        lyft_avg_price=np.mean(x_comp[-mean_back:,0,:], axis=0)
        uber_avg_price=np.mean(x_comp[-mean_back:,1,:], axis=0)
        lyft_avg_price_ig=np.mean(x_ig[-mean_back:,0,:], axis=0)
        uber_avg_price_ig=np.mean(x_ig[-mean_back:,1,:], axis=0)
        scale=1
        for ind,source in enumerate(self.locations_):
            #print(ind)
            for comp in [0,1]:
                if comp==1:

                    df_dic_all_a['centroid_lat'].append(locs_[source][0]+shift_amt)
                    df_dic_all_a['centroid_lon'].append(locs_[source][1])
                    change=(np.mean(rev_comp_p2_loc[ind][-mean_back:])-np.mean(rev_ig_p2_loc[ind][-mean_back:],axis=0))
                    df_dic_all_a['demand'].append(np.mean(demand_ig_p2[ind][-mean_back:],axis=0)*scale)
                    if change<=0:
                        df_dic_all_a['revenue_change'].append((change)*scale) #np.abs(change)*scale)
                        df_dic_all_a['company'].append(r'Uber')
                        df_dic_all_a['revenue_sign'].append(r'decrease')
                    else:
                        df_dic_all_a['revenue_change'].append(np.abs(change)*scale)
                        df_dic_all_a['company'].append(r'Uber')
                        df_dic_all_a['revenue_sign'].append(r'increase')
                    df_dic_all_a['price_change'].append(uber_avg_price[ind]-uber_avg_price_ig[ind])
                    change=(np.mean(demand_comp_p2[ind][-mean_back:],axis=0)-np.mean(demand_ig_p2[ind][-mean_back:],axis=0))
                    if change<=0:
                        df_dic_all_a['demand_change'].append((change)*scale) #np.abs(change)*scale)
                        #df_dic_all_a['company'].append(r'Uber')
                        df_dic_all_a['demand_sign'].append(r'decrease')
                    else:
                        df_dic_all_a['demand_change'].append(np.abs(change)*scale)
                        #df_dic_all_a['company'].append(r'Uber')
                        df_dic_all_a['demand_sign'].append(r'increase')

                else:

                    df_dic_all_b['centroid_lat'].append(locs_[source][0])
                    df_dic_all_b['centroid_lon'].append(locs_[source][1])
                    df_dic_all_b['demand'].append(np.mean(demand_ig_p1[ind][-mean_back:],axis=0)*scale)
                    change=(np.mean(rev_comp_p1_loc[ind][-mean_back:],axis=0)-np.mean(rev_ig_p1_loc[ind][-mean_back:],axis=0))
                    if change<=0:
                        df_dic_all_b['revenue_change'].append((change)*scale) #np.abs(change)*scale)
                        df_dic_all_b['company'].append(r'Lyft')
                        df_dic_all_b['revenue_sign'].append(r'decrease')
                    else:
                        df_dic_all_b['revenue_change'].append(np.abs(change)*scale)
                        df_dic_all_b['company'].append(r'Lyft')
                        df_dic_all_b['revenue_sign'].append(r'increase')
                        
                    df_dic_all_b['price_change'].append(lyft_avg_price[ind]-lyft_avg_price_ig[ind])
                    change=(np.mean(demand_comp_p1[ind][-mean_back:],axis=0)-np.mean(demand_ig_p1[ind][-mean_back:],axis=0))
                    if change<=0:
                        df_dic_all_b['demand_change'].append((change)*scale) #np.abs(change)*scale)
                        #df_dic_all_b['company'].append(r'Lyft')
                        df_dic_all_b['demand_sign'].append(r'decrease')
                    else:
                        df_dic_all_b['demand_change'].append(np.abs(change)*scale)
                        #df_dic_all_b['company'].append(r'Lyft')
                        df_dic_all_b['demand_sign'].append(r'increase')


        df_all_u=pd.DataFrame(df_dic_all_a)
        df_all_l=pd.DataFrame(df_dic_all_b)
        return df_all_u,df_all_l #df_dic_all_a, df_dic_all_b #
    
    def rungame_cor(self,A1_,A2_,Ac1_,Ac2_,x0,price_index=0,eta=0.001,BATCH=10,MAXITER=1000, verbose=False, reset=False):

        self.A1_=np.copy(self.A1)
        self.A2_=np.copy(self.A2)
        self.Ac1_=np.copy(self.Ac1)
        self.Ac2_=np.copy(self.Ac2)
        self.price_index=price_index
        print("Price we are running at : ", self.prices_[self.price_index])
        q_lyft_=self.ql_[:,:,self.price_index].T
        q_uber_=self.qu_[:,:,self.price_index].T
        
        ind_i,ind_j=np.nonzero(A1_)
        for ind_i_,ind_j_ in zip(ind_i,ind_j):
            self.A1[ind_i_,ind_j_]=A1_[ind_i_,ind_j_]
            #print(A1_[ind_i_,ind_j_],ind_i_,ind_j_)
        ind_i,ind_j=np.nonzero(A2_)
        for ind_i_,ind_j_ in zip(ind_i,ind_j):
            self.A2[ind_i_,ind_j_]=A2_[ind_i_,ind_j_]
        ind_i,ind_j=np.nonzero(Ac1_)
        for ind_i_,ind_j_ in zip(ind_i,ind_j):
            #print(self.Ac1[int(ind_i_),int(ind_j_)],ind_i_,ind_j_)
            self.Ac1[ind_i_,ind_j_]=Ac1_[ind_i_,ind_j_]
            
        ind_i,ind_j=np.nonzero(Ac2_)
        for ind_i_,ind_j_ in zip(ind_i,ind_j):
            self.Ac2[ind_i_,ind_j_]=Ac2_[ind_i_,ind_j_]
            #print(Ac2_[ind_i_,ind_j_],ind_i_,ind_j_)
            
        #for pair in cors.values():
        #    ind_i=pair[0][0]
        #    ind_j=pair[0][1]
        #    val= pair[1]
        #    if pair[3]=='mu' and pair[2]==0:
        #        self.A1[ind_i,ind_j]=val
        #    elif pair[3]=='mu' and pair[2]==1:
        #        self.A2[ind_i,ind_j]=val
        #    elif pair[3]=='gamma' and pair[2]==0:
        #        self.Ac1[ind_i,ind_j]=val
        #    elif pair[3]=='gamma' and pair[2]==1:
        #        self.Ac2[ind_i,ind_j]=val

        self.x_comp=[x0]; perform_comp=[True,True]

        self.rev_comp_p1=[self.revenue(self.x_comp[-1],0,q_lyft_)]
        self.rev_comp_p2=[self.revenue(self.x_comp[-1],1,q_uber_)]
        self.demand_comp_p1=[self.demand(self.x_comp[-1],0,q_lyft_)]
        self.demand_comp_p2=[self.demand(self.x_comp[-1],1,q_uber_)]
        for i in range(MAXITER):
            z1_=self.D_z(q_lyft_, batch=BATCH)
            z2_=self.D_z(q_uber_, batch=BATCH)
            self.x_comp.append(self.proj(self.x_comp[-1]-eta*self.getgrad(self.x_comp[-1],z1_,z2_,perform=perform_comp)))
            self.rev_comp_p1.append(self.revenue(self.x_comp[-1],0,q_lyft_,batch=BATCH))
            self.rev_comp_p2.append(self.revenue(self.x_comp[-1],1,q_uber_,batch=BATCH))
            self.demand_comp_p1.append(self.demand(self.x_comp[-1],0,q_lyft_))
            self.demand_comp_p2.append(self.demand(self.x_comp[-1],1,q_uber_))
        if verbose:
            print("A1 Lyft  : \n", self.A1)
            print("Am1 Lyft : \n", self.Ac1)
            print("A2 Uber  : \n", self.A2)
            print("Am2 Uber : \n", self.Ac2)

        if verbose:
            H1,H2=self.getHess(self.x_comp[-1])
            print("eigs p1  : ", la.eigvals(H1))
            print("eigs p2  : ", la.eigvals(H2))

            print("gradient : ", self.getgrad(self.x_comp[-1],z1_,z2_))
            print("Revenues when P1 considers performative effects")
            print("p1 revenue: \n", self.revenue(self.x_comp[-1],0,q_lyft_,batch=BATCH))
            print("p2 revenue: \n", self.revenue(self.x_comp[-1],1,q_uber_,batch=BATCH))
            print("x_comp : \n", self.x_comp[-1])
        if reset:
            self.A1=np.copy(self.A1_)
            self.A2=np.copy(self.A2_)
            self.Ac1=np.copy(self.Ac1_)
            self.Ac2=np.copy(self.Ac2_)
        return self.x_comp, self.rev_comp_p1, self.rev_comp_p2,  np.asarray(self.demand_comp_p1), np.asarray(self.demand_comp_p2)
    

    def update_estimate_decay(self, x,z1_,z2_, B=1,nu=0.01, UNCORR=False, RETURN_U=False, decay=1, idx=0):
        '''
        least squares update
        '''
        # query environment
        self.nu=nu
        u1 = np.random.normal(0,B,size=(self.d,))
        u2 = np.random.normal(0,B,size=(self.d,))
        v = np.vstack((u1,u2))*decay**idx
        q1=self.query_env_player(x+v,0, self.z1_base, batch=self.batch_agd)
        q2=self.query_env_player(x+v,1, self.z2_base, batch=self.batch_agd)
        z1=z1_+self.A1@x[0]+self.Ac1@x[1]
        z2=z2_+self.A2@x[1]+self.Ac2@x[0]
        
        # Update estimates
        barA1_hat = np.hstack((self.A1_hat[-1],self.Ac1_hat[-1]))
        #print("shape barA1_hat : ", np.shape(barA1_hat))
        if UNCORR:
            dA1=np.diag(np.diagonal(self.A1_hat[-1]))
            dAc1=np.diag(np.diagonal(self.Ac1_hat[-1]))
            barA1_hat = np.hstack((dA1,dAc1))

            #print("shape barA1_hat redesign : ", np.shape(barA1_hat))
        #print("shape     A1 hat: ", np.shape(self.A1_hat))
        #print("shape     A1    : ", np.shape(self.A1))
        #print("shape    Ac1 hat: ", np.shape(self.Ac1_hat))
        #print("shape    Ac1    : ", np.shape(self.Ac1))
        #print("shape bar A1 hat: ", np.shape(barA1_hat))
        v_temp=v.reshape(self.n*self.d,1)
        barA1_hat_new = barA1_hat + self.nu*((q1.reshape(self.d,1)-z1.reshape(self.d,1)-barA1_hat@v_temp)@(v_temp.T))

        barA2_hat = np.hstack((self.A2_hat[-1],self.Ac2_hat[-1]))
        if UNCORR:
            dA2=np.diag(np.diagonal(self.A2_hat[-1]))
            dAc2=np.diag(np.diagonal(self.Ac2_hat[-1]))
            barA2_hat = np.hstack((dA2,dAc2))
        v_ = np.vstack((u2,u1))
        v_temp=v_.reshape(self.n*self.d,1)*decay**idx
        barA2_hat_new = barA2_hat + self.nu*((q2.reshape(self.d,1)-z2.reshape(self.d,1)-barA2_hat@v_temp)@(v_temp.T))
        
        if UNCORR:
            A1_hat=np.diag(np.diagonal(barA1_hat_new[:,:self.d]))
            Ac1_hat=np.diag(np.diagonal(barA1_hat_new[:,self.d:]))
            A2_hat=np.diag(np.diagonal(barA2_hat_new[:,:self.d]))
            Ac2_hat=np.diag(np.diagonal(barA2_hat_new[:,self.d:]))
        else:
            A1_hat  = barA1_hat_new[:,:self.d]
            Ac1_hat = barA1_hat_new[:,self.d:]
            A2_hat = barA2_hat_new[:,:self.d]
            Ac2_hat = barA2_hat_new[:,self.d:]

        if RETURN_U:
            return A1_hat, Ac1_hat, A2_hat, Ac2_hat, v
        else:
            return A1_hat, Ac1_hat, A2_hat, Ac2_hat

    def update_estimate_decay_alt(self, x,u1,u2,z1_,z2_, B=1,nu=0.01, UNCORR=False, RETURN_U=False, decay=1, idx=0):
        '''
        least squares update
        '''
        # query environment
        self.nu=nu
        v = np.vstack((u1,u2))*decay**idx
        q1=self.query_env_player(x+v,0, self.z1_base, batch=self.batch_agd)
        q2=self.query_env_player(x+v,1, self.z2_base, batch=self.batch_agd)
        z1=z1_+self.A1@x[0]+self.Ac1@x[1]
        z2=z2_+self.A2@x[1]+self.Ac2@x[0]
        
        # Update estimates
        barA1_hat = np.hstack((self.A1_hat[-1],self.Ac1_hat[-1]))
        #print("shape barA1_hat : ", np.shape(barA1_hat))
        if UNCORR:
            dA1=np.diag(np.diagonal(self.A1_hat[-1]))
            dAc1=np.diag(np.diagonal(self.Ac1_hat[-1]))
            barA1_hat = np.hstack((dA1,dAc1))

            #print("shape barA1_hat redesign : ", np.shape(barA1_hat))
        #print("shape     A1 hat: ", np.shape(self.A1_hat))
        #print("shape     A1    : ", np.shape(self.A1))
        #print("shape    Ac1 hat: ", np.shape(self.Ac1_hat))
        #print("shape    Ac1    : ", np.shape(self.Ac1))
        #print("shape bar A1 hat: ", np.shape(barA1_hat))
        v_temp=v.reshape(self.n*self.d,1)
        barA1_hat_new = barA1_hat + self.nu*((q1.reshape(self.d,1)-z1.reshape(self.d,1)-barA1_hat@v_temp)@(v_temp.T))

        barA2_hat = np.hstack((self.A2_hat[-1],self.Ac2_hat[-1]))
        if UNCORR:
            dA2=np.diag(np.diagonal(self.A2_hat[-1]))
            dAc2=np.diag(np.diagonal(self.Ac2_hat[-1]))
            barA2_hat = np.hstack((dA2,dAc2))
        v_ = np.vstack((u2,u1))
        v_temp=v_.reshape(self.n*self.d,1)*decay**idx
        barA2_hat_new = barA2_hat + self.nu*((q2.reshape(self.d,1)-z2.reshape(self.d,1)-barA2_hat@v_temp)@(v_temp.T))
        
        if UNCORR:
            A1_hat=np.diag(np.diagonal(barA1_hat_new[:,:self.d]))
            Ac1_hat=np.diag(np.diagonal(barA1_hat_new[:,self.d:]))
            A2_hat=np.diag(np.diagonal(barA2_hat_new[:,:self.d]))
            Ac2_hat=np.diag(np.diagonal(barA2_hat_new[:,self.d:]))
        else:
            A1_hat  = barA1_hat_new[:,:self.d]
            Ac1_hat = barA1_hat_new[:,self.d:]
            A2_hat = barA2_hat_new[:,:self.d]
            Ac2_hat = barA2_hat_new[:,self.d:]

        if RETURN_U:
            return A1_hat, Ac1_hat, A2_hat, Ac2_hat, v
        else:
            return A1_hat, Ac1_hat, A2_hat, Ac2_hat

    def getgrad_in(self,x,u1,u2,z1_,z2_, perform=[True,True], MYOPIC=False):

        #p1=-(self.A1-self.lam1*self.I).T@x[0]-1/2*(z1_+self.Ac1@x[1])
        #p2=-(self.A2-self.lam2*self.I).T@x[1]-1/2*(z2_+self.Ac2@x[0])
        p1=-(self.A1).T@(x[0]+u1)-1/2*(z1_+self.Ac1@(x[1]+u2))+self.lam1*self.I.T@(x[0])
        p2=-(self.A2).T@(x[1]+u2)-1/2*(z2_+self.Ac2@(x[0]+u1))+self.lam2*self.I.T@(x[1])
        return np.vstack((p1,p2))

    def runAGDIncent(self,x0,A_dic, seed=42,price_index=0,eta=0.001,nu=0.01, BATCH=10,MAXITER=1000, 
                     verbose=False, perform_agd=[True,True], 
                     RETURN=True, INNERITER=1, B=1, UNCORR=False,tot_rev=1, decay=1,lam1=0,lam2=0
                     ):
        '''
        Runs for one price bin and all locations
        '''
        self.seed=seed
        #self.lam1=lam1
        #self.lam2=lam2
        np.random.seed(seed)
        self.stepsize_agd=eta
        self.nu=nu
        self.batch_agd=BATCH
        self.maxiter_agd=MAXITER
        self.perform_agd=perform_agd
        self.A1_hat=[A_dic['A1_hat']]
        self.Ac1_hat=[A_dic['Ac1_hat']]
        self.A2_hat=[A_dic['A2_hat']]
        self.Ac2_hat=[A_dic['Ac2_hat']]
        self.tot_rev=tot_rev
        self.uvals=[]
        self.price_index_agd=price_index
        print("Price we are running at : ", self.prices_[self.price_index_agd])
        self.z1_base=self.ql_[:,:,self.price_index_agd].T
        self.z2_base=self.qu_[:,:,self.price_index_agd].T
        #print(np.shape(q_lyft_))
        nu0=nu
        self.xo_agd=x0
        self.x_agd=[x0]; 
        eta=eta/(np.log(len(self.x_agd))+1)
        print(eta)
        nu=nu/len(self.x_agd) #(np.log(len(self.x_agd))+1)
        nu=1*2*nu/(len(self.x_agd)+2*3*self.d)
        print(nu0)
        
        #if len(self.x_agd)>100:
        #    nu=0.005
        #elif len(self.x_agd)>300:
        #    nu=0.005
        self.rev_agd_p1=[self.revenue(self.x_agd[-1],0,self.z1_base,price_index=self.price_index_agd)]
        self.rev_agd_p2=[self.revenue(self.x_agd[-1],1,self.z2_base,price_index=self.price_index_agd)]
        self.rev_agd_p1_loc=[self.revenue_loc(self.x_agd[-1],0,self.z1_base,price_index=self.price_index_agd)]
        self.rev_agd_p2_loc=[self.revenue_loc(self.x_agd[-1],1,self.z2_base,price_index=self.price_index_agd)]
        self.demand_agd_p1=[self.demand(self.x_agd[-1],0,self.z1_base)]
        self.demand_agd_p2=[self.demand(self.x_agd[-1],1,self.z2_base)]
        self.loss_agd_p1=[self.loss(self.x_agd[-1],0,self.z1_base)]
        self.loss_agd_p2=[self.loss(self.x_agd[-1],1,self.z2_base)]
        v=np.array([[0],[0]]) 
        for i in range(self.maxiter_agd):
            nu=1*2*nu0/(len(self.x_agd)+2*3*self.d)
            #print(nu)
            u1 = np.random.normal(0,B,size=(self.d,))
            u2 = np.random.normal(0,B,size=(self.d,))
            v = np.vstack((u1,u2))*decay**i
            u1_=u1*decay**i
            u2_=u2*decay**i
            for _ in range(INNERITER):
                z1=self.D_z(self.z1_base, batch=self.batch_agd)
                z2=self.D_z(self.z2_base,batch=self.batch_agd)

                self.x_agd.append(self.proj(self.x_agd[-1]-eta*self.getgrad_in(self.x_agd[-1],1*u1_,1*u2_, z1, z2)))
            
            A1_hat,Ac1_hat,A2_hat,Ac2_hat, v = self.update_estimate_decay_alt(self.x_agd[-1], 
                                                                              u1,u2,
                                                                              z1, z2, 
                                                                              B=B, 
                                                                              nu=nu,
                                                                              UNCORR=UNCORR, 
                                                                              RETURN_U=True, 
                                                                              decay=decay, 
                                                                              idx=i
                                                                              )
            self.uvals.append(v)
            self.A1_hat.append(A1_hat)
            self.Ac1_hat.append(Ac1_hat)
            self.A2_hat.append(A2_hat)
            self.Ac2_hat.append(Ac2_hat)
            self.rev_agd_p1.append(self.revenue(self.x_agd[-1],0, self.z1_base))
            self.rev_agd_p2.append(self.revenue(self.x_agd[-1],1, self.z2_base))

            self.rev_agd_p1_loc.append(self.revenue_loc(self.x_agd[-1],0,self.z1_base,batch=self.batch_agd,price_index=self.price_index_agd))
            self.rev_agd_p2_loc.append(self.revenue_loc(self.x_agd[-1],1,self.z2_base,batch=self.batch_agd,price_index=self.price_index_agd))
            self.demand_agd_p1.append(self.demand(self.x_agd[-1],0,self.z1_base))
            self.demand_agd_p2.append(self.demand(self.x_agd[-1],1,self.z2_base))
            self.loss_agd_p1.append(self.loss(self.x_agd[-1],0,self.z1_base))
            self.loss_agd_p2.append(self.loss(self.x_agd[-1],1,self.z2_base))
        if RETURN:
            dic={}
            dic['x']=self.x_agd
            dic['A1_hat']=self.A1_hat
            dic['Ac1_hat']=self.Ac1_hat
            dic['A2_hat']=self.A2_hat
            dic['Ac2_hat']=self.Ac2_hat
            dic['loss_p1']=np.asarray(self.loss_agd_p1)
            dic['loss_p2']=np.asarray(self.loss_agd_p2)
            dic['revenue_total_p1']=np.asarray(self.rev_agd_p1)
            dic['revenue_total_p2']=np.asarray(self.rev_agd_p2)
            dic['revenue_by_loc_p1']=np.asarray(self.rev_agd_p1_loc)
            dic['revenue_by_loc_p2']=np.asarray(self.rev_agd_p2_loc)
            dic['demand_p1']=np.asarray(self.demand_agd_p1)
            dic['demand_p2']=np.asarray(self.demand_agd_p2)
            dic['uvals']=np.asarray(self.uvals)
            return dic

def get_data_dic(filename='../data/datadic.p'):
    print(filename)
    return pickle.load(open(filename,'rb'))
        
        

    
    
def getcor(q_lyft_,q_uber_):
    q_lyft_loc_means=[]
    q_uber_loc_means=[]

    q_lyft_loc_means=np.mean(q_lyft_,axis=1)
    q_uber_loc_means=np.mean(q_uber_,axis=1)
    print(q_lyft_loc_means.shape, q_lyft_.shape)
    ql_=q_lyft_.T-q_lyft_loc_means
    ql_[:,0]=ql_[:,0]/np.std(ql_[:,0])
    ql_[:,1]=ql_[:,1]/np.std(ql_[:,1])

    qu_=q_uber_.T-q_uber_loc_means
    qu_[:,0]=qu_[:,0]/np.std(qu_[:,0])
    qu_[:,1]=qu_[:,1]/np.std(qu_[:,1])
    
    print(np.multiply(ql_[:,0],qu_[:,1]))
    
def getparams(loc_lst, floc_base="../data/"):
    mu=np.load(floc_base+'mu_est.npy')
    gamma=np.load(floc_base+'gamma_est.npy')
    who=['Lyft values','Uber values']
    A={}
    B={}
    for i in range(2):
        A[who[i]]=[]
        B[who[i]]=[]
        for j in loc_lst:
            B[who[i]].append(gamma[j][i][0,0])
            A[who[i]].append(mu[j][i][0,0])

        B[who[i]]=np.asarray(B[who[i]])
        A[who[i]]=np.asarray(A[who[i]])
    A1=np.diag(A[who[0]])
    A2=np.diag(A[who[1]])
    Ac1=np.diag(B[who[0]])
    Ac2=np.diag(B[who[1]])
    dic={}
    dic['A1']=A1
    dic['A2']=A2
    dic['Ac1']=Ac1
    dic['Ac2']=Ac2
    return dic




# print(u_so)
# u_so_=u_so.reshape(2,d)
# print(u_so_)


class incent():

    def __init__(self, x_so, ddgame, x0, MAXOUTER=1000, MAXINNER=1000, seed=42, d=11, n=2, eta=5e-5, m=1, gamma=1e-3, batch=1000,
                 nash=np.array([[5],[3]])):
        np.random.seed(seed)
        self.seed=seed
        self.batch=batch
        self.x0=x0
        self.MAXINNER=MAXINNER
        self.MAXOUTER=MAXOUTER
        self.x_sgd_s={}
        self.x_rgd_s={}
        self.x_sgd=[self.x0]
        self.x_rgd=[self.x0]
        self.VERBOSE=0
        self.x_so=x_so
        self.ddgame=ddgame
        self.eta=eta
        self.gamma=gamma
        self.err_sgd=[la.norm(self.x_sgd[-1]-self.x_so)]
        self.err_nash=[la.norm(self.x_sgd[-1]-nash)]
        
        self.m=m
        self.nash=np.asarray(nash)
        print(self.nash)

        G1=np.hstack((self.ddgame.lam1*np.eye(self.m)-self.ddgame.A1, -0.5*self.ddgame.Ac1))
        G2=np.hstack(( -0.5*self.ddgame.Ac2,self.ddgame.lam2*np.eye(self.m)-self.ddgame.A2))
        self.G=np.vstack((G1,G2))  
        
        self.mean_xi = np.vstack((self.ddgame.D_z(self.ddgame.ql_[:,:,self.ddgame.price_index_sgd].T, batch=self.batch*100), 
                     self.ddgame.D_z(self.ddgame.qu_[:,:,self.ddgame.price_index_sgd].T, batch=self.batch*100)))
        
    
        self.u_so =  2*self.G@self.x_so-self.mean_xi
        self.x_star=lambda u: 0.5*la.inv(self.G)@(u+self.mean_xi)
        
        
        G1ps = np.hstack((self.ddgame.lam1*np.eye(self.m)-0.5*self.ddgame.A1, -0.5*self.ddgame.Ac1)) 
        G2ps = np.hstack(( -0.5*self.ddgame.Ac2,self.ddgame.lam2*np.eye(self.m)-0.5*self.ddgame.A2))
        self.Gps=np.vstack((G1ps,G2ps))

        self.x_ps = self.ddgame.proj(0.5*la.inv(self.Gps)@self.mean_xi)
        self.err_rgd=[la.norm(self.x_rgd[-1]-self.x_so)]
        self.err_ps=[la.norm(self.x_rgd[-1]-self.x_ps)]
        self.u_so_rgp =  2*self.Gps@self.x_so-self.mean_xi 
        self.x_star_rgp=lambda u: 0.5*la.inv(self.Gps)@(u+self.mean_xi)

    def _getoptgradF(self, u,x):
        xi=np.vstack((self.ddgame.D_z(self.ddgame.ql_[:,:,self.ddgame.price_index_sgd].T, batch=1000), 
                     self.ddgame.D_z(self.ddgame.qu_[:,:,self.ddgame.price_index_sgd].T, batch=self.batch)))
        
        return 0.5*la.inv(self.G).T@(self.x_star(u)-self.x_so)
    
    def _getgradF(self, u,x):
        return 0.5*la.inv(self.G).T@(x-self.x_so)
    
    def _getgradFDFO(self,u,x):
        v = np.random.normal(0,1, size=(2,))
        v=v/la.norm(v)
        l = 0.5*la.norm(x-self.x_so)**2
        return 2*l*v
    
    def _getgradFRGP(self, u,x):
        return 0.5*la.inv(self.Gps).T@(x-self.x_so)
    
    def _getgrad_incentive(self,x,z1_,z2_,u):

        p1=-(self.ddgame.A1-self.ddgame.lam1*self.ddgame.I)@x[0]-1/2*(z1_+u[0]+self.ddgame.Ac1@x[1])
        p2=-(self.ddgame.A2-self.ddgame.lam2*self.ddgame.I)@x[1]-1/2*(z2_+u[1]+self.ddgame.Ac2@x[0])

        return np.vstack((p1,p2))
    
    def _getRGP_incentive(self,x,z1_,z2_,u):
        z1 = self.ddgame.A1@x[0]+self.ddgame.Ac1@x[1]+u[0]+z1_
        z2 = self.ddgame.A2@x[1]+self.ddgame.Ac2@x[0]+u[1]+z2_ 
        p1=self.ddgame.lam1*self.ddgame.I@x[0]-1/2*z1
        p2=self.ddgame.lam2*self.ddgame.I@x[1]-1/2*z2

        return np.vstack((p1,p2))
    
    def runSGP(self,x0,u, eta=0.001, MEAN=False):
        self.x_sgd=[x0]
        q_lyft_=self.ddgame.ql_[:,:,self.ddgame.price_index_sgd].T
        q_uber_=self.ddgame.qu_[:,:,self.ddgame.price_index_sgd].T
        for i in range(self.MAXINNER):
            if MEAN:
                z1_=self.mean_xi[0]
                z2_=self.mean_xi[1]
            else: 
                z1_=self.ddgame.D_z(q_lyft_, batch=self.batch)
                z2_=self.ddgame.D_z(q_uber_, batch=self.batch)
            self.x_sgd.append(
                self.ddgame.proj(self.x_sgd[-1]
                                 -eta*(self._getgrad_incentive(self.x_sgd[-1],1*z1_,1*z2_,u)
                                            +0.0*np.random.rand(2,1)
                                            )
                                )
                            )
            self.err_sgd.append(la.norm(self.x_sgd[-1]-self.x_so))
            self.err_nash.append(la.norm(self.x_sgd[-1]-self.nash))
        return None
    
    def runRGP(self,x0,u, eta=0.001, MEAN=False):
        self.x_rgd=[x0]
        q_lyft_=self.ddgame.ql_[:,:,self.ddgame.price_index_sgd].T
        q_uber_=self.ddgame.qu_[:,:,self.ddgame.price_index_sgd].T
        for i in range(self.MAXINNER):
            if MEAN:
                z1_=self.mean_xi[0]
                z2_=self.mean_xi[1]
            else: 
                z1_=self.ddgame.D_z(q_lyft_, batch=self.batch)
                z2_=self.ddgame.D_z(q_uber_, batch=self.batch)
            self.x_rgd.append(
                self.ddgame.proj(self.x_rgd[-1]
                                 -self.eta*(self._getRGP_incentive(self.x_rgd[-1],1*z1_,1*z2_,u)
                                            +0.0*np.random.rand(2,1)
                                            )
                                )
                            )
            self.err_rgd.append(la.norm(self.x_rgd[-1]-self.x_so))
            self.err_ps.append(la.norm(self.x_rgd[-1]-self.x_ps))
        return None
    
    def runIncentRGP(self,x0,u0, gamma=None, MAXOUTER=None, MEAN=False):
        np.random.seed(self.seed)
        self.err_xr=[]
        self.err_ur=[]
        self.err_p=[]
        self.us_rgp=[u0]
        x_star_rgp=self.x_star_rgp(self.u_so_rgp)
        self.x_r=[x0]
        if gamma!=None:
            self.gamma=gamma
        if MAXOUTER!=None:
            M=MAXOUTER
        else:
            M=self.MAXOUTER

        for i in range(M):
            self.runRGP(x0,self.us_rgp[-1], MEAN=MEAN)
            self.x_rgd_s[i]=self.x_rgd
            self.us_rgp.append(self.us_rgp[-1]-self.gamma*(self._getgradFRGP(self.us_rgp[-1],self.x_rgd[-1]))) #+np.random.rand(2,1)))
            x0=np.copy(self.x_rgd[-1])
            self.x_r.append(self.x_rgd[-1])
            self.err_xr.append(la.norm(self.x_r[-1]-x_star_rgp))
            self.err_ur.append(la.norm(self.us_rgp[-1]-self.u_so_rgp))
            self.err_p.append(la.norm(self.x_r[-1]-self.x_ps))

        self.us_rgp=np.asarray(self.us_rgp)
        self.x_r=np.asarray(self.x_r)
        self.err_xr=np.asarray(self.err_xr) 
        self.err_ur=np.asarray(self.err_ur)
        self.err_p=np.asarray(self.err_p)  

    def runIncent(self,x0,u0, gamma=None, MAXOUTER=None, MEAN=False):
        np.random.seed(self.seed)
        self.err_x=[]
        self.err_u=[]
        self.err_n=[]
        self.us=[u0]
        x_star=self.x_star(self.u_so)
        self.x_s=[x0]
        if gamma!=None:
            self.gamma=gamma
        if MAXOUTER!=None:
            M=MAXOUTER
        else:
            M=self.MAXOUTER

        for i in range(M):
            print(self.eta)
            self.runSGP(x0,self.us[-1], eta=self.eta, MEAN=MEAN)
            self.x_sgd_s[i]=self.x_sgd
            self.us.append(self.us[-1]-self.gamma*(self._getgradF(self.us[-1],self.x_sgd[-1]))) #+np.random.rand(2,1)))
            x0=np.copy(self.x_sgd[-1])
            self.x_s.append(self.x_sgd[-1])
            self.err_x.append(la.norm(self.x_s[-1]-x_star))
            self.err_u.append(la.norm(self.us[-1]-self.u_so))
            self.err_n.append(la.norm(self.x_s[-1]-self.nash))
            #print(i, ": ", self.us[-1], self._getgradF(self.us[-1],self.x_sgd[-1])+np.random.rand(2,1))
            #print(len(self.us))
            if self.VERBOSE:
                print(self.x_sgd[-1])
                print(self._getgradF(self.us[-1],self.x_sgd[-1]))
                print(self.us[-1])
                print()
        self.us=np.asarray(self.us)
        self.x_s=np.asarray(self.x_s)
        self.err_x=np.asarray(self.err_x) 
        self.err_u=np.asarray(self.err_u)
        self.err_n=np.asarray(self.err_n)  
    
    def runIncentDFO(self,x0,u0, gamma=None, MAXOUTER=None, MEAN=False, delta=10*0.5):
        np.random.seed(self.seed)
        self.err_x=[]
        self.err_u=[]
        self.err_n=[]
        self.us=[u0]
        self.grad_dfo=[]
        self.grads=[]
        x_star=self.x_star(self.u_so)
        self.x_s=[x0]
        if gamma!=None:
            self.gamma=gamma
        if MAXOUTER!=None:
            M=MAXOUTER
        else:
            M=self.MAXOUTER
        alpha=la.norm(la.inv(self.G))
        for i in range(M):
            #print(i)
            self.runSGP(x0,self.us[-1], MEAN=MEAN)
            self.x_sgd_s[i]=self.x_sgd
            grad=np.asarray(self._getgradFDFO(self.us[-1],self.x_sgd[-1])/delta).reshape(2,1)
            self.grad_dfo.append(self._getgradFDFO(self.us[-1],self.x_sgd[-1])/delta)
            self.grads.append(self._getgradF(self.us[-1],self.x_sgd[-1]))
            if i <10000:
                self.us.append(self.us[-1]-self.gamma/(alpha)*grad) #+np.random.rand(2,1)))
            else:
                self.us.append(self.us[-1]-self.gamma/(alpha*(i-20+1))*grad)
            x0=np.copy(self.x_sgd[-1])
            self.x_s.append(self.x_sgd[-1])
            self.err_x.append(la.norm(self.x_s[-1]-x_star))
            self.err_u.append(la.norm(self.us[-1]-self.u_so))
            self.err_n.append(la.norm(self.x_s[-1]-self.nash))
            #print(i, ": ", self.us[-1], self._getgradF(self.us[-1],self.x_sgd[-1])+np.random.rand(2,1))
            #print(len(self.us))
            if self.VERBOSE:
                print(self.x_sgd[-1])
                print(self._getgradF(self.us[-1],self.x_sgd[-1]))
                print(self.us[-1])
                print()
        self.us=np.asarray(self.us)
        self.x_s=np.asarray(self.x_s)
        self.err_x=np.asarray(self.err_x) 
        self.err_u=np.asarray(self.err_u)
        self.err_n=np.asarray(self.err_n)  

    def runIncentOpt(self,x0,u0, gamma=None, MAXOUTER=None):
        np.random.seed(self.seed)
        self.us=[u0]
        self.err_x=[]
        self.err_u=[]
        self.x_star_s=[self.x_star(u0)]
        x_star=self.x_star(self.u_so)
        if gamma!=None:
            self.gamma=gamma
        if MAXOUTER!=None:
            M=MAXOUTER
        else:
            M=self.MAXOUTER
        for i in range(M):
            #self.runSGP(x0,self.us[-1])
            #self.x_sgd_s[i]=self.x_sgd
            self.us.append(self.us[-1]-self.gamma*(self._getgradF(self.us[-1],self.x_star(self.us[-1]))+0.0*np.random.rand(2,1)))
            x0=np.copy(self.x_sgd[-1])
            self.x_star_s.append(self.x_star(self.us[-1]))
            #print(i, ": ", self.us[-1])
            #print(self._getgradF(self.us[-1],self.x_star(self.us[-1])))
            #print(len(self.us))
            self.err_x.append(la.norm(self.x_star_s[-1]-x_star))
            self.err_u.append(la.norm(self.us[-1]-self.u_so))
            if self.VERBOSE:
                print(self.x_sgd[-1])
                print(self._getgradF(self.us[-1],self.x_sgd[-1]))
                print(self.us[-1])
                print()         
        self.us=np.asarray(self.us)
        self.x_star_s=np.asarray(self.x_star_s)
        self.err_x=np.asarray(self.err_x)   
        self.err_u=np.asarray(self.err_u)

    def _runSGPBRstop(self,x0,u, eta=0.001, MEAN=False):
        self.x_sgd=[x0]
        q_lyft_=self.ddgame.ql_[:,:,self.ddgame.price_index_sgd].T
        q_uber_=self.ddgame.qu_[:,:,self.ddgame.price_index_sgd].T
        x_br=self.x_star(u)
        for i in range(10000*self.MAXINNER):
            if MEAN:
                z1_=self.mean_xi[0]
                z2_=self.mean_xi[1]
            else: 
                z1_=self.ddgame.D_z(q_lyft_, batch=self.batch)
                z2_=self.ddgame.D_z(q_uber_, batch=self.batch)
            self.x_sgd.append(
                self.ddgame.proj(self.x_sgd[-1]
                                 -eta*(self._getgrad_incentive(self.x_sgd[-1],1*z1_,1*z2_,u)
                                            +0.0*np.random.rand(2,1)
                                            )
                                )
                            )
            self.err_sgd.append(la.norm(self.x_sgd[-1]-self.x_so))
            self.err_nash.append(la.norm(self.x_sgd[-1]-self.nash))
            if la.norm(x_br-self.x_sgd[-1])<=self.stop_criteria:
                break
        return i
    
    def runIncentBRstop(self,x0,u0, gamma=None, MAXOUTER=None, MEAN=False, eps=1e-4, eta_outer=None):
        np.random.seed(self.seed)
        self.err_x=[]
        self.err_u=[]
        self.err_n=[]
        self.err_x_br=[]
        self.us=[u0]
        self.stop_criteria=eps
        x_star=self.x_star(self.u_so)
        self.x_s=[x0]
        self.stop_sequence=[]
        if gamma!=None:
            self.gamma=gamma
        if MAXOUTER!=None:
            M=MAXOUTER
        else:
            M=self.MAXOUTER
        if eta_outer==None:
            step_size=self.eta
        else:
            step_size=eta_outer
        for i in range(M):
            print(self.eta)
            stop_iter=self._runSGPBRstop(x0,self.us[-1], eta=step_size, MEAN=MEAN)
            self.stop_sequence.append(stop_iter)
            self.x_sgd_s[i]=self.x_sgd
            self.us.append(self.us[-1]-self.gamma*(self._getgradF(self.us[-1],self.x_sgd[-1]))) #+np.random.rand(2,1)))
            x0=np.copy(self.x_sgd[-1])
            self.x_s.append(self.x_sgd[-1])
            self.err_x.append(la.norm(self.x_s[-1]-x_star))
            self.err_x_br.append(la.norm(self.x_s[-1]-self.x_star(self.us[-2])))
            self.err_u.append(la.norm(self.us[-1]-self.u_so))
            self.err_n.append(la.norm(self.x_s[-1]-self.nash))
            #print(i, ": ", self.us[-1], self._getgradF(self.us[-1],self.x_sgd[-1])+np.random.rand(2,1))
            #print(len(self.us))
            if self.VERBOSE:
                print(self.x_sgd[-1])
                print(self._getgradF(self.us[-1],self.x_sgd[-1]))
                print(self.us[-1])
                print()
        self.us=np.asarray(self.us)
        self.x_s=np.asarray(self.x_s)
        self.err_x=np.asarray(self.err_x) 
        self.err_u=np.asarray(self.err_u)
        self.err_n=np.asarray(self.err_n)             
#for i in range(0,MAXOUTER):
# x_sgd=runSGP(u_so_)
# x_sgd=np.asarray(x_sgd)
# plt.plot(x_sgd[:,0,0])
# plt.plot(x_sgd[:,1,0])
# x_sgd[-1], x_so