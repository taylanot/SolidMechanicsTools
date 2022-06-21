from dolfin import *
import numpy as np
from ufl import algorithms
import copy as cp

class PeriodicBoundary(SubDomain):
    """
        GENERAL PERIODIC BOUNDARY CONDITIONS IMPLEMENTATION

                              #-----------# 
                             / |        / |
                            /  |       /  |
                           #----------#   |
                           *   |      |   |
                           *   #----------# 
                     [1]   *  *       |  / 
                           * * [2]    | / 
                           **         |/  
                           ***********# 
                               [0]

            *   : Master edges/nodes
            -   : Slave edges/nodes 
            [i] : directions for periodicity

   """

    def __init__(self,domain,periodicity=None,tolerance=DOLFIN_EPS):

        """ Initialize """

        SubDomain.__init__(self,tolerance)              

        ################################
        # Get the extrema of the domain for every direction
        ################################
        self.mins = np.array(domain.bounds[0])          
        self.maxs = np.array(domain.bounds[1])
        ################################
        # Mark non-zero directions
        ################################
        self.directions = np.flatnonzero(self.maxs - self.mins)     
        
        ################################
        # Definie periodic directions
        ################################
        if periodicity is None:
            self.periodic_dir = self.directions                 

        else:
            self.periodic_dir = periodicity

        self.tol = tolerance 
        self.master = []        # Master nodes 
        self.map_master = []    # Mapped master nodes
        self.map_slave = []     # Mapped slave nodes

    def inside(self,x,on_boundary):
        ################################
        # Mark the master nodes as True 
        ################################
        x_master=False                                                   
        if on_boundary:                                                         
            for axis in self.periodic_dir:                                      
                if near(x[axis],self.maxs[axis],self.tol):
                    x_master=False                                       
                    break
                elif near(x[axis],self.mins[axis],self.tol):
                    x_master=True                                        

        if x_master:
            self.master.append(cp.deepcopy(x))

        return x_master

    # Overwrite map method of SubDomain class master -> slave
    def map(self,x,y):
        ################################
        # Map the master nodes to slaves
        ################################
        x_slave=False                                       
        for axis in self.directions:                               
            if axis in self.periodic_dir:                          
                if near(x[axis],self.maxs[axis],self.tol):
                    y[axis]=self.mins[axis]                        
                    x_slave=True                            
                else:                                              
                    y[axis]=x[axis]                                
            else:                                                  
                y[axis]=x[axis]                                    
        ################################
        # add y to list mapped master coordinates
        # add x to list of mapped slave coordinates
        ################################
        if x_slave:
            self.map_master.append(cp.deepcopy(y))
            self.map_slave.append(cp.deepcopy(x)) 


