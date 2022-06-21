from dolfin import *
from fenics import variable
from ..util import *

class Material():
    """
        GENERAL MATERIAL RELATION IMPLEMENTATION DEFORMATION 

        (STRAIN) -> FORCE (STRESS)

        ***NOTE: macro is needed for multiscale formulations
    
    """
    def __init__(self, u):

        d = len(u)  # Dimension of the problem
        
        self.I = Identity(d)    # Identity tensor

        self.deformation_measure = dict()
        self.deformation_measure_invariants = dict()
        self.stress_measures = dict()

        # Deformation gradient
        self.deformation_measure['F'] = self.I + nabla_grad(u)   

    def update_deformation_measures(self):
        self.deformation_measure['eps'] = 0.5 * \
        (self.deformation_measure['F']+self.deformation_measure['F'].T) - \
        self.I
        # Right Cauchy-Green strain 
        self.deformation_measure['C'] = self.deformation_measure['F'].T * \
                self.deformation_measure['F']
        self.deformation_measure['C_inv'] = inv(self.deformation_measure['C'])
        # Left Cauchy-Green strain
        self.deformation_measure['b'] = self.deformation_measure['F'] * \
                self.deformation_measure['F'].T
        # Almansi strain
        self.deformation_measure['e'] = 0.5 * (self.I - \
                inv(self.deformation_measure['b']))
        # Green strain 
        self.deformation_measure['E'] = 0.5 * (self.deformation_measure[\
               \
               'C']\
                - self.I)
        # Invariants 
        for key in self.deformation_measure: 
            self.deformation_measure_invariants[key] = Invariants(self.deformation_measure[key])

        return self.deformation_measure, self.deformation_measure_invariants
 
