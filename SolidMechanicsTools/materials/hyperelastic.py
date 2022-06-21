from ..src.continuum import * 
from ..util import * 
import dolfin

class NeoHookean(Material):
    """
        Neo-Hookean material model implementation
    """
    def __init__(self, u, E, nu):

        """ Initialize """

        ################################
        # Initialize material properties
        ################################

        Material.__init__(self,u=u)    # Initialize base-class
        
        self.update_deformation_measures()
        mu , lmbda  = Lame(E, nu)
        self.mu = Constant(mu)
        self.lmbda= Constant(lmbda)

    def __call__(self, F):

        assert (type(F) == dolfin.function.expression.Expression) or \
                (type(F) == dolfin.function.constant.Constant), \
                "F should be dolfin Constant or Expression"
        self.deformation_measure['F'] += F
        self.deformation_measure['F'] = \
                variable(self.deformation_measure['F'])
        self.update_deformation_measures()

        J = self.deformation_measure_invariants['F'][2]
        I_C_1 = self.deformation_measure_invariants['C'][0]

        psi = self.mu/2. * (I_C_1 - 2 - 2*ln(J)) + self.lmbda/2.*(J-1)**2
        self.stress_measures['P'] = diff(psi, self.deformation_measure['F'])

class SVenantKirchhoff(Material):
    """
        SVenantKirchhoff material model implementation
    """
    def __init__(self, u, E, nu):

        """ Initialize """

        ################################
        # Initialize material properties
        ################################

        Material.__init__(self,u=u)    # Initialize base-class
        
        self.update_deformation_measures()
        mu , lmbda  = Lame(E, nu)
        self.mu = Constant(mu)
        self.lmbda= Constant(lmbda)

    def __call__(self, F):

        assert (type(F) == dolfin.function.expression.Expression) or \
                (type(F) == dolfin.function.constant.Constant), \
                "F should be dolfin Constant or Expression"

        self.deformation_measure['F'] += F
        self.deformation_measure['F'] = \
                variable(self.deformation_measure['F'])
        self.update_deformation_measures()

        E = self.deformation_measure['E']
        psi = self.lmbda/2*(tr(E))**2 + self.mu*tr(E*E.T) 
        self.stress_measures['P'] = diff(psi, self.deformation_measure['F'])

class Yeoh(Material):
    """
        Yeoh material model implementation
    """
    def __init__(self, u, cs, ds):

        """ Initialize """

        ################################
        # Initialize material properties
        ################################

        Material.__init__(self,u=u)    # Initialize base-class
        
        self.update_deformation_measures()

        assert len(cs) == 3, "There should be 3 constants in cs!"
        assert len(ds) == 3, "There should be 3 constants in ds!"

        self.cs = cs 
        self.ds = ds 

    def __call__(self, F):

        assert (type(F) == dolfin.function.expression.Expression) or \
                (type(F) == dolfin.function.constant.Constant), \
                "F should be dolfin Constant or Expression"

        self.deformation_measure['F'] += F

        self.update_deformation_measures()

        J = self.deformation_measure_invariants['F'][2]
        Ic = self.deformation_measure_invariants['C'][0]
        C = self.deformation_measure['C']
        Ic *= J**(-2./3.)

        term1 = sum([2*self.cs[i-1]*(Ic-3)**(i-1) for i in range(1,4)])\
                *J**(-2/3)*(self.I-Ic/3*inv(C))
        term2 = sum([2*i*self.d_s[i-1]*(J-1)**(2*i-1) for i in range(1,4)])*J*C
        S = term1+term2
        self.stress_measures['S'] = S 
        self.stress_measures['P'] = F * S 

class BlatzKo(Material):
    """
        BlatzKo material model implementation
    """
    def __init__(self, u, E, nu, f=1):

        """ Initialize """

        ################################
        # Initialize material properties
        ################################

        Material.__init__(self,u=u)    # Initialize base-class
        
        self.update_deformation_measures()

        mu , lmbda  = Lame(E, nu)

        beta = nu/(1-2*nu)

        self.mu = Constant(mu)
        self.lmbda= Constant(lmbda)
        self.beta= Constant(beta)
        self.f = f

    def __call__(self, F):

        assert (type(F) == dolfin.function.expression.Expression) or \
                (type(F) == dolfin.function.constant.Constant), \
                "F should be dolfin Constant or Expression"

        self.deformation_measure['F'] += F
        self.deformation_measure['F'] = \
                variable(self.deformation_measure['F'])

        self.update_deformation_measures()

        J = self.deformation_measure_invariants['F'][2]
        Ic = self.deformation_measure_invariants['C'][0]
        IIc = self.deformation_measure_invariants['C'][1]
        C = self.deformation_measure['C']

        psi = self.mu*0.5*(self.f*(Ic - 3 + 2/self.beta*(J**(-self.beta)-1))\
                +(1-self.f)*((IIc/J-3) + 2/self.beta*(J**(self.beta)-1)))
        self.stress_measures['P'] = diff(psi, F)

class Gao(Material):
    """
        Gao material model implementation
    """
    def __init__(self, u, E, nu, N=2):

        """ Initialize """

        ################################
        # Initialize material properties
        ################################

        Material.__init__(self,u=u)    # Initialize base-class
        
        self.update_deformation_measures()
        mu , lmbda  = Lame(E, nu)
        self.N = Constant(N)
        self.mu = Constant(mu)
        self.lmbda= Constant(lmbda)

    def __call__(self, F):

        assert (type(F) == dolfin.function.expression.Expression) or \
                (type(F) == dolfin.function.constant.Constant), \
                "F should be dolfin Constant or Expression"

        self.deformation_measure['F'] += F
        self.deformation_measure['F'] = \
                variable(self.deformation_measure['F'])

        self.update_deformation_measures()

        J = self.deformation_measure_invariants['F'][2]
        Ic = self.deformation_measure_invariants['C'][0]
        Ic_inv = self.deformation_measure_invariants['C_inv'][0]
        C = self.deformation_measure['C']

        psi = self.mu/2.*(Ic**self.N + Ic_inv**self.N) 
        self.stress_measures['P'] = diff(psi, F)


#class NeoHookean(Material):
#    """
#        Neo-Hookean material model implementation
#    """
#    def __init__(self, u, F_macro, E=None, nu=None, mu=None, lmbda=None):
#
#        """ Initialize """
#
#        ################################
#        # Initialize material properties
#        ################################
#
#        if E is not None and nu is not None:
#            self.mu, self.lmbda = Lame(E,nu)
#        else:
#            self.mu, self.lmbda = mu, lmbda
#        
#        self.mu = Constant(self.mu)
#
#        self.K = Constant(self.lmbda + 2./3. * self.mu)
#        self.C1 = self.mu/2. 
#        self.D1 = self.K /2.
#        
#        Material.__init__(self,u=u, F_macro=F_macro)    # Initialize base-class
#        
#    def Energy(self):
#
#        """ Method: Implement energy density function """
#
#        #psi = self.mu/2*(tr(self.b)-3-2*ln(self.J))+self.K/2*(self.J-1)**2
#        psi = (self.mu/2)*(tr(self.C)- 3) - self.mu*ln(self.J) + (self.lmbda/2)*(ln(self.J))**2
#        return psi
#
#class SVenantKirchhoff(Material):
#    """
#        Saint Venant-Kirchhoff material model implementation
#    """
#    def __init__(self, u, F_macro, E=None, nu=None, mu=None, lmbda=None):
#
#        """ Initialize """
#
#        ################################
#        # Initialize material properties
#        ################################
#
#        if E is not None and nu is not None:
#            self.mu, self.lmbda = Lame(E,nu)
#        else:
#            self.mu, self.lmbda = mu, lmbda
#        
#        self.mu = Constant(self.mu)
#
#        self.K = Constant(self.lmbda + 2./3. * self.mu)
#        self.C1 = self.mu/2. 
#        self.D1 = self.K /2.
#        
#        Material.__init__(self,u=u, F_macro=F_macro)    # Initialize base-class
#        
#    def Energy(self):
#
#        """ Method: Implement energy density function """
#
#        psi = self.lmbda/2*(tr(self.E))**2 + self.mu*tr(self.E*self.E.T) 
#
#        return psi

