from ..src.model import *
from ..materials.hyperelastic import *
from ..src.bc import *
import matplotlib.pyplot as plt

################################################################################################################
# Notes: General RVE models with periodic boundary conditions for composite materials. 3D and 2D are seperated 
# for the sake of computational time as the cached FFC's do not need to be compiled again for the models.
# To do: Material seperation from the problem can be thought in a more clever way to enable the usage of 
# different materials easliy with the same model. But, one should not forget that the variational form of the
# problem has to match!
################################################################################################################

class NeoHookean_Kirchhiff_RVE(RVE):
    """ 
        General RVE model implementation
    """
    def __init__(self, domain, Ve=None):

        """ Initialize """

        bc = PeriodicBoundary(domain,periodicity=list(range(domain.dim)),tolerance=1e-10)   # Initialize periodic boundary conditions
        RVE.__init__(self,domain,bc, Ve)              # Initialize base-class


        ################################
        # Mixed function space initialization with periodic boundary conditions
        ################################
        self.W = FunctionSpace(self.domain.mesh, MixedElement([self.Ve, self.Re]), constrained_domain=self.bc)

        #self.fileResults = XDMFFile("output.xdmf")
        #self.fileResults.parameters["flush_output"] = True
        #self.fileResults.parameters["functions_share_mesh"] = True

    def __call__(self,F_macro,time=0):
        
        """ Implement call method for function like usage """

        self.F_macro = F_macro
        self.convergence = True
        self.time = time
        self.solver()
        S, _ = self.postprocess()
        return S

    def problem(self):

        """ Method: Define the variational problem for with pure neumann boundary condition """

        v_,lamb_ = TestFunctions(self.W)                # Define test functions 
        dv, dlamb = TrialFunctions(self.W)              # Define trial functions 
        self.w = Function(self.W)
        
        ################################
        # F_macro should be defined locally because when passing in another way
        # it gives erroneous results! So for consistancy it is defined just before
        # passing as Constant from fenics.
        ################################
        F_macro = Constant(self.F_macro)                

        u,c = split(self.w)

        ################################
        # Define materials for phases
        ################################

        self.material = [NeoHookean(u,F_macro, E=300, nu=0.1),SVenantKirchhoff(u,F_macro,E=500,nu=0.3)]        # model-1


        dx = Measure('dx')(subdomain_data=self.domain.subdomains)       # Redefine dx for subdomains

        ################################
        # Variational problem definition -> Lagrangian Linear Momentum Equation
        ################################
        self.PI = inner(self.material[0].P,nabla_grad(v_))*dx(1) + inner(self.material[1].P,nabla_grad(v_))*dx(2)  
        
        self.PI += dot(lamb_,u)*dx + dot(c,v_)*dx

        return self.PI


    def solver(self, prm=None,cprm=None):

        """ Method: Define solver options with your solver """

        if prm == None:

            prm = {"newton_solver":
                    {"absolute_tolerance":1e-7,'relative_tolerance':1e-7,'relaxation_parameter':1.0,'linear_solver' : 'mumps'}}
        if cprm == None:
            cprm = {"optimize": True}
        try:
            solve(self.problem()==0, self.w, [],solver_parameters=prm,form_compiler_parameters=cprm)
            (self.v, lamb) = self.w.split(True)
        except:
            self.convergence = False
        
        self.__project_u()


    def postprocess(self):
        """ 
            Method: postprocess implementation to get the homogenized 
                    second Piola-Kirchhoff stress tensor 
        """
        if self.convergence is False:
            S = np.zeros((self.domain.dim,self.domain.dim)) * nan
        else:
            
            P = self.__project_P()                          # Project first Piola-Kirchhoff stress tensor 
            F = self.__project_F()                          # Project Deformation Gradient

            Piola = P.split(True)
            DG = F.split(True)
            P_hom = np.zeros(self.domain.dim**2) 
            F_hom = np.zeros(self.domain.dim**2) 
            
            for i in range(self.domain.dim**2):
                P_hom[i] = np.dot(Piola[i].vector().get_local(),self.domain.ele_vol)/self.domain.vol
                F_hom[i] = (DG[i].vector().get_local().mean())

            P_hom = P_hom.reshape(-1,self.domain.dim)
            F_hom = F_hom.reshape(-1,self.domain.dim)
            S = np.dot(np.linalg.inv(F_hom.T),P_hom)

        return S, F_hom


    def __project_P(self):

        """ 
            Method: Projecting first Piola-Kirchhoff stress tensor.
                    Another linear variational problem has to be solved.
        """

        V = TensorFunctionSpace(self.domain.mesh, "DG",0)           # Define Discontinuous Galerkin space

        ################################
        # Similar type of problem definition inside the model
        ################################
        dx = Measure('dx')(subdomain_data=self.domain.subdomains)   
        dx = dx(metadata={'quadrature_degree': 1})
        dv = TrialFunction(V)
        v_ = TestFunction(V)
        a_proj = inner(dv,v_)*dx
        b_proj = inner(self.material[0].P,v_)*dx(1) +inner(self.material[1].P,v_)*dx(2)
        P = Function(V,name='Piola')
        solve(a_proj==b_proj,P)
        #self.fileResults.write(P,self.time)
        return P

    def __project_u(self):

        """ 
            Method: Projecting displacement.
                    Another linear variational problem has to be solved.
        """

        V = FunctionSpace(self.domain.mesh, self.Ve)           # Define Discontinuous Galerkin space

        ################################
        # Similar type of problem definition inside the model
        ################################

        y = SpatialCoordinate(self.domain.mesh)
        write = dot(Constant(self.F_macro),y)+self.v
        dx = Measure('dx')(subdomain_data=self.domain.subdomains)   
        dx = dx(metadata={'quadrature_degree': 1})
        dv = TrialFunction(V)
        v_ = TestFunction(V)
        a_proj = inner(dv,v_)*dx
        b_proj = inner(write,v_)*dx
        u = Function(V,name='Displacement')
        solve(a_proj==b_proj,u,solver_parameters={"linear_solver": "mumps"} )
        #self.fileResults.write(u,self.time)
        return u
        

    def __project_F(self):

        """ 
            Method: Projecting deformation gradient.
                    Another linear variational problem has to be solved.
        """

        ################################
        # Similar type of problem definition inside the model
        ################################
        V = TensorFunctionSpace(self.domain.mesh, "DG",0)       # Define Discontinuous Galerkin space

        dx = Measure('dx')(subdomain_data=self.domain.subdomains)
        dv = TrialFunction(V)
        v_ = TestFunction(V)
        a_proj = inner(dv,v_)*dx
        b_proj = inner(self.material[0].F,v_)*dx(1) +inner(self.material[1].F,v_)*dx(2)
        F = Function(V)
        solve(a_proj==b_proj,F)
        return F


