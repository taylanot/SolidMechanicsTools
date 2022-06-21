from ..src.model import *
from ..materials.hyperelastic import *
from ..src.bc import *
from ..util import *
import matplotlib.pyplot as plt

available_materials = {
                'NeoHookean': NeoHookean, 
                'Yeoh': Yeoh,
                'BlatzKo': BlatzKo, 
                'SVenantKirchhoff': SVenantKirchhoff, 
                'Gao': Gao
                }

class MultiMaterial(RVE):
    """ 
        NeoHookean_Kirchoff_RVE
    """
    def __init__(self, domain, material_info={
        'id':{'matrix':'NeoHookean', 'inclusion':'SVenantKirchhoff'},
        'parameters':{'matrix':(30, 0.2), 'inclusion':(500, 0.3)}}
        ):

        """ Initialize """

        bc = PeriodicBoundary(domain,periodicity=list(range(domain.dim)),\
                tolerance=1e-12)   # Initialize periodic boundary conditions
        RVE.__init__(self,domain,bc)              # Initialize base-class

        self.material_id= material_info['id']

        self.parameters = material_info['parameters'] 

        #self.fileResults = XDMFFile("output.xdmf")
        #self.fileResults.parameters["flush_output"] = True
        #self.fileResults.parameters["functions_share_mesh"] = True

    def __call__(self,F_macro,time=0):
        
        """ Implement call method for function like usage """

        self.F_macro = F_macro
        self.convergence = True
        self.time = time
        self.solver()
        S, E = self.postprocess()
        return S, E

    def problem(self):

        """ Method: Define the variational problem """

        v_,lamb_ = TestFunctions(self.W)    # Define test functions 
        dv, dlamb = TrialFunctions(self.W)  # Define trial functions 
        self.w = Function(self.W)
        
        ################################
        # F_macro should be defined locally because when passing in another way
        # it gives erroneous results! So for consistancy it is defined just before
        # passing as Constant from fenics.
        ################################
        F_macro = Constant(self.F_macro)                

        u,c = split(self.w)

        self.material = [available_materials[self.material_id['matrix']]\
                (u, self.parameters['matrix'][0], self.parameters['matrix'][1])\
                ,available_materials[self.material_id['inclusion']]\
                (u, self.parameters['inclusion'][0], self.parameters['inclusion'][1])]

        for material in self.material:
            material(F_macro)

        dx = Measure('dx')(subdomain_data=self.domain.subdomains)       

        P_matrix = self.material[0].stress_measures['P']
        P_inclusion = self.material[1].stress_measures['P']
        ################################
        # Variational problem definition -> Lagrangian Linear Momentum Equation
        ################################
        self.PI = inner(P_matrix,nabla_grad(v_))*dx(1) + \
                inner(P_inclusion,nabla_grad(v_))*dx(2)  
        
        self.PI += dot(lamb_,u)*dx + dot(c,v_)*dx

        return self.PI


    def solver(self, prm=None,cprm=None):

        """ Method: Define solver options with your solver """

        if prm == None:

            prm = {"newton_solver":
                    {"absolute_tolerance":1e-5,\
                    'relative_tolerance':1e-5,
                    'maximum_iterations':25,
                    'relaxation_parameter':0.9,
                    'linear_solver' : 'mumps',
                    }}

                    #'preconditioner': 'ilu'}}

        if cprm == None:
            cprm = {"optimize": True}

        #solve(self.problem()==0, self.w, [],solver_parameters=prm,form_compiler_parameters=cprm)
        #(self.v, lamb) = self.w.split(True)
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
            S = np.zeros((self.domain.dim,self.domain.dim)) * np.nan

        else:
            Fs = [self.material[0].deformation_measure['F'],\
                    self.material[1].deformation_measure['F'] ]
            Ps = [self.material[0].stress_measures['P'],\
                    self.material[1].stress_measures['P'] ]

            P = self.__project_tensor(Ps, 'Piola')   
            F = self.__project_tensor(Fs, 'DefGrad')

            Piola = P.split(True)
            DG = F.split(True)
            P_hom = np.zeros(self.domain.dim**2) 
            F_hom = np.zeros(self.domain.dim**2) 
            
            for i in range(self.domain.dim**2):

                P_hom[i] = np.dot(Piola[i].vector().get_local(),self.domain.ele_vol)/self.domain.vol
                F_hom[i] = np.dot(DG[i].vector().get_local(),self.domain.ele_vol)/self.domain.vol


            P_hom = P_hom.reshape(-1,self.domain.dim)
            F_hom = F_hom.reshape(-1,self.domain.dim)
            S = np.dot(np.linalg.inv(F_hom),P_hom)
            
        F_macro = self.F_macro + np.eye(self.domain.dim)
        E = 0.5 * ( F_macro.T.dot(F_macro) -np.eye(self.domain.dim))

        return S, E 


    def __project_tensor(self, tensor, name):

        """ 
            Method: Projecting a tensor
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
        b_proj = inner(tensor[0],v_)*dx(1) +inner(tensor[1],v_)*dx(2)
        tens = Function(V,name=name)
        solve(a_proj==b_proj,tens)
        #self.fileResults.write(P,self.time)
        return tens

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
        a_proj = inner(dv, v_)*dx
        b_proj = inner(write,v_)*dx
        u = Function(V,name='Displacement')
        solve(a_proj==b_proj,u,solver_parameters={"linear_solver": "mumps"} )
        #self.fileResults.write(u,self.time)
        return u
 
class NeoHookean_Kirchhoff_RVE(RVE):
    """ 
        NeoHookean_Kirchoff_RVE
    """
    def __init__(self, domain):

        """ Initialize """

        bc = PeriodicBoundary(domain,periodicity=list(range(domain.dim)),\
                tolerance=1e-12)   # Initialize periodic boundary conditions
        RVE.__init__(self,domain,bc)              # Initialize base-class

        ################################
        # Define materials for phases
        ################################
        self.mu_neo, self.lmbda_neo, _ = Lame(30, 0.1)
        self.mu_kirc, self.lmbda_kirc, _ = Lame(500, 0.3)

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

        """ Method: Define the variational problem """

        v_,lamb_ = TestFunctions(self.W)    # Define test functions 
        dv, dlamb = TrialFunctions(self.W)  # Define trial functions 
        self.w = Function(self.W)
        
        ################################
        # F_macro should be defined locally because when passing in another way
        # it gives erroneous results! So for consistancy it is defined just before
        # passing as Constant from fenics.
        ################################
        F_macro = Constant(self.F_macro)                

        u,c = split(self.w)

        self.material = [NeoHookean(u, self.mu_neo, self.lmbda_neo),SVenantKirchhoff(u, self.mu_kirc, self.lmbda_kirc)]
        #$self.material = [NeoHookean(u, self.mu_neo, self.lmbda_neo),NeoHookean(u, self.mu_neo, self.lmbda_neo)]

        for material in self.material:
            material(F_macro)

        dx = Measure('dx')(subdomain_data=self.domain.subdomains)       

        P_matrix = self.material[0].stress_measures['P']
        P_inclusion = self.material[1].stress_measures['P']
        ################################
        # Variational problem definition -> Lagrangian Linear Momentum Equation
        ################################
        self.PI = inner(P_matrix,nabla_grad(v_))*dx(1) + \
                inner(P_inclusion,nabla_grad(v_))*dx(2)  
        
        self.PI += dot(lamb_,u)*dx + dot(c,v_)*dx

        return self.PI


    def solver(self, prm=None,cprm=None):

        """ Method: Define solver options with your solver """

        if prm == None:

            prm = {"newton_solver":
                    {"absolute_tolerance":1e-4,\
                    'relative_tolerance':1e-4,
                    'maximum_iterations':100,
                    'relaxation_parameter':0.9,
                    'linear_solver' : 'mumps',
                    }}

                    #'preconditioner': 'ilu'}}

        if cprm == None:
            cprm = {"optimize": True}

        solve(self.problem()==0, self.w, [],solver_parameters=prm,form_compiler_parameters=cprm)
        (self.v, lamb) = self.w.split(True)
        print(self.v)
        #try:
        #    solve(self.problem()==0, self.w, [],solver_parameters=prm,form_compiler_parameters=cprm)
        #    (self.v, lamb) = self.w.split(True)
        #except:
        #    self.convergence = False
        
        self.__project_u()


    def postprocess(self):
        """ 
            Method: postprocess implementation to get the homogenized 
                    second Piola-Kirchhoff stress tensor 
        """
        if self.convergence is False:
            S = np.zeros((self.domain.dim,self.domain.dim)) * nan
        else:
            Fs = [self.material[0].deformation_measure['F'],\
                    self.material[1].deformation_measure['F'] ]
            Ps = [self.material[0].stress_measures['P'],\
                    self.material[1].stress_measures['P'] ]

            P = self.__project_tensor(Ps, 'Piola')   
            F = self.__project_tensor(Fs, 'DefGrad')

            Piola = P.split(True)
            DG = F.split(True)
            P_hom = np.zeros(self.domain.dim**2) 
            F_hom = np.zeros(self.domain.dim**2) 
            
            for i in range(self.domain.dim**2):
                P_hom[i] = np.dot(Piola[i].vector().get_local(),self.domain.ele_vol)/self.domain.vol
                F_hom[i] = np.dot(DG[i].vector().get_local(),self.domain.ele_vol)/self.domain.vol


            P_hom = P_hom.reshape(-1,self.domain.dim)
            F_hom = F_hom.reshape(-1,self.domain.dim)
            S = np.dot(np.linalg.inv(F_hom.T),P_hom)

        return S, F_hom


    def __project_tensor(self, tensor, name):

        """ 
            Method: Projecting a tensor
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
        b_proj = inner(tensor[1],v_)*dx(1) +inner(tensor[1],v_)*dx(2)
        tens = Function(V,name=name)
        solve(a_proj==b_proj,tens)
        #self.fileResults.write(P,self.time)
        return tens

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
        a_proj = inner(dv, v_)*dx
        b_proj = inner(write,v_)*dx
        u = Function(V,name='Displacement')
        solve(a_proj==b_proj,u,solver_parameters={"linear_solver": "mumps"} )
        #self.fileResults.write(u,self.time)
        return u
        

#class NeoHookean_Kirchhoff_RVE(RVE):
#    """ 
#        NeoHookean_Kirchoff_RVE
#    """
#    def __init__(self, domain):
#
#        """ Initialize """
#
#        bc = PeriodicBoundary(domain,periodicity=list(range(domain.dim)),tolerance=1e-10)   # Initialize periodic boundary conditions
#        RVE.__init__(self,domain,bc, Ve)              # Initialize base-class
#
#        #self.fileResults = XDMFFile("output.xdmf")
#        #self.fileResults.parameters["flush_output"] = True
#        #self.fileResults.parameters["functions_share_mesh"] = True
#
#    def __call__(self,F_macro,time=0):
#        
#        """ Implement call method for function like usage """
#
#        self.F_macro = F_macro
#        self.convergence = True
#        self.time = time
#        self.solver()
#        S, _ = self.postprocess()
#        return S
#
#    def problem(self):
#
#        """ Method: Define the variational problem for with pure neumann boundary condition """
#
#        v_,lamb_ = TestFunctions(self.W)                # Define test functions 
#        dv, dlamb = TrialFunctions(self.W)              # Define trial functions 
#        self.w = Function(self.W)
#        
#        ################################
#        # F_macro should be defined locally because when passing in another way
#        # it gives erroneous results! So for consistancy it is defined just before
#        # passing as Constant from fenics.
#        ################################
#        F_macro = Constant(self.F_macro)                
#
#        u,c = split(self.w)
#
#        ################################
#        # Define materials for phases
#        ################################
#
#        self.material = [NeoHookean(u,F_macro, E=300, nu=0.1),SVenantKirchhoff(u,F_macro,E=500,nu=0.3)]
#
#        dx = Measure('dx')(subdomain_data=self.domain.subdomains)       # Redefine dx for subdomains
#
#        ################################
#        # Variational problem definition -> Lagrangian Linear Momentum Equation
#        ################################
#        self.PI = inner(self.material[0].P,nabla_grad(v_))*dx(1) + inner(self.material[1].P,nabla_grad(v_))*dx(2)  
#        
#        self.PI += dot(lamb_,u)*dx + dot(c,v_)*dx
#
#        return self.PI
#
#
#    def solver(self, prm=None,cprm=None):
#
#        """ Method: Define solver options with your solver """
#
#        if prm == None:
#
#            prm = {"newton_solver":
#                    {"absolute_tolerance":1e-7,'relative_tolerance':1e-7,'relaxation_parameter':1.0,'linear_solver' : 'mumps'}}
#        if cprm == None:
#            cprm = {"optimize": True}
#        try:
#            solve(self.problem()==0, self.w, [],solver_parameters=prm,form_compiler_parameters=cprm)
#            (self.v, lamb) = self.w.split(True)
#        except:
#            self.convergence = False
#        
#        self.__project_u()
#
#
#    def postprocess(self):
#        """ 
#            Method: postprocess implementation to get the homogenized 
#                    second Piola-Kirchhoff stress tensor 
#        """
#        if self.convergence is False:
#            S = np.zeros((self.domain.dim,self.domain.dim)) * nan
#        else:
#            
#            P = self.__project_P()                          # Project first Piola-Kirchhoff stress tensor 
#            F = self.__project_F()                          # Project Deformation Gradient
#
#            Piola = P.split(True)
#            DG = F.split(True)
#            P_hom = np.zeros(self.domain.dim**2) 
#            F_hom = np.zeros(self.domain.dim**2) 
#            
#            for i in range(self.domain.dim**2):
#                P_hom[i] = np.dot(Piola[i].vector().get_local(),self.domain.ele_vol)/self.domain.vol
#                F_hom[i] = (DG[i].vector().get_local().mean())
#
#            P_hom = P_hom.reshape(-1,self.domain.dim)
#            F_hom = F_hom.reshape(-1,self.domain.dim)
#            S = np.dot(np.linalg.inv(F_hom.T),P_hom)
#
#        return S, F_hom
#
#
#    def __project_P(self):
#
#        """ 
#            Method: Projecting first Piola-Kirchhoff stress tensor.
#                    Another linear variational problem has to be solved.
#        """
#
#        V = TensorFunctionSpace(self.domain.mesh, "DG",0)           # Define Discontinuous Galerkin space
#
#        ################################
#        # Similar type of problem definition inside the model
#        ################################
#        dx = Measure('dx')(subdomain_data=self.domain.subdomains)   
#        dx = dx(metadata={'quadrature_degree': 1})
#        dv = TrialFunction(V)
#        v_ = TestFunction(V)
#        a_proj = inner(dv,v_)*dx
#        b_proj = inner(self.material[0].P,v_)*dx(1) +inner(self.material[1].P,v_)*dx(2)
#        P = Function(V,name='Piola')
#        solve(a_proj==b_proj,P)
#        #self.fileResults.write(P,self.time)
#        return P
#
#    def __project_u(self):
#
#        """ 
#            Method: Projecting displacement.
#                    Another linear variational problem has to be solved.
#        """
#
#        V = FunctionSpace(self.domain.mesh, self.Ve)           # Define Discontinuous Galerkin space
#
#        ################################
#        # Similar type of problem definition inside the model
#        ################################
#
#        y = SpatialCoordinate(self.domain.mesh)
#        write = dot(Constant(self.F_macro),y)+self.v
#        dx = Measure('dx')(subdomain_data=self.domain.subdomains)   
#        dx = dx(metadata={'quadrature_degree': 1})
#        dv = TrialFunction(V)
#        v_ = TestFunction(V)
#        a_proj = inner(dv,v_)*dx
#        b_proj = inner(write,v_)*dx
#        u = Function(V,name='Displacement')
#        solve(a_proj==b_proj,u,solver_parameters={"linear_solver": "mumps"} )
#        #self.fileResults.write(u,self.time)
#        return u
#        
#
#    def __project_F(self):
#
#        """ 
#            Method: Projecting deformation gradient.
#                    Another linear variational problem has to be solved.
#        """
#
#        ################################
#        # Similar type of problem definition inside the model
#        ################################
#        V = TensorFunctionSpace(self.domain.mesh, "DG",0)       # Define Discontinuous Galerkin space
#
#        dx = Measure('dx')(subdomain_data=self.domain.subdomains)
#        dv = TrialFunction(V)
#        v_ = TestFunction(V)
#        a_proj = inner(dv,v_)*dx
#        b_proj = inner(self.material[0].F,v_)*dx(1) +inner(self.material[1].F,v_)*dx(2)
#        F = Function(V)
#        solve(a_proj==b_proj,F)
#        return F


