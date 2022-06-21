import numpy as np
import meshio
from  dolfin import *
import os 

class DOMAIN():
    """
        GENERAL DOMAIN HANDLING 

        * This class reads the mesh file, optionally convert it to xml 
            format (cannot be paralellized) to xml format.
        * Important geometric properties for the periodic boundary 
            conditions are also obtained from the mesh.
    """

    def __init__(self, filename):

        ################################
        # Filename handling
        ################################
        self.fname = filename
        self.name, self.ext =  os.path.splitext(filename)

        ################################
        # Read domain file depending on your extension
        ################################
        if self.ext == ".msh":
            dolfin_convert(self.fname)
            self.__read_xml()

        elif self.ext == ".xml":
            self.__read_xml()

        elif self.ext == ".xdmf":
            #xdmf_extract(self.fname)
            self.__read_xdmf()

        ################################
        # Dimension of the domain
        # Number of elements in the domain
        # Number of phases in the domain
        ################################
        self.dim = self.mesh.geometry().dim()           
        self.ele_num = self.mesh.num_cells()            
        self.phases = np.unique(self.subdomains.array()).astype(int)    

        ################################
        # Get the bounds and calculate the volume of the domain
        # Get volume of every element 
        ################################
        self.bounds, self.vol = self.__get_bounds()     
        self.__get_volume()                             
        
    def __read_xml(self):
        """

        Note:Legacy extension try not to use this method. 

        """

        self.mesh = Mesh(self.name+".xml")
        self.subdomains = MeshFunction("size_t", self.mesh, \
                self.name+"_physical_region.xml")
        self.facets = MeshFunction("size_t", self.mesh, \
                self.name+"_facet_region.xml")

    def __read_xdmf(self):
        """
            To do: name_to_read -> more specific names like subdomain!
        """

        ################################
        # Read main domain file and put it to self.mesh
        ################################
        self.mesh = Mesh()
        with XDMFFile(self.name+".xdmf") as infile:
            infile.read(self.mesh)

        ################################
        # Read physical region file and put it to self.subdomains
        ################################
        mvc = MeshValueCollection("size_t", self.mesh, 3) 
        with XDMFFile(self.name+"_physical_region.xdmf") as infile:
            infile.read(mvc, "name_to_read")
        self.subdomains = MeshFunction('size_t',self.mesh, mvc)

        ################################
        # Read facet region file and put it to self.facets
        ################################
        mfc = MeshValueCollection("size_t", self.mesh, 3) 
        with XDMFFile(self.name+"_facet_region.xdmf") as infile:
            infile.read(mfc, "name_to_read")
        self.facets = MeshFunction("size_t", self.mesh, mfc)
        
    def __get_vertices(self):
        """
            Note: Not using it any more!

             (x_min,y_max) #-----# (x_max,y_max)
                           |     |
                           |     |
                           |     |
             (x_min,y_min) #-----# (x_max,y_min)
        """

        if self.dim == 2:

            x_min = np.min(self.mesh.coordinates()[:,0]) 
            x_max = np.max(self.mesh.coordinates()[:,0]) 
                
            y_min = np.min(self.mesh.coordinates()[:,1]) 
            y_max = np.max(self.mesh.coordinates()[:,1]) 
            
            vert =  np.array([[x_min,y_min], [x_max,y_min], \
                             [x_max,y_max], [x_min,y_max]])
            vol = x_max * y_max

        elif self.dim == 3:
            raise ("Not implimented yet!")

        return vert, vol

    def __get_bounds(self):
        """
            Method: Get the bounds of your domain
                                        (x_max,y_max,z_max)
                              #-----------# 
                             / |        / |
                            /  |       /  |
                           #----------#   |
                           |   |      |   |
                           |   #----------# 
                           |  /       |  / 
                           | /        | / 
                           |/         |/  
                           #----------# 
         (x_min,y_min,z_min)
        """
        ################################
        # Bounds for 2D domains
        ################################
        if self.dim == 2:

            x_min = np.min(self.mesh.coordinates()[:,0]) 
            x_max = np.max(self.mesh.coordinates()[:,0]) 
                
            y_min = np.min(self.mesh.coordinates()[:,1]) 
            y_max = np.max(self.mesh.coordinates()[:,1]) 
            
            vol = x_max * y_max

            bounds = np.array([[x_min, y_min],[x_max, y_max]])

        ################################
        # Bounds for 3D domains
        ################################
        elif self.dim == 3:

            x_min = np.min(self.mesh.coordinates()[:,0]) 
            x_max = np.max(self.mesh.coordinates()[:,0]) 
                
            y_min = np.min(self.mesh.coordinates()[:,1]) 
            y_max = np.max(self.mesh.coordinates()[:,1]) 

            z_min = np.min(self.mesh.coordinates()[:,2]) 
            z_max = np.max(self.mesh.coordinates()[:,2]) 
            
            vol = x_max * y_max * z_max

            bounds = np.array([[x_min, y_min, z_min],[x_max, y_max, z_max]])

        return bounds, vol



    def __get_volume(self):
        """
            Method: Get volume/area of all the elements in a numpy array
        """
        
        self.ele_vol = np.zeros(self.ele_num)
        for i in range(self.ele_num):
            cell = Cell(self.mesh, i)
            self.ele_vol[i] = cell.volume()

