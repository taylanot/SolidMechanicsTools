from gmshModel.Model import RandomInclusionRVE
from ..src.domain import *
import numpy as np
import os 
import meshio 

class Create_RVE_gmshModel():

    """

        RVE GENERATOR wrapper from gmshModel repository

    """

    def __init__(self, dim=2, directory=None, name='gmshModelRVE-collect', ext='.msh'):
        
        """ Initialize """

        self.dim = dim                      # Dimension of RVE 2 or 3 

        self.write_ext = ext                # Extension for saving 
        self.read_ext = '.xdmf'             # Extension for reading 

        
        if directory is None:
            self.directory_base = name
        else:
            self.directory_base = directory



    def __call__(self, Lc, r, Vf, tag=None):              

        """ Call for the volume fraction of desire """

        self.directory = self.directory_base + '/Vf:'+str(int(Vf*100))

        self.init_Vf = Vf 
        self.Lc = Lc
        self.r = r

        self.directory += '-'+str(self.dim)+'D-L:'+str(self.Lc)+'-r:'+str(self.r)

        try:

            self.create(tag)

            
        except KeyboardInterrupt:
            print(tag," is not created")

        self.extract_info()
        
        self.domain = DOMAIN(self.directory+self.read_ext)
        return self.domain, self.directory


    def create(self,tag,filename= "rve"):

        """ Method: Creation of RVE """ 

        ################################
        # Number of inclusion with the given radius
        ################################
        if self.dim == 2:
            self.inc_type = "Circle"
            self.size = [self.Lc, self.Lc, 0]        
            self.vol = self.Lc**2
            self.no = self.vol * self.init_Vf / (np.pi * self.r**2)
        
        elif self.dim == 3:
            self.inc_type = "Sphere"
            self.size = [self.Lc, self.Lc, self.Lc]
            self.vol = self.Lc**3
            self.no = self.vol * self.init_Vf / (4./3.*np.pi * self.r**3)

        self.set = [[self.r, np.rint(self.no)]] 

        ################################
        # gmshModel create and save RVE
        ################################
        self.initParameters = {                                                                
            "inclusionSets": self.set,
            "inclusionType": self.inc_type,
            "size": self.size,
            "origin": [0, 0, 0],                                                        
            "periodicityFlags": [1, 1, 1],                                              
            "domainGroup": "domain",                                                    
            "inclusionGroup": "inclusions",                                             
            "gmshConfigChanges": {"General.Terminal": 0,                                
                                  "General.Verbosity":0,
                                  "General.AbortOnError": 2,
                                  "Geometry.OCCBoundsUseStl": 0,
                                  "Geometry.Tolerance": 1e-6,
                                  "Mesh.CharacteristicLengthExtendFromBoundary": 0}}
        self.modelingParameters = {                                                            
            "placementOptions": {"maxAttempts": 100000,                                  
                                 "minRelDistBnd": 0.01,                                  
                                 "minRelDistInc": 0.01}}


        self.RVE = RandomInclusionRVE(**self.initParameters) 

        self.RVE.createGmshModel(**self.modelingParameters)
 
        meshingParameters={                                                             
            "threads": None,                                                            
            "refinementOptions": {"maxMeshSize": 0.1,                                
                                  "inclusionRefinement": True,                          
                                  "interInclusionRefinement": False,                    
                                  "elementsPerCircumference": 10,                       
                                  "elementsBetweenInclusions": 10,                       
                                  "inclusionRefinementWidth": 5,                        
                                  "transitionElements": "auto",                         
                                  "aspectRatio": 1.5}}

        self.RVE.createMesh(**meshingParameters)


        self.placed = np.sum(self.RVE.placementInfo)

        if self.dim == 2:
            self.final_Vf = self.placed * np.pi * self.r**2 / self.vol
        else:
            self.final_Vf = self.placed * 4./3. * np.pi * self.r**3 / self.vol


        ################################
        # Directory Handling for the Vf
        ################################


        if tag != None:
            self.directory += '/'+str(tag)

        self.directory += '/'+filename

        self.RVE.saveMesh(self.directory+self.write_ext)

        self.RVE.close()

    def extract_info(self):

        """ Method: Extract all the information needed from xdmf file """

        xdmf_extract(self.directory+self.write_ext)

##############################################################################################
# UTILITIES for preparing the domain for fenics
##############################################################################################

def dolfin_convert(filename):
    """
        Convert the .msh file with msh2 format to xml using dolfin-convert
    *** Legacy format try not to use it!
    """
    name, _ =  os.path.splitext(filename)
    os.system('dolfin-convert '+str(filename)+' '+str(name)+'.xml')

def xdmf_extract(filename):
    """
        dolfin-convert like funnction for the output from gmshModel
    *** TO DO: Extend the order of elements that can be extracted.
    
    """
    def extract(mesh,cell_type):
        cells = np.vstack([cell.data for cell in mesh.cells if cell.type==cell_type])
        data = np.hstack([mesh.cell_data_dict["gmsh:physical"][key]
                               for key in mesh.cell_data_dict["gmsh:physical"].keys() if key==cell_type])
        mesh = meshio.Mesh(points=mesh.points, cells={cell_type: cells},
                                   cell_data={"name_to_read":[data]})
        return mesh


    name, _ =  os.path.splitext(filename)
    
    mesh = meshio.read(filename)
    
    dim = (np.sum(np.max(mesh.points, axis=0) - np.min(mesh.points, axis=0) > 1e-15))

    if dim == 2:
        physical = extract(mesh,"triangle")
        facet= extract(mesh,"line")
        meshio.write(name+"_physical_region.xdmf", physical)
        meshio.write(name+"_facet_region.xdmf", facet)
        mesh.remove_lower_dimensional_cells()
        mesh.prune_z_0()
        mesh = extract(mesh,'triangle')
        meshio.write(name+".xdmf", mesh)

    elif dim == 3:
        physical = extract(mesh,"tetra")
        facet= extract(mesh,"triangle")
        meshio.write(name+"_physical_region.xdmf", physical)
        meshio.write(name+"_facet_region.xdmf", facet)
        mesh = extract(mesh,'tetra')
        meshio.write(name+".xdmf", mesh)
    else:
        raise Exception("Sorry, not implimented yet!")


