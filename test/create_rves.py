import numpy as np
import gmsh
import meshio
import os
import matplotlib.pyplot as plt
from SALib.sample import sobol_sequence

N = 7
d = 1
mesh_size = 0.02
points = sobol_sequence.sample(N,d)
limit = 0.5/N
r = 0.05

points = points[(points+r < 1-mesh_size) & (points-r > mesh_size)]

Vf = len(points)*r

assert r <= limit 

name = 'rve'
directory= 'rve_gallery/parallel'

if not os.path.exists(directory):
    os.makedirs(directory)

gmsh.initialize()
gmsh.model.add(name)

tag_matrix = 0 
gmsh.model.occ.addRectangle(0, 0, 0, 1, 1, tag_matrix)

tags_inclusions = 1
for x in points:
    gmsh.model.occ.addRectangle(x, 0, 0, r, 1, tags_inclusions)
    tags_inclusions += 1
gmsh.model.occ.synchronize()

translation_x = [1, 0, 0, 1,\
        0, 1, 0, 0,\
        0, 0, 1, 0,\
        0, 0, 0, 1]

translation_y = [1, 0, 0, 0,\
        0, 1, 0, 1,\
        0, 0, 1, 0,\
        0, 0, 0, 1]

gmsh.model.mesh.setPeriodic(1, [2], [4], translation_x)
gmsh.model.mesh.setPeriodic(1, [3], [1], translation_y)
gmsh.model.occ.fragment([(2, 0)], [(2, i) for i in range(1, tags_inclusions)])
gmsh.model.occ.synchronize()

inc = gmsh.model.addPhysicalGroup(dim=2,tags=[i for i in range(1, tags_inclusions)],tag=2)
matrix = gmsh.model.addPhysicalGroup(dim=2,tags=[i for i in range(tags_inclusions, 2*tags_inclusions+1)],tag=1)
box = gmsh.model.getBoundary(gmsh.model.getEntities(2), combined=True, oriented=False, recursive=False)
boundary = gmsh.model.addPhysicalGroup(dim=1,tags=[ent[1] for ent in box],tag=3)

gmsh.model.setPhysicalName(2,inc, "Inclusion")
gmsh.model.setPhysicalName(2,matrix, "Matrix")
gmsh.model.setPhysicalName(2,matrix, "Boundary")


gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

gmsh.model.mesh.generate(2)
gmsh_file = os.path.join(directory,name+".msh")
gmsh.write(gmsh_file)

gmsh.finalize()


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


xdmf_extract(gmsh_file)
from SolidMechanicsTools.src.domain import *
from SolidMechanicsTools.models.rve import *
from SolidMechanicsTools.models.rve_utils import *

import pandas as pd
import matplotlib.pyplot as plt

set_log_level(30)
rve = Create_RVE_gmshModel(dim=2, name='rve_gallery/perpendicular',save=True)
L = 1
rve_domain = rve(L,r,Vf)

#
#domain = DOMAIN("rve_dir/t18.xdmf")
#
#FEM_model = NeoHookean_Kirchhiff_RVE(domain)
#
#F11, F12, F22 = 0., 0., 0.1
#F_macro = np.array([[F11,F12],[F12,F22]])
#
#S = FEM_model(F_macro,0)
#print(S)
