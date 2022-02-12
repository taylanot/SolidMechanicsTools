from SolidMechanicsTools.src.domain import *
from SolidMechanicsTools.models.rve import *
from SolidMechanicsTools.models.rve_utils import *

import pandas as pd
import matplotlib.pyplot as plt

import torch 
import pandas as pd
import os 

#set_log_level(30)
np.random.seed(24)
for i in range(1000):
    rve = Create_RVE_gmshModel(dim=2, name='test_gmshModel',save=False)
    L = np.random.uniform(4,8)
    r = np.random.uniform(0.2,0.3)
    Vf = np.random.uniform(0.1,0.4)
    #L, r, Vf = 4.567013426947733 0.2219420158308871 0.20698145830717937
    print(i, L, r, Vf)
    rve_domain = rve(L,r,Vf)

#domain = DOMAIN("gmshModelRVE/2Drve40.xdmf")
#
#FEM_model = NeoHookean_Kirchhiff_RVE(domain)
#
#F11, F12, F22 = 0., 0., 0.1
#F_macro = np.array([[F11,F12],[F12,F22]])
#
#S = FEM_model(F_macro,0)
