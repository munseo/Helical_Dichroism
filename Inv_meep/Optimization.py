import sys
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import meep.adjoint as mpa
import os
from mpi4py import MPI
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
sys.path.append('/home/msb/msopt')
sys.path.append('/home/user/msb/msopt')
sys.path.append('../msopt')
import Opt_MS
import module

comm= MPI.COMM_WORLD
rank= comm.Get_rank()

design_dir = "./A/"
local_best_dir = "./Local_bests/"
mp.verbosity(1)
if rank == 1:
    if not os.path.exists(design_dir):
        os.makedirs(design_dir)
    if not os.path.exists(local_best_dir):
        os.makedirs(local_best_dir)

mode_l=[3, -3]       #OAM modes
N_mode=1

meep_unit=0.5

Air = mp.Medium(index=1.0)
poly=mp.Medium(index=2.024)
SiO2 = mp.Medium(index=1.45)
#####################################  Parameters  #################################################
resolution = int(20)                 # simulation resolution
design_region_resolution = resolution   # design resolution
Lpml= round(5.0/ resolution, 2) # 10 pixels
Mask_thick= 0 # total 20 pixels
""" Simulation volume """
# X (width)
HD_w= round(5.0/meep_unit, 2) # Geometry width (Beam waist * 4)
# Y (height)
HD_h= round(5.0/meep_unit, 2) # Geometry height (Beam waist * 4)
# Z (length: propagation of OAM beam)
pml_2_source = round(4/resolution, 2)
source_2_geo = round(0.8/meep_unit, 2) # 0.2 um
HD_t= round(0.8/meep_unit, 2) # Design thickness 0.3 um
SiO2_t= round(0.6/meep_unit, 2) # Substrate thickness w/o pml : 0.2 um

Air_Pad_side= 0#round(0.25/meep_unit - Mask_thick/resolution, 2) # 0.5 um - mask size

Sx= Air_Pad_side+ HD_w+ Air_Pad_side
Sy= Air_Pad_side+ HD_h+ Air_Pad_side
Sz= pml_2_source + source_2_geo+ HD_t+ SiO2_t

""" Custom coordinate """
X_min= round(0.5*-Sx, 2)       
Y_min= round(0.5*-Sy, 2)           
Z_min= round(0.5*-Sz, 2)           

X_max= round(0.5*Sx, 2)              
Y_max= round(0.5*Sy, 2)             
Z_max= round(0.5*Sz, 2)          

""" Source information """
w0= round(1.6, 2) # beam waist: 504 nm
z_pos= Z_min+ pml_2_source # source position (real)
z_geo= z_pos+ source_2_geo           # -z side boundary of geometry. Now, same with bw position
z0= -source_2_geo  # |z0| is distance between beam waist(if z_pos= z0, beam waist= 0) and source position

geo_center= z_geo+ 0.5*HD_t       # z center of geometry

R_center= mp.Vector3(0, 0, z_pos- pml_2_source/2)
T_center= mp.Vector3(0, 0, z_geo+ HD_t+ pml_2_source/2)
Monitor_size= mp.Vector3(Sx, Sy, 0)
######################################################################################################
cell = mp.Vector3(Sx+ 2*Lpml, Sy+ 2*Lpml, Sz+ 2*Lpml)
pml_layers = [mp.PML(thickness=Lpml)]

# Frequancy, Lambda, k ,source_frequancy_width
Wavelength= 0.8/meep_unit
fcen= 1.0/ Wavelength
k= 2.0*np.pi*fcen
width = 0.2
fwidth = width* fcen

############################  SOURCE (LG BEAM)  ########################
N_fom=len(mode_l)

Use_Tran= False
# EVEN order: positive, ODD order: negative
Lx= Sx+ 2*Lpml
Ly= Sy+ 2*Lpml
# Input: Lx, Ly, Lpml, resolution, mode, freq, fwidth, CP(1: RCP, -1: LCP, 0: x-pol)

# Mode +-4 (x-pol)
Sources=[0]*N_fom
Sources2=[0]*N_fom
for i in range(N_fom):
    Sources[i]=module.LG_source(Lx, Ly, Lpml, resolution, mode_l[i], fcen, fwidth, 0, w0, z0, z_pos)

############################  SOURCE END  ##############################

if True: #--------------------------Design region, Substrates
    Nx= int(design_region_resolution* (HD_w))+ 1
    Ny= int(design_region_resolution* (HD_h))+ 1
    #Nz= int(design_region_resolution* (HD_t))+ 1 
    Nz= int(1)

    design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny, Nz), Air, poly, grid_type="U_MEAN")
    design_region = mpa.DesignRegion(
        design_variables,
        volume=mp.Volume(
            center=mp.Vector3(0, 0, geo_center),
            size=mp.Vector3(HD_w, HD_h, HD_t),
        ),
    )
    # 구조물 Z축을 따라 pml 영역까지 배치
    geometry= [
        mp.Block(#SiO2 substrate
            center= mp.Vector3(0, 0, Z_max - (SiO2_t + Lpml)/2),
            material= SiO2,
            size= mp.Vector3(Sx+ 2*Lpml, Sy+ 2*Lpml, SiO2_t+ Lpml) 
        ),    
        mp.Block(#Design region
            center=design_region.center, size=design_region.size, material=design_variables
        ),  
    ]
Input_power= [0]*N_fom
Input_E_Fields= [0]*N_fom
Input_H_Fields= [0]*N_fom
if True: # Normalization
    for i in range(N_fom):
        sim_free = mp.Simulation(
            cell_size=cell,
            boundary_layers=pml_layers,
            sources=Sources[i],
            eps_averaging=False,
            resolution=resolution,
            force_complex_fields=True,
        )

        # Input DFT field and flux
        Input_Ex=sim_free.add_dft_fields([mp.Ex],fcen,0,1,center=R_center,size=Monitor_size)
        Input_Ey=sim_free.add_dft_fields([mp.Ey],fcen,0,1,center=R_center,size=Monitor_size)
        Input_Ez=sim_free.add_dft_fields([mp.Ez],fcen,0,1,center=R_center,size=Monitor_size)

        Input_Hx=sim_free.add_dft_fields([mp.Hx],fcen,0,1,center=R_center,size=Monitor_size)
        Input_Hy=sim_free.add_dft_fields([mp.Hy],fcen,0,1,center=R_center,size=Monitor_size)
        Input_Hz=sim_free.add_dft_fields([mp.Hz],fcen,0,1,center=R_center,size=Monitor_size)

        sim_free.run(until_after_sources= mp.stop_when_dft_decayed(1e-4, 0))

        Input_E_Field= [0]*3
        Input_H_Field= [0]*3

        Input_E_Field[0]= sim_free.get_dft_array(Input_Ex, mp.Ex, 0)
        Input_E_Field[1]= sim_free.get_dft_array(Input_Ey, mp.Ey, 0)
        Input_E_Field[2]= sim_free.get_dft_array(Input_Ez, mp.Ez, 0)

        Input_H_Field[0]= sim_free.get_dft_array(Input_Hx, mp.Hx, 0)
        Input_H_Field[1]= sim_free.get_dft_array(Input_Hy, mp.Hy, 0)
        Input_H_Field[2]= sim_free.get_dft_array(Input_Hz, mp.Hz, 0)

        Input_E_Fields[i]= Input_E_Field
        Input_H_Fields[i]= Input_H_Field
        Input_power[i]= Opt_MS.Cross_product(Input_E_Field, Input_H_Field)

        sim_free.reset_meep()


if True: # Optimization setup for positive source
    sim=[0]*N_fom
    ob_list=[0]*N_fom
    opt=[0]*N_fom
    Js=[0]*N_fom
    for i in range(N_fom):
        sim[i] = mp.Simulation(
            cell_size= cell,
            resolution= resolution,
            boundary_layers= pml_layers,
            sources= Sources[i],
            eps_averaging= False,
            geometry= geometry,
            default_material=Air,
            force_complex_fields=True,
            extra_materials=[poly, SiO2],
        )
        if round(i/2)==i/2 and Use_Tran:
            Eff_center=T_center
        else:
            Eff_center=R_center

        FoM_Ex = mpa.FourierFields(sim[i], mp.Volume(center= Eff_center, size= Monitor_size), mp.Ex ) #<--FoM calc monitor
        FoM_Ey = mpa.FourierFields(sim[i], mp.Volume(center= Eff_center, size= Monitor_size), mp.Ey )
        FoM_Ez = mpa.FourierFields(sim[i], mp.Volume(center= Eff_center, size= Monitor_size), mp.Ez )

        FoM_Hx = mpa.FourierFields(sim[i], mp.Volume(center= Eff_center, size= Monitor_size), mp.Hx ) #<--FoM calc monitor
        FoM_Hy = mpa.FourierFields(sim[i], mp.Volume(center= Eff_center, size= Monitor_size), mp.Hy )
        FoM_Hz = mpa.FourierFields(sim[i], mp.Volume(center= Eff_center, size= Monitor_size), mp.Hz )    

        ob_list[i] = [FoM_Ex, FoM_Ey, FoM_Ez, FoM_Hx, FoM_Hy, FoM_Hz] # input of J   
        del FoM_Ex, FoM_Ey, FoM_Ez
        del FoM_Hx, FoM_Hy, FoM_Hz

    if Use_Tran:
        def J_0(E_x, E_y, E_z, H_x, H_y, H_z): # Figure of Merit function
            Output_E_field=[0]*3
            Output_E_field[0]=E_x[0]
            Output_E_field[1]=E_y[0]
            Output_E_field[2]=E_z[0]

            Output_H_field=[0]*3
            Output_H_field[0]=H_x[0]
            Output_H_field[1]=H_y[0]
            Output_H_field[2]=H_z[0]

            Power=Opt_MS.Cross_product(Output_E_field, Output_H_field)
            FoM=(Power)/Input_power[0]
            return FoM
    else: 
        def J_0(E_x, E_y, E_z, H_x, H_y, H_z): # Figure of Merit function
            Output_E_field=[0]*3
            Output_E_field[0]=E_x[0]
            Output_E_field[1]=E_y[0]
            Output_E_field[2]=E_z[0]

            Output_H_field=[0]*3
            Output_H_field[0]=H_x[0]
            Output_H_field[1]=H_y[0]
            Output_H_field[2]=H_z[0]
            Substracted_E_field= Opt_MS.Substract_field(Input_E_Fields[0], Output_E_field)
            Substracted_H_field= Opt_MS.Substract_field(Input_H_Fields[0], Output_H_field)
            #Purity=Opt_MS.Overlap_intg(Input_E_Fields[1], Substracted_E_field, normalization=True)
            Power=Opt_MS.Cross_product(Substracted_E_field, Substracted_H_field)
            FoM= (Power)/Input_power[0] # R
            return FoM
    def J_1(E_x, E_y, E_z, H_x, H_y, H_z): # Figure of Merit function
        Output_E_field=[0]*3
        Output_E_field[0]=E_x[0]
        Output_E_field[1]=E_y[0]
        Output_E_field[2]=E_z[0]

        Output_H_field=[0]*3
        Output_H_field[0]=H_x[0]
        Output_H_field[1]=H_y[0]
        Output_H_field[2]=H_z[0]
        Substracted_E_field= Opt_MS.Substract_field(Input_E_Fields[1], Output_E_field)
        Substracted_H_field= Opt_MS.Substract_field(Input_H_Fields[1], Output_H_field)
        #Purity=Opt_MS.Overlap_intg(Input_E_Fields[1], Substracted_E_field, normalization=True)
        Power=Opt_MS.Cross_product(Substracted_E_field, Substracted_H_field)
        FoM= (Power)/Input_power[1] # R
        return FoM
    Js[0]=J_0
    Js[1]=J_1
    if N_mode>1:
        if Use_Tran:
            def J_2(E_x, E_y, E_z, H_x, H_y, H_z): # Figure of Merit function
                Output_E_field=[0]*3
                Output_E_field[0]=E_x[0]
                Output_E_field[1]=E_y[0]
                Output_E_field[2]=E_z[0]

                Output_H_field=[0]*3
                Output_H_field[0]=H_x[0]
                Output_H_field[1]=H_y[0]
                Output_H_field[2]=H_z[0]

                Power=Opt_MS.Cross_product(Output_E_field, Output_H_field)
                FoM=(Power)/Input_power[2]
                return FoM
        else: 
            def J_2(E_x, E_y, E_z, H_x, H_y, H_z): # Figure of Merit function
                Output_E_field=[0]*3
                Output_E_field[0]=E_x[0]
                Output_E_field[1]=E_y[0]
                Output_E_field[2]=E_z[0]

                Output_H_field=[0]*3
                Output_H_field[0]=H_x[0]
                Output_H_field[1]=H_y[0]
                Output_H_field[2]=H_z[0]

                Substracted_E_field= Opt_MS.Substract_field(Input_E_Fields[2], Output_E_field)
                Substracted_H_field= Opt_MS.Substract_field(Input_H_Fields[2], Output_H_field)
                Power=Opt_MS.Cross_product(Substracted_E_field, Substracted_H_field)
                FoM= (Power)/Input_power[2] # R
                return FoM
        def J_3(E_x, E_y, E_z, H_x, H_y, H_z): # Figure of Merit function
            Output_E_field=[0]*3
            Output_E_field[0]=E_x[0]
            Output_E_field[1]=E_y[0]
            Output_E_field[2]=E_z[0]

            Output_H_field=[0]*3
            Output_H_field[0]=H_x[0]
            Output_H_field[1]=H_y[0]
            Output_H_field[2]=H_z[0]

            Substracted_E_field= Opt_MS.Substract_field(Input_E_Fields[3], Output_E_field)
            Substracted_H_field= Opt_MS.Substract_field(Input_H_Fields[3], Output_H_field)
            Power=Opt_MS.Cross_product(Substracted_E_field, Substracted_H_field)
            FoM= (Power)/Input_power[3] # R
            return FoM
        Js[2]=J_2
        Js[3]=J_3
        if N_mode>2:
            if Use_Tran:
                def J_4(E_x, E_y, E_z, H_x, H_y, H_z): # Figure of Merit function
                    Output_E_field=[0]*3
                    Output_E_field[0]=E_x[0]
                    Output_E_field[1]=E_y[0]
                    Output_E_field[2]=E_z[0]

                    Output_H_field=[0]*3
                    Output_H_field[0]=H_x[0]
                    Output_H_field[1]=H_y[0]
                    Output_H_field[2]=H_z[0]

                    Power=Opt_MS.Cross_product(Output_E_field, Output_H_field)
                    FoM=(Power)/Input_power[4]
                    return FoM
            else: 
                def J_4(E_x, E_y, E_z, H_x, H_y, H_z): # Figure of Merit function
                    Output_E_field=[0]*3
                    Output_E_field[0]=E_x[0]
                    Output_E_field[1]=E_y[0]
                    Output_E_field[2]=E_z[0]

                    Output_H_field=[0]*3
                    Output_H_field[0]=H_x[0]
                    Output_H_field[1]=H_y[0]
                    Output_H_field[2]=H_z[0]

                    Substracted_E_field= Opt_MS.Substract_field(Input_E_Fields[4], Output_E_field)
                    Substracted_H_field= Opt_MS.Substract_field(Input_H_Fields[4], Output_H_field)
                    Power=Opt_MS.Cross_product(Substracted_E_field, Substracted_H_field)
                    FoM= (Power)/Input_power[4] # R
                    return FoM
            def J_5(E_x, E_y, E_z, H_x, H_y, H_z): # Figure of Merit function
                Output_E_field=[0]*3
                Output_E_field[0]=E_x[0]
                Output_E_field[1]=E_y[0]
                Output_E_field[2]=E_z[0]

                Output_H_field=[0]*3
                Output_H_field[0]=H_x[0]
                Output_H_field[1]=H_y[0]
                Output_H_field[2]=H_z[0]

                Substracted_E_field= Opt_MS.Substract_field(Input_E_Fields[5], Output_E_field)
                Substracted_H_field= Opt_MS.Substract_field(Input_H_Fields[5], Output_H_field)
                Power=Opt_MS.Cross_product(Substracted_E_field, Substracted_H_field)
                FoM= (Power)/Input_power[5] # R
                return FoM
            Js[4]=J_4
            Js[5]=J_5
    for i in range(N_fom):
        opt[i] = mpa.OptimizationProblem(
            simulation=sim[i],
            objective_functions=Js[i],
            objective_arguments=ob_list[i],
            design_regions=[design_region],
            fcen=fcen,
            df=0,
            nf=1,
            minimum_run_time=10,
            decay_by=1e-4,
        )

####################################################################################################
if True: #------------------------------------Mapping & Symmetry
    DR_info= [HD_w, HD_h, HD_t, 0, 1, 2]
    DR_N_info= [Nx, Ny, Nz, design_region_resolution]
    MFS= round(0.2/meep_unit, 2) 
    MGS= round(0.1/meep_unit, 2)

    evaluation_history_p = [[0],[0],[0],[0],[0],[0],[0]]
    evaluation_history_n = [[0],[0],[0],[0],[0],[0],[0]]

    mapping = Opt_MS.Mapping(
        Symmetry_sim= False,
        Sym_geo_width= False,
        Sym_geo_length= False,
        Sym_geo_C2= True,
        DR_info= DR_info,
        DR_N_info= DR_N_info,
        Mask_pixels= Mask_thick,
        MFS= MFS,
        MGS= MGS,
    )

####################################################################################################-----|
if True:    #--------------------------Hyper parameters for Main function & Optimization         #
    evaluation_history = [0]                                                                             #
    wrong_evaluation_history = [[0],[0],[0],[0],[0],[0]]                                                 #
    wrong_evaluation_history2 = [[0],[0],[0],[0],[0],[0]]                                                #
    grad_mean_history = []                                                                               #
    grad_max_history = []                                                                                #
    learning_rate_history = []                                                                           #
    binarization_history = []                                                                            #
    beta_history = []                                                                                    #
    cur_iter = [0]                                                                                       #
    numevl = 1                                                                                           #
    #----------------------------------------------------------------------------------------------------#
    def f(v, gJ, g1_o, g2_o, beta, alpha, Outer_M, flag: bool = False):                                  #
        global numevl                                                                                    #
        print("\n Current iteration: {}".format(cur_iter[0] + 1))                                        #
        f0s, dJ_dus=[0]*N_fom, [0]*N_fom                                                                 #
        v_new, X, beta, beta_o, h_list = Opt_MS.Design_update(v, alpha, gJ, mapping, beta, flag=flag, is_init=True)   #
        beta_history.append(h_list[0])                                                                   #
        binarization_history.append(h_list[1])                                                           #
        if beta == npa.inf: #----------------------------------------------------------------------------|

            ff0s=[0]*N_mode # only for HD
            for i in range(N_fom):
                f0s[i], dJ_dus[i] = opt[i](rho_vector=[npa.clip(X, 0.0, 1.0)], need_gradient=False)
            for i in range(N_mode):
                if Use_Tran:
                    ff0s[i]=2*((f0s[2*i+1]-(1-f0s[2*i]))/(f0s[2*i+1]+(1-f0s[2*i])))
                else:
                    ff0s[i]=2*((f0s[2*i+1]-f0s[2*i])/(f0s[2*i+1]+f0s[2*i]))
            f0=np.mean(ff0s)
            for i in range(7):
                if i < N_mode:
                    if Use_Tran:
                        evaluation_history_p[i].append(np.real(1-f0s[2*i]))
                    else:
                        evaluation_history_p[i].append(np.real(f0s[2*i]))
                    evaluation_history_n[i].append(np.real(f0s[2*i+1]))
                else:
                    evaluation_history_p[i].append(0)
                    evaluation_history_n[i].append(0)

            #################################################################################################------|
            numevl += 1                                                                                            #
            evaluation_history.append(np.real(f0))                                                                 #
            print("First FoM: {}".format(evaluation_history[1]))                                                   #
            print("Last FoM: {}".format(np.real(f0)))                                                              #
            np.savetxt(design_dir+"lastdesign.txt", npa.clip(X, 0.0, 1.0))                                         #
            # np.savetxt(design_dir+"last_v.txt", v_new)                                                             #
            print("-----------------Optimization Complete------------------")                                      #
            return v_new, gJ, g1_o, g2_o, beta, alpha, np.real(f0), Outer_M                                        #
        else: #----------------------------------------------------------------------------------------------------#
            last_fom= evaluation_history[cur_iter[0]]                                                              #
            Case, tol, last_fom, beta_n = Opt_MS.Beta_update(flag, numevl, Outer_M[4], last_fom, beta)             #
            if Case < 3:                                                                                           #
                v_new, X, Outer_M = Opt_MS.Design_update_AC(Outer_M, alpha, beta_n, beta_o, mapping, v, v_new, X)  #
                alpha_scale= 10                                                                                    #
            else:
                alpha_scale= 0.1
            X0=mapping(npa.clip(v, 0.0, 1.0), beta) # 3D geo w/o upt                                               #
            grad_suff=npa.sum(npa.abs(npa.where(((X0==0)&(g1_o<0)),0, npa.where(((X0==1)&(g1_o>0)),0,g1_o))))      #
            #------------------------------------------------------------------------------------------------------|

            ff0s=[0]*N_mode # only for HD
            for i in range(N_fom):
                f0s[i], dJ_dus[i] = opt[i](rho_vector=[npa.clip(X, 0.0, 1.0)])
            for i in range(N_mode):
                if Use_Tran:
                    ff0s[i]=2*((f0s[2*i+1]-(1-f0s[2*i]))/(f0s[2*i+1]+(1-f0s[2*i])))
                else:
                    ff0s[i]=2*((f0s[2*i+1]-f0s[2*i])/(f0s[2*i+1]+f0s[2*i]))
            f0=np.mean(ff0s)
            ff1s=[0]*N_mode # only for HD

            # Sufficient condition ###########################################################################----|
            if np.real(f0) < last_fom+ (1/v.size)*alpha*grad_suff and cur_iter[0] > 0:                            #
                Backtraking_count=0                                                                               #
                while Backtraking_count <6 or f0 < last_fom+ (1/v.size)*alpha*grad_suff:                          #
                    if Backtraking_count == 0:                                                                    #
                        grad_info, foms_info, fom_info, trig, stop= Opt_MS.Back_tracking_init(f0, f0s, dJ_dus)    #
                        f1s=[0]*N_fom                                                                             #
                        dJ_du1s=[0]*N_fom                                                                         #
                    if tol==0 and np.real(f0) > last_fom:                                                         #
                        break                                                                                     #
                    elif tol==0 and alpha < 0.2:                                                                  #
                        break                                                                                     #
                    Backtraking_count +=1                                                                         #
                    alpha *=alpha_scale                                                                           #
                    v_new, X = Opt_MS.Design_update(v, alpha, gJ, mapping, beta)                                  #
                    #---------------------------------------------------------------------------------------------|

                    for i in range(N_fom):
                        f1s[i], dJ_du1s[i] = opt[i](rho_vector=[npa.clip(X, 0.0, 1.0)])
                    for i in range(N_mode):
                        if Use_Tran:
                            ff1s[i]=2*((f1s[2*i+1]-(1-f1s[2*i]))/(f1s[2*i+1]+(1-f1s[2*i])))
                        else:
                            ff1s[i]=2*((f1s[2*i+1]-f1s[2*i])/(f1s[2*i+1]+f1s[2*i]))
                    foms_info[Backtraking_count]=f1s
                    grad_info[Backtraking_count]=dJ_du1s
                    fom_info[Backtraking_count]=np.mean(ff1s)

                    # Backtracking ----------------------------------------------------------------------------------   #
                    trig, stop= Opt_MS.Back_tracking(grad_info, foms_info, fom_info, Backtraking_count, trig, tol)      #
                    if alpha_scale < 1:                                                                                 #
                        wrong_evaluation_history[Backtraking_count-1].append(np.real(fom_info[Backtraking_count]))      #
                    else:                                                                                               #
                        wrong_evaluation_history2[Backtraking_count-1].append(np.real(fom_info[Backtraking_count]))     #
                    if stop and trig:                                                                                   #
                        alpha *= (1/alpha_scale)**Backtraking_count                                                     #
                        v_new, X = Opt_MS.Design_update(v, alpha, gJ, mapping, beta)                                    #
                        f0, f0s, dJ_dus = Opt_MS.Back_tracking_call(fom_info, foms_info, grad_info, 0)                  #
                        if last_fom > 999 and alpha_scale>1:                                                            #
                            alpha_scale= (1/alpha_scale)                                                                #
                            if Backtraking_count <6:                                                                    #
                                for i in range(6-Backtraking_count):                                                    #
                                    wrong_evaluation_history[Backtraking_count+i].append(np.real(f0))                   #
                            Backtraking_count=0                                                                         #
                            print("Restart backtraking with downscaling factor")                                        #
                        else:                                                                                           #
                            print("Continue with current LR: ",alpha)                                                   #
                            break                                                                                       #
                    elif stop:                                                                                          #
                        alpha *= (1/alpha_scale)                                                                        #
                        v_new, X = Opt_MS.Design_update(v, alpha, gJ, mapping, beta)                                    #
                        f0, f0s, dJ_dus = Opt_MS.Back_tracking_call(fom_info, foms_info, grad_info, Backtraking_count-1)#
                        print("Continue with ",Backtraking_count-1,"times reduced LR: ", alpha)                         #
                        break                                                                                           #
                    else:                                                                                               #
                        f0, f0s, dJ_dus = Opt_MS.Back_tracking_call(fom_info, foms_info, grad_info, Backtraking_count)  #
                        print("Continue with ",Backtraking_count,"times reduced LR: ", alpha)                           #
                        if Backtraking_count==6:                                                                        #
                            break                                                                                       #
                if Backtraking_count <6:                                                                                #
                    if alpha_scale < 1:                                                                                 #
                        for i in range(6-Backtraking_count):                                                            #
                            wrong_evaluation_history[Backtraking_count+i].append(np.real(f0))                           #
                        for i in range(6):                                                                              #
                            wrong_evaluation_history2[i].append(np.real(f0))                                            #  
                    else:                                                                                               #
                        for i in range(6-Backtraking_count):                                                            #
                            wrong_evaluation_history2[Backtraking_count+i].append(np.real(f0))                          #
            else:                                                                                                       #
                for i in range(6):                                                                                      #
                    wrong_evaluation_history[i].append(np.real(f0))                                                     #
                    wrong_evaluation_history2[i].append(np.real(f0))                                                    #
            #--------------------------------------------------------------------------------------------------------   #
            # Stack the gradient
            dt_dus=[0]*N_mode
            for i in range(N_mode):
                if Use_Tran:
                    weight_p=4*(1-f0s[2*i])/(f0s[2*i+1]+(1-f0s[2*i])) # 1-T
                else:
                    weight_p=2*((-f0s[2*i])/(f0s[2*i+1]+f0s[2*i])) # R
                weight_n= ((f0s[2*i+1])/(f0s[2*i+1]+f0s[2*i])) # R
                dt_dus[i]=weight_n*dJ_dus[2*i+1] + weight_p*dJ_dus[2*i]
            if N_mode >1:    
                dJ_du= Opt_MS.Goal_Attainment(g1_o, ff0s, dt_dus, f0)
            else:
                dJ_du= dt_dus[0]
            for i in range(7):
                if i < N_mode:
                    print("OAM mode |l|:",mode_l[2*i])
                    if Use_Tran:
                        evaluation_history_p[i].append(np.real(1-f0s[2*i]))
                        print("Current R+: {}".format(np.real(1-f0s[2*i])))
                    else:
                        evaluation_history_p[i].append(np.real(f0s[2*i]))
                        print("Current R+: {}".format(np.real(f0s[2*i])))
                    evaluation_history_n[i].append(np.real(f0s[2*i+1]))
                    print("Current R-: {}".format(np.real(f0s[2*i+1])))
                else:
                    evaluation_history_p[i].append(0)
                    evaluation_history_n[i].append(0)

            # Momentum (Adam) ---------------------------------------------------------#         
            g_m, grad_adj, grad_adj_2 = Opt_MS.Momentum(dJ_du, g1_o, g2_o, numevl)     #
            # Jacobian grad --------------------------------------------------------------------------#-------|
            if v.size > 0:                                                                            #
                gradient = v*0                                                                        #
                gradient[:] = tensor_jacobian_product(mapping, 0)(                                    #   C
                    v_new, beta_n, g_m,                                                               #   a
                )  # backprop ------------------------------------------------------------------------#   p
            evaluation_history.append(np.real(f0))                                                    #   s
            learning_rate_history.append(alpha)                                                       #   u
            numevl += 1                                                                               #   l
            print("First FoM: {}".format(evaluation_history[1]))                                      #   i
            print("Current FoM: {}".format(np.real(f0)))                                              #   z
            print(f"Learning rate: {(alpha)}\n")                                                      #   e
            grad_mean_history.append(npa.mean(npa.abs(gradient)))                                     #   d
            grad_max_history.append(npa.max(npa.abs(gradient)))                                       #
            cur_iter[0] = cur_iter[0] + 1                                                             #
            return v_new, gradient, grad_adj, grad_adj_2, beta_n, alpha, np.real(f0), Outer_M         #
            #-----------------------------------------------------------------------------------------#-------|

    ##########################
    n= Nx * Ny #* Nz
    x0 = np.ones((n,)) * 0.5  # initial geometry
    dJ_0 = np.zeros((n,))
    #-----------------------------------------------------------------|
    As, Ps, Best, Best2, OMs, F_his= Opt_MS.Initializer(x0, dJ_0) #   |
    Max_iter = 777                                                #   |
    inner_iter = 0                                                #   |
    is_converged = False                                          #   |
    flag= False                                                   #   |
    Re_roll=False                                                 #   |
    Load_init=False                                               #   |
    if Load_init:                                                 #   |
        Load_iter= 119                                            #   |
        As, Ps, numevl= Opt_MS.Load_params(Load_iter, As, Ps)     #   |
        flag= True                                                #   |
    while inner_iter < Max_iter: #------------------------------------|  Gradient Ascent Optimizer
        # Update main parameters                                      ---------------------------------------------------------|
        As_o= Opt_MS.Updater(As, Ps, Best)                                                                                   # |
        # main simulation                                                                                                    # |
        As[0], As[1], As[2], As[3], As[4], Ps[0], Ps[1], OMs= f(As_o[0], As_o[1], As_o[2], As_o[3], As_o[4], Ps[0], OMs, flag)#|
        # geo,  g_J,   g_1,   g_2,  beta,   LR,    FoM          geo_old, g_J_old, g_1_old, g_2_old, cur_beta, LR               |
        #if inner_iter/10==int(inner_iter/10):
        #    np.savetxt(design_dir+f"Grayscale_geo{inner_iter}.txt", design_variables.weights)
        if As[4] == npa.inf:                                                                                                  #|
            break                                                                                                            # |
        Best2[0]= As_o[4]                                                                                                    # |
        Best2[1]= numevl-1                                                                                                   # |
        Best2[2]= cur_iter[0]                                                                                                # |
        if is_converged:                                                                                                     # |
            Ps, As, OMs, numevl, Re_roll = Opt_MS.After_conversion(Ps, As, Best, OMs, numevl, Re_roll)                       # |
            Best[0]=0
        else:
            As, Ps, Best = Opt_MS.Warm_restarter(As, Ps, Best, Best2)                                                        # |        
        if not Re_roll:                                                                                                      # |
            F_his, As, Ps, Best, is_converged = Opt_MS.Conversion_check(F_his, As, Ps, Best, inner_iter)                     # |
            if is_converged:                                                                                                 # |
                OMs[3]=Ps[4]                                                                                                 # |
                flag= True # hold beta                                                                                       # |
                numevl=1                                                                                                     # |
        inner_iter +=1                                                                                                       # |
    #--------------------------------------------------------------------------------------------------------------------------|

############################# Output data #########################################
if True:
    plt.figure()
    for i in range(6):
        plt.plot(wrong_evaluation_history[i], "r-")
    for i in range(6):
        plt.plot(wrong_evaluation_history2[i], "b-")
    plt.plot(evaluation_history, "k-")
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("FoM")
    plt.savefig(design_dir+"result1.png")
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure

    plt.figure()
    for i in range(N_mode):
        plt.plot(evaluation_history_p[i], "r-")
        plt.plot(evaluation_history_n[i], "b-")
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("FoM")
    plt.savefig(design_dir+"result0.png")
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure

    plt.figure()
    plt.plot(binarization_history, "o-")
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("Binarized")
    plt.savefig(design_dir+"result2.png")
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure

    plt.figure()
    plt.plot(learning_rate_history, "o-")
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("LR")
    plt.yscale('log')
    plt.savefig(design_dir+"result3.png")
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure
    
    plt.figure()
    plt.plot(beta_history, "o-")
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("Beta")
    plt.savefig(design_dir+"beta.png")
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure

    np.savetxt(design_dir+"evaluation.txt", evaluation_history)
    # np.savetxt(design_dir+"evaluation2.txt", wrong_evaluation_history)
    np.savetxt(design_dir+'FoM_p.txt',evaluation_history_p)
    np.savetxt(design_dir+'FoM_n.txt',evaluation_history_n)

    np.savetxt(design_dir+"binarization.txt", binarization_history)
    np.savetxt(design_dir+"learning_rate.txt", learning_rate_history)
    # np.savetxt(design_dir+"grad_mean.txt", grad_mean_history)
    # np.savetxt(design_dir+"grad_max.txt", grad_max_history)
    np.savetxt(design_dir+"beta.txt", beta_history)

