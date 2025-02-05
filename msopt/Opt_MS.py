"""
Module for gradient ascent optimization
History:
24/11/07 - Created
24/11/12 - Outer momentum added
24/11/13 - Design & Beta updater and Mapping class added

Last update: 2024/11/13 by munseong
"""
import numpy as np
import meep.adjoint as mpa
import autograd.numpy as npa
from autograd import tensor_jacobian_product, grad
import os
import Sub_Mapping

"""
Optimizer for gradient ascent optimization

1. Initializer: Define initial value of main parameters
2. Load_params: Load data from previous optimization and use it as initial parameters
3. Updater: Update the main parameters or local best data during the optimization 
4. Warm_restarter: Reset the learning rate(warm restart) to escape from local optima and save the local best data
5. Opt_tol: Evaluate the conversion condition via FoM history and update it
6. Conversion_check: Increasing the beta(constraint factor) with warm restart after geometry conversion
"""
""" 1 """
def Initializer(Initial_geo, Initial_grad):
    Array=[0]*5 
    # 0: Geometry
    # 1: Jacobian gradient 
    # 2: 1st order momentum gradient
    # 3: 2nd order momentum gradient
    # 4: Beta for mapping function
    Array[0]= Initial_geo
    Array[1]= Initial_geo*0
    Array[2]= Initial_grad       
    Array[3]= Initial_grad 
    Array[4]= 1.0

    Parameters=[0]*6
    # 0: Learning rate
    # 1: FoM
    # 2: Global best FoM
    # 3: Warm restart counter
    # 4: Convergence counter
    Parameters[0]=0.2
    Parameters[1]=0
    Parameters[2]=0
    Parameters[3]=0
    Parameters[4]=0

    Best=[0]*8
    # 0: Local best FoM
    # 1: Local best geometry
    # 2: Local best 1st order grad
    # 3: Local best 2nd order grad
    # 4: Local best beta
    # 5: Local best LR
    # 6: Local best grad stack
    # 7: Local best iters
    Best[0]= 0
    Best[1]= Initial_geo
    Best[2]= Initial_grad
    Best[3]= Initial_grad
    Best[4]= Array[4]
    Best[5]= Parameters[0]
    Best[6]= 0
    Best[7]= 0 

    Best2=[0]*3
    # 0: cur beta
    # 1: cur grad stack
    # 2: cur iters
    Best2[0]= 0
    Best2[1]= 0
    Best2[2]= 0

    Outer_M=[0]*5
    # 0: Jacobian gradient 
    # 1: 1st order outer momentum gradient
    # 2: 2nd order outer momentum gradient
    # 3: Convergence counter
    # 4: Outer momentum triger
    Outer_M[0]= Array[1]
    Outer_M[1]= Array[2]
    Outer_M[2]= Array[3]
    Outer_M[3]= Parameters[4]
    Outer_M[4]= False

    F_history=[0, 0, 0, 0, 0, 0, 0, 0]

    return Array, Parameters, Best, Best2, Outer_M, F_history 

""" 2 """
def Load_params(Load_iter, Array, Param):
    local_best_dir = "./Local_bests/"
    os.chdir(local_best_dir)
    Parameter=np.loadtxt(f"Param_iter{Load_iter}.txt")
    Array[0]= np.loadtxt(f"Ref_layer_iter{Load_iter}.txt")
    Array[1]= Array[0]*0
    Array[2]= np.loadtxt(f"Grad_iter{Load_iter}.txt")
    Array[3]= np.loadtxt(f"Grad2_iter{Load_iter}.txt")
    Array[4] = Parameter[0]
    Param[0] = Parameter[1]
    os.chdir("..")
    return Array, Param, Parameter[2]

""" 3 """
def Updater(Array_cur, Parameters, Best, is_Best: bool = False, is_Worst: bool = False):
    if is_Best:
        Best[0]= Parameters[1]
        Best[1]= Array_cur[0]
        Best[2]= Array_cur[2]
        Best[3]= Array_cur[3]
        Best[5]= Parameters[0]
        return Best
    elif is_Worst:
        Array_cur[0] = Best[1]
        Array_cur[1] = Best[1]*0
        Array_cur[2] = Array_cur[2]*0
        Array_cur[3] = Array_cur[3]*0
        Array_cur[4] = Best[4]
        Parameters[0] = 0.2 
        return Array_cur, Parameters
    else:
        Array_old=Array_cur
        return Array_old

""" 4 """
def Warm_restarter(Array_cur, Parameters, Best, Best2, Save_small: bool = True):
    if Parameters[1] < Best[0]*1.01: # 1% improvement?
        if Parameters[1] > Best[0] and Save_small:
            Best= Updater(Array_cur, Parameters, Best, is_Best=True)
            Best[4]= Best2[0]
            Best[6]= Best2[1]
            Best[7]= Best2[2]
            if Parameters[3] ==2:
                Parameters[3] -=1
        Parameters[3] +=1
        if Parameters[3] == 3:
            #local_best_dir = "./Local_bests/"
            #np.savetxt(f"{local_best_dir}Ref_layer_iter{Best[7]}.txt", Best[1])
            #np.savetxt(f"{local_best_dir}Grad_iter{Best[7]}.txt", Best[2])
            #np.savetxt(f"{local_best_dir}Grad2_iter{Best[7]}.txt", Best[3])
            #np.savetxt(f"{local_best_dir}Param_iter{Best[7]}.txt", [Best[4], Best[5], Best[6], Best[0]])
            if Best[4] > 50:
                Array_cur[0]= Best[1]
                Array_cur[4]= Best[4]
                Parameters[0]= 0
            else:
                Parameters[0]= 0.2
            Parameters[3]= 0
            print('Warm restart with LR: 0.2')
    else:
        Best= Updater(Array_cur, Parameters, Best, is_Best=True)
        Best[4]= Best2[0]
        Best[6]= Best2[1]
        Best[7]= Best2[2]
        Parameters[3]= 0
    return Array_cur, Parameters, Best

""" 5 """
def Opt_tol(F_history, F, Count, iters):
    F_history[len(F_history)-1]=F
    for i in range (0, len(F_history)-1):
        F_history[i]=F_history[i+1]
    mean_F=npa.mean(F_history)
    if F > mean_F+ mean_F*1e-4:
        return F_history, Count, False
    else:
        if iters < 5:
            return F_history, Count, False
        else:
            Count +=1
            return F_history, Count, True

""" 6 """
def Conversion_check(F_history, Array_cur, Param, Best, iters):
    F_history, Param[4], is_converged = Opt_tol(F_history, Param[1], Param[4], iters)
    if is_converged:
        F_history=[0, 0, 0, 0, 0, 0, 0]
        Array_cur[0] = Best[1]
        Array_cur[1] = Best[1]*0
        Array_cur[2] = Best[2]*0
        Array_cur[3] = Best[2]*0
        Array_cur[4] = Best[4]*1.5

        Param[0]= 1.0
        Param[3]=0
        Param[2]=Best[0]
        Best[0]=Param[1]*0.9
        print('FoM converged')
        if Best[4] > 50:
            Array_cur[0] = Best[1]
            Array_cur[4] = Best[4]
            Param[0] = 0        
    return F_history, Array_cur, Param, Best, is_converged

""" 7 """
def After_conversion(Parameters, Array_cur, Best, Outer_M, numevl, Re_roll):
    if Parameters[1] < Parameters[2]*0.85: # FoM < 85% of Global best FoM
        Array_cur, Parameters = Updater(Array_cur, Parameters, Best, is_Worst= True)
        Outer_M[4] = True
        if Outer_M[3] == Parameters[4]:
            Parameters[0] = 1.0
            Array_cur[4] *= 1.25
            Parameters[2] = 0 
        numevl= 1
        Re_roll= True
    else:
        Re_roll= False
        Parameters[0] = 0.2
    return Parameters, Array_cur, Outer_M, numevl, Re_roll

""" 8 """
def Design_update_AC(Outer_M, alpha, beta_n, beta_o, mapping, v, v_new, X):
    if Outer_M[4]:
        Outer_M[4] = False
        Outer_M[3] += 1
        if alpha == 0.2:
            beta_scale = 1.25
        elif alpha == 1.0:
            beta_scale = 2.0
        else:
            beta_scale = 0.9
    else:
        beta_scale = 1.5

    if beta_n == 55:
        beta_best= beta_o/beta_scale
    else:
        beta_best= beta_n/beta_scale

    X_best = mapping(v_new, beta_best)
    grad_temp = X_best - X                                      
    gJ = v*0                                           
    gJ[:] = tensor_jacobian_product(mapping, 0)(      
        v_new, beta_n, grad_temp,                            
    )  # backprop                                     
    v_new= npa.clip(v + alpha*gJ, 0.0, 1.0)                
    X=mapping(v_new, beta_n)                          

    return v_new, X, Outer_M


"""
Optimization methods for various cases

O1. Goal_Attainment: for multi-objective optimization -> use 1st order momentum gradient
O2. Minimax: for multi-objective optimization -> use minimum gradients only 
O3. Momentum: Adaptive momentum estimation (Adam) -> for escape from saddle point and bias correction
O4. Back_tracking_init: Define the parameters for backtracking
O5. Back_tracking: evaluate the current learning rate with iterative process
"""
""" O1 """
def Goal_Attainment(dT_old, fom, dJ_du,
): # Method for multiobjective optimization
    T_mean= np.mean(fom)
    dT_cur= dT_old*0
    for i in range (0, len(fom)): # for n th object
        print(fom[i])
        if fom[i] - T_mean <= 0: # constrain
            dT_cur += dJ_du[i]#/(npa.clip(npa.max(npa.abs(dJ_du[i])), 1e-9, npa.inf)) # + normaized dj_du
        else:
            dT_cur += dT_old # + dt_du
    dT_cur=dT_cur/len(fom) # dt_du
    return dT_cur

""" O2 """
def Minimax(fom, dJ_du,
): # Method for multiobjective optimization
    T_mean= np.mean(fom)
    dT_cur= dJ_du[0]*0
    Weight_sum= 0
    for i in range (0, len(fom)): # for n th object
        print(fom[i])
        #Weight= 1-fom[i]
        if fom[i] - T_mean <= 0: # constrain
            dT_cur += dJ_du[i]#/(npa.clip(npa.max(npa.abs(dJ_du[i])), 1e-9, npa.inf)) # + normaized dj_du
            Weight_sum +=1
    dT_cur=dT_cur/Weight_sum # dt_du
    return dT_cur

""" O3 """
def Momentum(dF_cur, dF_old, dF_old2, numevl
): # Momentum (Adam)
    N_grad = 9
    bt1= N_grad/10
    grad_adj = bt1*dF_old + (1-bt1)*dF_cur
    RMSprop = (1-(1-bt1)**2)*(dF_old2) + ((1-bt1)**2)*(dF_cur**2)
    Bias_corr = RMSprop/(1-bt1**(numevl+1))
    grad_prop = grad_adj/ (np.sqrt(Bias_corr) + 1e-8)
    return grad_prop, grad_adj, RMSprop

""" O4 """
def Back_tracking_init(f0, f0s, dJ_dus):
    foms_info=[0]*7
    grad_info=[0]*7
    fom_info=[0]*7    

    grad_info[0]= dJ_dus
    foms_info[0]= f0s
    fom_info[0]=f0
    
    return grad_info, foms_info, fom_info, False, False

""" O5 """
def Back_tracking(grad_info, foms_info, fom_info, Backtraking_count, trig, N):
    print("fom without backtracking")
    print(fom_info[0])
    if fom_info[0] > fom_info[Backtraking_count]:
        print("Current fom is smaller than first one")
        print(fom_info[Backtraking_count])
        if trig:
            return trig, True
        else:
            if N==0 and fom_info[Backtraking_count-1] > fom_info[0]:
                print("Current fom is smaller than last one")
                print(fom_info[Backtraking_count])
                return trig, True
            else :
                if Backtraking_count >N-1:
                    trig=True
                return trig, False
    else:
        print("Current fom is larger than first one")
        trig=False
        if fom_info[Backtraking_count-1] > fom_info[Backtraking_count]:
            print("Current fom is smaller than last one")
            print(fom_info[Backtraking_count])
            return trig, True
        else:
            print("Current fom is larglist one")
            print(fom_info[Backtraking_count])
            return trig, False

        
def Back_tracking_call(fom_info, foms_info, grad_info, Backtraking_count):
    f0=fom_info[Backtraking_count]      
    f0s=foms_info[Backtraking_count]    
    dJ_dus=grad_info[Backtraking_count] 
    return f0, f0s, dJ_dus

"""
Calculation methods for FoM

C1. Cross_product: Cross product for power calculation
C2. Substract_field: Substract the fields to calculate the scattered fields <- needs pre-simulation for total fields
C3. Overlap_intg: Overlap integration to calculate purity of current field profile <- needs pre-simulation for target fields
"""
""" C1 """
def Cross_product(E, H, is_3D: bool = True):
    if is_3D:
        axis=2
    else:
        axis=1
    S=[0]*3
    S[0]= E[1]*npa.conjugate(H[2]) - E[2]*npa.conjugate(H[1])
    S[1]= E[2]*npa.conjugate(H[0]) - E[0]*npa.conjugate(H[2])
    S[2]= E[0]*npa.conjugate(H[1]) - E[1]*npa.conjugate(H[0])
    power= npa.abs(npa.sum(S[axis]))
    return power

""" C2 """
def Substract_field(Total_field, Scattered_field):
    Substracted_field= [0]*3
    Substracted_field[0]= Scattered_field[0]-Total_field[0]
    Substracted_field[1]= Scattered_field[1]-Total_field[1]
    Substracted_field[2]= Scattered_field[2]-Total_field[2]
    return Substracted_field

""" C3 """
def Overlap_intg(Target, Output, normalization=False, Reflection=False):
    if Reflection:
        X=Output[0]*(Target[0])
        Y=Output[1]*(Target[1])
        Z=Output[2]*(Target[2])
    else:
        X=Output[0]*npa.conjugate(Target[0])
        Y=Output[1]*npa.conjugate(Target[1])
        Z=Output[2]*npa.conjugate(Target[2])
    FoM=npa.abs(npa.sum(X+Y+Z))**2
    if normalization:
        Tn=(npa.abs(Target[0])**2) + (npa.abs(Target[1])**2) + (npa.abs(Target[2])**2)
        On= (npa.abs(Output[0])**2) + (npa.abs(Output[1])**2) + (npa.abs(Output[2])**2)
        Purity=FoM/((npa.sum(Tn))*(npa.sum(On)))
        return Purity
    else:
        return FoM

""" 
Mapping
"""
class Mapping:
    def __init__(
        self,
        Symmetry_sim= False,
        Sym_geo_width= False,
        Sym_geo_length= False,
        Sym_geo_C2= False,
        Is_waveguide= [False, False, False, 2], # Is_wg?, Is_middle?, Is_3D?, number of local region
        DR_info= [None, None, None, 1, 2, 0], # x,y,z, width-axis, length-axis, hight-axis: 0(x), 1(y), 2(z)
        DR_N_info= [None, None, None, None], # Nx, Ny, Nz, resolution
        Mask_info= [None, None], # wg_left, wg_right
        Mask_pixels= 0,
        MFS= None,
        MGS= None,
    ):
        self.MFS = MFS
        self.MGS = MGS

        self.MFS0 = MFS
        self.MGS0 = MGS
        self.N_height = DR_N_info[DR_info[5]]
        self.DR_res = DR_N_info[3]
        self.Mask_info = Mask_info
        self.Mask_pixels = Mask_pixels
        self.Is_waveguide = Is_waveguide

        self.Symmetry_sim = Symmetry_sim
        self.Sym_geo_width = Sym_geo_width
        self.Sym_geo_length = Sym_geo_length
        self.Sym_geo_C2 = Sym_geo_C2

        if DR_info[5] == DR_info[4] or DR_info[5] == DR_info[3]:
            raise ValueError("Design layer includes height axis")
        elif DR_info[3] >= DR_info[4]:
            raise ValueError("Design layer width axis should be low axis")

        self.Mask_info[0]=0
        self.Mask_info[1]=0

        self.DR_width = DR_info[DR_info[3]]
        self.N_width = DR_N_info[DR_info[3]]

        self.DR_length = DR_info[DR_info[4]]
        self.N_length = DR_N_info[DR_info[4]]

    def __call__(
        self,
        x,
        beta,
        Is_optimization = True,
    ) -> np.ndarray:
        if beta == npa.inf:
            beta= 55
            Is_optimization= False #last iter
        x_copy = Sub_Mapping.get_reference_layer(
            Symmetry_in_Sim= self.Symmetry_sim,
            Width_symmetry= self.Sym_geo_width,
            Length_symmetry= self.Sym_geo_length,
            C2_symmetry= self.Sym_geo_C2,
            Is_waveguide= self.Is_waveguide[0],
            DR_width= self.DR_width, 
            N_width= self.N_width, 
            DR_length= self.DR_length, 
            N_length= self.N_length, 
            DR_res= self.DR_res,
            Min_size_top= self.MFS0,
            Min_gap= self.MGS0,
            input_w_top= self.Mask_info[0],
            w_top= self.Mask_info[1],
            x= x,
            beta= beta,
            Mask_pixels= self.Mask_pixels,             
        )
        x= x_copy#(x_copy.reshape(Ny,Nz)).transpose()
        if not Is_optimization:
            x= mpa.conic_filter(
                npa.clip(x, 0.0, 1.0), 
                self.MFS0/2, # constrain
                self.DR_width, 
                self.DR_length, 
                self.DR_res,
            )
        return x.flatten()

def Design_update(v, alpha, g, mapping, beta, flag=False, is_init=False):
    if is_init:
        h_list=[0]*2
        if beta > 55:        
            beta_o=beta 
            beta= 55     
        else:
            beta_o=beta
        h_list[0]= beta
        Updated_v= npa.clip(v+ alpha*g, 0.0, 1.0)                       
        X=mapping(Updated_v, beta)
        if alpha == 0 and flag:
            beta= npa.inf
            X=mapping(Updated_v, beta)
            X= mpa.tanh_projection(X, beta, 0.5)
        Bi_idx = npa.sum(npa.where(((1e-4<X)&(X<(1-1e-4))), 0, 1))/X.size
        print("Current beta: ", beta)                    
        print(f"Binarization rate: {round(Bi_idx*100,2)}%\n")        
        h_list[1]= Bi_idx    
        return Updated_v, X, beta, beta_o, h_list
    else:
        Updated_v= npa.clip(v+ alpha*g, 0.0, 1.0)
        X= mapping(Updated_v, beta)
        return Updated_v, X

def inv_tanh_proj(beta, eta):
    return (npa.tanh(beta * eta) + np.log(np.abs(npa.cosh(beta)))/beta) / (npa.tanh(beta * eta) + npa.tanh(beta * (1 - eta)))

def projection_error(beta, eta):
    if beta>50:
        return 0.5
    A= inv_tanh_proj(beta, eta)
    for db in range(5000, 0, -1):
        dB=db/10000
        B= inv_tanh_proj(beta+dB, eta)
        Err= np.abs((B-A)/A)
        if Err < 0.002:
            break
    if dB==0.5:
        for db in range(10000, 20000, +1):
            dB=db/20000
            B= inv_tanh_proj(beta+dB, eta)
            Err= np.abs((B-A)/A)
            if Err > 0.0019:
                break  
    return dB

def Beta_update(flag, numevl, OM4, last_fom, beta):
    if flag:
        beta_n = beta
        if numevl == 1:
            tol = 1
            last_fom = 9999
            if OM4:
                print("Beta updated with rescaled step")
                Case=1
                return Case, tol, last_fom, beta_n
            else:
                print("Beta updated with 1.5 step")
                Case=2
                return Case, tol, last_fom, beta_n
        else:
            tol = 0
            print("Continue with fixed beta")
            Case=3
            return Case, tol, last_fom, beta_n
    else:
        del_beta= projection_error(beta, 0.05)
        beta_n = beta + del_beta
        tol = 0
        print("Beta upated with gradual step")
        Case=4
        return Case, tol, last_fom, beta_n
