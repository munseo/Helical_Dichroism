"""
Last update: 2025/02/05 by munseong
"""
import numpy as np
import meep.adjoint as mpa
import autograd.numpy as npa

def get_conic_radius(b, eta_e):
    if (eta_e >= 0.5) and (eta_e < 0.75):
        return b / (2 * np.sqrt(eta_e - 0.5))
    elif (eta_e >= 0.75) and (eta_e <= 1):
        return b / (2 - 2 * np.sqrt(1 - eta_e))
    else:
        raise ValueError("(eta_e) must be between 0.5 and 1.")
#------------------------------------------------------------------
def tanh_projection_m(x: np.ndarray, beta: float, eta: float) -> np.ndarray:
    if beta == npa.inf:
        if eta == 0.5:
            return npa.where(x < eta, 0.0, 1.0)    
        else:
            return npa.where(x > eta, 1.0, 0.0) # save less -> more erosion
    
    else:
        return (npa.tanh(beta * eta) + npa.tanh(beta * (x - eta))) / (
            npa.tanh(beta * eta) + npa.tanh(beta * (1 - eta))
        )

def get_reference_layer(#-- Input parameters ---------------------|
    Symmetry_in_Sim, # Check the symmetry in simulation           |
    Width_symmetry,  # Width symmetry in geometry                 |
    Length_symmetry, # Length symmetry in geometry                |
    C2_symmetry,     # C2 symmetry in geometry                    |
    Is_waveguide,    #                                            |
    #                                                             |
    DR_width, # Disign region width                               |
    N_width,  # Number of DR width grids                          |
    #                                                             |
    DR_length, # Disign region length                             |
    N_length,  # Number of DR length grids                        |
    #                                                             |
    DR_res, # Design region resolution                            |
    #                                                             |
    Min_size_top, # Minimum linwidth of top layer                 |
    Min_gap, # Minimum gap width of ref layer                     |
    #                                                             |
    input_w_top, # Top width of input waveguide                   |
    w_top, # Top width of output waveguide                        |
    #                                                             |
    x, # Input material density array                             |
    #                                                             |
    beta,     # Binrization parameter                             |
    Mask_pixels, # Number of pixels for mask region               |
):                                                                #-- Impose MFS and MGS on x ------------------------|
    eta_Ref = 0.05                                                      # Thresholding point of tanh projection       |
    Single_pixel= round(1/DR_res,2)                                                          #                        |
    Gap_Ref= Min_gap                                                                         #                        |
    Total_R = get_conic_radius(Min_size_top+ Gap_Ref, 1-eta_Ref)                             # Radius for MFS + MGS   |
    MGS_R = get_conic_radius((Gap_Ref), 1-eta_Ref)                                           # Radius for MGS         |
    N_total = round(Total_R/(4*Single_pixel))                     # Total number of Erosion & Dilation with R= 4pixel |
    N_Half = round(Total_R/(8*Single_pixel))                      # Half number of Erosion & Dilation with R= 4pixel  |
    # Set filter R, number of dilation & erosion & smoothing ---------------------------------------------------------|
    if N_total > 1:                               # Total_R > 5 pixel ------------------------------------------------|
        N_total = 2                                                                          #                        |
        if N_Half < 1:                                      # Half_R < 5 pixel ---------------------------------------|
            N_Half = 1                                                                       #                        |
        N_dilation = N_total*N_Half                               # Number of dilation = 2*N_Half                     |
        Dilation_R = get_conic_radius((Gap_Ref + Min_size_top)/(N_dilation), 1-eta_Ref)      # R for unit dilation    |
        Half_R = (Gap_Ref + Min_size_top)/2                                                  # R for Half smoothing   |
        if Gap_Ref > Min_size_top:                         # MGS > MFS  ----------------------------------------------|
            MGS_R = get_conic_radius((Gap_Ref - Min_size_top)/2, 1-eta_Ref)                  # R for half erosion     |
            N_MGS = round(MGS_R/(4*Single_pixel))              # Number of unit erosion with R= 4pixel                |
            if N_MGS < 1:                                      # MGS_R < 2 pixel -------------------------------------|
                N_MGS = 1                                                                    #                        |
            Erosion_R = get_conic_radius((Gap_Ref - Min_size_top)/(2*N_MGS), 1-eta_Ref)      # R for unit erosion     |
            Smoothing_E = True                                                   # Half smoothing during erosion      |
            Smoothing_D = False                                                  #                                    |
            N_erosion = N_Half + N_MGS                                           # Number of erosion = N_Half + N_MGS |
        else:                                               # MGS =< MFS ---------------------------------------------|
            N_MGS = round(MGS_R/(4*Single_pixel))              # Number of unit erosion with R= 4pixel                |
            if N_MGS < 1:                                      # MGS_R < 2 pixel -------------------------------------|
                N_MGS = 1                                                                    #                        |
            Erosion_R = get_conic_radius((Gap_Ref)/(N_MGS), 1-eta_Ref)                       # R for unit erosion     |
            Smoothing_E = False                                                  #                                    |
            Smoothing_D = True                                                   # Half smoothing during dilation     |
            N_erosion = N_MGS                                                    # Number of erosion = N_MGS          |
    else:                                         # Total_R <= 5 pixel -----------------------------------------------|
        N_dilation = 1                                                           # Single dilation                    |
        N_erosion = 1                                                            # Single erosion                     |
        Dilation_R = Total_R                                                     # R for dilation = MGS + MFS         |
        Erosion_R = MGS_R                                                        # R for erosion = MGS                |
        Smoothing_D = False                                                      # Deactivate half smoothing          |
        Smoothing_E = False                                                      # Deactivate half smoothing          |
    #-----------------------------------------------------------------------------------------------------------------|
    if Mask_pixels == 0 and not Is_waveguide:
        Air_pad= int(((Gap_Ref)/2+ Dilation_R + Single_pixel)/Single_pixel)+1                # Size of Air pad        |
        x= npa.pad(npa.reshape(x,(N_width,N_length)),(Air_pad,Air_pad),mode="constant", constant_values=0)
        x= x.flatten() 
        N_length += 2*Air_pad
        N_width  += 2*Air_pad
        DR_width += 2*round(Air_pad/DR_res,2)
        DR_length += 2*round(Air_pad/DR_res,2)
        #Minimum_mask= (Min_size_top)/2
        print("Use Air pad")
    else:
        Air_pad= 0
    Minimum_mask= (Gap_Ref + Min_size_top)/2+ Dilation_R + Single_pixel                      # Minimum size of Mask   |
    if Mask_pixels <= Minimum_mask/Single_pixel:                                             #                        |
        Mask_pixels = int(Minimum_mask/Single_pixel)+1                                       #                        |
        print(f"Mask size changed to {Mask_pixels} Pixels ")                                 #                        |
    width_grid = np.linspace(-DR_width /2, DR_width /2, N_width)                             #                        |
    length_grid = np.linspace(-DR_length /2, DR_length /2, N_length)                         #                        |
    W_g, L_g = np.meshgrid(width_grid, length_grid, sparse=True, indexing="ij")              #                        |
    border_mask = (                                                                          # Cladding mask ---------|
        (W_g >= W_g[N_width-1-Mask_pixels])                                                  #                        |
        | (W_g <= W_g[Mask_pixels])                                                          #                        |
        | (L_g <= L_g[:,Mask_pixels])                                                        #                        |
        | (L_g >= L_g[:,N_length-1-Mask_pixels])                                             #                        |
    )                                                                                        #------------------------|
    Air_mask = border_mask.copy()                                                            # Mask ------------------|
    if Is_waveguide: #-----------------------------------------------------------------------# Define the mask region |
        Mask_sz_ii = round(0.5*(input_w_top)/Single_pixel)                                   # Core mask -------------|
        Mask_sz_io = round(0.5*(w_top)/Single_pixel)                                         #                        |
        down_wg_mask_init= (                                                                 #                        |
            (L_g <=L_g[:,Mask_pixels])                                                       #                        |
            & (np.abs(W_g)<=W_g[int((N_width-1)/2+ Mask_sz_ii)])                             #                        |
            )                                                                                #                        |
        up_wg_mask_init= (                                                                   #                        |
            (L_g >=L_g[:,N_length-1-Mask_pixels])                                            #                        |
            & (np.abs(W_g)<=W_g[round((N_width-1)/2+Mask_sz_io)])                            #                        |
            )                                                                                #                        |
        Li_mask_init = down_wg_mask_init | up_wg_mask_init                                   #                        |
        border_mask = (                                                                      # Cladding mask ---------|
            (W_g >= W_g[N_width-1-Mask_pixels])                                              #                        |
            | (W_g <= W_g[Mask_pixels])                                                      #                        |
            | (L_g <= L_g[:,Mask_pixels])                                                    #                        |
            | (L_g >= L_g[:,N_length-1-Mask_pixels])                                         #                        |
        )                                                                                    #------------------------|
        Air_mask[Li_mask_init] = False                                                       #                        |
    #-----------------------------------------------------------------------------------------------------------------|
    x_ref = npa.reshape(x,(N_width,N_length))  # Reshape for flip ----------------------------------------------------|
    if Width_symmetry:                                                                       # Width Mirror ----------|
        if Symmetry_in_Sim:                                                                  #   Y Mirror Sim Case    |
            print('Gradient has Width symmetry')                                             #                        |
        else:                                                                                #                        |
            if N_width/2 == int(N_width/2):                                                  #                        |
                x_ref = ((npa.flipud(x_ref)) + x_ref)/2                                      #                        |
                print('ODD pixels & EVEN grids')                                             #                        |
            else:                                                                            #                        |
                x_ref = ((npa.flipud(x_ref)) + x_ref)/2                                      #                        |
                print('EVEN pixels & ODD grids')                                             #                        |
    if Length_symmetry:                                                                      # Length Mirror ---------|
        x_ref = (npa.fliplr(x_ref) + x_ref)/2                                                #                        |
    if C2_symmetry:                                                                          # C2 --------------------|
        x_ref = (npa.rot90(x_ref,2)+ x_ref)/2                                                #                        |
    x_ref = x_ref.flatten()                                                                  #                        |
    if Is_waveguide:                                                                         # Masking            ----|
        x_ref = npa.where(Li_mask_init.flatten(), 1, npa.where(Air_mask.flatten(), 0, x_ref))#                        |
    else:                                                                                    #                        |
        x_ref = npa.where(Air_mask.flatten(), 0, x_ref)                                      #                        |
    #-----------------------------------------------------------------------------------------------------------------|
    x_copy = mpa.tanh_projection(x_ref, beta, 0.5)                                           # Binarization ----------|
    Dilation_index= 0                                                                        #                        |
    while Dilation_index < N_dilation:    #------- Activate multiple dilation N_dilation times -----------------------|
        x_copy= mpa.conic_filter(npa.clip(x_copy, 0.0, 1.0), Dilation_R, DR_width, DR_length, DR_res)   #             |
        x_copy = tanh_projection_m(x_copy, beta, eta_Ref)                                 # Unit dilation             |
        if Smoothing_D and Dilation_index == round(N_dilation/2)-1: # MGS < MFS, Smoothing when MFS = MGS ------------|
            x_copy= mpa.conic_filter(npa.clip(x_copy, 0.0, 1.0), Half_R, DR_width, DR_length, DR_res)             #   |
            x_copy = tanh_projection_m(x_copy, beta, 0.5)                                 # Half smoothing        #   |
            x_copy= mpa.conic_filter(npa.clip(x_copy, 0.0, 1.0), Half_R, DR_width, DR_length, DR_res)             #   |
            x_copy = tanh_projection_m(x_copy, beta, 0.5)                                 # Compensation smoothing    |
        Dilation_index +=1                                                                                        #   |
    Erosion_index= 0                                                                                              #   |
    while Erosion_index < N_erosion:      #--------- Activate multiple erosion N_erosion times -----------------------|
        if Smoothing_E:                             # MGS >= MFS -----------------------------------------------------|
            while Erosion_index < N_Half:            # Erosion with Dilation radius ----------------------------------|
                x_copy= mpa.conic_filter(npa.clip(x_copy, 0.0, 1.0), Dilation_R, DR_width, DR_length, DR_res)     #   |
                x_copy = tanh_projection_m(x_copy, beta, 1-eta_Ref)                       # Unit erosion          #   |
                Erosion_index +=1                                                                                 #   |
            if Erosion_index == N_Half:              # Smoothing when MFS = MGS --------------------------------------|
                x_copy= mpa.conic_filter(npa.clip(x_copy, 0.0, 1.0), Half_R, DR_width, DR_length, DR_res)         #   |
                x_copy = tanh_projection_m(x_copy, beta, 0.5)                             # Half smoothing        #   |
                x_copy= mpa.conic_filter(npa.clip(x_copy, 0.0, 1.0), Half_R, DR_width, DR_length, DR_res)         #   |
                x_copy = tanh_projection_m(x_copy, beta, 0.5)                             # Compensation smoothing    |
        x_copy= mpa.conic_filter(npa.clip(x_copy, 0.0, 1.0), Erosion_R, DR_width, DR_length, DR_res)              #   |
        x_copy = tanh_projection_m(x_copy, beta, 1-eta_Ref)                               # Unit erosion              |
        Erosion_index +=1                                                                                         #   |
    if Air_pad > 0:
        x_copy= x_copy[Air_pad:(N_width-Air_pad),Air_pad:(N_length-Air_pad)]
        print(x_copy.shape)
    x_copy = x_copy.flatten()                                                                                     #   |
    del x_ref                                     # Clear Memory -----------------------------------------------------|
    return x_copy                                                                                # Reference layer ---|
#---------------------------------------------------------------------------------------------------------------------|