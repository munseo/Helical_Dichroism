import sys
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import meep.adjoint as mpa
import os
from mpi4py import MPI
from autograd import numpy as npa
sys.path.append('/home/msb/msopt')
sys.path.append('/home/user/msb/msopt')
sys.path.append('../msopt')
import module

comm= MPI.COMM_WORLD
rank= comm.Get_rank()

mp.verbosity(1)

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
source_2_geo = round(0.8/meep_unit, 2) # 0.8 um
HD_t= round(0.8/meep_unit, 2) # Design thickness 0.25 um
SiO2_t= round(0.6/meep_unit, 2) # Substrate thickness w/o pml : 0.6 um

Air_Pad_side= 2.0#round(0.25/meep_unit - Mask_thick/resolution, 2) # 0.5 um - mask size

Sx= Air_Pad_side+ HD_w+ Air_Pad_side
Sy= Air_Pad_side+ HD_h+ Air_Pad_side
Sz= pml_2_source+ source_2_geo+ HD_t+ SiO2_t

""" Custom coordinate """
X_min= round(0.5*-Sx, 2)       
Y_min= round(0.5*-Sy, 2)           
Z_min= round(0.5*-Sz, 2)           

X_max= round(0.5*Sx, 2)              
Y_max= round(0.5*Sy, 2)             
Z_max= round(0.5*Sz, 2)          

""" Source information """
w0= round(1.6, 2) # beam waist: 1008 nm
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
# EVEN order: positive, ODD order: negative
Lx= Sx+ 2*Lpml
Ly= Sy+ 2*Lpml
# Input: Lx, Ly, Lpml, resolution, mode, freq, fwidth, CP(1: RCP, -1: LCP, 0: x-pol)

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
            center= mp.Vector3(0, 0, Z_max - (SiO2_t - Lpml)/2),
            material= SiO2,
            size= mp.Vector3(Sx+ 2*Lpml, Sy+ 2*Lpml, SiO2_t+ Lpml) 
        ),    
        mp.Block(#Design region
            center=design_region.center, size=design_region.size, material=design_variables
        ),  
    ]
######################################## Post process ####################################################
os.chdir('A')
Opt_design=np.loadtxt('lastdesign.txt')
design_variables.update_weights(Opt_design)

z_slice = npa.reshape(Opt_design,(Nx*Ny,Nz))
eps_data = npa.reshape(z_slice[:,0],(Nx,Ny))
plt.figure()
plt.imshow(eps_data.transpose() , cmap='binary')
plt.axis('off')
plt.savefig('eps_last.png')
# np.savetxt("lastdesign.txt", design_variables.weights)
result_dir="./Results/"
if rank==0 and not os.path.exists(result_dir):
    os.makedirs(result_dir)
os.chdir(result_dir)
# np.savetxt("lastdesign.txt", design_variables.weights)
############################## create list ##############################
R_p=[]
T_p=[]
R_n=[]
T_n=[]
P_p=[]
P_n=[]
for sign in [0, 1]:
    for i in range(11):
        if sign==0:
            print("OAM mode: ", i)
            #  LG Beam source configuration
            l=i 
        else:
            print("OAM mode: ", -i)
            #  LG Beam source configuration
            l=-i 
        sourcess=module.LG_source(Lx, Ly, Lpml, resolution, l, fcen, fwidth, 0, w0, z0, z_pos)
        Target=module.LG_source(Lx, Ly, Lpml, resolution, l, fcen, fwidth, 0, w0, -z0, z_pos)
        if True: #free space
            print("Free space")
            #  Reflection and Transmission Simulation
            sim_free = mp.Simulation(cell_size=cell,
                            boundary_layers=pml_layers,
                            sources=sourcess,
                            eps_averaging=False,
                            resolution=resolution,
                            force_complex_fields=True,
                            default_material = Air,
                        )
            
            input_f = sim_free.add_flux(fcen, 0, 1, mp.FluxRegion(center=R_center, size=Monitor_size ))
            
            input_dft_fields_x = sim_free.add_dft_fields([mp.Ex],fcen,0,1,center=R_center,size=Monitor_size)
            input_dft_fields_y = sim_free.add_dft_fields([mp.Ey],fcen,0,1,center=R_center,size=Monitor_size)
            input_dft_fields_z = sim_free.add_dft_fields([mp.Ez],fcen,0,1,center=R_center,size=Monitor_size)

            input_dft_fields_Hx = sim_free.add_dft_fields([mp.Hx],fcen,0,1,center=R_center,size=Monitor_size)
            input_dft_fields_Hy = sim_free.add_dft_fields([mp.Hy],fcen,0,1,center=R_center,size=Monitor_size)

            sim_free.run(until_after_sources=mp.stop_when_dft_decayed(1e-6,0))

            Subst_E_Field= [0]*3
            Subst_E_Field[0]= sim_free.get_dft_array(input_dft_fields_x, mp.Ex, 0)
            Subst_E_Field[1]= sim_free.get_dft_array(input_dft_fields_y, mp.Ey, 0)
            Subst_E_Field[2]= sim_free.get_dft_array(input_dft_fields_z, mp.Ez, 0)

            incidence_flux = np.abs(mp.get_fluxes(input_f))
            incidentFluxToSubtract = sim_free.get_flux_data(input_f)

            Ex_Namei=f"Ex_input_field{l}"
            Ey_Namei=f"Ey_input_field{l}"
            #Ez_Namei=f"Ez_input_field{l}"
            Hx_Namei=f"Hx_input_field{l}"
            Hy_Namei=f"Hy_input_field{l}"
            sim_free.output_dft(input_dft_fields_x,Ex_Namei)
            sim_free.output_dft(input_dft_fields_y,Ey_Namei)
            # sim_free.output_dft(input_dft_fields_z,Ez_Namei)
            sim_free.output_dft(input_dft_fields_Hx,Hx_Namei)
            sim_free.output_dft(input_dft_fields_Hy,Hy_Namei)
            sim_free.reset_meep() #<-Must reset the simulation after run it / If not, flux would be accumulated by load_minus_flux_data 
        if False:
            sim_target = mp.Simulation(cell_size=cell,
                            boundary_layers=pml_layers,
                            sources=Target,
                            eps_averaging=False,
                            resolution=resolution,
                            force_complex_fields=True,
                            default_material = Air,
                        )
            
            target_dft_fields_x = sim_target.add_dft_fields([mp.Ex],fcen,0,1,center=R_center,size=Monitor_size)
            target_dft_fields_y = sim_target.add_dft_fields([mp.Ey],fcen,0,1,center=R_center,size=Monitor_size)
            target_dft_fields_z = sim_target.add_dft_fields([mp.Ey],fcen,0,1,center=R_center,size=Monitor_size)

            sim_target.run(until_after_sources=mp.stop_when_dft_decayed(1e-6,0))

            Target_E_Field= [0]*3
            Target_E_Field[0]= sim_target.get_dft_array(target_dft_fields_x, mp.Ex, 0)
            Target_E_Field[1]= sim_target.get_dft_array(target_dft_fields_y, mp.Ey, 0)
            Target_E_Field[2]= sim_target.get_dft_array(target_dft_fields_z, mp.Ez, 0)

            sim_target.reset_meep() #<-Must reset the simulation after run it / If not, flux would be accumulated by load_minus_flux_data 
        
        if True: # Geo run
            print("Material space")
            sim = mp.Simulation(cell_size=cell,
                            boundary_layers=pml_layers,
                            sources=sourcess,
                            geometry=geometry,
                            eps_averaging=False,
                            resolution=resolution,
                            force_complex_fields=True,
                            default_material = Air,
                            extra_materials = [poly,SiO2]
                        )
            #  Transmission monitors
            trans_dft_fields_x = sim.add_dft_fields([mp.Ex],fcen,0,1,center=T_center,size=Monitor_size)
            trans_dft_fields_y = sim.add_dft_fields([mp.Ey],fcen,0,1,center=T_center,size=Monitor_size)

            trans_dft_fields_Hx = sim.add_dft_fields([mp.Hx],fcen,0,1,center=T_center,size=Monitor_size)
            trans_dft_fields_Hy = sim.add_dft_fields([mp.Hy],fcen,0,1,center=T_center,size=Monitor_size)
            #  Reflection monitors
            reflec_dft_fields_x = sim.add_dft_fields([mp.Ex],fcen,0,1,center=R_center,size=Monitor_size)
            reflec_dft_fields_y = sim.add_dft_fields([mp.Ey],fcen,0,1,center=R_center,size=Monitor_size)
            reflec_dft_fields_z = sim.add_dft_fields([mp.Ez],fcen,0,1,center=R_center,size=Monitor_size)            

            reflec_dft_fields_Hx = sim.add_dft_fields([mp.Hx],fcen,0,1,center=R_center,size=Monitor_size)
            reflec_dft_fields_Hy = sim.add_dft_fields([mp.Hy],fcen,0,1,center=R_center,size=Monitor_size)
            # Z axis monitor
            middle_dft_fields_x = sim.add_dft_fields([mp.Ex],fcen,0,1,center=mp.Vector3(0, 0, 0),size=mp.Vector3(0, Sy, Sz))
            middle_dft_fields_y = sim.add_dft_fields([mp.Ey],fcen,0,1,center=mp.Vector3(0, 0, 0),size=mp.Vector3(0, Sy, Sz))

            middle_dft_fields_Hx = sim.add_dft_fields([mp.Hx],fcen,0,1,center=mp.Vector3(0, 0, 0),size=mp.Vector3(0, Sy, Sz))
            middle_dft_fields_Hy = sim.add_dft_fields([mp.Hy],fcen,0,1,center=mp.Vector3(0, 0, 0),size=mp.Vector3(0, Sy, Sz))
            # Fluxes
            trans = sim.add_flux(fcen, 0, 1, mp.FluxRegion(center=T_center, size=Monitor_size ))
            reflec = sim.add_flux(fcen, 0, 1, mp.FluxRegion(center=R_center, size=Monitor_size ))
            sim.load_minus_flux_data(reflec,incidentFluxToSubtract)
            
            sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-6,0)) # <- run until dft decayed in overall simulation region to Source*1e-6
            
            # Refl_E_Field= [0]*3
            # Refl_E_Field[0]= sim.get_dft_array(reflec_dft_fields_x, mp.Ex, 0)
            # Refl_E_Field[1]= sim.get_dft_array(reflec_dft_fields_y, mp.Ey, 0)
            # Refl_E_Field[2]= sim.get_dft_array(reflec_dft_fields_z, mp.Ez, 0)

            # Substracted_E_field= Opt_MS.Substract_field(Subst_E_Field, Refl_E_Field)
            # Purity=Opt_MS.Overlap_intg(Target_E_Field, Substracted_E_field, normalization=True)

            # Transmitted flux
            trans_flux = np.array(mp.get_fluxes(trans))
            Transmission= trans_flux/incidence_flux
            # Reflected flux
            reflec_flux = np.array(mp.get_fluxes(reflec))
            Reflectance= reflec_flux/incidence_flux

            if sign==0:
                T_p.append(Transmission*100)
                R_p.append(-Reflectance*100)
                #P_p.append(Purity*100)
            else:
                T_n.append(Transmission*100)
                R_n.append(-Reflectance*100)
                #P_n.append(Purity*100)

            
            Ex_Namei=f"Ex_R_field{l}"
            Ey_Namei=f"Ey_R_field{l}"
            # Ez_Namei=f"Ez_R_field{l}"
            Hx_Namei=f"Hx_R_field{l}"
            Hy_Namei=f"Hy_R_field{l}"
            sim.output_dft(reflec_dft_fields_x,Ex_Namei)
            sim.output_dft(reflec_dft_fields_y,Ey_Namei)
            # sim.output_dft(reflec_dft_fields_z,Ez_Namei)
            sim.output_dft(reflec_dft_fields_Hx,Hx_Namei)
            sim.output_dft(reflec_dft_fields_Hy,Hy_Namei)

            Ex_Namei=f"Ex_T_field{l}"
            Ey_Namei=f"Ey_T_field{l}"
            # Ez_Namei=f"Ez_T_field{l}"
            Hx_Namei=f"Hx_T_field{l}"
            Hy_Namei=f"Hy_T_field{l}"
            sim.output_dft(trans_dft_fields_x,Ex_Namei)
            sim.output_dft(trans_dft_fields_y,Ey_Namei)
            # sim.output_dft(trans_dft_fields_z,Ez_Namei)
            sim.output_dft(trans_dft_fields_Hx,Hx_Namei)
            sim.output_dft(trans_dft_fields_Hy,Hy_Namei)


            Ex_Namei=f"Ex_M_field{l}"
            Ey_Namei=f"Ey_M_field{l}"
            # Ez_Namei=f"Ez_M_field{l}"
            Hx_Namei=f"Hx_M_field{l}"
            Hy_Namei=f"Hy_M_field{l}"
            sim.output_dft(middle_dft_fields_Hx,Hx_Namei)
            sim.output_dft(middle_dft_fields_Hy,Hy_Namei)
            # sim.output_dft(middle_dft_fields_z,Ez_Namei)
            sim.output_dft(middle_dft_fields_x,Ex_Namei)
            sim.output_dft(middle_dft_fields_y,Ey_Namei)
            
            sim.reset_meep() #<-Must reset the simulation after run it / If not, flux would be accumulated by load_minus_flux_data 

np.savetxt('Tran_pos.txt',T_p)
np.savetxt('Tran_neg.txt',T_n)
np.savetxt('Refl_pos.txt',R_p)
np.savetxt('Refl_neg.txt',R_n)
#np.savetxt('Purity_pos.txt',P_p)
#np.savetxt('Purity_neg.txt',P_n)

plt.figure()
plt.plot(R_p, "r-")
plt.plot(R_n, "b-")
plt.grid(True)
plt.xlabel("|l|")
plt.ylabel("R")
plt.savefig("result1.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure

plt.figure()
plt.plot(T_p, "r-")
plt.plot(T_n, "b-")
plt.grid(True)
plt.xlabel("|l|")
plt.ylabel("T")
plt.savefig("result2.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure

# plt.figure()
# plt.plot(P_p, "r-")
# plt.plot(P_n, "b-")
# plt.grid(True)
# plt.xlabel("|l|")
# plt.ylabel("R")
# plt.savefig("result0.png")
# plt.cla()   # clear the current axes
# plt.clf()   # clear the current figure
# plt.close() # closes the current figure


HD=[0]*11
for i in range(11):
    HD[i]=2*(R_n[i]-R_p[i])/(R_n[i]+R_p[i])

# HD_oam=[0]*11
# for i in range(11):
#     HD_oam[i]=2*(R_n[i]*P_n[i]-R_p[i]*P_p[i])/(R_n[i]*P_n[i]+R_p[i]*P_p[i])

plt.figure()
plt.plot(HD, "g-")
plt.grid(True)
plt.xlabel("|l|")
plt.ylabel("HD")
plt.savefig("result3.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure


# plt.figure()
# plt.plot(HD_oam, "g-")
# plt.grid(True)
# plt.xlabel("|l|")
# plt.ylabel("HD")
# plt.savefig("result4.png")
# plt.cla()   # clear the current axes
# plt.clf()   # clear the current figure
# plt.close() # closes the current figure

# os.chdir("..")
# os.chdir("..")
