# Helical_Dichroism
Optimization and simulation code for "Inverse Design of Chiral Structures for Giant Helical Dichroism" (https://arxiv.org/abs/2501.12825)   
FDTD simulations for forward, adjoint, and post-processing were implemented with Meep, Python interface (https://github.com/NanoComp/meep)   
The adjoint gradient was calculated with autograd (https://github.com/HIPS/autograd)   

The Laguerre-Gaussian source, design constraints, and gradient ascent algorithm were implemented with custom modules (/msopt)   

The data sources on the "Inverse Design of Chiral Structures for Giant Helical Dichroism" were upploaded in sub-directory of \inv_meep (\inv_meep\A)   
