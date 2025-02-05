import meep as mp
import math
import numpy as np

def lg(l,w,lamb,x,y,z,r):
    #Rayleigh range
    z_r=np.pi*w**2/lamb
    #Beam waist
    w_z=w*np.sqrt((z**2+z_r**2)/z_r**2)
    #Gouy phase
    psi=(l+1)*np.arctan2(z/z_r,1)
    #Normalizing factor
    lg_func = np.sqrt(2/(np.pi*math.factorial(np.abs(l))))
    lg_func *= 1/w_z*(r*np.sqrt(2)/w_z)**(np.abs(l))*np.exp(-r**2/w_z**2)*np.exp(-1j*l*np.arctan2(y,x))
    #Focusing
    lg_func *= np.exp(-1j*psi)*np.exp(1j*2*np.pi/lamb*(r**2)*z/(2*(z**2+z_r**2)))
    #Associate Laguerre
    #lg_func *= scipy.special.assoc_laguerre(2*r**2/w_z**2,0,k=abs(l)) #if p=0 LG_l0(x)=1
    return lg_func

# Lx, Ly, Lpml, resolution, mode, freq, fwidth, CP(1: RCP, -1: LCP), w0, z0
def LG_source(Lx, Ly, Lpml, resolution, mode, freq, fwidth, CP, w0, z0, z_pos):
    wave_length=1/freq
    src = mp.GaussianSource(frequency=freq, fwidth=fwidth, is_integrated=True)
    #src = mp.ContinuousSource(frequency=freq, fwidth=fwidth, is_integrated=True)
    x=Lx/2-Lpml -1/(2*resolution) #(-half pixel)
    y=Ly/2-Lpml -1/(2*resolution)
    source_0=[]
    Amp_0=0
    # LG field (l=0, p=0, z_pos plane, beam waist plane is z=z0)
    #Ex+Ey (LCP)
    while x >= -1*Lx/2+Lpml+1/(2*resolution):
        while y >= -1*Ly/2+Lpml+1/(2*resolution):
            r_source = np.sqrt(x**2 + y**2)
            amplitude = lg(mode,w0,wave_length,x,y,z0,r_source)
            source_0.append(mp.Source(src,
                        component=mp.Ex,
                        center=mp.Vector3(x,y,z_pos),
                        amplitude = amplitude)) 
            if CP**2 == 1:
                source_0.append(mp.Source(src,
                            component=mp.Ey,
                            center=mp.Vector3(x,y,z_pos),
                            amplitude = amplitude*1j*CP)) #<---- this (Ex + Ey*1j) makes RCP
            Amp_0+=np.abs(amplitude)**2
            y-=1/resolution
        x-=1/resolution
        y=Ly/2-Lpml -1/(2*resolution)
    print(Amp_0)
    return source_0
