import numpy as np
from numpy import linalg as npla

def phase_kr(kx0,ky0,r0):
    return np.exp(1j*(kx0*r0[0]+ky0*r0[1]))
def hex(a0):
    a1=a0*np.array([3/2,np.sqrt(3)/2,0]);a2=a0*np.array([3/2,-np.sqrt(3)/2,0]);a3=a0*np.array([0,0,1])
    V=np.dot(np.cross(a1,a2),a3)
    b1=np.delete(2*np.pi*np.cross(a2,a3)/V,-1) #array([3.62759873, 6.28318531]) # K(2/3,1/3) M (0.5,0) 
    b2=np.delete(2*np.pi*np.cross(a3,a1)/V,-1) #array([3.62759873, -6.28318531])
    return b1,b2 
### get cartesian coordinates, kx[i,j],ky[i,j]
def BZ_mesh(kx1,ky1,a0):
    b1=hex(a0)[0];b2=hex(a0)[1]
    ky2,kx2=np.meshgrid(ky1,kx1) 
    kx=kx2*b1[0]+ky2*b2[0]
    ky=kx2*b1[1]+ky2*b2[1]
    return kx,ky 
def eigen(A):
    l1=A.shape[0];l2=A.shape[1]
    eigenValues, eigenVectors = npla.eigh(A) # for eigh, an arbitrary phase will be added to the eigenvectors, which makes me unhappy
    # for i in range(l1):
    #     for j in range(l2): 
    #         phi=np.angle(eigenVectors[i,j,0,0])
    #         eigenVectors[i,j,:,0]=eigenVectors[i,j,:,0]*np.exp(-1j*phi)
    #         eigenVectors[i,j,:,1]=eigenVectors[i,j,:,1]*np.exp(-1j*phi)
    #         #eigenVectors[i,j,1,1]=eigenVectors[i,j,0,0].conjugate()
    # #print(eigenVectors[2,3,0,0])
    return eigenValues, eigenVectors

def k_cartesion(k_fac,a):
    b1=hex(a)[0]
    b2=hex(a)[1]
    k_cart=k_fac[0]*b1+k_fac[1]*b2
    return k_cart


def get_L_as_chi(chi0,hbarw,theta):
    hbar1=6.62607015*1E-34/(2*np.pi)
    hbar=6.62607015*1E-34/(2*np.pi)*6.242*1E18
    eps0=8.8541878128*1E-12;eps=1
    omeg0=hbarw*0.001/hbar #the unit of hbarw:mev
    theta=1.2*np.pi/180;lat=3.472/(2*np.sin(theta/2));l=np.sqrt(3)/3*lat
    A0=hbar*chi0/l
    V=hbar1/(A0*1E10)**2/(omeg0*eps*eps0)
    L=(V*1e8)**(1/2)*1E10
    return L

def Gaussion_1D(x0,mu0,sigma):
    g=1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x0-mu0)/sigma)**2)
    return g

def Gaussion_2D(x,y):
    sigma=0.1
    g=(1/(sigma*np.sqrt(2*np.pi)))**2*np.exp(-0.5*((x/sigma)**2+(y/sigma)**2))
    return g


import matplotlib.colors as colors
from scipy.interpolate import griddata
from scipy import interpolate
from pylab import *
from matplotlib.path import Path
from matplotlib.patches import PathPatch
def plot_way2(kx1,ky1,z1,a0):
    kx2=np.append(kx1,1);ky2=np.append(ky1,1)
    z2=np.concatenate((z1, np.array([z1[0,:]])), axis=0)
    z3=np.concatenate((z2, np.array([z2[:,0]]).T), axis=1)
    ####list of k points
    G1=hex(a0)[0];G2=hex(a0)[1]
    K1_x=1/3*G1[0]+2/3*G2[0];K1_y=(1/3*G1[1]+2/3*G2[1])
    K2_x=2/3*G1[0]+1/3*G2[0];K2_y=(2/3*G1[1]+1/3*G2[1])
    K3_x=1/3*G1[0]-1/3*G2[0];K3_y=(1/3*G1[1]-1/3*G2[1])
    K4_x=-1/3*G1[0]-2/3*G2[0];K4_y=(-1/3*G1[1]-2/3*G2[1])
    K5_x=-2/3*G1[0]-1/3*G2[0];K5_y=(-2/3*G1[1]-1/3*G2[1])
    K6_x=-1/3*G1[0]+1/3*G2[0];K6_y=(-1/3*G1[1]+1/3*G2[1])
    M1_x=0*G1[0]+0.5*G2[0];M1_y=0*G1[1]+0.5*G2[1]
    M2_x=0.5*G1[0]+0.5*G2[0];M2_y=0.5*G1[1]+0.5*G2[1]
    M3_x=0.5*G1[0]+0*G2[0];M3_y=0.5*G1[1]+0*G2[1]
    Sigma_x=0.1566332999666333*G1[0]+0.4299933266599933*G2[0]
    Sigma_y=0.1566332999666333*G1[1]+0.4299933266599933*G2[1]

    ####interpolate
    ####
    # f1=interpolate.interp2d(kx1,ky1,z1, kind='linear')
    # x1=np.linspace(kx2[0],kx2[-1],300)
    # y1=np.linspace(ky2[0],ky2[-1],300)
    # z4=f1(x1,y1)
    # kx3,ky3=BZ_mesh(x1,y1,a0)
    kx3,ky3=BZ_mesh(kx2,ky2,a0)
    z4=np.copy(z3)
    
    #plt.rcParams.update({'font.family':'Times New Roman','font.size': 22})
    #plt.rcParams.update({'font.family':'Times New Roman','font.size': 10})

    ## Fig 2
    plt.rcParams.update({'font.family':'Times New Roman','font.size': 10})
    cm = 0.1*1/2.54 
    #fig, ax = subplots(figsize=(47*cm, 47 *cm)) ## Fig.2
    fig, ax = subplots(figsize=(29*cm, 29*cm)) ## Fig. 3C

    cNorm = colors.Normalize(vmin=0,vmax=2.4) ## Fig.2 A,B
    #cNorm = colors.Normalize(vmin=0,vmax=0.5) ## Fig. 2C,Fig.3C
    #cNorm = colors.Normalize(vmin=np.min(z4),vmax=np.max(z4)) 
    #cm='hot'
    #cm='viridis'
    #cm='cividis'
    #CorMar='g'
    cm='jet'
  
 

    path = Path([[K1_x, K1_y],[K2_x, K2_y],[K3_x, K3_y], [K4_x, K4_y],[K5_x, K5_y], [K6_x, K6_y]])
    patch = PathPatch(path,transform = ax.transData)
    p = ax.pcolor(kx3,ky3,z4,cmap=cm,norm=cNorm,clip_path = patch)
    p = ax.pcolor(kx3-G1[0],ky3-G1[1],z4,cmap=cm,norm=cNorm,clip_path = patch)
    p = ax.pcolor(kx3-G2[0],ky3-G2[1],z4,cmap=cm,norm=cNorm,clip_path = patch)
    p = ax.pcolor(kx3-G2[0]-G1[0],ky3-G2[1]-G1[1],z4,cmap=cm,norm=cNorm,clip_path = patch)
    #cb = fig.colorbar(p) ## for classical ## Fig.2
    cax = ax.inset_axes([1.03, 0, 0.04, 1]) ## Fig.2
    cb = fig.colorbar(p, cax=cax)
    #cb.set_ticks([0,0.6,1.2,1.8,2.4]) ## Fi2. 2A
    #cb.set_ticks([0,0.1,0.2,0.3,0.4,0.5]) ##Fig. 2C

    # cax = ax.inset_axes([1.03, 0, 0.1, 0.5]) ## Fig.3
    # cb = fig.colorbar(p,cax=cax,ticks=[0,0.5],shrink=0.3,aspect=5)

 
    ax.set_xlim(K4_x,K1_x)
    ax.set_ylim(K6_y,K3_y)
    ax.set_position([0.2, 0.2, 0.65, 0.65])
    ax.set_aspect('equal')


    #kpx=[0,1,2,3,4,7,10,13,14]
    #kpy=[0,3,6,9,10,9,8,7,7]
    kpx=[0,1,2,3,4,5,6,7,7]
    kpy=[0,3,6,9,10,11,12,13,14] ##data 17
    # kpx=[0,1,2,3,3,3,4,5]
    # kpy=[0,2,4,6,7,8,9,10] ##data 20
    # for kn in range (len(kpx)):
    #     i=kpx[kn];j=kpy[kn]
    #     K_x,K_y=kx3[i,j],ky3[i,j]
    #     plt.scatter(K_x,K_y,color=CorMar,s=3)
    #     #print(i,j)

    #ax.set_xticks([]) ##fig 3
    #ax.set_yticks([]) ## fig 3
    #ax.axis("off")
 

    
    