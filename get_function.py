import numpy as np
from Tools import *
from setting import *
l0=l0_1
def get_v(func,kx1,ky1,*args):
    dkx1=1e-6;dky1=1e-6
    b1=hex(l0)[0];b2=hex(l0)[1]
    kxd=np.linalg.norm(b1+b2)*dkx1;kyd=np.linalg.norm((b1-b2))*dky1
    h0=func(kx1,ky1,*args)[1]
    S=eigen(h0)[-1]
    h0x1=func(kx1+dkx1/2,ky1+dky1/2,*args)[1]
    h0x2=func(kx1-dkx1/2,ky1-dky1/2,*args)[1]
    h0y1=func(kx1+dkx1/2,ky1-dky1/2,*args)[1]
    h0y2=func(kx1-dkx1/2,ky1+dky1/2,*args)[1]
    vx0=(h0x1-h0x2)/kxd;vy0=(h0y1-h0y2)/kyd
    vx=np.matmul(np.matmul(np.conjugate(np.transpose(S,(0,1,3,2))),vx0),S)
    vy=np.matmul(np.matmul(np.conjugate(np.transpose(S,(0,1,3,2))),vy0),S)
    #H=np.matmul(np.matmul(np.conjugate(np.transpose(S,(0,1,3,2))),h0),S)
    return vx,vy

def get_M(func,kx1,ky1,*args): 
    h0=func(kx1,ky1,*args)[1]
    EK,S=eigen(h0)

    Ec1=np.where(EK[:,:,1]==np.min(EK[:,:,1]))[0][0]
    Ec2=np.where(EK[:,:,1]==np.min(EK[:,:,1]))[1][0]
    Ev1=np.where(EK[:,:,0]==np.max(EK[:,:,0]))[0][0]
    Ev2=np.where(EK[:,:,0]==np.max(EK[:,:,0]))[1][0]
    mu=(EK[Ec1,Ec2,1]+EK[Ev1,Ev2,0])/2
    #print("mu:",mu)

    EK1m=np.stack([EK]*EK.shape[-1],axis=3)
    EK2m=np.stack([EK]*EK.shape[-1],axis=2)
    IDE_inf=np.tile(np.identity(EK.shape[-1]),(len(kx1),len(ky1),1,1))*1E16
    EKnm=np.divide(-(EK1m-EK2m),(np.multiply(EK1m-EK2m,EK1m-EK2m)+IDE_inf)) ###mk
    #EKnm=np.divide(-(EK1m+EK2m)+2*mu,(np.multiply(EK1m-EK2m,EK1m-EK2m)+IDE_inf)) # multiply the last term by 0 in line 124
    vx=get_v(func,kx1,ky1,*args)[0];vy=get_v(func,kx1,ky1,*args)[1]
    cx=np.multiply(EKnm,vx);cy=np.multiply(EKnm,vy)    
    b1=hex(l0)[0];b2=hex(l0)[1]
    dkx=kx1[1]-kx1[0];dky=ky1[1]-ky1[0]
    dv=abs(np.cross(dkx*b1,dky*b2))
    m=1j*1/(2*(2*np.pi)**2)*(cx@vy-cy@vx)
    #M=np.sum(np.sum(np.trace(m[:,:,start:occu,start:occu],axis1=-2,axis2=-1),-1),-1)*dv+(mu)/(2*np.pi)*0
    M=np.sum(np.sum(m[:,:,0,0],-1),-1)*dv+(mu)/(2*np.pi)*0

    EKnm2=np.divide(2,(np.multiply(EK1m-EK2m,EK1m-EK2m)+IDE_inf))*2*np.pi #use this formula, the  Chern number
    cx=np.multiply(EKnm2,vx);cy=np.multiply(EKnm2,vy) 
    m=1j*1/(2*(2*np.pi)**2)*(cx@vy-cy@vx)
    C_num=np.sum(np.sum(m,0),0)*dv

    return m,M,C_num



def get_GME(func,kx1,ky1,mu,sigma0,*args):
    mk=get_M(func,kx1,ky1,*args)[0]*1e-3
    vx=get_v(func,kx1,ky1,*args)[0]*1e-3/hbar;vy=get_v(func,kx1,ky1,*args)[1]/hbar*1e-3
    H0=func(kx1,ky1,*args)[0]
    band_n=0
    En=eigen(H0)[0]
    unit_f=0.13123
    b1=hex(l0)[0];b2=hex(l0)[1]
    dkx=kx1[1]-kx1[0];dky=ky1[1]-ky1[0]
    dv=abs(np.cross(dkx*b1,dky*b2)) 

    GME_xz=dv*np.sum(vx[:,:,band_n,band_n]*mk[:,:,band_n,band_n]*Gaussion_1D(En[:,:,band_n],mu,sigma0)*1e3)*unit_f
    GME_yz=dv*np.sum(vy[:,:,band_n,band_n]*mk[:,:,band_n,band_n]*Gaussion_1D(En[:,:,band_n],mu,sigma0)*1e3)*unit_f
    return GME_xz,GME_yz

# def get_GME(func,kx1,ky1,*args):
#     mk=get_M(func,kx1,ky1,*args)[0]*1e-3
#     vx=get_v(func,kx1,ky1,*args)[0]*1e-3/hbar;vy=get_v(func,kx1,ky1,*args)[1]/hbar
#     print("test: ",np.max(vx)*hbar)
#     fermi=-1;dfermi=0.1
#     H0=func(kx1,ky1,*args)[0]
#     band_n=0
#     En=eigen(H0)[0]
#     fermi_k=np.where((En[:,:,0] >= fermi) & (En[:,:,0] <= fermi+dfermi))
#     N_k=Nsqu
#     print(N_k)
#     b1=hex(l0)[0];b2=hex(l0)[1]
#     print("test",b1)
#     dkx=kx1[1]-kx1[0];dky=ky1[1]-ky1[0]
#     dv=abs(np.cross(dkx*b1,dky*b2))
#     unit_f=0.13123 ##unit:uB.A-1.S-1  
#     GME_xz=np.sqrt(dv)*np.sum(vx[:,:,band_n,band_n][fermi_k]*mk[:,:,band_n,band_n][fermi_k]/np.abs(hbar*vx[:,:,band_n,band_n][fermi_k]))*unit_f
#     GME_yz=np.sqrt(dv)*np.sum(vy[:,:,band_n,band_n][fermi_k]*mk[:,:,band_n,band_n][fermi_k]/np.abs(hbar*vy[:,:,band_n,band_n][fermi_k]))*unit_f
#     return GME_xz,GME_yz

"""
The dimension of tha Hamiltonian should be N x N
"""
def Chern_Fr_Discrete(Ham): 
    n1,n2=Ham.shape[0:2]
    Ham1=np.zeros((n1+1,n2+1,2,2),dtype=complex)
    Ham1[0:n1,0:n2]=np.copy(Ham)
    Ham1[-1,0:-1]=np.copy(Ham[0])
    Ham1[0:-1,-1,:,:]=np.copy(Ham[:,0,:,:])
    Ham1[-1,-1]=np.copy(Ham[0,0,:,:])

    EK,EV=eigen(Ham1)
    ind1=0
    C_num=0
    for i in range(n1):
        for j in range(n2):
            temp=np.dot(np.conj(EV[i,j,:,ind1]),EV[i+1,j,:,ind1])*np.dot(np.conj(EV[i+1,j,:,ind1]),EV[i+1,j+1,:,ind1])*np.dot(np.conj(EV[i+1,j+1,:,ind1]),EV[i,j+1,:,ind1])*np.dot(np.conj(EV[i,j+1,:,ind1]),EV[i,j,:,ind1])
            C_num=C_num+np.imag(np.log(temp))/(2*np.pi)
    return C_num