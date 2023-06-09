from bdb import effective
from numbers import Complex
import numpy as np
from numpy import append, linalg as npla
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sympy import E
from Tools import *
from numpy.linalg import inv
from setting import *


#a=10/np.sqrt(3);h_bar=1

"""
Define functions for Hamiltonian along klines
"""
EA=EA_1;EB=EB_1;EC=EC_1;t1=t1_1;t2=t2_1;t3=t3_1;tB=tB_1;tC=tC_1;phi_B=phi_B_1;phi_C=phi_C_1 #Hongyi's
l0=l0_1
chi=chi_1
def plot_Haldane_BC(kx1,ky1): 
    lx=len(kx1)
    D=2
    H0=np.zeros((lx,D,D),dtype=complex)
    d1=l0*np.array([1,0]);d3=l0*np.array([-1/2,np.sqrt(3)/2]);d5=l0*np.array([-1/2,-np.sqrt(3)/2])
    nu1=d3-d5;nu2=d5-d1;nu3=d1-d3
    Next_BB=tB*np.exp(1j*phi_B)*(phase_kr(kx1,ky1,nu1)+phase_kr(kx1,ky1,nu2)+phase_kr(kx1,ky1,nu3))
    Next_CC=tC*np.exp(1j*phi_C)*(phase_kr(kx1,ky1,nu1)+phase_kr(kx1,ky1,nu2)+phase_kr(kx1,ky1,nu3))
    H0[:,0,0]=np.ones((lx))*EB+Next_BB+Next_BB.conjugate()
    H0[:,1,1]=np.ones((lx))*EC+Next_CC+Next_CC.conjugate()
    H0[:,1,0]=t3*(phase_kr(kx1,ky1,d1)+phase_kr(kx1,ky1,d3)+phase_kr(kx1,ky1,d5))
    H0[:,0,1]=H0[:,1,0].conjugate()     
    return H0

def plot_Haldane_BC_Hv(kx1,ky1): 
    lx=len(kx1)
    D=2
    d1=l0*np.array([1,0]);d3=l0*np.array([-1/2,np.sqrt(3)/2]);d5=l0*np.array([-1/2,-np.sqrt(3)/2])
    nu1=d3-d5;nu2=d5-d1;nu3=d1-d3
 
    Hv=np.zeros((lx,D,D),dtype=complex)
    n1=np.array([np.cos(theta),np.sin(theta)])
    Next_BB=1j*tB*np.exp(1j*phi_B)*(phase_kr(kx1,ky1,nu1)*np.dot(n1,nu1)/l0+phase_kr(kx1,ky1,nu2)*np.dot(n1,nu2)/l0+phase_kr(kx1,ky1,nu3)*np.dot(n1,nu3)/l0)
    Next_CC=1j*tC*np.exp(1j*phi_C)*(phase_kr(kx1,ky1,nu1)*np.dot(n1,nu1)/l0+phase_kr(kx1,ky1,nu2)*np.dot(n1,nu2)/l0+phase_kr(kx1,ky1,nu3)*np.dot(n1,nu3)/l0)
    #Next_BB=tB*np.exp(1j*phi_B)*(phase_kr(kx1,ky1,nu1)+phase_kr(kx1,ky1,nu2)+phase_kr(kx1,ky1,nu3))
    #Next_CC=tC*np.exp(1j*phi_C)*(phase_kr(kx1,ky1,nu1)+phase_kr(kx1,ky1,nu2)+phase_kr(kx1,ky1,nu3))
    Hv[:,0,0]=Next_BB+Next_BB.conjugate()
    Hv[:,1,1]=Next_CC+Next_CC.conjugate()
    Hv[:,1,0]=1j*t3*(phase_kr(kx1,ky1,d1)*np.dot(n1,d1)/l0+phase_kr(kx1,ky1,d3)*np.dot(n1,d3)/l0+phase_kr(kx1,ky1,d5)*np.dot(n1,d5)/l0)
    Hv[:,0,1]=Hv[:,1,0].conjugate()
    #Hv[:,1,0]=t3*(phase_kr(kx1,ky1,d1)+phase_kr(kx1,ky1,d3)+phase_kr(kx1,ky1,d5));H0[:,0,1]=H0[:,1,0].conjugate()

    return Hv

def plot_Effective_model(kx2,ky2):    
    H0=plot_Haldane_BC(kx2,ky2)
    Hv=plot_Haldane_BC_Hv(kx2,ky2)
    EK,EV=npla.eigh(H0)
    Hv_eig=inv(EV)@Hv@EV
    Heff=np.zeros((EK.shape[0],2,2),dtype=complex)
    M1=Hv_eig[:,0,0];M2=Hv_eig[:,1,1];N1=Hv_eig[:,0,1]
    Heff[:,0,0]=EK[:,0]-chi**2*M1*M1/hbarw+chi**2/(EK[:,0]-EK[:,1]-hbarw)*abs(N1)*abs(N1)
    Heff[:,1,1]=EK[:,1]-chi**2*M2*M2/hbarw+chi**2/(EK[:,1]-EK[:,0]-hbarw)*abs(N1)*abs(N1)
    Heff[:,1,0]=-chi**2*N1.conjugate()*(M1+M2)/(2*hbarw)+chi**2*M1*N1.conjugate()/(2*(EK[:,1]-EK[:,0]-hbarw))-chi**2*M2*N1.conjugate()/(2*(EK[:,1]-EK[:,0]+hbarw))
    Heff[:,0,1]=Heff[:,1,0].conjugate()
    return Heff


def plot_EK(*kpoints,func):
    N_k=60
    kx=np.array([]);ky=np.array([]);KL=np.array([])
    k_length=0
    for i in range(len(kpoints)-1):
        print(i)
        KL1=k_length
        kx=np.append(kx,np.linspace(kpoints[i][0],kpoints[i+1][0],N_k))
        ky=np.append(ky,np.linspace(kpoints[i][1],kpoints[i+1][1],N_k))
        k_length=k_length+np.linalg.norm(kpoints[i]-kpoints[i+1])
        KL=np.append(KL,np.linspace(KL1,k_length,N_k))
    H0=func(kx,ky) 
    EK,S = npla.eigh(H0) # for eigh, an arbitrary phase will be added to the eigenvectors, which makes me unhappy
    # for i in range(EK.shape[0]):
    #     idx = np.argsort(EK[i])
    #     EK[i,:] = EK[i,idx]
    #     S[i,:,:] = S[i,:,idx]
    #H1=plot_Tri_orbit(kx,ky)
    #EK1= npla.eigh(H1)[0]
    #Del1= -EK1[0,0]+EK[0,0]
    D=EK.shape[-1]
    #plt.figure(figsize=(4,4))
    #plt.ylim(np.min(EK[:,0]), np.min(EK[:,0])+2)
    
    YY1=-2.5;YY2=3
    plt.ylim(YY1, YY2)
    plt.xlim(0, KL[-1])
    
    for i in range(D):
        plt.plot(KL, EK[:,i])
        #plt.plot(KL, EK1[:,i]+Del1,'--')
    print(np.max(EK[:,1]-EK[:,0]))
    #plt.show()
    plt.plot([KL[N_k],KL[N_k]],[YY1,YY2],'r--',lw=1)
    plt.plot([KL[2*N_k],KL[2*N_k]],[YY1,YY2],'r--',lw=1)
    plt.plot([KL[3*N_k],KL[3*N_k]],[YY1,YY2],'r--',lw=1)

    textsize=15
    #Kline 3
    plt.figtext(0.12, 0.06, "$\Gamma$")
    plt.figtext(0.38, 0.06, "K")
    plt.figtext(0.50, 0.06, "M")
    plt.figtext(0.64, 0.06, "K'")
    plt.figtext(0.90, 0.06, "$\Gamma$")

    plt.xticks([], [])
    plt.ylabel('$E$ (meV)', fontsize=textsize)
    plt.legend(loc='upper right', prop={'size': 10})

    return EK

#EA=5;EB=0;EC=0;t1=1;t2=1;t3=0.5;tB=t1**2/(EA-EB);tC=t2**2/(EA-EC);phi_B=2*np.pi/3;phi_C=-2*np.pi/3 #Hongyi's
#EB=0;EC=-0;t3=0.29;tB=0.06;tC=0.06;phi_B=2*np.pi/3;phi_C=-2*np.pi/3  # parameters from Fengcheng Wu
#EB=3.67/3;EC=-3.67/3;t3=1;tB=1/3;tC=1/3;phi_B=1*np.pi/4;phi_C=-1*np.pi/4 #parameters from PHYSICAL REVIEW B 74, 235111 2006
#EB=0;EC=-0;t3=0.5;tB=0.06;tC=0.06;phi_B=2*np.pi/3;phi_C=-2*np.pi/3 # ajusted parameters from Fengcheng Wu
hbarw=w
#chi=chi0
"""
for Hamiltonian along k lines
Plot band structure and Berry curvature
"""

k1=np.array([-1/3,-2/3])
k2=np.array([0,0])
k3=np.array([1/3,2/3])
k4=np.array([2/3,1/3])
km=np.array([1/2,1/2])

#k_all_points=[k_cartesion(GM,a),k_cartesion(MM1,a),k_cartesion(K,a),k_cartesion(MM2,a),k_cartesion(K2,a),k_cartesion(MM3,a),k_cartesion(GM,a)]
k_all_points=[k_cartesion(k2,l0),k_cartesion(k3,l0),k_cartesion(km,l0),k_cartesion(k4,l0),k_cartesion(k2,l0)] #Kline 3
plot_EK(*k_all_points, func=plot_Haldane_BC)
#plot_EK(*k_all_points, func=plot_Effective_model)

plt.show()





