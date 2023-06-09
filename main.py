from bdb import effective
import numpy as np
from regex import E
from Haldane_BC import *
from Tools import *
from get_function import *
from scipy.optimize import root
from functools import reduce

import matplotlib.colors as colors

"""
x0[0]:⟨g^†_(k_1)g_(k_1)⟩;x0[1]:⟨e^†_(k_1)g_(k_1)⟩;1-x0[0]:⟨e^†_(k_1)e_(k_1)⟩
"""

def Gap_Eq(x):
    x0=np.split(x,[3*N_1*N_1,3*N_1*N_1+3*N_2*N_2])
    x1=np.reshape(x0[0],(3,N_1,N_1))
    x2=np.reshape(x0[1],(3,N_2,N_2))
    order1=[x1[0],x1[1],x1[2]]
    order2=[x2[0],x2[1],x2[2]]
    
    Ht=Many_Ham2sys(m_1,n_1,m_2,n_2,w,chi_1,chi_2,S1,S2,order1,order2)
    H_int1= Ht[0]
    H_int2= Ht[2]
    Evint1=eigen(H_int1)[1]
    Evint2=eigen(H_int2)[1]

    results1=np.zeros((3,N_1,N_1),complex)
    results1[0]=x1[0]-Evint1[:,:,0,0].conjugate()*Evint1[:,:,0,0]
    results1[1]=x1[1]-Evint1[:,:,0,0].conjugate()*Evint1[:,:,1,0]
    results1[2]=x1[2]-Evint1[:,:,1,0].conjugate()*Evint1[:,:,1,0]
    result1_group=np.reshape(results1,3*N_1*N_1)

    results2=np.zeros((3,N_2,N_2),complex)
    results2[0]=x2[0]-Evint2[:,:,0,0].conjugate()*Evint2[:,:,0,0]
    results2[1]=x2[1]-Evint2[:,:,0,0].conjugate()*Evint2[:,:,1,0]
    results2[2]=x2[2]-Evint2[:,:,1,0].conjugate()*Evint2[:,:,1,0]
    result2_group=np.reshape(results2,3*N_2*N_2)
    result=np.concatenate([result1_group,result2_group])
    return result

def Gap_out():
    input_nm=1e4
    guess10=np.ones((3,N_1,N_1),dtype=complex)*input_nm
    guess20=np.ones((3,N_2,N_2),dtype=complex)*input_nm

    # guess10=np.random.randn(3,N_1,N_1)
    # guess20=np.random.randn(3,N_2,N_2)

    guess1=np.reshape(guess10,3*N_1*N_1)
    guess2=np.reshape(guess20,3*N_2*N_2)
    guess=np.concatenate((guess1,guess2))

    sol = root(Gap_Eq, guess, method='df-sane',tol=1e-9,options={'disp': True}) ##krylov;df-sane
    output=np.split(sol.x,[3*N_1*N_1,3*N_1*N_1+3*N_2*N_2])
    aa1=np.reshape(output[0],(3,N_1,N_1))[0]
    ab1=np.reshape(output[0],(3,N_1,N_1))[1]
    bb1=np.reshape(output[0],(3,N_1,N_1))[2]
    aa2=np.reshape(output[1],(3,N_2,N_2))[0]
    ab2=np.reshape(output[1],(3,N_2,N_2))[1]
    bb2=np.reshape(output[1],(3,N_2,N_2))[2]
    return aa1,ab1,bb1,aa2,ab2,bb2

def plot_MF(*args):
    En0,En1,Hint00,Hint11,aa0,bb0=args
    G1=hex(l0_1)[0];G2=hex(l0_1)[1]
    scat_amp=8# 60 for band, 200 for enlarger
    
    line_L=np.array([]);line_En0=[];line_En1=[]
    line_aa0=[];line_bb0=[];line_aa1=[];line_bb1=[];line_Hint00=[];line_Hint11=[]
     
    #kpx=[0,1,2,3,4,7,10,13,14]
    #kpy=[0,3,6,9,10,9,8,7,7]

    kpx=[0,1,2,3,4,5,6,7,7]
    kpy=[0,3,6,9,10,11,12,13,14] ## data 17,21
    # kpx=[0,1,2,3,3,4,5]
    # kpy=[0,2,4,6,8,9,10] ##data 20
    k_length=0
    for kn in range (len(kpx)):
        i=kpx[kn];j=kpy[kn]
        line_En0.append(En0[i,j]);line_En1.append(En1[i,j])
        line_aa0.append(aa0[i,j].real*scat_amp)
        line_aa1.append((1-aa0[i,j].real)*scat_amp)
        line_bb0.append(bb0[i,j].real*scat_amp)
        line_bb1.append((1-bb0[i,j].real)*scat_amp)
        line_Hint00.append(Hint00[i,j])
        line_Hint11.append(Hint11[i,j])
        if kn==0:
            k_length=0
        else:
            k_length=k_length+np.linalg.norm((m_1[i]-m_1[i-1])*G1+(n_1[j]-n_1[j-1])*G2)
        line_L=np.append(line_L,k_length)
    
    YY1=-1.2;YY2=1;alpha0=0.5;alpha1=0.8;alpha2=1;textsize=15
    plt.plot(line_L,line_En0) ## ,label=r'$E_{\alpha}$' "2" for band, 10 for enlarge
    plt.plot(line_L,line_En1)
    plt.plot(line_L,line_Hint00,'r--',label=r'$\Delta H_{\rm 11}$')
    plt.plot(line_L,line_Hint11,'b--',label=r'$\Delta H_{\rm 22}$')

    plt.scatter(line_L,line_En0) ## ,label=r'$E_{\alpha}$' "2" for band, 10 for enlarge
    plt.scatter(line_L,line_En1)
    plt.scatter(line_L,line_Hint00,color='r')
    plt.scatter(line_L,line_Hint11,color='b')

    # plt.scatter(line_L,line_En0,line_aa0,c='b',alpha=alpha1,marker="o",label=r'$|u_k|^2$')
    # plt.scatter(line_L,line_En0,line_bb0,c='r',alpha=alpha2,marker="o")
    # plt.scatter(line_L,line_En1,line_bb1,c='r',alpha=alpha1,marker="o",label=r'$|v_k|^2$')
    # plt.scatter(line_L,line_En1,line_aa1,c='b',alpha=alpha2,marker="o")
    plt.show()

    En_data=np.zeros((len(kpx),7))
    En_data[:,0]=line_L;En_data[:,1]=line_En0;En_data[:,2]=line_En1
    En_data[:,3]=line_aa0;En_data[:,4]=line_bb0;En_data[:,5]=line_aa1;En_data[:,6]=line_bb1
    line_En0_df=pd.DataFrame(En_data)
    #line_En0_df.to_csv("./Data/Data22/line_En0_moire1.dat",header=None,index=None)
    #line_En0_df.to_csv("./Data/Data17/line_En0_moire1_trivial.dat",header=None,index=None)
    #line_En0_df.to_csv("./Data/Data22/line_En0_moire1_chi0.dat",header=None,index=None)



"""
Single-particle Hamiltonian
"""
b1=hex(l0_1)[0];b2=hex(l0_1)[1]
dkx1=m_1[1]-m_1[0];dky1=n_1[1]-n_1[0]
dv=abs(np.cross(dkx1*b1,dky1*b2))
ratia1=S1/((2*np.pi)**2)*dv
b1=hex(l0_2)[0];b2=hex(l0_2)[1]
dkx1=m_2[1]-m_2[0];dky1=n_2[1]-n_2[0]
dv=abs(np.cross(dkx1*b1,dky1*b2))
ratia2=S2/((2*np.pi)**2)*dv

H01=Haldane_BC(m_1,n_1,1)
Hv1=Haldane_BC_Hv(m_1,n_1,1)
EK1,EV1=eigen(H01)
Hv1_eig=inv(EV1)@Hv1@EV1
M1aa=Hv1_eig[:,:,0,0];M1ab=Hv1_eig[:,:,0,1]
M1ba=Hv1_eig[:,:,1,0];M1bb=Hv1_eig[:,:,1,1]
w1ab=EK1[:,:,0]-EK1[:,:,1];w1ba=EK1[:,:,1]-EK1[:,:,0]
HD1=Haldane_BC_D(m_1,n_1,1)
HD1_eig=inv(EV1)@HD1@EV1
D1aa=HD1_eig[:,:,0,0];D1ab=HD1_eig[:,:,0,1]
D1ba=HD1_eig[:,:,1,0];D1bb=HD1_eig[:,:,1,1]

H02=Haldane_BC(m_2,n_2,2)
Hv2=Haldane_BC_Hv(m_2,n_2,2)
EK2,EV2=eigen(H02)
Hv2_eig=inv(EV2)@Hv2@EV2
M2aa=Hv2_eig[:,:,0,0];M2ab=Hv2_eig[:,:,0,1]
M2ba=Hv2_eig[:,:,1,0];M2bb=Hv2_eig[:,:,1,1]
w2ab=EK2[:,:,0]-EK2[:,:,1];w2ba=EK2[:,:,1]-EK2[:,:,0]
HD2=Haldane_BC_D(m_2,n_2,2)
HD2_eig=inv(EV2)@HD2@EV2
D2aa=HD2_eig[:,:,0,0];D2ab=HD2_eig[:,:,0,1]
D2ba=HD2_eig[:,:,1,0];D2bb=HD2_eig[:,:,1,1]



z=abs(EK1[:,:,1]-EK1[:,:,0])
plot_way2(m_1,n_1,z.real,l0_1)
plt.show()


"""
mean-field Hamiltonian
"""

import pandas as pd
ordpa=Gap_out()
aa1,ab1,bb1=ordpa[0:3]
aa2,ab2,bb2=ordpa[3:6]
H_int1,H_int1_base,H_int2,H_int2_base,CE=Many_Ham2sys(m_1,n_1,m_2,n_2,w,chi_1,chi_2,S1,S2,ordpa[0:3],ordpa[3:6])
En=eigen(H_int1)[0]


z=abs(ab1)
z=En[:,:,1]-En[:,:,0]

plot_way2(m_1,n_1,z.real,l0_1)
plt.show()


###plot section 2 for moire 1


En_diff=En[:,:,1]-En[:,:,0]
Sigma=np.where(En_diff==np.min(En_diff))### get the special kpoint
print("Sigma: ",m_1[Sigma[0][0]],n_1[Sigma[1][0]])
K1=[0,0];K2=[m_1[Sigma[0][0]],n_1[Sigma[1][0]]];K3=[1/3,2/3];K4=[0,1]##list all k point
kpoints=[K1,K2,K3,K4]

plot_MF(En[:,:,0],En[:,:,1],H_int1[:,:,0,0],H_int1[:,:,1,1],aa1,bb1)

plt.show()


