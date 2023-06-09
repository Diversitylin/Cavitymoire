from numbers import Complex
import numpy as np
from numpy import linalg as npla
from numpy.linalg import inv
from Tools import *
from get_function import *
from setting import *

"""
Define functions for 2D Hamiltonian 
"""
def Haldane_BC(kx2,ky2,flag):
    if flag==1: 
        EA=EA_1;EB=EB_1;EC=EC_1;t1=t1_1;t2=t2_1;t3=t3_1;tB=tB_1;tC=tC_1;phi_B=phi_B_1;phi_C=phi_C_1 #Hongyi's
        l0=l0_1
    else:
        EA=EA_2;EB=EB_2;EC=EC_2;t1=t1_2;t2=t2_2;t3=t3_2;tB=tB_2;tC=tC_2;phi_B=phi_B_2;phi_C=phi_C_2 #Hongyi's
        l0=l0_2
    lx=len(kx2);ly=len(ky2)
    kx1,ky1=BZ_mesh(kx2,ky2,l0)
    D=2
    H0=np.zeros((lx,ly,D,D),dtype=complex)
    d1=l0*np.array([1,0]);d3=l0*np.array([-1/2,np.sqrt(3)/2]);d5=l0*np.array([-1/2,-np.sqrt(3)/2])
    nu1=d3-d5;nu2=d5-d1;nu3=d1-d3
 
    Next_BB=tB*np.exp(1j*phi_B)*(phase_kr(kx1,ky1,nu1)+phase_kr(kx1,ky1,nu2)+phase_kr(kx1,ky1,nu3))
    Next_CC=tC*np.exp(1j*phi_C)*(phase_kr(kx1,ky1,nu1)+phase_kr(kx1,ky1,nu2)+phase_kr(kx1,ky1,nu3))
    H0[:,:,0,0]=np.ones((lx))*EB+Next_BB+Next_BB.conjugate()
    H0[:,:,1,1]=np.ones((lx))*EC+Next_CC+Next_CC.conjugate()
    H0[:,:,1,0]=t3*(phase_kr(kx1,ky1,d1)+phase_kr(kx1,ky1,d3)+phase_kr(kx1,ky1,d5))
    H0[:,:,0,1]=H0[:,:,1,0].conjugate() 
    return H0

def Haldane_BC_Hv(kx2,ky2,flag): 
    if flag==1: 
        EA=EA_1;EB=EB_1;EC=EC_1;t1=t1_1;t2=t2_1;t3=t3_1;tB=tB_1;tC=tC_1;phi_B=phi_B_1;phi_C=phi_C_1 #Hongyi's
        l0=l0_1
    else:
        EA=EA_2;EB=EB_2;EC=EC_2;t1=t1_2;t2=t2_2;t3=t3_2;tB=tB_2;tC=tC_2;phi_B=phi_B_2;phi_C=phi_C_2 #Hongyi's
        l0=l0_2
    lx=len(kx2);ly=len(ky2)
    kx1,ky1=BZ_mesh(kx2,ky2,l0)
    D=2
    d1=l0*np.array([1,0]);d3=l0*np.array([-1/2,np.sqrt(3)/2]);d5=l0*np.array([-1/2,-np.sqrt(3)/2])
    nu1=d3-d5;nu2=d5-d1;nu3=d1-d3
    Hv=np.zeros((lx,ly,D,D),dtype=complex)

    n1=np.array([np.cos(theta),np.sin(theta)])
    phase=1
    Next_BB=phase*1j*tB*np.exp(1j*phi_B)*(phase_kr(kx1,ky1,nu1)*np.dot(n1,nu1)/l0+phase_kr(kx1,ky1,nu2)*np.dot(n1,nu2)/l0+phase_kr(kx1,ky1,nu3)*np.dot(n1,nu3)/l0)
    Next_CC=phase*1j*tC*np.exp(1j*phi_C)*(phase_kr(kx1,ky1,nu1)*np.dot(n1,nu1)/l0+phase_kr(kx1,ky1,nu2)*np.dot(n1,nu2)/l0+phase_kr(kx1,ky1,nu3)*np.dot(n1,nu3)/l0)

    Hv[:,:,0,0]=Next_BB+Next_BB.conjugate()
    Hv[:,:,1,1]=Next_CC+Next_CC.conjugate()
    Hv[:,:,1,0]=phase*1j*t3*(phase_kr(kx1,ky1,d1)*np.dot(n1,d1)/l0+phase_kr(kx1,ky1,d3)*np.dot(n1,d3)/l0+phase_kr(kx1,ky1,d5)*np.dot(n1,d5)/l0)
    Hv[:,:,0,1]=Hv[:,:,1,0].conjugate()
    return Hv

def Haldane_BC_D(kx2,ky2,flag): 
    if flag==1: 
        EA=EA_1;EB=EB_1;EC=EC_1;t1=t1_1;t2=t2_1;t3=t3_1;tB=tB_1;tC=tC_1;phi_B=phi_B_1;phi_C=phi_C_1 #Hongyi's
        l0=l0_1
    else:
        EA=EA_2;EB=EB_2;EC=EC_2;t1=t1_2;t2=t2_2;t3=t3_2;tB=tB_2;tC=tC_2;phi_B=phi_B_2;phi_C=phi_C_2 #Hongyi's
        l0=l0_2
    lx=len(kx2);ly=len(ky2)
    kx1,ky1=BZ_mesh(kx2,ky2,l0)
    D=2
    d1=l0*np.array([1,0]);d3=l0*np.array([-1/2,np.sqrt(3)/2]);d5=l0*np.array([-1/2,-np.sqrt(3)/2])
    nu1=d3-d5;nu2=d5-d1;nu3=d1-d3
    Hv=np.zeros((lx,ly,D,D),dtype=complex)

    n1=np.array([np.cos(theta),np.sin(theta)])
    phase=1j
    Next_BB=phase*1j*tB*np.exp(1j*phi_B)*(phase_kr(kx1,ky1,nu1)*(np.dot(n1,nu1)/l0)**2+phase_kr(kx1,ky1,nu2)*(np.dot(n1,nu2)/l0)**2+phase_kr(kx1,ky1,nu3)*(np.dot(n1,nu3)/l0)**2)
    Next_CC=phase*1j*tC*np.exp(1j*phi_C)*(phase_kr(kx1,ky1,nu1)*(np.dot(n1,nu1)/l0)**2+phase_kr(kx1,ky1,nu2)*(np.dot(n1,nu2)/l0)**2+phase_kr(kx1,ky1,nu3)*(np.dot(n1,nu3)/l0)**2)

    Hv[:,:,0,0]=Next_BB+Next_BB.conjugate()
    Hv[:,:,1,1]=Next_CC+Next_CC.conjugate()
    Hv[:,:,1,0]=phase*1j*t3*(phase_kr(kx1,ky1,d1)*(np.dot(n1,d1)/l0)**2+phase_kr(kx1,ky1,d3)*(np.dot(n1,d3)/l0)**2+phase_kr(kx1,ky1,d5)*(np.dot(n1,d5)/l0)**2)
    Hv[:,:,0,1]=Hv[:,:,1,0].conjugate()
    return Hv


def Effective_model(kx2,ky2,flag,*args):
    hbarw,chi=args
    H0=Haldane_BC(kx2,ky2,flag)
    Hv=Haldane_BC_Hv(kx2,ky2,flag)
    EK,EV=eigen(H0)
    Hv_eig=inv(EV)@Hv@EV
    apa=2.8**2*0
    Heff=np.zeros((EK.shape[0],EK.shape[1],2,2),dtype=complex)
    M1=Hv_eig[:,:,0,0];M2=Hv_eig[:,:,1,1];N1=Hv_eig[:,:,0,1]
    Heff[:,:,0,0]=EK[:,:,0]-(chi**2*M1*M1/hbarw+chi**2/(EK[:,:,0]-EK[:,:,1]-hbarw)*N1*N1.conjugate())\
    +apa*chi**2*N1*N1.conjugate()*(-1/(EK[:,:,1]-EK[:,:,0]-hbarw)-1/(EK[:,:,1]-EK[:,:,0]+hbarw))
    Heff[:,:,1,1]=EK[:,:,1]-(chi**2*M2*M2/hbarw+chi**2/(EK[:,:,1]-EK[:,:,0]-hbarw)*N1*N1.conjugate())\
    +apa*chi**2*N1*N1.conjugate()*(1/(EK[:,:,1]-EK[:,:,0]-hbarw)+1/(EK[:,:,1]-EK[:,:,0]+hbarw))
    Heff[:,:,1,0]=(-chi**2*N1.conjugate()*(M1+M2)/(2*hbarw)+chi**2*M1*N1.conjugate()/(2*(EK[:,:,1]-EK[:,:,0]-hbarw))+chi**2*M2*N1.conjugate()/(2*(EK[:,:,0]-EK[:,:,1]-hbarw)))
    +apa*chi**2*N1.conjugate()*(M1*(-1/(EK[:,:,1]-EK[:,:,0]-hbarw)-1/(EK[:,:,1]-EK[:,:,0]+hbarw))\
                                +M2*(1/(EK[:,:,1]-EK[:,:,0]-hbarw)+1/(EK[:,:,1]-EK[:,:,0]+hbarw)))
    Heff[:,:,0,1]=Heff[:,:,1,0].conjugate()
    Heff_base=EV@Heff@inv(EV)
    return Heff,Heff_base #if calculate Berry curvature, use Heff_base

def Ham_int(kx2,ky2,flag,*args):
    hbarw,chi,S,ordpa=args
    aa1,ab1,bb1=ordpa;ba1=ab1.conjugate()
    H0=Haldane_BC(kx2,ky2,flag)
    Hv=Haldane_BC_Hv(kx2,ky2,flag)
    EK,EV0=eigen(H0)
    Hv_eig=inv(EV0)@Hv@EV0
    Maa=Hv_eig[:,:,0,0];Mbb=Hv_eig[:,:,1,1];Mab=Hv_eig[:,:,0,1];Mba=Hv_eig[:,:,1,0]
    wab=EK[:,:,1]-EK[:,:,0];wba=EK[:,:,0]-EK[:,:,1]
    HD=Haldane_BC_D(kx2,ky2,flag)
    HD_eig=inv(EV0)@HD@EV0
    factD=0
    Daa=HD_eig[:,:,0,0]*factD;Dbb=HD_eig[:,:,1,1]*factD;Dab=HD_eig[:,:,0,1]*factD;Dba=HD_eig[:,:,1,0]*factD

    if flag==1:
        l0=l0_1
    if flag==2:
        l0=l0_2
    #print("l0: ",l0)
    b1=hex(l0)[0];b2=hex(l0)[1]
    dkx=kx2[1]-kx2[0];dky=ky2[1]-ky2[0]
    dv=abs(np.cross(dkx*b1,dky*b2))
    ratia=S/((2*np.pi)**2)*dv

    Heff=Effective_model(kx2,ky2,flag,hbarw,chi)[0]

    Inter_H=np.zeros((H0.shape[0],H0.shape[1],2,2),dtype=complex)
 
    Inter_H=np.zeros((H0.shape[0],H0.shape[1],2,2),dtype=complex)
 
    Inter_H[:,:,0,0]=Heff[:,:,0,0]+ratia*chi**2/2*np.sum(2*Maa*aa1/(-w)+2*Mbb*bb1/(-w)+(Mab*ab1+Mba*ba1)/(wab-w)+(Mba*ba1+Mab*ab1)/(wba-w))*Maa \
        +ratia*chi**2/2*2*np.sum(Maa*aa1+Mbb*bb1+Mab*ab1+Mba*ba1)*Maa/(-w)\
            +Daa*chi**2/2
        
    Inter_H[:,:,0,1]=Heff[:,:,0,1]+ratia*chi**2/2*np.sum(2*Maa*aa1/(-w)+2*Mbb*bb1/(-w)+(Mab*ab1+Mba*ba1)/(wab-w)+(Mba*ba1+Mab*ab1)/(wba-w))*Mab \
        +ratia*chi**2/2*np.sum(Maa*aa1+Mbb*bb1+Mab*ab1+Mba*ba1)*(Mab/(wab-w)+Mab/(wba-w))\
            +Dab*chi**2/2

    Inter_H[:,:,1,0]=Inter_H[:,:,0,1].conjugate()

    Inter_H[:,:,1,1]=Heff[:,:,1,1]+ratia*chi**2/2*np.sum(2*Maa*aa1/(-w)+2*Mbb*bb1/(-w)+(Mab*ab1+Mba*ba1)/(wab-w)+(Mba*ba1+Mab*ab1)/(wba-w))*Mbb \
        +ratia*chi**2/2*2*np.sum(Maa*aa1+Mbb*bb1+Mab*ab1+Mba*ba1)*Mbb/(-w)\
           +Dbb*chi**2/2 
       
    Inter_H_base=EV0@Inter_H@inv(EV0)
    ## calculate constant energy (CE)
    #f1=h,i1=g
    CE_aa=-Maa*aa1*ratia*chi**2/2*np.sum(2*Maa*aa1/(-w)+2*Mbb*bb1/(-w)+(Mab*ab1+Mba*ba1)/(wab-w)+(Mba*ba1+Mab*ab1)/(wba-w))
    CE_ab=-Mab*ab1*ratia*chi**2/2*np.sum(2*Maa*aa1/(-w)+2*Mbb*bb1/(-w)+(Mab*ab1+Mba*ba1)/(wab-w)+(Mba*ba1+Mab*ab1)/(wba-w))
    CE_ba=-Mba*ba1*ratia*chi**2/2*np.sum(2*Maa*aa1/(-w)+2*Mbb*bb1/(-w)+(Mab*ab1+Mba*ba1)/(wab-w)+(Mba*ba1+Mab*ab1)/(wba-w))
    CE_bb=-Mbb*bb1*ratia*chi**2/2*np.sum(2*Maa*aa1/(-w)+2*Mbb*bb1/(-w)+(Mab*ab1+Mba*ba1)/(wab-w)+(Mba*ba1+Mab*ab1)/(wba-w))

    const_energy=np.sum(CE_aa+CE_ab+CE_ba+CE_bb)*ratia


    return Inter_H,Inter_H_base,const_energy #if calculate Berry curvature, use H_1st_base and use Heff in Effective_model

def Many_Ham2sys(kx1,ky1,kx2,ky2,*args):
    w,chi_1,chi_2,S1,S2,ordpa1,ordpa2=args
    aa1,ab1,bb1=ordpa1;ba1=ab1.conjugate()
    aa2,ab2,bb2=ordpa2;ba2=ab2.conjugate()
    H_int1=Ham_int(kx1,ky1,1,w,chi_1,S1,ordpa1)[0]
    H_int2=Ham_int(kx2,ky2,2,w,chi_2,S2,ordpa2)[0]
    CE1=Ham_int(kx1,ky1,1,w,chi_1,S1,ordpa1)[-1]
    b1=hex(l0_1)[0];b2=hex(l0_1)[1]
    dkx1=kx1[1]-kx1[0];dky1=ky1[1]-ky1[0]
    dv=abs(np.cross(dkx1*b1,dky1*b2))
    ratia1=S1/((2*np.pi)**2)*dv
    
    b1=hex(l0_2)[0];b2=hex(l0_2)[1]
    dkx2=kx2[1]-kx2[0];dky2=ky2[1]-ky2[0]
    dv=abs(np.cross(dkx2*b1,dky2*b2))
    ratia2=S2/((2*np.pi)**2)*dv

    H01=Haldane_BC(kx1,ky1,1)
    Hv1=Haldane_BC_Hv(kx1,ky1,1)
    EK1,EV1=eigen(H01)
    Hv1_eig=inv(EV1)@Hv1@EV1
    M1aa=Hv1_eig[:,:,0,0];M1bb=Hv1_eig[:,:,1,1];M1ab=Hv1_eig[:,:,0,1];M1ba=Hv1_eig[:,:,1,0]
    w1ab=EK1[:,:,0]-EK1[:,:,1];w1ba=EK1[:,:,1]-EK1[:,:,0]
    
    H02=Haldane_BC(kx2,ky2,2)
    Hv2=Haldane_BC_Hv(kx2,ky2,2)
    EK2,EV2=eigen(H02)
    Hv2_eig=inv(EV2)@Hv2@EV2
    M2aa=Hv2_eig[:,:,0,0];M2bb=Hv2_eig[:,:,1,1];M2ab=Hv2_eig[:,:,0,1];M2ba=Hv2_eig[:,:,1,0]
    w2ab=EK2[:,:,0]-EK2[:,:,1];w2ba=EK2[:,:,1]-EK2[:,:,0]
    chi_eff=chi_1*chi_2

    H_int1_tot=np.copy(H_int1)
    H_int2_tot=np.copy(H_int2)

    ## the impact of system 2 on system 1
    H_int_21=np.zeros((H_int1.shape[0],H_int1.shape[1],2,2),dtype=complex)
    H_int_21[:,:,0,0]=ratia2*chi_eff/2*np.sum(2*M2aa*aa2/(-w)+2*M2bb*bb2/(-w)+(M2ab*ab2+M2ba*ba2)/(w2ab-w)+(M2ba*ba2+M2ab*ab2)/(w2ba-w))*M1aa \
        +ratia2*chi_eff/2*2*np.sum(M2aa*aa2+M2bb*bb2+M2ab*ab2+M2ba*ba2)*M1aa/(-w)\
        
    H_int_21[:,:,0,1]=ratia2*chi_eff/2*np.sum(2*M2aa*aa2/(-w)+2*M2bb*bb2/(-w)+(M2ab*ab2+M2ba*ba2)/(w2ab-w)+(M2ba*ba2+M2ab*ab2)/(w2ba-w))*M1ab \
        +ratia2*chi_eff/2*np.sum(M2aa*aa2+M2bb*bb2+M2ab*ab2+M2ba*ba2)*(M1ab/(w1ab-w)+M1ab/(w1ba-w))
    H_int_21[:,:,1,0]=H_int_21[:,:,0,1].conjugate()

    H_int_21[:,:,1,1]=ratia2*chi_eff/2*np.sum(2*M2aa*aa2/(-w)+2*M2bb*bb2/(-w)+(M2ab*ab2+M2ba*ba2)/(w2ab-w)+(M2ba*ba2+M2ab*ab2)/(w2ba-w))*M1bb \
        +ratia2*chi_eff/2*2*np.sum(M2aa*aa2+M2bb*bb2+M2ab*ab2+M2ba*ba2)*M1bb/(-w)

    CE_aa=-M1aa*aa1*ratia2*chi_eff/2*np.sum(2*M2aa*aa2/(-w)+2*M2bb*bb2/(-w)+(M2ab*ab2+M2ba*ba2)/(w2ab-w)+(M2ba*ba2+M2ab*ab2)/(w2ba-w))
    CE_ab=-M1ab*ab1*ratia2*chi_eff/2*np.sum(2*M2aa*aa2/(-w)+2*M2bb*bb2/(-w)+(M2ab*ab2+M2ba*ba2)/(w2ab-w)+(M2ba*ba2+M2ab*ab2)/(w2ba-w))
    CE_ba=-M1ba*ba1*ratia2*chi_eff/2*np.sum(2*M2aa*aa2/(-w)+2*M2bb*bb2/(-w)+(M2ab*ab2+M2ba*ba2)/(w2ab-w)+(M2ba*ba2+M2ab*ab2)/(w2ba-w))
    CE_bb=-M1bb*bb1*ratia2*chi_eff/2*np.sum(2*M2aa*aa2/(-w)+2*M2bb*bb2/(-w)+(M2ab*ab2+M2ba*ba2)/(w2ab-w)+(M2ba*ba2+M2ab*ab2)/(w2ba-w))
    
    CE_21=np.sum(CE_aa+CE_ab+CE_ba+CE_bb)*ratia1
    
    ## the impact of system 1 on system 2
    H_int_12=np.zeros((H_int2.shape[0],H_int2.shape[1],2,2),dtype=complex)
    H_int_12[:,:,0,0]=ratia1*chi_eff/2*np.sum(2*M1aa*aa1/(-w)+2*M1bb*bb1/(-w)+(M1ab*ab1+M1ba*ba1)/(w1ab-w)+(M1ba*ba1+M1ab*ab1)/(w1ba-w))*M2aa \
        +ratia1*chi_eff/2*2*np.sum(M1aa*aa1+M1bb*bb1+M1ab*ab1+M1ba*ba1)*M2aa/(-w)\
        
    H_int_12[:,:,0,1]=ratia1*chi_eff/2*np.sum(2*M1aa*aa1/(-w)+2*M1bb*bb1/(-w)+(M1ab*ab1+M1ba*ba1)/(w1ab-w)+(M1ba*ba1+M1ab*ab1)/(w1ba-w))*M2ab \
        +ratia1*chi_eff/2*np.sum(M1aa*aa1+M1bb*bb1+M1ab*ab1+M1ba*ba1)*(M2ab/(w2ab-w)+M2ab/(w2ba-w))
    H_int_12[:,:,1,0]=H_int_12[:,:,0,1].conjugate()

    H_int_12[:,:,1,1]=ratia1*chi_eff/2*np.sum(2*M1aa*aa1/(-w)+2*M1bb*bb1/(-w)+(M1ab*ab1+M1ba*ba1)/(w1ab-w)+(M1ba*ba1+M1ab*ab1)/(w1ba-w))*M2bb \
        +ratia1*chi_eff/2*2*np.sum(M1aa*aa1+M1bb*bb1+M1ab*ab1+M1ba*ba1)*M2bb/(-w)
    H_int1_tot=np.copy(H_int1)+np.copy(H_int_21);H_int1_tot_base=np.copy(EV1@H_int1_tot@inv(EV1))
    H_int2_tot=np.copy(H_int2)+np.copy(H_int_12);H_int2_tot_base=np.copy(EV2@H_int2_tot@inv(EV2))

    return H_int1_tot,H_int1_tot_base,H_int2_tot,H_int2_tot_base,CE_21+CE1

def Clas_Field(kx2,ky2,Effe_a,flag):
    H0=Haldane_BC(kx2,ky2,flag)
    Hv=Haldane_BC_Hv(kx2,ky2,flag)
    E0,EV=eigen(H0)
    Hv_eig=inv(EV)@Hv@EV
    N=H0.shape[0]
    H_eff=np.zeros((N,N,2,2),dtype=complex)
    H_eff[:,:,0,0]=np.copy(chi_1*Effe_a*Hv_eig[:,:,0,0]+E0[:,:,0])
    H_eff[:,:,0,1]=np.copy(chi_1*Effe_a*Hv_eig[:,:,0,1])
    H_eff[:,:,1,0]=np.copy(chi_1*Effe_a*Hv_eig[:,:,1,0])
    H_eff[:,:,1,1]=np.copy(chi_1*Effe_a*Hv_eig[:,:,1,1]+E0[:,:,1])
    H_base=H0+chi_1*Effe_a*Hv
    return H_eff,H_base

