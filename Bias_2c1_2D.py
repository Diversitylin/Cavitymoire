from audioop import bias
from email import header
from turtle import color
from scipy.optimize import root
from Tools import *
from functools import reduce
from numpy.linalg import inv


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
    phase=1
    Next_BB=phase*1j*tB*np.exp(1j*phi_B)*(phase_kr(kx1,ky1,nu1)*(np.dot(n1,nu1)/l0)**2+phase_kr(kx1,ky1,nu2)*(np.dot(n1,nu2)/l0)**2+phase_kr(kx1,ky1,nu3)*(np.dot(n1,nu3)/l0)**2)
    Next_CC=phase*1j*tC*np.exp(1j*phi_C)*(phase_kr(kx1,ky1,nu1)*(np.dot(n1,nu1)/l0)**2+phase_kr(kx1,ky1,nu2)*(np.dot(n1,nu2)/l0)**2+phase_kr(kx1,ky1,nu3)*(np.dot(n1,nu3)/l0)**2)

    Hv[:,:,0,0]=Next_BB+Next_BB.conjugate()
    Hv[:,:,1,1]=Next_CC+Next_CC.conjugate()
    Hv[:,:,1,0]=phase*1j*t3*(phase_kr(kx1,ky1,d1)*(np.dot(n1,d1)/l0)**2+phase_kr(kx1,ky1,d3)*(np.dot(n1,d3)/l0)**2+phase_kr(kx1,ky1,d5)*(np.dot(n1,d5)/l0)**2)
    Hv[:,:,0,1]=Hv[:,:,1,0].conjugate()
    return Hv


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
    Inter_H=np.zeros((H0.shape[0],H0.shape[1],2,2),dtype=complex)
 
    Inter_H=np.zeros((H0.shape[0],H0.shape[1],2,2),dtype=complex)
 
    Inter_H[:,:,0,0]=EK[:,:,0]+ratia*chi**2/2*np.sum(2*Maa*aa1/(-w)+2*Mbb*bb1/(-w)+(Mab*ab1+Mba*ba1)/(wab-w)+(Mba*ba1+Mab*ab1)/(wba-w))*Maa \
        +ratia*chi**2/2*2*np.sum(Maa*aa1+Mbb*bb1+Mab*ab1+Mba*ba1)*Maa/(-w)\
            +Daa*chi**2/2
        
    Inter_H[:,:,0,1]=ratia*chi**2/2*np.sum(2*Maa*aa1/(-w)+2*Mbb*bb1/(-w)+(Mab*ab1+Mba*ba1)/(wab-w)+(Mba*ba1+Mab*ab1)/(wba-w))*Mab \
        +ratia*chi**2/2*np.sum(Maa*aa1+Mbb*bb1+Mab*ab1+Mba*ba1)*(Mab/(wab-w)+Mab/(wba-w))\
            +Dab*chi**2/2

    Inter_H[:,:,1,0]=Inter_H[:,:,0,1].conjugate()

    Inter_H[:,:,1,1]=EK[:,:,1]+ratia*chi**2/2*np.sum(2*Maa*aa1/(-w)+2*Mbb*bb1/(-w)+(Mab*ab1+Mba*ba1)/(wab-w)+(Mba*ba1+Mab*ab1)/(wba-w))*Mbb \
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

    # guess10[0,:,:]=(pd.read_csv("./aa1.csv").values).astype(complex)
    # guess10[1,:,:]=(pd.read_csv("./ab1.csv").values).astype(complex)
    # guess10[2,:,:]=(pd.read_csv("./bb1.csv").values).astype(complex)
    # guess20[0,:,:]=(pd.read_csv("./aa2.csv").values).astype(complex)
    # guess20[1,:,:]=(pd.read_csv("./ab2.csv").values).astype(complex)
    # guess20[2,:,:]=(pd.read_csv("./bb2.csv").values).astype(complex)

    guess1=np.reshape(guess10,3*N_1*N_1)
    guess2=np.reshape(guess20,3*N_2*N_2)
    guess=np.concatenate((guess1,guess2))

    sol = root(Gap_Eq, guess, method='df-sane') ##krylov;df-sane
    output=np.split(sol.x,[3*N_1*N_1,3*N_1*N_1+3*N_2*N_2])
    aa1=np.reshape(output[0],(3,N_1,N_1))[0]
    ab1=np.reshape(output[0],(3,N_1,N_1))[1]
    bb1=np.reshape(output[0],(3,N_1,N_1))[2]
    aa2=np.reshape(output[1],(3,N_2,N_2))[0]
    ab2=np.reshape(output[1],(3,N_2,N_2))[1]
    bb2=np.reshape(output[1],(3,N_2,N_2))[2]
    return aa1,ab1,bb1,aa2,ab2,bb2

def count_fun(Bias1,Bias2,ordpa):
    aa1,ab1,bb1=ordpa[0:3];ba1=ab1.conjugate()
    aa2,ab2,bb2=ordpa[3:6];ba2=ab2.conjugate()
    EB_1=0.5*Bias1;EC_1=-0.5*Bias1
    EB_2=0.5*Bias2;EC_2=-0.5*Bias2
    H01=Haldane_BC(m_1,n_1,1)
    Hv1=Haldane_BC_Hv(m_1,n_1,1)
    EK1,EV1=eigen(H01)
    Hv1_eig=inv(EV1)@Hv1@EV1
    M1aa=Hv1_eig[:,:,0,0];M1ab=Hv1_eig[:,:,0,1]
    M1ba=Hv1_eig[:,:,1,0];M1bb=Hv1_eig[:,:,1,1]
    w1ab=EK1[:,:,0]-EK1[:,:,1];w1ba=EK1[:,:,1]-EK1[:,:,0]
    H02=Haldane_BC(m_2,n_2,2)
    Hv2=Haldane_BC_Hv(m_2,n_2,2)
    EK2,EV2=eigen(H02)
    Hv2_eig=inv(EV2)@Hv2@EV2
    M2aa=Hv2_eig[:,:,0,0];M2ab=Hv2_eig[:,:,0,1]
    M2ba=Hv2_eig[:,:,1,0];M2bb=Hv2_eig[:,:,1,1]
    w2ab=EK2[:,:,0]-EK2[:,:,1];w2ba=EK2[:,:,1]-EK2[:,:,0] 

    
    N1aa=chi_1fix*np.sum(M1aa*aa1/(-w))*ratia1
    N1ab=chi_1fix*np.sum(M1ab*ab1/(w1ba-w))*ratia1
    N1bb=chi_1fix*np.sum(M1bb*bb1/(-w))*ratia1
    N1ba=chi_1fix*np.sum(M1ba*ba1/(w1ab-w))*ratia1
    photon1=N1aa+N1ab+N1bb+N1ba
    
    N2aa=chi_2fix*np.sum(M2aa*aa2/(-w))*ratia2
    N2ab=chi_2fix*np.sum(M2ab*ab2/(w2ba-w))*ratia2
    N2bb=chi_2fix*np.sum(M2bb*bb2/(-w))*ratia2
    N2ba=chi_2fix*np.sum(M2ba*ba2/(w2ab-w))*ratia2
    photon2=N2aa+N2ab+N2bb+N2ba
    photon_t=np.sum(np.kron(photon1+photon2,photon1+photon2))

    H_int1,H_int1_base,H_int2,H_int2_base,CE=Many_Ham2sys(m_1,n_1,m_2,n_2,w,chi_1,chi_2,S1,S2,ordpa[0:3],ordpa[3:6])
    Eint=eigen(H_int1)[0]
    GE1=CE.real+np.sum(Eint[:,:,0])*ratia1
    GE0=np.sum(EK1[:,:,0])*ratia1
    GE=(GE1-GE0)

    Egap0=np.min(Eint[:,:,1]-Eint[:,:,0])
    z1=(H_int1_base[:,:,0,0]-H_int1_base[:,:,1,1])/2
    z2=abs(H_int1_base[:,:,0,1])
    Del_H=H_int1-H01
    Del_Hz_max=np.max((Del_H[:,:,0,0]-Del_H[:,:,1,1]).real)
    
    return photon1,photon2,abs(photon_t),GE,z1,z2,Del_Hz_max,Egap0,H_int1
    


import pandas as pd
import os
from Data.Data23.setting import *
print(r"$\chi_2$: ",chi_2)
print('volume/lambd3: ',V1/lambd**3)
data_name="Data23/Bias_2c1_2D"
if not os.path.exists("Data/"+data_name):
    os.makedirs("Data/"+data_name)
Bias_L1=np.linspace(0.5,0.72,23) 
Bias_L2=np.linspace(0.8,2.3,23) 
chi_1fix=chi_1;chi_2fix=chi_2;ratia1=1;ratia2=1
photon1_L=np.zeros((len(Bias_L2),len(Bias_L1)));photon2_L=np.zeros((len(Bias_L2),len(Bias_L1)));photon_tot=np.zeros((len(Bias_L2),len(Bias_L1)))
HZ=np.zeros((len(Bias_L2),len(Bias_L1)));Egap=np.zeros((len(Bias_L2),len(Bias_L1)));Del_Hz_max_L=np.zeros((len(Bias_L2),len(Bias_L1)));Conden_E=np.zeros((len(Bias_L2),len(Bias_L1)));C1_L=np.zeros((len(Bias_L2),len(Bias_L1)));C2_L=np.zeros((len(Bias_L2),len(Bias_L1)))
Chern_L1=np.zeros((len(Bias_L2),len(Bias_L1)));Chern_L2=np.zeros((len(Bias_L2),len(Bias_L1)))

aa1,ab1,bb1=np.ones((3,N_1,N_1),dtype=complex)*101
aa2,ab2,bb2=np.ones((3,N_2,N_2),dtype=complex)*101

aa1_df=pd.DataFrame(aa1);ab1_df=pd.DataFrame(ab1);bb1_df=pd.DataFrame(bb1)
aa2_df=pd.DataFrame(aa2);ab2_df=pd.DataFrame(ab2);bb2_df=pd.DataFrame(bb2)
aa1_df.to_csv("./aa1.csv",index=False);ab1_df.to_csv("./ab1.csv",index=False);bb1_df.to_csv("./bb1.csv",index=False)
aa2_df.to_csv("./aa2.csv",index=False);ab2_df.to_csv("./ab2.csv",index=False);bb2_df.to_csv("./bb2.csv",index=False)


for i,Bias2 in enumerate(Bias_L2):
    for j,Bias1 in enumerate(Bias_L1):
        print(i,j)
        EB_1=0.5*Bias1;EC_1=-0.5*Bias1
        EB_2=0.5*Bias2;EC_2=-0.5*Bias2
        ordpa=Gap_out()
        aa1,ab1,bb1=ordpa[0:3];ba1=ab1.conjugate()
        aa2,ab2,bb2=ordpa[3:6];ba2=ab2.conjugate()
        photon1,photon2,photon_t,GE,z1,z2,Del_Hz_max,EGap0,H_int1=count_fun(Bias1,Bias2,ordpa)
        Mean_a=(photon1+photon2)*2
        H_clas1=Clas_Field(m_1,n_1,Mean_a,1)[1]
        Chern1=Chern_Fr_Discrete(H_clas1)
        H_clas2=Clas_Field(m_2,n_2,Mean_a,2)[1]
        Chern2=Chern_Fr_Discrete(H_clas2)

        photon1_L[i,j]=abs(photon1)
        photon2_L[i,j]=(abs(photon2))
        photon_tot[i,j]=(photon_t)
        Conden_E[i,j]=(GE)
        Egap[i,j]=(EGap0)
        Del_Hz_max_L[i,j]=(Del_Hz_max)
        Chern_L1[i,j]=(Chern1)
        Chern_L2[i,j]=(Chern2)

        if j==len(Bias_L1)-1:
            aa1,ab1,bb1=np.ones((3,N_1,N_1),dtype=complex)*101
            aa2,ab2,bb2=np.ones((3,N_2,N_2),dtype=complex)*101

        aa1_df=pd.DataFrame(aa1);ab1_df=pd.DataFrame(ab1);bb1_df=pd.DataFrame(bb1)
        aa2_df=pd.DataFrame(aa2);ab2_df=pd.DataFrame(ab2);bb2_df=pd.DataFrame(bb2)
        aa1_df.to_csv("./aa1.csv",index=False);ab1_df.to_csv("./ab1.csv",index=False);bb1_df.to_csv("./bb1.csv",index=False)
        aa2_df.to_csv("./aa2.csv",index=False);ab2_df.to_csv("./ab2.csv",index=False);bb2_df.to_csv("./bb2.csv",index=False)

# photon1_L_df=pd.DataFrame(photon1_L);photon1_L_df.to_csv("./Data/"+data_name+"/photon1_L.csv",header=None,index=False)
# photon2_L_df=pd.DataFrame(photon2_L);photon2_L_df.to_csv("./Data/"+data_name+"/photon2_L.csv",header=None,index=False)
# photon_tot_L_df=pd.DataFrame(photon_tot);photon_tot_L_df.to_csv("./Data/"+data_name+"/photon_tot_L.csv",header=None,index=False)
# Chern_L1_df=pd.DataFrame(Chern_L1);Chern_L1_df.to_csv("./Data/"+data_name+"/Chern_L1.csv",header=None,index=False)
# Chern_L2_df=pd.DataFrame(Chern_L2);Chern_L2_df.to_csv("./Data/"+data_name+"/Chern_L2.csv",header=None,index=False)
# Hz_L_df=pd.DataFrame(HZ);Hz_L_df.to_csv("./Data/"+data_name+"/HZ_L.csv",header=None,index=False)
# Egap_L_df=pd.DataFrame(Egap);Egap_L_df.to_csv("./Data/"+data_name+"/Egap_L.csv",header=None,index=False)
# Del_Hz_max_df=pd.DataFrame(Del_Hz_max_L);Del_Hz_max_df.to_csv("./Data/"+data_name+"/Del_Hz_max_L.csv",header=None,index=False)
# Conden_E_df=pd.DataFrame(Conden_E);Conden_E_df.to_csv("./Data/"+data_name+"/Conden_E_L.csv",header=None,index=False)



# photon1_L=pd.read_csv("./Data/"+data_name+"/photon1_L.csv", header=None).values
# photon2_L=pd.read_csv("./Data/"+data_name+"/photon2_L.csv", header=None).values
# Chern_L1=pd.read_csv("./Data/"+data_name+"/Chern_L1.csv", header=None).values
# Chern_L2=pd.read_csv("./Data/"+data_name+"/Chern_L2.csv", header=None).values

C1_V2=[];C1_V1=[]
for j in range(len(Bias_L1)):   
    if Bias_L1[j]>=0.545:
        dC1=np.diff(Chern_L1[:,j])
        temp=np.where(abs(dC1)>=0.9)[0]
        if len(temp)>0:
            bd=temp[0]+1
            C1_V2.append(Bias_L2[bd])
            C1_V1.append(Bias_L1[j])
        else:
            if abs(Chern_L1[0,j]-1)<=0.1:
                C1_V2.append(Bias_L2[-1])
                C1_V1.append(Bias_L1[j])
            if abs(Chern_L1[0,j]-0)<=0.1:
                C1_V2.append(Bias_L2[0])
                C1_V1.append(Bias_L1[j])
        
C2_V2=[];C2_V1=[]
for i in range(len(Bias_L2)):   
    if Bias_L2[i]>=1.68:
        dC2=np.diff(Chern_L2[i,:])
        temp=np.where(abs(dC2)>=0.9)[0]
        if len(temp)>0:
            bd=temp[0]+1
            C2_V2.append(Bias_L2[i])
            C2_V1.append(Bias_L1[bd])
        else:
            if abs(Chern_L2[i,0]-1)<=0.1:
                C2_V2.append(Bias_L2[i])
                C2_V1.append(Bias_L1[-1])
            if abs(Chern_L2[i,0]-0)<=0.1:
                C2_V2.append(Bias_L2[i])
                C2_V1.append(Bias_L1[0])


from matplotlib.colors import LinearSegmentedColormap, Normalize,ListedColormap
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.colors import TwoSlopeNorm

cm = 1/2.54  # centimeters in inches
fig, ax1 = plt.subplots(figsize=(10*cm, 8*cm))
plt.rcParams.update({'font.family':'Times New Roman','font.size': 10})

f1=interpolate.interp2d(Bias_L2,Bias_L1,photon1_L + photon2_L,kind='linear')
x1=linspace(Bias_L2[0],Bias_L2[-1],101)
y1=linspace(Bias_L1[0],Bias_L1[-1],101)
z=f1(x1,y1)
y,x = np.meshgrid(y1,x1)

# z=photon1_L + photon2_L
# y,x=np.meshgrid(Bias_L1,Bias_L2)


cmap = plt.get_cmap('BrBG')
#cmap = plt.get_cmap('plasma')


start, end = 0.2, 0.8
#start, end = 0, 1
colors = cmap(np.linspace(start, end, int((end-start)*1000)))


# Set the colorbar limits based on the data range
vmin = np.min(z)
vmax = np.max(z)
vcenter = np.min(z) + (np.max(z)- np.min(z)) * 0.5
#vcenter = vmin + (vmax - vmin) * 0.8 ##D terms
ticks = [vmin,(vmin+vmax)/2,vmax]

# Use a two-slope normalization to center the colormap around a value
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

p=ax1.pcolor(x,y,z,cmap=ListedColormap(colors),norm=norm,linewidth=0,rasterized=True)
ax1.scatter(C1_V2,C1_V1,c='r',s=10,marker="v")
ax1.scatter(C2_V2,C2_V1,c='b',s=10,marker="*")
ax1.set_xticks(np.linspace(0.8,2.3,6))
ax1.set_xlim([0.8,2.25])


# cax = fig.add_axes([0.77,0.88,0.15,0.04]) # left, bottom, width, height
# cb = fig.colorbar(p, cax=cax, orientation='horizontal',ticks=ticks)
#cb.ax.set_xticklabels(['{:.1f}'.format(tick) for tick in ticks])

cax = fig.add_axes([0.90,0.1,0.04,0.2]) # left, bottom, width, height
cb = fig.colorbar(p, cax=cax, ticks=ticks)
cb.ax.set_yticklabels(['{:.1f}'.format(tick) for tick in ticks])

cb.ax.tick_params(labelsize=8)


#cb.set_label('Colorbar Label', fontsize=10)
cax.yaxis.set_ticks_position('right')
cax.xaxis.set_ticks_position('top')
#cb.ax.set_xticks([np.min(y_new), np.max(y_new)])
plt.subplots_adjust(left=0.12,bottom=0.1,right=0.88,top=0.85, wspace=0.4,hspace=0.4)
#ax1.set_yticks([])

# plt.savefig("./Data/"+data_name+'/Bias_2c1_2D.eps',transparent=True)
# plt.savefig("./Data/"+data_name+'/Bias_2c1_2D.png',dpi=1200)
plt.show()








