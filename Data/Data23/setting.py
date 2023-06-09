import numpy as np
"""
globale setting
"""
hbar=6.62607015*1E-34/(2*np.pi)*6.242*1E18
hbar1=6.62607015*1E-34/(2*np.pi)
eps0=8.8541878128*1E-12;eps=1;v_light=1/np.sqrt(eps0*1.25663706212*1e-6)


##system parameters
#Bias1=0.66 ##fix value when ajusting V2
Bias1=0.7 ## Data 23
#Bias1=0.62 ### for D term
theta1=1.2*np.pi/180;lat_1=3.472/(2*np.sin(theta1/2))
EA_1=5;EB_1=-0.5*Bias1;EC_1=0.5*Bias1;t1_1=1;t2_1=1;t3_1=0.29;tB_1=0.06;tC_1=0.06;phi_B_1=2*np.pi/3;phi_C_1=-2*np.pi/3 
num_ctr_1=22
m_1=np.delete(np.linspace(0,1,num_ctr_1),-1);n_1=np.delete(np.linspace(0,1,num_ctr_1),-1);N_1=len(m_1) ##to get K points, use (0,1,10),to get M points,

fact1=10*60/60## changing the fact means changing the sample size.
S1=np.sqrt(3)/2*(21*lat_1)**2
### cavity paramters w means omg
w=8.1;theta=0 ## origin 19.4
#w=12.5
omg0=w*1e-3/hbar;f0=omg0/(2*np.pi);lambd=v_light/(f0);Lz=30


##system 2 parameters
Bias2=2.1## the fixed value for moire2
#Bias2=0.2 #fig 2
#Bias2=1.64## Fig 3
theta2=1.2*np.pi/180;lat_2=3.472/(2*np.sin(theta2/2))
EA_2=5;EB_2=-0.5*Bias2;EC_2=0.5*Bias2;t1_2=1;t2_2=1;t3_2=0.5;tB_2=0.2;tC_2=0.2;phi_B_2=2*np.pi/3;phi_C_2=-2*np.pi/3
num_ctr_2=11
m_2=np.delete(np.linspace(0,1,num_ctr_2),-1);n_2=np.delete(np.linspace(0,1,num_ctr_2),-1);N_2=len(m_2) ##to get K points, use (0,1,10),to get M points,


fact2=10*60/150

S2=np.sqrt(3)/2*(10*lat_2)**2

V1=np.sqrt(3)/2*(99*lat_1)**2*Lz*1e-30
l0_1=np.sqrt(3)/3*lat_1
A01=np.sqrt(hbar1/(omg0*eps*eps0*V1))*1E-10
chi_1=A01*l0_1/hbar

l0_2=np.sqrt(3)/3*lat_2

V2=V1
A02=np.sqrt(hbar1/(omg0*eps*eps0*V2))*1E-10

chi_2=A02*l0_2/hbar
