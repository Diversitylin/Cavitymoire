import matplotlib.pyplot as plt
cm = 0.1*1/2.54 
fig = plt.figure(figsize=(20*cm,20*cm))


#plt.text(-0.1, 0.4,'($C_1=1$, $C_2=1$)',color='k')
plt.text(-0.1, 0.4,'$\sim 10^{-2}$ um$^3$',color='g')
#plt.text(-0.1, 0.4,'$\sim 17$ nm',color='k')
#plt.text(-0.1, 0.4,'$\hat{\mathbf{A}}=\mathbf{A}_0(\hat{a}+\hat{a}^{\dagger})$',color='k')
#plt.text(-0.1, 0.4,'$\sim 10^3$ nm',color='k')


plt.xticks([])
plt.yticks([])
plt.axis("off")
plt.rcParams.update({'font.family':'Times New Roman','font.size': 10})
#plt.savefig("C1C2.eps",transparent=True)
plt.savefig("Cavity_field.eps",transparent=True)
#plt.savefig("Lateral.eps",transparent=True)
plt.show()
