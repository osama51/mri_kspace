'''
Tr
Te
Rf :sinc pulse amp : 1.5, time :14
gz:[time : 14 ,c_time :, amp:0.8]
Gpe'y' : [time :10 ,c_time ,amp: 1.4-1.4/no of rows]
Gx : [time:20 , amp : 0.8]
RO : sinc + noise
Te = ()
Tr
numpy.concatenate
total time 60
'''
import json
import numpy as np
import matplotlib.pyplot as plt

time_step=0.1
start_time=50
Gz_amp=0.4
Gz_time=15
Rf_time=Gz_time
Gph_max_amp=.7
Gph_time=7
Gx_amp = 0.4
Gx_time=24
no_of_rows =5
scaling_factor=2
noise_scaling=5


time = np.arange(0,60,time_step)
zeros=np.full(time.shape,0,dtype=float)
Rf_x_axis=int(Rf_time/time_step)
fin_Gz_time_pos=int((Gz_time/time_step)+(start_time))
fin_Gph_time=int(fin_Gz_time_pos+(Gph_time/time_step))
fin_Gz_time_neg=int(fin_Gz_time_pos+(Gz_time/(2*time_step)))
# str_Gx_time_neg=int(fin_Gph_time-(Gx_time/(2*time_step)))
str_Gx_time_neg=fin_Gz_time_pos
fin_Gx_time_neg=int(str_Gx_time_neg+(Gx_time/(2*time_step)))

fin_Gx_time_pos=int(fin_Gx_time_neg+(Gx_time/time_step))
fin_RO_time_pos=int(fin_Gx_time_pos-(Gx_time/(2*time_step))+(Rf_time/(2*time_step)))
str_RO_time_pos=int(fin_RO_time_pos-(Rf_time/(time_step)))
# np.zeros(600)
# Rf_amp = 1.5
# Rf_time= 14



Gz_zeros=zeros.copy()

Gz_zeros[start_time:fin_Gz_time_pos]=Gz_amp
Gz_zeros[fin_Gz_time_pos:fin_Gz_time_neg]=-Gz_amp



# amp: 1.4-1.4/no_of_rows
Gph_zeros=zeros.copy()#2
for row in range(1,no_of_rows+1):
	Gph_amp= ((Gph_max_amp/no_of_rows)*row)
	Gph_zeros[fin_Gz_time_pos:fin_Gph_time]=Gph_amp
	Gph_zeros_neg = - Gph_zeros
	Gph_zeros = Gph_zeros +(2*scaling_factor)
	Gph_zeros_neg = Gph_zeros_neg +(2*scaling_factor)
	plt.plot(time,Gph_zeros)
	plt.plot(time,Gph_zeros_neg)
	Gph_zeros_neg=zeros.copy()
	Gph_zeros=zeros.copy()



x=np.linspace(int(-Rf_time/2),int(Rf_time/2),Rf_x_axis)
y=np.sinc(x)/1.5


Gx_zeros=zeros.copy()
Gx_zeros[str_Gx_time_neg:fin_Gx_time_neg]=-Gx_amp
Gx_zeros[fin_Gx_time_neg:fin_Gx_time_pos]=Gx_amp




# write json file
# with open('filename.json','w') as f :
# 	data = json.dump(data,f)

# read json file
# with open('sample4.json') as f :
# 	data2 = json.load(f)




# rf_zeros=np.full(time.shape,0,dtype=float)
print(y.shape[0])
# Rf_row = np.concatenate(y,rf_zeros)
rf_zeros=zeros.copy()
rf_zeros[start_time:Rf_x_axis+start_time]=y

xx=x.shape[0]
# print(xx)
ran=np.random.rand(x.shape[0])/noise_scaling
y_ran = y+ran
RO_zeros=zeros.copy()
RO_zeros[str_RO_time_pos:fin_RO_time_pos]=y_ran

Gx_zeros = Gx_zeros + (1*scaling_factor)
Gz_zeros = Gz_zeros + (3*scaling_factor)
rf_zeros = rf_zeros + (4*scaling_factor)
# Gx_zeros


plt.plot(time,Gx_zeros)
plt.plot(time,Gz_zeros)
plt.plot(time,rf_zeros)
plt.plot(time,RO_zeros)
plt.show()