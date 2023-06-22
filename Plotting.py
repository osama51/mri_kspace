# -*- coding: utf-8 -*-
"""
This example demonstrates many of the 2D plotting capabilities
in pyqtgraph. All of the plots may be panned/scaled by dragging with 
the left/right mouse buttons. Right click on any plot to show a context menu.
"""

# import initExample ## Add path to library (just for examples; you do not need this)


from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import json
# import numpy as np


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
scaling_factor=1.8
noise_scaling=5
Rf_amp_inv = 1.5
prep_length=40

app = pg.mkQApp("Plotting Example")
#mw = QtGui.QMainWindow()
#mw.resize(800,800)

win = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples")
win.resize(1000,600)
win.setWindowTitle('pyqtgraph example: Plotting')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

# p1 = win.addPlot(title="Basic array plotting", y=np.random.normal(size=100))

p2 = win.addPlot(title="Multiple curves")

time_prep = np.arange(0,prep_length,time_step,dtype=float)
zeros_prep=np.full(time_prep.shape,0,dtype=float)
Rf_zeros_prep=np.full(time_prep.shape,0,dtype=float)
time = np.arange(prep_length,60+prep_length,time_step)
# print('time shape ', time.shape)
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
    # plt.plot(time,Gph_zeros)
    # plt.plot(time,Gph_zeros_neg)
    p2.plot(time,Gph_zeros,pen=(255,255,255))
    p2.plot(time,Gph_zeros_neg,pen=(255,255,255))

    Gph_zeros_neg=zeros.copy()
    Gph_zeros=zeros.copy()



x=np.linspace(int(-Rf_time/2),int(Rf_time/2),Rf_x_axis)
y=np.sinc(x)/Rf_amp_inv
# print('ana y',y.shape)
def create_rf(Rf_time,Rf_amp_inv):
    Rf_x_axis=int(Rf_time/time_step)
    x=np.linspace(int(-Rf_time/2),int(Rf_time/2),Rf_x_axis)
    y=np.sinc(x)/Rf_amp_inv
    return x,y


Gx_zeros=zeros.copy()
Gx_zeros[str_Gx_time_neg:fin_Gx_time_neg]=-Gx_amp
Gx_zeros[fin_Gx_time_neg:fin_Gx_time_pos]=Gx_amp


def create_prep_hump(hump_amp,cen_hump_time,hump_interval):
    hump = zeros_prep.copy()
    str_hump_time = int((cen_hump_time - (hump_interval/2))/time_step)
    fin_hump_time = int((cen_hump_time + (hump_interval/2))/time_step)
    hump[str_Gx_time_neg:fin_hump_time]=hump_amp
    return hump
def tagging_draw(angle,cen_hump_time,hump_interval):
    Gx_amp= angle/150
    Gy_amp=-Gx_amp+.6
    x_hump=create_prep_hump(Gx_amp,cen_hump_time,hump_interval)+ (1*scaling_factor)
    y_hump=create_prep_hump(Gy_amp,cen_hump_time,hump_interval)+ (2*scaling_factor)
    p2.plot(time_prep,x_hump, pen=(255,0,0), name="Red curve")
    p2.plot(time_prep,y_hump, pen=(255,255,255), name="Red curve")
tagging_draw(45,20,10)
# write json file
# with open('filename.json','w') as f :
#   data = json.dump(data,f)

# read json file
# with open('sample4.json') as f :
#   data2 = json.load(f)

#prep_inv
def prep_inv(TI=30):
    Rf_time_prep=Rf_time-TI
    # print('el7222',time_prep.shape)
    # Rf_zeros_prep=np.full(time_prep.shape,0,dtype=float)
    Rf_prep=Rf_zeros_prep.copy()
    Rf_prep[int((Rf_time_prep/time_step)-.5*Rf_x_axis):int((Rf_time_prep/time_step)+.5*Rf_x_axis)]=y*2
    print(int((Rf_time_prep/time_step)+.5*Rf_x_axis))
    return Rf_prep
def prep_T2(duration):
    if duration < 10 : duration=10
    if duration > 30 : duration=30
    duration = duration / time_step
    x,y=create_rf(10,1.5)
    Rf_prep=Rf_zeros_prep.copy()
    Rf_prep[0:y.shape[0]]=y
    x,y=create_rf(10,1.5)
    Rf_prep[int(duration):int(y.shape[0]+duration)]=y
    return Rf_prep



# rf_zeros=np.full(time.shape,0,dtype=float)
# print(y.shape[0])


# prep_row = np.concatenate(y,zeros_prep)
rf_zeros=zeros.copy()
rf_zeros[start_time:Rf_x_axis+start_time]=y

# incorporate prep inv
# Rf_prep=prep_inv(47.4)#22.71 min,47.69 max, we will know why
# rf_zeros = np.concatenate((Rf_prep,rf_zeros), axis=0) + (4*scaling_factor)
# incorporate prep_T2
Rf_prep=prep_T2(30)
rf_zeros = np.concatenate((Rf_prep,rf_zeros), axis=0) + (4*scaling_factor)


xx=x.shape[0]
# print(xx)
ran=np.random.rand(x.shape[0])/noise_scaling
y_ran = y+ran
RO_zeros=zeros.copy()
RO_zeros[str_RO_time_pos:fin_RO_time_pos]=y_ran

#Scaling section
RO_zeros = np.concatenate((Rf_zeros_prep,RO_zeros), axis=0)+ (0*scaling_factor)
Gx_zeros = Gx_zeros+ (1*scaling_factor)
# Gx_zeros = np.concatenate((Rf_zeros_prep,Gx_zeros), axis=0)+ (1*scaling_factor)
Gz_zeros = np.concatenate((Rf_zeros_prep,Gz_zeros), axis=0) + (3*scaling_factor)
# Rf_prep = Rf_prep + (4*scaling_factor)

# plt.plot(time,Gx_zeros)
# plt.plot(time,Gz_zeros)
# plt.plot(time,rf_zeros)
# plt.plot(time,RO_zeros)
# plt.show()
total_time= np.concatenate((time_prep,time))
p2.plot(time,Gx_zeros, pen=(255,0,0), name="Red curve")
p2.plot(total_time,Gz_zeros, pen=(0,255,0), name="Green curve")
p2.plot(total_time,rf_zeros, pen=(0,0,255), name="Blue curve")
# print(total_time.shape,RO_zeros.shape)
p2.plot(total_time,RO_zeros, pen=(230,17,50), name="banafseg curve")
# p2.plot(time_prep,Rf_prep, pen=(20,17,250), name="banafseg curve")


if __name__ == '__main__':
    pg.exec()
