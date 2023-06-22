

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import json
# import numpy as np

class Plotting:
    def __init__(self):

        super(Plotting,self).__init__()

        self.time_step=0.1
        self.start_time=50
        self.Gz_amp=0.4
        self.Gz_time=15
        self.Rf_time=self.Gz_time
        
        self.Gph_max_amp=.7
        self.Gph_time=7
        self.Gx_amp = 0.4
        self.Gx_time=24
        self.no_of_rows =5
        self.scaling_factor=1.8
        self.noise_scaling=5
        self.Rf_amp_inv = 1.5
        self.prep_length=40
        
        self.TE = 30
        
        
        self.time_prep = np.arange(0,self.prep_length,self.time_step,dtype=float)
        self.zeros_prep=np.full(self.time_prep.shape,0,dtype=float)
        self.Rf_zeros_prep=np.full(self.time_prep.shape,0,dtype=float)
        self.time = np.arange(self.prep_length,60+self.prep_length,self.time_step)
        # print('time shape ', time.shape)
        self.zeros=np.full(self.time.shape,0,dtype=float)
        self.Rf_x_axis=int(self.Rf_time/self.time_step)
        self.fin_Gz_time_pos=int((self.Gz_time/self.time_step)+(self.start_time))
        self.fin_Gph_time=int(self.fin_Gz_time_pos+(self.Gph_time/self.time_step))
        self.fin_Gz_time_neg=int(self.fin_Gz_time_pos+(self.Gz_time/(2*self.time_step)))
        
        self.str_Gx_time_neg=self.fin_Gz_time_pos
        self.fin_Gx_time_neg=int(self.str_Gx_time_neg+(self.Gx_time/(2*self.time_step)))
        
        self.fin_Gx_time_pos=int(self.fin_Gx_time_neg+(self.Gx_time/self.time_step))
        
        # fin_RO_time_pos=int(fin_Gx_time_pos-(Gx_time/(2*time_step))+(Rf_time/(2*time_step)))
        self.fin_RO_time_pos = self.start_time+self.Rf_x_axis+int(self.TE/self.time_step)
        
        self.str_RO_time_pos = int(self.fin_RO_time_pos-(self.Rf_time/(self.time_step)))
        
        
        self.Gz_zeros=self.zeros.copy()
        
        self.Gz_zeros[self.start_time:self.fin_Gz_time_pos]=self.Gz_amp
        self.Gz_zeros[self.fin_Gz_time_pos:self.fin_Gz_time_neg]=-self.Gz_amp
    
    
        #########################################################################
        
        self.x=np.linspace(int(-self.Rf_time/2),int(self.Rf_time/2),self.Rf_x_axis)
        
        
        self.Gx_zeros=self.zeros.copy()
        self.Gx_zeros[self.str_Gx_time_neg:self.fin_Gx_time_neg]=-self.Gx_amp
        self.Gx_zeros[self.fin_Gx_time_neg:self.fin_Gx_time_pos]=self.Gx_amp
        
        self.rf_zeros=self.zeros.copy()
        self.y=np.sinc(self.x)/self.Rf_amp_inv
        self.rf_zeros[self.start_time:self.Rf_x_axis+self.start_time]=self.y
        
        
        
        
        self.xx=self.x.shape[0]
        # print(xx)
        self.ran=np.random.rand(self.x.shape[0])/self.noise_scaling
        self.y_ran = self.y+self.ran
        self.RO_zeros=self.zeros.copy()
        self.RO_zeros[self.str_RO_time_pos:self.fin_RO_time_pos]=self.y_ran
        
        #Scaling section
        self.RO_zeros = np.concatenate((self.Rf_zeros_prep,self.RO_zeros), axis=0)+ (0*self.scaling_factor)
        self.Gx_zeros = self.Gx_zeros+ (1*self.scaling_factor)
        # Gx_zeros = np.concatenate((Rf_zeros_prep,Gx_zeros), axis=0)+ (1*scaling_factor)
        self.Gz_zeros = np.concatenate((self.Rf_zeros_prep,self.Gz_zeros), axis=0) + (3*self.scaling_factor)
        # self.rf_zeros = np.concatenate((self.Rf_zeros_prep,self.rf_zeros), axis=0) + (4*self.scaling_factor)
        
    
    
    def draw_Gy(self, graph):
        self.Gph_zeros=self.zeros.copy()#2
        for row in range(1,self.no_of_rows+1):
            self.Gph_amp= ((self.Gph_max_amp/self.no_of_rows)*row)
            self.Gph_zeros[self.fin_Gz_time_pos:self.fin_Gph_time]=self.Gph_amp
            self.Gph_zeros_neg = - self.Gph_zeros
            self.Gph_zeros = self.Gph_zeros +(2*self.scaling_factor)
            self.Gph_zeros_neg = self.Gph_zeros_neg +(2*self.scaling_factor)
        
        
            """ Draw here """
            timeee = np.arange(0, 100, .1)
            self.Gph_zeros = np.concatenate((self.Rf_zeros_prep + (2*self.scaling_factor),self.Gph_zeros), axis=0)
            graph.plot(timeee,self.Gph_zeros,pen=(255,255,255))
            graph.plot(self.time,self.Gph_zeros_neg,pen=(255,255,255))
        
            self.Gph_zeros_neg=self.zeros.copy()
            self.Gph_zeros=self.zeros.copy()
    
    

    def update_y(self): 
        self.y=np.sinc(self.x) * self.Rf_amp_inv
        self.rf_zeros[self.start_time:self.Rf_x_axis+self.start_time]=self.y
        self.rf_zeros_final = np.concatenate((self.Rf_zeros_prep,self.rf_zeros), axis=0) + (4*self.scaling_factor)
    
    def update_TE(self):
        # self.fin_RO_time_pos = self.start_time+self.Rf_x_axis+int(self.TE/self.time_step)
        self.fin_RO_time_pos = int((25 * self.TE + 3550)/8)
        # if(self.fin_RO_time_pos >= 600):
        #     self.fin_RO_time_pos = 600
        self.str_RO_time_pos = int(self.fin_RO_time_pos-(self.Rf_time/(self.time_step)))
        self.RO_zeros=self.zeros.copy()
        self.RO_zeros[self.str_RO_time_pos:self.fin_RO_time_pos]=self.y_ran
        self.RO_zeros_final = np.concatenate((self.Rf_zeros_prep,self.RO_zeros), axis=0)+ (0*self.scaling_factor)
        
        
    """ draw here """
    def draw_the_rest(self, graph):
        self.update_y()
        self.update_TE()
        # print("hi i'm drawing rn", Rf_amp_inv)
        self.total_time= np.concatenate((self.time_prep,self.time))
        Gx_zeros = np.concatenate((self.Rf_zeros_prep + (1 *self.scaling_factor),self.Gx_zeros), axis=0) 
        # print(self.Rf_zeros_prep.shape)
        graph.plot(self.total_time,Gx_zeros, pen=(255,0,0), name="Red curve")
        graph.plot(self.total_time,self.Gz_zeros, pen=(0,255,0), name="Green curve")
        graph.plot(self.total_time,self.rf_zeros_final, pen=(71,185,235), name="Blue curve")
        # print(total_time.shape,RO_zeros.shape)
        graph.plot(self.total_time,self.RO_zeros_final, pen=(230,17,50), name="banafseg curve")
        
    
    def create_rf(self, Rf_time,Rf_amp_inv):
        Rf_x_axis=int(Rf_time/self.time_step)
        x=np.linspace(int(-Rf_time/2),int(Rf_time/2),Rf_x_axis)
        y=np.sinc(x)/Rf_amp_inv
        return x, y


    def create_prep_hump(self, hump_amp,cen_hump_time,hump_interval):
        hump = self.zeros_prep.copy()
        str_hump_time = int((cen_hump_time - (hump_interval/2))/self.time_step)
        fin_hump_time = int((cen_hump_time + (hump_interval/2))/self.time_step)
        hump[self.str_Gx_time_neg:fin_hump_time]=hump_amp
        return hump
    
    def tagging_draw(self, angle,cen_hump_time,hump_interval, graph):
        Gx_amp= angle/150
        Gy_amp=-Gx_amp+.6
        x_hump= self.create_prep_hump(Gx_amp,cen_hump_time,hump_interval)+ (1*self.scaling_factor)
        y_hump= self.create_prep_hump(Gy_amp,cen_hump_time,hump_interval)+ (2*self.scaling_factor)
        
        graph.clear()
        self.draw_Gy(graph)
        self.draw_the_rest(graph)
        graph.plot(self.time_prep,x_hump, pen=(255,0,0), name="Red curve")
        graph.plot(self.time_prep,y_hump, pen=(255,255,255), name="Red curve")
    # tagging_draw(45,20,10)

    #prep_inv
    def prep_inv(self, TI, graph):
        TI = int(0.01256 * TI - 0.1256)
        self.update_y()
        Rf_inv_start = self.start_time - self.Rf_x_axis - (TI / self.time_step) + 350
        
        if(Rf_inv_start <= 5.1):
            Rf_inv_start = 5.1
            
        Rf_inv_stop = self.Rf_x_axis + Rf_inv_start

        Rf_prep = self.Rf_zeros_prep.copy()

        # print(int(Rf_inv_start), int(Rf_inv_stop))
        Rf_prep[int(Rf_inv_start):int(Rf_inv_stop)]= self.y * 2

        graph.clear()
        self.draw_Gy(graph)
        self.draw_the_rest(graph)
        graph.plot(self.time_prep,Rf_prep + (4*self.scaling_factor), pen=(71,185,235), name="Blue curve")
        # return Rf_prep
    
    def prep_T2(self, duration, graph):
        duration = 0.1333 * duration + 10
        if duration < 10 : duration=10
        if duration > 30 : duration=30
        duration = duration / self.time_step
        x,y = self.create_rf(10,1.5)
        Rf_prep = self.Rf_zeros_prep.copy()
        Rf_prep[0:y.shape[0]]=y
        # self.x,self.y = self.create_rf(10,1.5)
        Rf_prep[int(duration):int(y.shape[0]+duration)] = y
        
        graph.clear()

        self.draw_Gy(graph)
        self.draw_the_rest(graph)
        graph.plot(self.time_prep,graph.plot(self.time_prep,Rf_prep + (4*self.scaling_factor), pen=(71,185,235), name="Blue curve"))
        # return Rf_prep
