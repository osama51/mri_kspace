import cv2
import pprint
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt
import scipy.fftpack as spfft
from enum import Enum

class Temp: 
    m16 = np.ndarray((16,16,3))
    m32 = np.ndarray((32,32,3))
    m64 = np.ndarray((64,64,3))
    temp_counter = 0
    
class Parameters:
    TE = 0
    TR = 0
    RF = 0
    TI = 0
    duration = 0
    
class Prep_Pulses(Enum):
    NONE = 0    # Done
    IR = 1      # Done
    T2 = 2      # Done
    TAGGING = 3
    
class ACQ_Seq(Enum):
    GRE = 0     # Done
    SPOILED_GRE = 1
    BALANCED = 2
    SE = 3      # Done
    TSE = 4
    
    
    

class KSpace:
    def __init__(self, phantom):
        super(KSpace,self).__init__()
        
        self.pp = pprint.PrettyPrinter(sort_dicts=True)
    
        # Define phantom
        # phantom =cv2.imread("phantoms/brain16.png",0)
        # self.phantom = np.array(phantom)
        # print(self.phantom.shape, 'phantom')
        
        self.phantom = phantom
        self.num_of_rows = self.phantom.shape[0]
        self.num_of_cols = self.phantom.shape[1]
        
        self.dkx = (2 * np.pi / (self.num_of_rows )) 
        self.dky = (2 * np.pi / (self.num_of_cols )) 
        
        x_pos, y_pos = np.meshgrid(np.linspace(0, self.num_of_cols-1, self.num_of_cols),
                                    np.linspace(0,self.num_of_rows-1, self.num_of_rows))
        
        # self.k_space = np.ones((self.num_of_rows, self.num_of_cols), dtype=np.complex64)
        self.real_kspace = np.fft.fft2(self.phantom)
        
        # in ms
        # self.T1 = 400
        # self.T2 = 40
        self.TR = Parameters.TR
        self.TE = Parameters.TE
        
        # self.tissue1_T2 = 70
        # self.tissue2_T2 = 80
        # # self.tissue3_T2 = 300
        # self.tissue3_T2 = 110
        
        # self.tissue1_T1 = 500
        # self.tissue2_T1 = 800
        # # self.tissue3_T1 = 2500
        # self.tissue3_T1 = 1200
        
        self.tissue1_T2 = 100
        self.tissue2_T2 = 80
        # self.tissue3_T2 = 300
        self.tissue3_T2 = 20
        
        self.tissue1_T1 = 250
        self.tissue2_T1 = 350
        # self.tissue3_T1 = 2500
        self.tissue3_T1 = 1800
        
        
        # self.m = self.magnetization_vector(self.phantom)
    

    # Define rotation matrices
    def Rx(self, theta):
        return np.array([[1, 0, 0],
                          [0, np.cos(theta), -np.sin(theta)],
                          [0, np.sin(theta), np.cos(theta)]])
    
    def Rz(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]])
    
    def decay(self, TE, TR, T1, T2):
        return np.array([[np.exp(-TE / T2), 0, 0],
                         [0, np.exp(-TE / T2), 0],
                         [0, 0, 1-np.exp(-TR / T1)]])
    
    def decay_mag(self, m, decay):
        m_reshaped = m.reshape((-1, 3))
        m_decayed = np.dot(decay, m_reshaped.T).T
        m = m_decayed.reshape((self.num_of_rows, self.num_of_cols, 3))
        return m

    def decay_power(self, m, t):
        for row in range(self.num_of_rows):
            for col in range(self.num_of_cols):
                if(self.phantom[row, col]==85):
                    m[row, col, 0] = m[row, col, 0] * np.exp(-t / self.tissue1_T2)
                    m[row, col, 1] = m[row, col, 1] * np.exp(-t / self.tissue1_T2)
                    m[row, col, 2] = self.phantom[row, col] - ((self.phantom[row, col] - m[row, col, 2]) * np.exp(-t / self.tissue1_T1))
                elif(self.phantom[row, col]==170):
                    m[row, col, 0] = m[row, col, 0] * np.exp(-t / self.tissue2_T2)
                    m[row, col, 1] = m[row, col, 1] * np.exp(-t / self.tissue2_T2)
                    m[row, col, 2] = self.phantom[row, col] - ((self.phantom[row, col] - m[row, col, 2]) * np.exp(-t / self.tissue2_T1))
                
                elif(self.phantom[row, col]==255):
                    m[row, col, 0] = m[row, col, 0] * np.exp(-t / self.tissue3_T2)
                    m[row, col, 1] = m[row, col, 1] * np.exp(-t / self.tissue3_T2)
                    m[row, col, 2] = self.phantom[row, col] - ((self.phantom[row, col] - m[row, col, 2]) * np.exp(-t / self.tissue3_T1))
                
                else:
                    m[row, col, 0] = 0
                    m[row, col, 1] = 0
                    m[row, col, 2] = 0
        return m

    def RF_pulse(self, m, theta):
        m_reshaped = m.reshape((-1, 3))
        m_rotated_rf = np.dot(self.Rx(theta), m_reshaped.T).T
        m = m_rotated_rf.reshape((self.num_of_rows, self.num_of_cols, 3))
        return m
    
    def magnetization_vector(self, phantom):
        mx = np.zeros_like(phantom)
        my = np.zeros_like(phantom)
        # mz = np.zeros_like(self.phantom)
        mz = phantom
        m = np.stack([mx, my, mz], axis=-1)
        return m
    
    """
    For each point in k-space, we will change the phase of all vector pixels
    by a gradient of a certain slope (strength), depending on your number of row
    in k-space, the gradient slope will increase, so you basically have Ny gradient slopes 
    for each row in k-space.
    
    Note: 
          A single gradient with a single slope is a range of angles (in radian), meaning
          for row 0 in k-space, you create a gradient that covers (2 * pi / Ny) degrees which means 
          the difference between the angles applied to the image rows will be the gradient range 
          divided by Ny ((2 * pi / Ny) / Ny), the ranges for this specific gradient for this 
          specific row in k-space will be np.linspace(-np.pi/Ny, np.pi/Ny, Ny), to center 0 angle.
    """
    
    def build_kspace(self, counter, kspace, prep_pulse=Prep_Pulses.NONE, ACQ=ACQ_Seq.GRE):
        self.num_of_rows = self.phantom.shape[0]
        self.num_of_cols = self.phantom.shape[1]
        returned_kspace = kspace
        
        if(prep_pulse==Prep_Pulses.NONE):
            m = self.NONE_prep_pulse(counter)
        elif(prep_pulse==1):
            returned_kspace = self.IR_prep_pulse(counter, kspace, Parameters.TI)
        elif(prep_pulse==2):
            returned_kspace = self.T2_prep_pulse(counter, kspace, Parameters.duration)
        
        
        if(ACQ==ACQ_Seq.GRE):   # GRE
            returned_kspace = self.basic_GRE(counter, kspace, m)
        elif(ACQ==1): # 
            pass
        elif(ACQ==2): #
            pass
        elif(ACQ==ACQ_Seq.SE): # SE_Seq
            returned_kspace = self.SE_Seq(counter, kspace, m)
        return returned_kspace
    
    def basic_GRE(self, counter, kspace, m_prep):
        # y and x are iterators in K-Space
        for y in range(counter, counter + 1):
            # if(counter == 0):
            #     m = self.magnetization_vector(self.phantom)
            # else:
            #     if(self.num_of_cols==16):
            #         m = Temp.m16
            #     elif(self.num_of_cols==32):
            #         m = Temp.m32
            #     elif(self.num_of_cols==64):
            #         m = Temp.m64
                                    
            #     m = self.decay_power(m, self.TR)
                            
            m = m_prep
            
            """ comment the next line when you use decay """
            # m = self.magnetization_vector(self.phantom)
            
            """" OPTIMIZATION """
            # m[self.phaton == 0] = 0
            # m[self.phantom == 85] = self.phantom - ((self.phantom - m[..., 2]) * np.exp(-self.TR / self.tissue1_T1))
            # m[self.phantom == 170] = self.phantom - ((self.phantom - m[..., 2]) * np.exp(-self.TR / self.tissue2_T1))
            # m[self.phantom == 255] = self.phantom - ((self.phantom - m[..., 2]) * np.exp(-self.TR / self.tissue3_T1))
            
           
            m = self.RF_pulse(m, (Parameters.RF * np.pi)/180)
            
            # if(ACQ==3):
            #     m = self.decay_power(m, self.TE/2)
            #     m = self.RF_pulse(m, np.pi)
            
            # m = self.RF_pulse_rotation(np.pi/2)
            m_rot = m.copy()
            m_rotx = m.copy()
        
            _, Gy = np.meshgrid(np.linspace(0, (y * self.dkx) * (self.num_of_cols - 1), self.num_of_cols),
                                np.linspace(0, (y * self.dky) * (self.num_of_rows - 1), self.num_of_rows))
            
            for row in range(self.num_of_rows):
               for col in range(self.num_of_cols):  
                   # Phase Encoding 
                   m_rot[row, col, :] = np.dot(self.Rz(Gy[row, col]), m[row, col, :])

            m_rot = self.decay_power(m_rot, self.TE)
            for x in range(self.num_of_cols):
                
                Gx, _ = np.meshgrid(np.linspace(0, (x * self.dkx) * (self.num_of_cols - 1), self.num_of_cols),
                                    np.linspace(0, (y * self.dky) * (self.num_of_rows - 1), self.num_of_rows))
    

                for row in range(self.num_of_rows):
                    for col in range(self.num_of_cols):       

                        # Frequency Encoding
                        m_rotx[row, col, :] = np.dot(self.Rz(Gx[row, col]), m_rot[row, col, :])
                    
                if(self.num_of_cols==16):
                    Temp.m16 = m_rotx
                elif(self.num_of_cols==32):
                    Temp.m32 = m_rotx
                elif(self.num_of_cols==64):
                    Temp.m64 = m_rotx
                
                x_sum = np.sum(m_rotx[..., 0])
                y_sum = np.sum(m_rotx[..., 1])
                kspace[y, x] = complex(y_sum, x_sum)
                # self.pp.pprint(Gy)
    
        # image = np.abs((np.fft.ifft2(self.k_space)))
        # self.k_space = self.k_space / np.max(np.abs(self.k_space)) * 255
        return kspace

    def SE_Seq(self, counter, kspace, m_prep):
        # y and x are iterators in K-Space
        for y in range(counter, counter + 1):
           
            m = m_prep
            m = self.RF_pulse(m, (Parameters.RF * np.pi)/180)
            

            m = self.decay_power(m, self.TE/2)
            m = self.RF_pulse(m, np.pi)

            m_rot = m.copy()
            m_rotx = m.copy()
        
            _, Gy = np.meshgrid(np.linspace(0, (y * self.dkx) * (self.num_of_cols - 1), self.num_of_cols),
                                np.linspace(0, (y * self.dky) * (self.num_of_rows - 1), self.num_of_rows))
            
            for row in range(self.num_of_rows):
               for col in range(self.num_of_cols):  
                   # Phase Encoding 
                   m_rot[row, col, :] = np.dot(self.Rz(Gy[row, col]), m[row, col, :])

            m_rot = self.decay_power(m_rot, self.TE/2)
            for x in range(self.num_of_cols):
                
                Gx, _ = np.meshgrid(np.linspace(0, (x * self.dkx) * (self.num_of_cols - 1), self.num_of_cols),
                                    np.linspace(0, (y * self.dky) * (self.num_of_rows - 1), self.num_of_rows))
    

                for row in range(self.num_of_rows):
                    for col in range(self.num_of_cols):       

                        # Frequency Encoding
                        m_rotx[row, col, :] = np.dot(self.Rz(Gx[row, col]), m_rot[row, col, :])
                    
                if(self.num_of_cols==16):
                    Temp.m16 = m_rotx
                elif(self.num_of_cols==32):
                    Temp.m32 = m_rotx
                elif(self.num_of_cols==64):
                    Temp.m64 = m_rotx
                
                x_sum = np.sum(m_rotx[..., 0])
                y_sum = np.sum(m_rotx[..., 1])
                kspace[y, x] = complex(y_sum, x_sum)
                # self.pp.pprint(Gy)
    
        # image = np.abs((np.fft.ifft2(self.k_space)))
        # self.k_space = self.k_space / np.max(np.abs(self.k_space)) * 255
        return kspace
    
    # not implemented yet
    def balnced_ssfp(self, counter, kspace):
        # y and x are iterators in K-Space
        for y in range(counter, counter + 1):
            if(counter == 0):
                m = self.magnetization_vector(self.phantom)
            else:
                if(self.num_of_cols==16):
                    m = Temp.m16
                elif(self.num_of_cols==32):
                    m = Temp.m32
                elif(self.num_of_cols==64):
                    m = Temp.m64
                                    
                m = self.decay_power(m, self.TR)
                            
                                       
            m = self.RF_pulse(m, (Parameters.RF * np.pi)/180)
            
            # if(ACQ==3):
            #     m = self.decay_power(m, self.TE/2)
            #     m = self.RF_pulse(m, np.pi)
            
            # m = self.RF_pulse_rotation(np.pi/2)
            m_rot = m.copy()
            m_rotx = m.copy()
        
            _, Gy = np.meshgrid(np.linspace(0, (y * self.dkx) * (self.num_of_cols - 1), self.num_of_cols),
                                np.linspace(0, (y * self.dky) * (self.num_of_rows - 1), self.num_of_rows))
            
            for row in range(self.num_of_rows):
               for col in range(self.num_of_cols):  
                   # Phase Encoding 
                   m_rot[row, col, :] = np.dot(self.Rz(Gy[row, col]), m[row, col, :])

            m_rot = self.decay_power(m_rot, self.TE)
            for x in range(self.num_of_cols):
                
                Gx, _ = np.meshgrid(np.linspace(0, (x * self.dkx) * (self.num_of_cols - 1), self.num_of_cols),
                                    np.linspace(0, (y * self.dky) * (self.num_of_rows - 1), self.num_of_rows))
    

                for row in range(self.num_of_rows):
                    for col in range(self.num_of_cols):       

                        # Frequency Encoding
                        m_rotx[row, col, :] = np.dot(self.Rz(Gx[row, col]), m_rot[row, col, :])
                    
                if(self.num_of_cols==16):
                    Temp.m16 = m_rotx
                elif(self.num_of_cols==32):
                    Temp.m32 = m_rotx
                elif(self.num_of_cols==64):
                    Temp.m64 = m_rotx
                
                x_sum = np.sum(m_rotx[..., 0])
                y_sum = np.sum(m_rotx[..., 1])
                kspace[y, x] = complex(y_sum, x_sum)
                # self.pp.pprint(Gy)
    
        # image = np.abs((np.fft.ifft2(self.k_space)))
        # self.k_space = self.k_space / np.max(np.abs(self.k_space)) * 255
        return kspace
        
    
    
    
    """"">_______________________________________________<"""""
    """"|             WELCOME TO PREP PULSES             |"""
    """|________________________________________________|"""

    def NONE_prep_pulse(self, counter):
        if(counter == 0):
            m = self.magnetization_vector(self.phantom)
        else:
            if(self.num_of_cols==16):
                m = Temp.m16
            elif(self.num_of_cols==32):
                m = Temp.m32
            elif(self.num_of_cols==64):
                m = Temp.m64
                                
            m = self.decay_power(m, self.TR)
            
        return m

    def IR_prep_pulse(self, counter, kspace, TI):
        # y and x are iterators in K-Space
        for y in range(counter, counter + 1):
            if(counter == 0):
                m = self.magnetization_vector(self.phantom)
            else:
                if(self.num_of_cols==16):
                    m = Temp.m16
                elif(self.num_of_cols==32):
                    m = Temp.m32
                elif(self.num_of_cols==64):
                    m = Temp.m64
                                    
                m = self.decay_power(m, self.TR-TI)
                            
            m = self.RF_pulse(m, np.pi)
            m = self.decay_power(m, TI)
            m = self.RF_pulse(m, (Parameters.RF * np.pi)/180)
            
            # if(ACQ==3):
            #     m = self.decay_power(m, self.TE/2)
            #     m = self.RF_pulse(m, np.pi)
            
            # m = self.RF_pulse_rotation(np.pi/2)
            m_rot = m.copy()
            m_rotx = m.copy()
        
            _, Gy = np.meshgrid(np.linspace(0, (y * self.dkx) * (self.num_of_cols - 1), self.num_of_cols),
                                np.linspace(0, (y * self.dky) * (self.num_of_rows - 1), self.num_of_rows))
            
            for row in range(self.num_of_rows):
               for col in range(self.num_of_cols):  
                   # Phase Encoding 
                   m_rot[row, col, :] = np.dot(self.Rz(Gy[row, col]), m[row, col, :])

            m_rot = self.decay_power(m_rot, self.TE)
            for x in range(self.num_of_cols):
                
                Gx, _ = np.meshgrid(np.linspace(0, (x * self.dkx) * (self.num_of_cols - 1), self.num_of_cols),
                                    np.linspace(0, (y * self.dky) * (self.num_of_rows - 1), self.num_of_rows))
    

                for row in range(self.num_of_rows):
                    for col in range(self.num_of_cols):       

                        # Frequency Encoding
                        m_rotx[row, col, :] = np.dot(self.Rz(Gx[row, col]), m_rot[row, col, :])
                    
                if(self.num_of_cols==16):
                    Temp.m16 = m_rotx
                elif(self.num_of_cols==32):
                    Temp.m32 = m_rotx
                elif(self.num_of_cols==64):
                    Temp.m64 = m_rotx
                
                x_sum = np.sum(m_rotx[..., 0])
                y_sum = np.sum(m_rotx[..., 1])
                kspace[y, x] = complex(y_sum, x_sum)
                # self.pp.pprint(Gy)
    
        # image = np.abs((np.fft.ifft2(self.k_space)))
        # self.k_space = self.k_space / np.max(np.abs(self.k_space)) * 255
        return kspace
    
    
    def T2_prep_pulse(self, counter, kspace, duration):
         # y and x are iterators in K-Space
        for y in range(counter, counter + 1):
            if(counter == 0):
                m = self.magnetization_vector(self.phantom)
            else:
                if(self.num_of_cols==16):
                    m = Temp.m16
                elif(self.num_of_cols==32):
                    m = Temp.m32
                elif(self.num_of_cols==64):
                    m = Temp.m64
                                    
                m = self.decay_power(m, self.TR-duration)
                            
            m = self.RF_pulse(m, np.pi/2)
            m = self.decay_power(m, duration)
            m = self.RF_pulse(m, 3*np.pi/2)
            m = self.RF_pulse(m, (Parameters.RF * np.pi)/180)
            
            # if(ACQ==3):
            #     m = self.decay_power(m, self.TE/2)
            #     m = self.RF_pulse(m, np.pi)
            
            # m = self.RF_pulse_rotation(np.pi/2)
            m_rot = m.copy()
            m_rotx = m.copy()
        
            _, Gy = np.meshgrid(np.linspace(0, (y * self.dkx) * (self.num_of_cols - 1), self.num_of_cols),
                                np.linspace(0, (y * self.dky) * (self.num_of_rows - 1), self.num_of_rows))
            
            for row in range(self.num_of_rows):
               for col in range(self.num_of_cols):  
                   # Phase Encoding 
                   m_rot[row, col, :] = np.dot(self.Rz(Gy[row, col]), m[row, col, :])

            m_rot = self.decay_power(m_rot, self.TE)
            for x in range(self.num_of_cols):
                
                Gx, _ = np.meshgrid(np.linspace(0, (x * self.dkx) * (self.num_of_cols - 1), self.num_of_cols),
                                    np.linspace(0, (y * self.dky) * (self.num_of_rows - 1), self.num_of_rows))
    

                for row in range(self.num_of_rows):
                    for col in range(self.num_of_cols):       

                        # Frequency Encoding
                        m_rotx[row, col, :] = np.dot(self.Rz(Gx[row, col]), m_rot[row, col, :])
                    
                if(self.num_of_cols==16):
                    Temp.m16 = m_rotx
                elif(self.num_of_cols==32):
                    Temp.m32 = m_rotx
                elif(self.num_of_cols==64):
                    Temp.m64 = m_rotx
                
                x_sum = np.sum(m_rotx[..., 0])
                y_sum = np.sum(m_rotx[..., 1])
                kspace[y, x] = complex(y_sum, x_sum)
                # self.pp.pprint(Gy)
    
        # image = np.abs((np.fft.ifft2(self.k_space)))
        # self.k_space = self.k_space / np.max(np.abs(self.k_space)) * 255
        return kspace
# phantom =cv2.imread("phantoms/brain16.png",0)
# phantom = np.array(phantom)
# m1 = KSpace(phantom).RF_pulse()
# m2 = KSpace(phantom).RF_pulse_rotation(np.pi/2)