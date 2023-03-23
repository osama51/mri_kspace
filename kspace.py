import cv2
import pprint
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt
import scipy.fftpack as spfft

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
    

    # Define rotation matrices
    def Rx(self, theta):
        return np.array([[1, 0, 0],
                          [0, np.cos(theta), -np.sin(theta)],
                          [0, np.sin(theta), np.cos(theta)]])
    
    def Rz(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]])
    
    def RF_pulse(self):
        # RF
        # Compute magnetization vectors for all pixels
        mx = np.cos(np.pi/2) * self.phantom
        my = np.sin(np.pi/2) * self.phantom
        mz = np.zeros_like(self.phantom)
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
        
    
    def build_kspace(self, counter, kspace):
        self.num_of_rows = self.phantom.shape[0]
        self.num_of_cols = self.phantom.shape[1]
        
        # y and x are iterators in K-Space
        for y in range(counter, counter + 1):
            m = self.RF_pulse()
            m_rot = m.copy()
            m_rotx = m.copy()
        
            _, Gy = np.meshgrid(np.linspace(0, (y * self.dkx) * (self.num_of_cols - 1), self.num_of_cols),
                                np.linspace(0, (y * self.dky) * (self.num_of_rows - 1), self.num_of_rows))
            
            
            for x in range(self.num_of_cols):
                
                Gx, _ = np.meshgrid(np.linspace(0, (x * self.dkx) * (self.num_of_cols - 1), self.num_of_cols),
                                    np.linspace(0, (y * self.dky) * (self.num_of_rows - 1), self.num_of_rows))
    
                for col in range(self.num_of_cols):
                    for row in range(self.num_of_rows):
                        # Phase Encoding 
                        m_rot[row, col, :] = np.dot(self.Rz(Gy[row, col]), m[row, col, :])
                        # Frequency Encoding
                        m_rotx[row, col, :] = np.dot(self.Rz(Gx[row, col]), m_rot[row, col, :])
                        
                x_sum = np.sum(m_rotx[..., 0])
                y_sum = np.sum(m_rotx[..., 1])
                kspace[y, x] = complex(y_sum, x_sum)
                
                # image = np.abs((np.fft.ifft2(self.k_space)))
                
    
        # image = np.abs((np.fft.ifft2(self.k_space)))
        # self.k_space = self.k_space / np.max(np.abs(self.k_space)) * 255
        return kspace
    
    
    # # Display image
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 5)) # , figsize=(10, 5)
    # ax1.set_title('Phantom')
    # ax1.imshow(self.phantom, cmap='gray')
    # ax2.imshow(image, cmap='gray')
    # ax2.set_title('Reconstructed')
    # ax3.imshow(np.log(np.abs((self.k_space))), cmap='gray')
    # ax3.set_title('My Poor K-Space')
    # ax4.imshow(np.log(np.abs((self.real_kspace))), cmap='gray')
    # ax4.set_title('Real K-Space')
    # plt.show()

