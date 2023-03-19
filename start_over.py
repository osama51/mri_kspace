import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
import scipy.fftpack as spfft
import cv2
import pprint

# Define rotation matrices
def Rx(theta):
    return np.array([[1, 0, 0],
                      [0, np.cos(theta), -np.sin(theta)],
                      [0, np.sin(theta), np.cos(theta)]])

def Rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])

def RF_pulse():
    # RF
    # Compute magnetization vectors for all pixels
    mx = np.cos(np.pi/2) * phantom
    my = np.sin(np.pi/2) * phantom
    mz = np.zeros_like(phantom)
    m = np.stack([mx, my, mz], axis=-1)
    
    return m

pp = pprint.PrettyPrinter(sort_dicts=True)


# Define phantom
image =cv2.imread("sword2_11p.JPG",0)
phantom = np.array(image)

print(phantom.shape, 'phantom')
N = phantom.shape[0]


dkx = (2 * np.pi / (N )) 
dky = (2 * np.pi / (N )) 


# m = RF_pulse()
# m_rot = m.copy()

x_pos, y_pos = np.meshgrid(np.linspace(0, N-1, N),
                            np.linspace(0, N-1, N))

k_space = np.zeros((N, N), dtype=np.complex64)


# For each point in k-space, we will change the phase of all vector pixels
# by a gradient of a certain slope (strength), depending on your number of row
# in k-space, the gradient slope will increase, so you basically have Ny gradient slopes 
# for each row in k-space.
#
# Note: 
#      A single gradient with a single slope is a range of angles (in radian), meaning
#      for row 0 in k-space, you create a gradient that covers (2 * pi / Ny) degrees which means 
#      the difference between the angles applied to the image rows will be the gradient range 
#      divided by Ny ((2 * pi / Ny) / Ny), the ranges for this specific gradient for this 
#      specific row in k-space will be np.linspace(-np.pi/Ny, np.pi/Ny, Ny), to center 0 angle.
    
real_kspace = np.fft.fft2(phantom)

# y and x are iterators in K-Space
for y in range(N):
    
    m = RF_pulse()
    m_rot = m.copy()
    m_rotx = m.copy()

    # print(m_rot, 'mmmmm')

    _, Gy = np.meshgrid(np.linspace(0, (y * dkx) * (N - 1), N),
                        np.linspace(0, (y * dky) * (N - 1), N))
    
    # if(y == 0):
    #     Gy = np.linspace(0,0,N) # 0    0    0
    # elif(y == 1):
    #     Gy = np.linspace(0,240,N) # 0   120   240
    # elif(y == 2):
    #     Gy = np.linspace(0,480,N) # 0   240   480
    
    # pp.pprint(Gy)
    # print('Gy ', y)
    
    
    # Phase Encoding (for the entire image)
    # for row in range(N):
    #     for col in range(N):
    #         m_rot[row, col, :] = np.dot(Rz(Gy[row, col]), m[row, col, :])
    #         # print(m, 'mmmm_newwwwww1')
    
    for x in range(N):
        
        Gx, _ = np.meshgrid(np.linspace(0, (x * dkx) * (N - 1), N),
                            np.linspace(0, (y * dky) * (N - 1), N))
        
        # if(x == 0):
        #     Gx = np.linspace(0,0,N)    # step 0
        #     print('x = 0')
        # elif(x == 1):
        #     Gx = np.linspace(0,240,N)  # step 120
        #     print('x = 1')
        # elif(x == 2):
        #     Gx = np.linspace(0,480,N)  # step 240
        #     print('x = 2')
            
        # pp.pprint(Gx)
        # print('Gx ', x)
        
        # Frequency Encoding (for the entire image)
        for col in range(N):
            for row in range(N):
                m_rot[row, col, :] = np.dot(Rz(Gy[row, col]), m[row, col, :])
                m_rotx[row, col, :] = np.dot(Rz(Gx[row, col]), m_rot[row, col, :])
                
        # print(m, 'mmmm_newwwwww_final')
        x_sum = np.sum(m_rotx[..., 0])
        y_sum = np.sum(m_rotx[..., 1])
        k_space[y, x] = np.complex(y_sum, x_sum)
        
        image = np.abs((np.fft.ifft2(k_space)))
        
        # Display image
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 5)) # , figsize=(10, 5)
        ax1.set_title('Phantom')
        ax1.imshow(phantom, cmap='gray')
        ax2.imshow(image, cmap='gray')
        ax2.set_title('Reconstructed')
        ax3.imshow(np.log(np.abs((k_space))), cmap='gray')
        ax3.set_title('My Poor K-Space')
        ax4.imshow(np.log(np.abs((real_kspace))), cmap='gray')
        ax4.set_title('Real K-Space')
    
    
# k_space = k_space / np.max(np.abs(k_space))
image = np.abs((np.fft.ifft2(k_space)))


# Display image
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 5)) # , figsize=(10, 5)
ax1.set_title('Phantom')
ax1.imshow(phantom, cmap='gray')
ax2.imshow(image, cmap='gray')
ax2.set_title('Reconstructed')
ax3.imshow(np.log(np.abs((k_space))), cmap='gray')
ax3.set_title('My Poor K-Space')
ax4.imshow(np.log(np.abs((real_kspace))), cmap='gray')
ax4.set_title('Real K-Space')

plt.show()








# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image 
# import scipy.fftpack as spfft
# import cv2
# import pprint

# # Define rotation matrices
# def Rx(theta):
#     return np.array([[1, 0, 0],
#                       [0, np.cos(theta), -np.sin(theta)],
#                       [0, np.sin(theta), np.cos(theta)]])

# def Rz(theta):
#     return np.array([[np.cos(theta), -np.sin(theta), 0],
#                       [np.sin(theta), np.cos(theta), 0],
#                       [0, 0, 1]])


# pp = pprint.PrettyPrinter(sort_dicts=True)


# # Define phantom
# image =cv2.imread("sword2_3p.JPG",0)
# phantom = np.array(image)

# print(phantom.shape, 'phantom')
# N = phantom.shape[0]

# # x, y = np.meshgrid(np.linspace(-np.pi,np.pi, N),
# #                    np.linspace(-np.pi,np.pi, N))

# dkx = 2 * np.pi / (N )
# dky = 2 * np.pi / (N )

# kx = np.linspace(-N/2, N/2, N)*dkx
# ky = np.linspace(-N/2, N/2, N)*dky


# kx, ky = np.meshgrid(kx, ky)




# # RF
# # Compute magnetization vectors for all pixels
# mx = np.cos(np.pi/2) * phantom
# my = np.sin(np.pi/2) * phantom
# mz = np.zeros_like(phantom)


# x_pos, y_pos = np.meshgrid(np.linspace(1, N, N),
#                             np.linspace(1, N, N))

# k_space = np.zeros((N, N), dtype=np.complex64)

# # pp.pprint(m_rot)
# for y in range(N):

#     m = np.empty((N,N,3))
#     m = np.stack([mx, my, mz], axis=-1)
#     m_rot = m 
#     print(m_rot, 'mmmmmm')
#     _, Gy = np.meshgrid(np.linspace(0, (x_pos[0, y] * dkx) - (dkx/(N-0)), N),
#                         np.linspace(0, (y_pos[y, 0] * dky) - (dky/(N-y)), N))
    
#     # pp.pprint(Gx)
    
#     # For each point in k-space, we will change the phase of all vector pixels
#     # by a gradient of a certain slope (strength), depending on your number of row
#     # in k-space, the gradient slope will increase, so you basically have Ny gradient slopes 
#     # for each row in k-space.
#     #
#     # Note: 
#     #      A single gradient with a single slope is a range of angles (in radian), meaning
#     #      for row 0 in k-space, you create a gradient that covers (2 * pi / Ny) degrees which means 
#     #      the difference between the angles applied to the image rows will be the gradient range 
#     #      divided by Ny ((2 * pi / Ny) / Ny), the ranges for this specific gradient for this 
#     #      specific row in k-space will be np.linspace(-np.pi/Ny, np.pi/Ny, Ny), to center 0 angle.
    
#     # Phase Encoding (for the entire image)
#     for row in range(N):
#         for col in range(N):
#             m_rot[row, col, :] = np.dot(Rz(Gy[row, col]), m[row, col, :])
    
#     for x in range(N):
#         Gx, _ = np.meshgrid(np.linspace(0, (x_pos[y, x] * dkx) - (dkx/(N-x)), N),
#                             np.linspace(0, (y_pos[y, x] * dky) - (dky/(N-y)), N))
    
#         pp.pprint(Gx)
#         # Frequency Encoding (for the entire image)
#         for col in range(N):
#             # Gx, Gy = np.meshgrid(np.linspace(-x_pos[col] * dkx/2, x_pos[col] * dkx/2, N),
#             #                  np.linspace(-y_pos[y] * dky/2, y_pos[y] * dky/2, N))
#             for row in range(N):
#                 m_rot[row, col, :] = np.dot(Rz(Gx[row, col]), m_rot[row, col, :])
                
#         x_sum = np.sum(m_rot[..., 0])
#         y_sum = np.sum(m_rot[..., 1])
#         k_space[y, x] = np.complex(x_sum, y_sum)
    
# image = np.abs((np.fft.ifft2(k_space)))

# real_kspace = np.fft.fft2(phantom)
# # Display image
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 5)) # , figsize=(10, 5)
# ax1.set_title('Phantom')
# ax1.imshow(phantom, cmap='gray')
# ax2.imshow(image, cmap='gray')
# ax2.set_title('Reconstructed')
# ax3.imshow(np.log(np.abs(spfft.fftshift(k_space))), cmap='gray')
# ax3.set_title('My Poor K-Space')
# ax4.imshow(np.log(np.abs(spfft.fftshift(real_kspace))), cmap='gray')
# ax4.set_title('Real K-Space')

# plt.show()

