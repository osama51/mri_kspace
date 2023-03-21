import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
import scipy.fftpack as spfft
import cv2

# Define rotation matrices
def Rx(theta):
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])

def Rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

# Define phantom
image =cv2.imread("sword2_101p.JPG",0)
phantom = np.array(image)

print(phantom.shape, 'phantom')
N = phantom.shape[0]

# phantom = np.zeros((N, N))
x, y = np.meshgrid(np.linspace(-np.pi,np.pi, N), np.linspace(-np.pi,np.pi, N))
# phantom[(x ** 2 + y ** 2) < 0.5 ** 2] = 1

gamma = 42.58  # gyromagnetic ratio in MHz/T for Hydrogen

# Define gradient parameters
Gx = 1 / N # gradient strength in x direction
Gy = 1 / N # gradient strength in y direction

# Define k-space parameters
M = N  # number of samples in k-space

dx = x[0, 1] - x[0, 0] # pixel size for sampling in the k-space domain
dy = y[1, 0] - y[0, 0] # pixel size for sampling in the k-space domain 
dkx = 2 * np.pi / (N )
dky = 2 * np.pi / (N )
kx = np.linspace(-M//2, M//2, M)*dkx
ky = np.linspace(-M//2, M//2, M)*dky


# The meshgrid function returns
# two 2-dimensional arrays
# with kx having the n of rows as ky
# and ky having the n of columns as kx
kx, ky = np.meshgrid(kx, ky)
print(kx.shape, ky.shape, 'k-space')
# print(kx, ky, 'KXKY')


# RF
# Compute magnetization vectors for all pixels
mx = np.cos(np.pi/2) * phantom
my = np.sin(np.pi/2) * phantom
mz = np.zeros_like(phantom)
m = np.stack([mx, my, mz], axis=-1)

# Initialize k-space matrix
kspace = np.zeros((M, M), dtype=np.complex64)

# Loop over all k-space points
for i in range(M):      # i for phase       (vertical)
    for j in range(M):  # j for frequency   (horizontal)
        
        # Rotate magnetization vectors by corresponding gradients
        theta_x = Gx * kx[i, j]
        theta_y = Gy * ky[i, j]
        # print(np.rad2deg(theta_x))
        # m_rot = m @ Rx(theta_y) @ Rx(theta_x)
        m_rot = np.matmul(m, Rz(theta_y))
        m_rot = np.matmul(m_rot, Rz(theta_x))
        
        # Compute dot product with gradient rotation vectors and sum over all pixels
        # Element-wise multiplication
        exp = np.exp(-2j * np.pi * (kx[i, j] * x + ky[i, j] * y))
        
        kspace[i, j] = np.sum(m_rot[..., 0] * exp) + \
                     np.sum(m_rot[..., 1] * exp)
                     

image = np.abs(np.fft.ifftshift(np.fft.ifft2(kspace)))

# Display image
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5)) # , figsize=(10, 5)
ax1.set_title('Phantom')
ax1.imshow(phantom, cmap='gray')
ax2.imshow(image, cmap='gray')
ax2.set_title('Reconstructed')
ax3.imshow(np.log(np.abs((kspace))), cmap='gray')
ax3.set_title('K-Space')

plt.show()


### BACKUP ###

# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image 
# import scipy.fftpack as spfft
# import cv2

# # Define rotation matrices
# def Rx(theta):
#     return np.array([[1, 0, 0],
#                      [0, np.cos(theta), -np.sin(theta)],
#                      [0, np.sin(theta), np.cos(theta)]])

# def Rz(theta):
#     return np.array([[np.cos(theta), -np.sin(theta), 0],
#                      [np.sin(theta), np.cos(theta), 0],
#                      [0, 0, 1]])

# # Define phantom
# image =cv2.imread("sword2_45p.JPG",0)
# phantom = np.array(image)

# print(phantom.shape, 'phantom')
# N = phantom.shape[0]

# # phantom = np.zeros((N, N))
# x, y = np.meshgrid(np.linspace(-N//2,N//2, N), np.linspace(-N//2,N//2, N))
# # phantom[(x ** 2 + y ** 2) < 0.5 ** 2] = 1

# gamma = 42.58  # gyromagnetic ratio in MHz/T for Hydrogen

# # Define gradient parameters
# Gx = 2 * np.pi / N # gradient strength in x direction
# Gy = 2 * np.pi / N # gradient strength in y direction

# # Define k-space parameters
# M = N  # number of samples in k-space

# dx = x[0, 1] - x[0, 0] # pixel size for sampling in the k-space domain
# dy = y[1, 0] - y[0, 0] # pixel size for sampling in the k-space domain 
# dkx = 2 * np.pi / (N * dx)
# dky = 2 * np.pi / (N * dy)
# kx = np.linspace(-M//2, M//2, M)*dkx
# ky = np.linspace(-M//2, M//2, M)*dky


# # The meshgrid function returns
# # two 2-dimensional arrays
# # with kx having the n of rows as ky
# # and ky having the n of columns as kx
# kx, ky = np.meshgrid(kx, ky)
# print(kx.shape, ky.shape, 'k-space')
# # print(kx, ky, 'KXKY')


# # RF
# # Compute magnetization vectors for all pixels
# mx = np.cos(np.pi/2) * phantom
# my = np.sin(np.pi/2) * phantom
# mz = np.zeros_like(phantom)
# m = np.stack([mx, my, mz], axis=-1)

# # Initialize k-space matrix
# kspace = np.zeros((M, M), dtype=np.complex64)

# # Loop over all k-space points
# for i in range(M):      # i for phase       (vertical)
#     for j in range(M):  # j for frequency   (horizontal)
        
#         # Rotate magnetization vectors by corresponding gradients
#         theta_x = Gx * kx[i, j]
#         theta_y = Gy * ky[i, j]
#         # print(np.rad2deg(theta_x))
#         # m_rot = m @ Rx(theta_y) @ Rx(theta_x)
#         m_rot = np.matmul(m, Rx(theta_y))
#         m_rot = np.matmul(m_rot, Rx(theta_x))
        
#         # Compute dot product with gradient rotation vectors and sum over all pixels
#         # Element-wise multiplication
#         kspace[i, j] = np.sum(m_rot[..., 0] * np.exp(-2j * np.pi * (kx[i, j] * Gx * x + ky[i, j] * Gy * y))) + \
#                      np.sum(m_rot[..., 1] * np.exp(-2j * np.pi * (kx[i, j] * Gx * x + ky[i, j] * Gy * y)))
                     
                     
#         # # m_rot[..., 0] and m_rot[..., 1]  (Original)
#         # kspace[i, j] = np.sum(m_rot[..., 0] * np.exp(-2j * np.pi * (kx[i, j] * x + ky[i, j] * y))) + \
#         #                np.sum(m_rot[..., 1] * np.exp(-2j * np.pi * (kx[i, j] * x + ky[i, j] * y)))
                       
#         # kspace[i, j] = np.sum(m_rot[..., 0] * np.exp(1j * 2 * np.pi * gamma * i * y * ky[i, j] / N )) + \
#         #                np.sum(m_rot[..., 1] * np.exp(1j * 2 * np.pi * gamma * j * x * kx[i, j] / N ))

# image = np.abs(np.fft.ifftshift(np.fft.ifft2(kspace)))

# # Display image
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5)) # , figsize=(10, 5)
# ax1.set_title('Phantom')
# ax1.imshow(phantom, cmap='gray')
# ax2.imshow(image, cmap='gray')
# ax2.set_title('Reconstructed')
# ax3.imshow(np.log(np.abs((kspace))), cmap='gray')
# ax3.set_title('K-Space')

# plt.show()







