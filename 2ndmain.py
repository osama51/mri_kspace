# import numpy as np
# from skimage.transform import radon, iradon
# import matplotlib.pyplot as plt
# import pandas as pd
# from PIL import Image 
# import cv2
# import scipy.fftpack as spfft

# def shepp_logan(size):
#     """
#     Generate a Shepp-Logan phantom.
#     size: The size of the square image.
#     """
#     # Define the ellipses
#     ellipses = [
#         (0.69, 0.92, 0.9, 0.9, 0, 1),
#         (-0.6624, -0.0844, 0.21, 0.21, 0, 1),
#         (0.6624, -0.0844, 0.21, 0.21, 0, 1),
#         (0, -0.6, 0.23, 0.23, 90, 1),
#         (0, 0.1, 0.23, 0.23, 0, 1),
#         (-0.017, 0.1, 0.23, 0.23, 0, 1),
#         (0, 0.45, 0.1, 0.1, 0, 1),
#         (0, -0.45, 0.1, 0.1, 0, 1),
#         (-0.08, -0.605, 0.1, 0.1, 0, 1),
#         (0.08, -0.605, 0.1, 0.1, 0, 1)
#     ]

#     # Create the image
#     image = np.zeros((size, size), dtype = np.float64)
#     x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
#     for a, b, c, d, theta, intensity in ellipses:
#         mask = (((x-a)**2)/c**2 + ((y-b)**2)/d**2) <= 1
#         image[mask] = intensity
    
    
#     return image


# # Define the T1 and T2 values for each material
# t1 = [1000, 500, 2000] # ms
# t2 = [100, 50, 300] # ms
# size = 150
# # Assign T1 and T2 values to the phantom intensities
# image = shepp_logan(size)

# materials = np.zeros_like(image)
# materials[image < 0.2] = 0
# materials[(image >= 0.2) & (image < 0.6)] = 1
# materials[image >= 0.6] = 2
# t1_map = np.zeros_like(image)
# t2_map = np.zeros_like(image)
# t1_map[materials == 0] = t1[0]
# t1_map[materials == 1] = t1[1]
# t1_map[materials == 2] = t1[2]
# t2_map[materials == 0] = t2[0]
# t2_map[materials == 1] = t2[1]
# t2_map[materials == 2] = t2[2]



# # Define the parameters
# FOV = 1 # Field of view
# N = image.shape[0] # Number of pixels
# dy = dx = FOV / N # Pixel size
# Gx = np.zeros((N, N))
# Gy = np.zeros((N, N))
# k_space = np.zeros((N, N), dtype=np.complex128)

# # Apply the RF pulse to rotate the magnetization vector
# theta = np.pi/2
# M = np.zeros_like(image, dtype=np.complex128)
# M.real = np.cos(theta)
# M.imag = np.sin(theta)
# image_m = np.multiply(image, M)
# print(image.dtype)

# # Apply the gradients
# for kx in range(N):
#     for ky in range(N):
#         # Apply the Gy gradient
#         phase = np.exp(-1j * 2 * np.pi * dx)
#         k_space[:, ky] += image_m[kx, ky] * phase
        
#         # Apply the Gx gradient
#         phase = np.exp(-1j * 2 * np.pi * dx)
#         k_space[kx, :] += k_space[kx, ky] * phase
        
        
        

# # # Take the Radon transform of the k-space data
# # theta = np.linspace(0, 180, N, endpoint=False)
# # radon_transform = radon(k_space.real, theta=theta, circle=False)

# # # Reconstruct the image using the inverse Radon transform
# # reconstruction = iradon(radon_transform, theta=theta, circle=False)

# # image =cv2.imread("sword2.JPG",0)
# # image = np.array(image)

# # Inverse Fourier Transform
# # k_space = spfft.fft2(image)
# # k_space = spfft.fftshift(k_space)
# reconstruction = spfft.ifft2(k_space).real

# print(image.dtype)

# # Plot the original image
# plt.figure()
# plt.imshow(image, cmap='gray')
# plt.title('Original Image')

# # Plot the k-space data
# plt.figure()
# plt.imshow(np.log(np.abs(k_space)), cmap='gray')
# plt.title('k-space')

# # Plot the reconstructed image
# plt.figure()
# plt.imshow(reconstruction, cmap='gray')
# plt.title('Reconstructed Image')


import numpy as np
import matplotlib.pyplot as plt

# Define the dimensions of the Shepp-Logan phantom
Nx = 50
Ny = 50

# Define the sampling intervals in the object space
dx = 1
dy = 1

# Define the sampling intervals in the k-space domain
dkx = 2*np.pi/(Nx*dx)
dky = 2*np.pi/(Ny*dy)

# Define the magnetization values in the object space (Shepp-Logan phantom)
x = np.linspace(-Nx//2, Nx//2, Nx)*dx
y = np.linspace(-Ny//2, Ny//2, Ny)*dy
X, Y = np.meshgrid(x, y)
M = np.zeros((Nx, Ny))
M[X**2 + Y**2 < 0.25*(Nx*dx)**2] = 1   # Circle
M[X**2/((0.6*Nx*dx)**2) + Y**2/((0.2*Ny*dy)**2) < 1] = 2   # Ellipse
M[X**2/((0.2*Nx*dx)**2) + Y**2/((0.6*Ny*dy)**2) < 1] = 3   # Ellipse
M[X**2/((0.4*Nx*dx)**2) + Y**2/((0.4*Ny*dy)**2) < 1] = 4   # Ellipse
M[X < 0] = 5   # Rectangle

# Calculate the k-space data using the MRI equation
kx = np.linspace(-Nx//2, Nx//2-1, Nx)*dkx
ky = np.linspace(-Ny//2, Ny//2-1, Ny)*dky
Kx, Ky = np.meshgrid(kx, ky)
K = np.zeros((Nx, Ny), dtype=np.complex64)
for i in range(Nx):
    for j in range(Ny):
        K[i, j] = np.trapz(np.trapz(M*np.exp(-2j*np.pi*(X[i,j]*Kx + Y[i,j]*Ky)), dx=dx), dx=dy)
        print(K)

image = np.fft.ifft(K).real
# Visualize the Shepp-Logan phantom and its k-space representation
plt.subplot(121)
plt.imshow(M, cmap='gray')
plt.title('Shepp-Logan Phantom')
plt.axis('off')

# plt.subplot(122)
# plt.imshow(np.abs(K), cmap='gray')
# plt.title('K-Space Data')
# plt.axis('off')

plt.subplot(122)
plt.imshow(image, cmap='gray')
plt.title('Image Reconstructed')
plt.axis('off')

plt.show()
