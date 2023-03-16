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


pp = pprint.PrettyPrinter(sort_dicts=True)


# Define phantom
image =cv2.imread("sword2_11p.JPG",0)
phantom = np.array(image)

print(phantom.shape, 'phantom')
N = phantom.shape[0]

# x, y = np.meshgrid(np.linspace(-np.pi,np.pi, N),
#                    np.linspace(-np.pi,np.pi, N))

dkx = 2 * np.pi / (N )
dky = 2 * np.pi / (N )

kx = np.linspace(-N/2, N/2, N)*dkx
ky = np.linspace(-N/2, N/2, N)*dky


kx, ky = np.meshgrid(kx, ky)


# RF
# Compute magnetization vectors for all pixels
mx = np.cos(np.pi/2) * phantom
my = np.sin(np.pi/2) * phantom
mz = np.zeros_like(phantom)
m = np.stack([mx, my, mz], axis=-1)


m_rot = m 

x_pos, y_pos = np.meshgrid(np.linspace(1, N, N),
                           np.linspace(1, N, N))

k_space = np.zeros((N, N), dtype=np.complex64)

# pp.pprint(m_rot)
for y in range(N):
    Gx, Gy = np.meshgrid(np.linspace(-x_pos[y, 0] * dkx/2, (x_pos[y, 0] * dkx/2) - (dkx/(N-0)), N),
                         np.linspace(-y_pos[y, 0] * dky/2, (y_pos[y, 0] * dky/2) - (dky/(N-y)), N))
    
    # pp.pprint(Gx)
    # Phase Encoding (for the entire image)
    for row in range(N):
        for col in range(N):
            m_rot[row, col, :] = np.dot(Rz(Gy[row, col]), m[row, col, :])
    
    for x in range(N):
        Gx, Gy = np.meshgrid(np.linspace(-x_pos[y, x] * dkx/2, (x_pos[y, x] * dkx/2) - (dkx/(N-x)), N),
                             np.linspace(-y_pos[y, x] * dky/2, (y_pos[y, x] * dky/2) - (dky/(N-y)), N))
    
        # Frequency Encoding (for the entire image)
        for col in range(N):
            # Gx, Gy = np.meshgrid(np.linspace(-x_pos[col] * dkx/2, x_pos[col] * dkx/2, N),
            #                  np.linspace(-y_pos[y] * dky/2, y_pos[y] * dky/2, N))
            for row in range(N):
                m_rot[row, col, :] = np.dot(Rz(Gx[row, col]), m_rot[row, col, :])
                
        x_sum = np.sum(m_rot[..., 0])
        y_sum = np.sum(m_rot[..., 1])
        k_space[y, x] = np.complex(x_sum, y_sum)
    
image = np.abs((np.fft.ifft2(k_space)))

real_kspace = np.fft.fft2(phantom)
# Display image
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 5)) # , figsize=(10, 5)
ax1.set_title('Phantom')
ax1.imshow(phantom, cmap='gray')
ax2.imshow(image, cmap='gray')
ax2.set_title('Reconstructed')
ax3.imshow(np.log(np.abs(spfft.fftshift(k_space))), cmap='gray')
ax3.set_title('My Poor K-Space')
ax4.imshow(np.log(np.abs(spfft.fftshift(real_kspace))), cmap='gray')
ax4.set_title('Real K-Space')

plt.show()