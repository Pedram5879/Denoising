import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io, util

def add_speckle_noise(image, variance):
    row, col, ch = image.shape
    noisy = util.random_noise(image, mode='speckle', var=variance)
    return (noisy * 255).astype(np.uint8)


# Read the image
image_path = "../../DS/Med/1.png"
original_image = cv2.imread(image_path)
# Display the original image
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Apply noise with different variances
variances = [0.0001, 0.0005, 0.1, 0.2]

# Add noise with density 0.0001
noisy_image_0001 = add_speckle_noise(original_image, 0.0001)

# Display the noisy image
plt.subplot(2, 3, 2), plt.imshow(cv2.cvtColor(noisy_image_0001, cv2.COLOR_BGR2RGB))
plt.title('(Variance={0.0001})')

# Save the noisy image to a new file
cv2.imwrite('Speckle_noisy_image_variance_0.0001.png', noisy_image_0001)


# Add noise with density 0.002
noisy_image_0005 = add_speckle_noise(original_image, 0.0005)

# Display the noisy image
plt.subplot(2, 3, 3), plt.imshow(cv2.cvtColor(noisy_image_0005, cv2.COLOR_BGR2RGB))
plt.title('(Variance={0.002})')

# Save the noisy image to a new file
cv2.imwrite('Speckle_noisy_image_variance_0.0005.png', noisy_image_0005)

# Add noise with density 0.001
noisy_image_01 = add_speckle_noise(original_image, 0.1)

# Display the noisy image
plt.subplot(2, 3, 4), plt.imshow(cv2.cvtColor(noisy_image_01, cv2.COLOR_BGR2RGB))
plt.title('(Variance={0.01})')

# Save the noisy image to a new file
cv2.imwrite('Speckle_noisy_image_variance_0.01.png', noisy_image_01)

# Add noise with density 0.0005
noisy_image_02 = add_speckle_noise(original_image, 0.2)

# Display the noisy image
plt.subplot(2, 3, 5), plt.imshow(cv2.cvtColor(noisy_image_02, cv2.COLOR_BGR2RGB))
plt.title('(Variance={0.02})')

# Save the noisy image to a new file
cv2.imwrite('Speckle_noisy_image_variance_0.02.png', noisy_image_02)

# Apply median filter
denoised_image_0005 = cv2.medianBlur(noisy_image_0005, 5)  # The second parameter is the kernel size, which should be an odd integer
denoised_image_0001 = cv2.medianBlur(noisy_image_0001, 5)  # The second parameter is the kernel size, which should be an odd integer
denoised_image_01 = cv2.medianBlur(noisy_image_01, 5)  # The second parameter is the kernel size, which should be an odd integer
denoised_image_02 = cv2.medianBlur(noisy_image_02, 5)  # The second parameter is the kernel size, which should be an odd integer

# Save the noisy image to a new file
cv2.imwrite('speckel_noise_Denoised_median_filter_density_0.0005.png', denoised_image_0005)
cv2.imwrite('speckel_noise_Denoised_median_filter_density_0.0001.png', denoised_image_0001)
cv2.imwrite('speckel_noise_Denoised_median_filter_density_0.001.png', denoised_image_01)
cv2.imwrite('speckel_noise_Denoised_median_filter_density_0.002.png', denoised_image_02)

# Apply a Gaussian blur filter
blurred_image_0005 = cv2.GaussianBlur(noisy_image_0005, (5, 5), 0)
blurred_image_0001 = cv2.GaussianBlur(noisy_image_0001, (5, 5), 0)
blurred_image_01 = cv2.GaussianBlur(noisy_image_01, (5, 5), 0)
blurred_image_02 = cv2.GaussianBlur(noisy_image_02, (5, 5), 0)

# Save the Gaussian blur filter a new file
cv2.imwrite('speckel_noise_Denoised_blur_filter_density_0.005.png', blurred_image_0005)
cv2.imwrite('speckel_noise_Denoised_blur_filter_density_0.001.png', blurred_image_0001)
cv2.imwrite('speckel_noise_Denoised_blur_filter_density_0.01.png', blurred_image_01)
cv2.imwrite('speckel_noise_Denoised_blur_filter_density_0.02.png', blurred_image_02)

# Apply a bilateral filter
bilateralFilter_image_0005 = cv2.bilateralFilter(noisy_image_0005, 9, 75, 75)
bilateralFilter_image_0001 = cv2.bilateralFilter(noisy_image_0001, 9, 75, 75)
bilateralFilter_image_01 = cv2.bilateralFilter(noisy_image_01, 9, 75, 75)
bilateralFilter_image_02 = cv2.bilateralFilter(noisy_image_02, 9, 75, 75)

# Save the bilateral filter a new file
cv2.imwrite('speckel_noise_Denoised_bilateral_filter_density_0.05.png', bilateralFilter_image_0005)
cv2.imwrite('speckel_noise_Denoised_bilateral_filter_density_0.1.png', bilateralFilter_image_0001)
cv2.imwrite('speckel_noise_Denoised_bilateral_filter_density_0.2.png', bilateralFilter_image_01)
cv2.imwrite('speckel_noise_Denoised_bilateral_filter_density_0.4.png', bilateralFilter_image_02)

# Apply sharpening filter (Laplacian) in spatial domain
sharpened_image_0005 = cv2.Laplacian(denoised_image_0005, cv2.CV_64F)
sharpened_image_0005 = np.uint8(np.absolute(sharpened_image_0005))

sharpened_image_0001 = cv2.Laplacian(denoised_image_0001, cv2.CV_64F)
sharpened_image_0001 = np.uint8(np.absolute(sharpened_image_0001))

sharpened_image_001 = cv2.Laplacian(denoised_image_01, cv2.CV_64F)
sharpened_image_001 = np.uint8(np.absolute(sharpened_image_001))

sharpened_image_002 = cv2.Laplacian(denoised_image_02, cv2.CV_64F)
sharpened_image_002 = np.uint8(np.absolute(sharpened_image_002))

# Save the sharpened images to new files
cv2.imwrite('Speckel_noise_laplacian_Sharpened_density_0.0005.png', sharpened_image_0005 + denoised_image_0005)
cv2.imwrite('Speckel_noise_laplacian_Sharpened_density_0.0001.png', sharpened_image_0001 + denoised_image_0001)
cv2.imwrite('Speckel_noise_laplacian_Sharpened_density_0.001.png', sharpened_image_001 + denoised_image_01)
cv2.imwrite('Speckel_noise_laplacian_Sharpened_density_0.002.png', sharpened_image_002 + denoised_image_02)

# Apply Sobel filter as a sharpening filter in spatial domain
sobel_sharpened_image_0005_x = cv2.Sobel(denoised_image_0005, cv2.CV_64F, 1, 0, ksize=3)
sobel_sharpened_image_0005_y = cv2.Sobel(denoised_image_0005, cv2.CV_64F, 0, 1, ksize=3)
sobel_sharpened_image_0005 = np.sqrt(sobel_sharpened_image_0005_x**2 + sobel_sharpened_image_0005_y**2)

sobel_sharpened_image_0001_x = cv2.Sobel(denoised_image_0001, cv2.CV_64F, 1, 0, ksize=3)
sobel_sharpened_image_0001_y = cv2.Sobel(denoised_image_0001, cv2.CV_64F, 0, 1, ksize=3)
sobel_sharpened_image_0001 = np.sqrt(sobel_sharpened_image_0001_x**2 + sobel_sharpened_image_0001_y**2)

sobel_sharpened_image_01_x = cv2.Sobel(denoised_image_01, cv2.CV_64F, 1, 0, ksize=3)
sobel_sharpened_image_01_y = cv2.Sobel(denoised_image_01, cv2.CV_64F, 0, 1, ksize=3)
sobel_sharpened_image_01 = np.sqrt(sobel_sharpened_image_01_x**2 + sobel_sharpened_image_01_y**2)

sobel_sharpened_image_02_x = cv2.Sobel(denoised_image_02, cv2.CV_64F, 1, 0, ksize=3)
sobel_sharpened_image_02_y = cv2.Sobel(denoised_image_02, cv2.CV_64F, 0, 1, ksize=3)
sobel_sharpened_image_02 = np.sqrt(sobel_sharpened_image_02_x**2 + sobel_sharpened_image_02_y**2)

# Save the Sobel sharpened images to new files
cv2.imwrite('Speckel_noise_Sobel_Sharpened_density_0.0005.png', sobel_sharpened_image_0005 + denoised_image_0005)
cv2.imwrite('Speckel_noise_Sobel_Sharpened_density_0.0001.png', sobel_sharpened_image_0001 + denoised_image_0001)
cv2.imwrite('Speckel_noise_Sobel_Sharpened_density_0.01.png', sobel_sharpened_image_01 + denoised_image_01)
cv2.imwrite('Speckel_noise_Sobel_Sharpened_density_0.02.png', sobel_sharpened_image_02 + denoised_image_02)

# Apply the Roberts Cross operator
roberts_x_0005 = cv2.filter2D(denoised_image_0005, -1, np.array([[1, 0], [0, -1]]))
roberts_y_0005 = cv2.filter2D(denoised_image_0005, -1, np.array([[0, 1], [-1, 0]]))
roberts_x_0001 = cv2.filter2D(denoised_image_0001, -1, np.array([[1, 0], [0, -1]]))
roberts_y_0001 = cv2.filter2D(denoised_image_0001, -1, np.array([[0, 1], [-1, 0]]))
roberts_x_01 = cv2.filter2D(denoised_image_01, -1, np.array([[1, 0], [0, -1]]))
roberts_y_01 = cv2.filter2D(denoised_image_01, -1, np.array([[0, 1], [-1, 0]]))
roberts_x_02 = cv2.filter2D(denoised_image_02, -1, np.array([[1, 0], [0, -1]]))
roberts_y_02 = cv2.filter2D(denoised_image_02, -1, np.array([[0, 1], [-1, 0]]))

# Combine the results to get the final edges
roberts_edges_0005 = np.sqrt(np.square(roberts_x_0005) + np.square(roberts_y_0005))
roberts_edges_0001 = np.sqrt(np.square(roberts_x_0001) + np.square(roberts_y_0001))
roberts_edges_01 = np.sqrt(np.square(roberts_x_01) + np.square(roberts_y_01))
roberts_edges_02 = np.sqrt(np.square(roberts_x_02) + np.square(roberts_y_02))

# Save the Sobel sharpened images to new files
cv2.imwrite('Speckel_Noise_Denoisd_Robert_Sharpened_density_0.0005.png', roberts_edges_0005 + denoised_image_0005)
cv2.imwrite('Speckel_Noise_Denoisd_Robert_Sharpened_density_0.0001.png', roberts_edges_0001 + denoised_image_0001)
cv2.imwrite('Speckel_Noise_Denoisd_Robert_Sharpened_density_0.01.png', roberts_edges_01 + denoised_image_01)
cv2.imwrite('Speckel_Noise_Denoisd_Robert_Sharpened_density_0.02.png', roberts_edges_02 + denoised_image_02)

# Define a high-pass filter kernel (Laplacian)
kernel = np.array([[0, -1, 0],
                   [-1, 4, -1],
                   [0, -1, 0]])

# Apply the high-pass filter using convolution
high_pass_0005 = cv2.filter2D(denoised_image_0005, -1, kernel)
high_pass_0001 = cv2.filter2D(denoised_image_0001, -1, kernel)
high_pass_01 = cv2.filter2D(denoised_image_01, -1, kernel)
high_pass_02 = cv2.filter2D(denoised_image_02, -1, kernel)

# Save the Sobel sharpened images to new files
cv2.imwrite('Speckel_noise_High_pass_Sharpened_density_0.005.png', high_pass_0005 + denoised_image_0005)
cv2.imwrite('Speckel_noise_High_pass_Sharpened_density_0.0001.png', high_pass_0001 + denoised_image_0001)
cv2.imwrite('Speckel_noise_High_pass_Sharpened_density_0.01.png', high_pass_01 + denoised_image_01)
cv2.imwrite('Speckel_noise_High_pass_Sharpened_density_0.02.png', high_pass_02 + denoised_image_02)

# Convert the image to grayscale
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Calculate the high-pass component (sharpening)
high_pass = gray_image - blurred_image

# Specify the unsharp masking parameters
strength = 1.5  # Adjust this parameter to control the sharpening effect
unsharp_masked_image = cv2.addWeighted(gray_image, 1 + strength, blurred_image, -strength, 0)

# Clip values to the valid range [0, 255]
unsharp_masked_image = np.clip(unsharp_masked_image, 0, 255).astype(np.uint8)

# Save the Sobel sharpened images to new files
cv2.imwrite('Speckel_noise_unsharp_masked_image.png', unsharp_masked_image)

plt.show()

