import cv2
import numpy as np
from matplotlib import pyplot as plt1
# import shearlet

def add_salt_and_pepper_noise(image, density):
    row, col, _ = image.shape
    noisy_image = np.copy(image)

    # Generate random positions for salt and pepper noise
    salt_pixels = np.random.randint(0, row - 1, int(row * col * density))
    pepper_pixels = np.random.randint(0, col - 1, int(row * col * density))

    # Add salt noise
    noisy_image[salt_pixels, pepper_pixels, :] = 255  # White

    # Add pepper noise
    noisy_image[pepper_pixels, salt_pixels, :] = 0  # Black

    return noisy_image

# Load the original image
original_image = cv2.imread('../../DS/Normal/8.bmp')

# Create a matplotlib figure for noisy images
plt1.figure(figsize=(10, 8))
plt1.suptitle('Denoised Images')

# Create a matplotlib figure for denoised images
plt1.figure(figsize=(10, 8))
plt1.suptitle('Blured Images')

# Create a matplotlib figure for blured images
plt1.figure(figsize=(10, 8))
plt1.suptitle('bilateralFilter Images')


# Create a matplotlib figure for bilateralFilter images
plt1.figure(figsize=(10, 8))
plt1.suptitle('Noisy Images')

# Display the original image
plt1.subplot(2, 3, 1), plt1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt1.title('Original Image')

# Add noise with density 0.05
noisy_image_005 = add_salt_and_pepper_noise(original_image, 0.05)

# Display the noisy image
plt1.subplot(2, 3, 2), plt1.imshow(cv2.cvtColor(noisy_image_005, cv2.COLOR_BGR2RGB))
plt1.title('Density 0.05')

# Save the noisy image to a new file
cv2.imwrite('Salt_and_Pepper_noisy_image_density_0.05.png', noisy_image_005)

# Add noise with density 0.1
noisy_image_01 = add_salt_and_pepper_noise(original_image, 0.1)

# Display the noisy image
plt1.subplot(2, 3, 3), plt1.imshow(cv2.cvtColor(noisy_image_01, cv2.COLOR_BGR2RGB))
plt1.title('Density 0.1')

# Save the noisy image to a new file
cv2.imwrite('Salt_and_Pepper_noisy_image_density_0.1.png', noisy_image_01)

# Add noise with density 0.1
noisy_image_02 = add_salt_and_pepper_noise(original_image, 0.2)

# Display the noisy image
plt1.subplot(2, 3, 4), plt1.imshow(cv2.cvtColor(noisy_image_02, cv2.COLOR_BGR2RGB))
plt1.title('Density 0.2')

# Save the noisy image to a new file
cv2.imwrite('Salt_and_Pepper_noisy_image_density_0.2.png', noisy_image_02)

# Add noise with density 0.1
noisy_image_04 = add_salt_and_pepper_noise(original_image, 0.4)

# Display the noisy image
plt1.subplot(2, 3, 5), plt1.imshow(cv2.cvtColor(noisy_image_04, cv2.COLOR_BGR2RGB))
plt1.title('Density 0.4')

# Save the noisy image to a new file
cv2.imwrite('Salt_and_Pepper_noisy_image_density_0.4.png', noisy_image_04)

# Apply median filter
denoised_image_005 = cv2.medianBlur(noisy_image_005, 5)  # The second parameter is the kernel size, which should be an odd integer
denoised_image_01 = cv2.medianBlur(noisy_image_01, 5)  # The second parameter is the kernel size, which should be an odd integer
denoised_image_02 = cv2.medianBlur(noisy_image_02, 5)  # The second parameter is the kernel size, which should be an odd integer
denoised_image_04 = cv2.medianBlur(noisy_image_04, 5)  # The second parameter is the kernel size, which should be an odd integer

# Display the denoised images
plt1.figure(1)  # Switch to the second figure
plt1.subplot(2, 3, 1), plt1.imshow(denoised_image_005, cmap='gray')
plt1.title('Denoised (Density 0.05)')
plt1.subplot(2, 3, 2), plt1.imshow(denoised_image_01, cmap='gray')
plt1.title('Denoised (Density 0.1)')
plt1.subplot(2, 3, 3), plt1.imshow(denoised_image_02, cmap='gray')
plt1.title('Denoised (Density 0.2)')
plt1.subplot(2, 3, 4), plt1.imshow(denoised_image_04, cmap='gray')
plt1.title('Denoised (Density 0.4)')

# Save the noisy image to a new file
cv2.imwrite('Salt_and_Pepper_Denoised_median_density_density_0.05.png', denoised_image_005)
cv2.imwrite('Salt_and_Pepper_Denoised_median_density_density_0.1.png', denoised_image_01)
cv2.imwrite('Salt_and_Pepper_Denoised_median_density_density_0.2.png', denoised_image_02)
cv2.imwrite('Salt_and_Pepper_Denoised_median_density_density_0.4.png', denoised_image_04)


# # Convert the image to grayscale
# gray_image_005 = cv2.cvtColor(noisy_image_005, cv2.COLOR_BGR2GRAY)
# gray_image_01 = cv2.cvtColor(noisy_image_01, cv2.COLOR_BGR2GRAY)
# gray_image_02 = cv2.cvtColor(noisy_image_02, cv2.COLOR_BGR2GRAY)
# gray_image_04 = cv2.cvtColor(noisy_image_04, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur filter
blurred_image_005 = cv2.GaussianBlur(noisy_image_005, (5, 5), 0)
blurred_image_01 = cv2.GaussianBlur(noisy_image_01, (5, 5), 0)
blurred_image_02 = cv2.GaussianBlur(noisy_image_02, (5, 5), 0)
blurred_image_04 = cv2.GaussianBlur(noisy_image_04, (5, 5), 0)

# Save the Gaussian blur filter a new file
cv2.imwrite('Salt_and_Pepper_Denoised_blur_filter_density_0.05.png', blurred_image_005)
cv2.imwrite('Salt_and_Pepper_Denoised_blur_filter_density_0.1.png', blurred_image_01)
cv2.imwrite('Salt_and_Pepper_Denoised_blur_filter_density_0.2.png', blurred_image_02)
cv2.imwrite('Salt_and_Pepper_Denoised_blur_filter_density_0.4.png', blurred_image_04)

# Display the Blurred Image
plt1.figure(2)  # Switch to the second figure
plt1.subplot(2, 3, 1), plt1.imshow(blurred_image_005, cmap='gray')
plt1.title('blurred_image (Density 0.05)')
plt1.subplot(2, 3, 2), plt1.imshow(blurred_image_01, cmap='gray')
plt1.title('blurred_image (Density 0.1)')
plt1.subplot(2, 3, 3), plt1.imshow(blurred_image_02, cmap='gray')
plt1.title('blurred_image (Density 0.02)')
plt1.subplot(2, 3, 4), plt1.imshow(blurred_image_04, cmap='gray')
plt1.title('blurred_image (Density 0.04)')

# Apply a bilateral filter
bilateralFilter_image_005 = cv2.bilateralFilter(noisy_image_005, 9, 75, 75)
bilateralFilter_image_01 = cv2.bilateralFilter(noisy_image_01, 9, 75, 75)
bilateralFilter_image_02 = cv2.bilateralFilter(noisy_image_02, 9, 75, 75)
bilateralFilter_image_04 = cv2.bilateralFilter(noisy_image_04, 9, 75, 75)

# Save the bilateral filter a new file
cv2.imwrite('Salt_and_Pepper_Denoised_bilateral_filter_density_0.05.png', bilateralFilter_image_005)
cv2.imwrite('Salt_and_Pepper_Denoised_bilateral_filter_density_0.1.png', bilateralFilter_image_01)
cv2.imwrite('Salt_and_Pepper_Denoised_bilateral_filter_density_0.2.png', bilateralFilter_image_02)
cv2.imwrite('Salt_and_Pepper_Denoised_bilateral_filter_density_0.4.png', bilateralFilter_image_04)

# Display the bilateral filter Image
plt1.figure(3)  # Switch to the second figure
plt1.subplot(2, 3, 1), plt1.imshow(bilateralFilter_image_005, cmap='gray')
plt1.title('bilateralFilter (Density 0.05)')
plt1.subplot(2, 3, 2), plt1.imshow(bilateralFilter_image_01, cmap='gray')
plt1.title('bilateralFilter (Density 0.1)')
plt1.subplot(2, 3, 3), plt1.imshow(bilateralFilter_image_02, cmap='gray')
plt1.title('bilateralFilter (Density 0.02)')
plt1.subplot(2, 3, 4), plt1.imshow(bilateralFilter_image_04, cmap='gray')
plt1.title('bilateralFilter (Density 0.04)')

# Apply sharpening filter (Laplacian) in spatial domain
sharpened_image_005 = cv2.Laplacian(denoised_image_005, cv2.CV_64F)
sharpened_image_005 = np.uint8(np.absolute(sharpened_image_005))

sharpened_image_01 = cv2.Laplacian(denoised_image_01, cv2.CV_64F)
sharpened_image_01 = np.uint8(np.absolute(sharpened_image_01))

sharpened_image_02 = cv2.Laplacian(denoised_image_02, cv2.CV_64F)
sharpened_image_02 = np.uint8(np.absolute(sharpened_image_02))

sharpened_image_04 = cv2.Laplacian(denoised_image_04, cv2.CV_64F)
sharpened_image_04 = np.uint8(np.absolute(sharpened_image_04))

# Save the sharpened images to new files
cv2.imwrite('Salt_and_Pepper_Sharpened_laplacian_density_0.05.png', sharpened_image_005 + denoised_image_005)
cv2.imwrite('Salt_and_Pepper_Sharpened_laplacian_density_0.1.png', sharpened_image_01 + denoised_image_01)
cv2.imwrite('Salt_and_Pepper_Sharpened_laplacian_density_0.2.png', sharpened_image_02 + denoised_image_02)
cv2.imwrite('Salt_and_Pepper_Sharpened_laplacian_density_0.4.png', sharpened_image_04 + denoised_image_04)

# Apply Sobel filter as a sharpening filter in spatial domain
sobel_sharpened_image_005_x = cv2.Sobel(denoised_image_005, cv2.CV_64F, 1, 0, ksize=3)
sobel_sharpened_image_005_y = cv2.Sobel(denoised_image_005, cv2.CV_64F, 0, 1, ksize=3)
sobel_sharpened_image_005 = np.sqrt(sobel_sharpened_image_005_x**2 + sobel_sharpened_image_005_y**2)

sobel_sharpened_image_01_x = cv2.Sobel(denoised_image_01, cv2.CV_64F, 1, 0, ksize=3)
sobel_sharpened_image_01_y = cv2.Sobel(denoised_image_01, cv2.CV_64F, 0, 1, ksize=3)
sobel_sharpened_image_01 = np.sqrt(sobel_sharpened_image_01_x**2 + sobel_sharpened_image_01_y**2)

sobel_sharpened_image_02_x = cv2.Sobel(denoised_image_02, cv2.CV_64F, 1, 0, ksize=3)
sobel_sharpened_image_02_y = cv2.Sobel(denoised_image_02, cv2.CV_64F, 0, 1, ksize=3)
sobel_sharpened_image_02 = np.sqrt(sobel_sharpened_image_02_x**2 + sobel_sharpened_image_02_y**2)

sobel_sharpened_image_04_x = cv2.Sobel(denoised_image_04, cv2.CV_64F, 1, 0, ksize=3)
sobel_sharpened_image_04_y = cv2.Sobel(denoised_image_04, cv2.CV_64F, 0, 1, ksize=3)
sobel_sharpened_image_04 = np.sqrt(sobel_sharpened_image_04_x**2 + sobel_sharpened_image_04_y**2)

# Save the Sobel sharpened images to new files
cv2.imwrite('Salt_and_Pepper_Sobel_Sharpened_density_0.05.png', sobel_sharpened_image_005 + denoised_image_005)
cv2.imwrite('Salt_and_Pepper_Sobel_Sharpened_density_0.1.png', sobel_sharpened_image_01 + denoised_image_01)
cv2.imwrite('Salt_and_Pepper_Sobel_Sharpened_density_0.2.png', sobel_sharpened_image_02 + denoised_image_02)
cv2.imwrite('Salt_and_Pepper_Sobel_Sharpened_density_0.4.png', sobel_sharpened_image_04 + denoised_image_04)

# Apply the Roberts Cross operator
roberts_x_005 = cv2.filter2D(denoised_image_005, -1, np.array([[1, 0], [0, -1]]))
roberts_y_005 = cv2.filter2D(denoised_image_005, -1, np.array([[0, 1], [-1, 0]]))
roberts_x_01 = cv2.filter2D(denoised_image_01, -1, np.array([[1, 0], [0, -1]]))
roberts_y_01 = cv2.filter2D(denoised_image_01, -1, np.array([[0, 1], [-1, 0]]))
roberts_x_02 = cv2.filter2D(denoised_image_02, -1, np.array([[1, 0], [0, -1]]))
roberts_y_02 = cv2.filter2D(denoised_image_02, -1, np.array([[0, 1], [-1, 0]]))
roberts_x_04 = cv2.filter2D(denoised_image_04, -1, np.array([[1, 0], [0, -1]]))
roberts_y_04 = cv2.filter2D(denoised_image_04, -1, np.array([[0, 1], [-1, 0]]))

# Combine the results to get the final edges
roberts_edges_005 = np.sqrt(np.square(roberts_x_005) + np.square(roberts_y_005))
roberts_edges_01 = np.sqrt(np.square(roberts_x_01) + np.square(roberts_y_01))
roberts_edges_02 = np.sqrt(np.square(roberts_x_02) + np.square(roberts_y_02))
roberts_edges_04 = np.sqrt(np.square(roberts_x_04) + np.square(roberts_y_04))

# Save the Sobel sharpened images to new files
cv2.imwrite('Salt_and_Pepper_Robert_Sharpened_density_0.005.png', roberts_edges_005 + denoised_image_005)
cv2.imwrite('Salt_and_Pepper_Robert_Sharpened_density_0.01.png', roberts_edges_01 + denoised_image_01)
cv2.imwrite('Salt_and_Pepper_Robert_Sharpened_density_0.02.png', roberts_edges_02 + denoised_image_02)
cv2.imwrite('Salt_and_Pepper_Robert_Sharpened_density_0.04.png', roberts_edges_04 + denoised_image_04)

# Define a high-pass filter kernel (Laplacian)
kernel = np.array([[0, -1, 0],
                   [-1, 4, -1],
                   [0, -1, 0]])

# Apply the high-pass filter using convolution
high_pass_005 = cv2.filter2D(denoised_image_005, -1, kernel)
high_pass_01 = cv2.filter2D(denoised_image_01, -1, kernel)
high_pass_02 = cv2.filter2D(denoised_image_02, -1, kernel)
high_pass_04 = cv2.filter2D(denoised_image_04, -1, kernel)

# Save the Sobel sharpened images to new files
cv2.imwrite('Salt_and_Pepper_High_pass_Sharpened_density_0.005.png', high_pass_005 + denoised_image_005)
cv2.imwrite('Salt_and_Pepper_High_pass_Sharpened_density_0.01.png', high_pass_01 + denoised_image_01)
cv2.imwrite('Salt_and_Pepper_High_pass_Sharpened_density_0.02.png', high_pass_02 + denoised_image_02)
cv2.imwrite('Salt_and_Pepper_High_pass_Sharpened_density_0.04.png', high_pass_04 + denoised_image_04)

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
cv2.imwrite('Salt_and_Pepper_unsharp_masked_image.png', unsharp_masked_image)

# Show the plots
plt1.show()
