import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_poisson_noise(image, scale):
    # Generate Poisson noise with the same shape as the image
    noise = np.random.poisson(image * scale) / scale
    
    # Add Poisson noise to the image
    noisy = image + noise
    
    # Clip pixel values to the range [0, 255]
    noisy = np.clip(noisy, 0, 255)
    
    # Convert the noisy image to the data type of the original image (uint8)
    noisy = noisy.astype(np.uint8)
    
    return noisy


# Read the image
image_path = "../../DS/NORMAL/12.bmp"
original_image = cv2.imread(image_path)

# Display the original image
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Apply noise with different scales
scales = [2, 1, 0.5, 0.1]

for i, scale in enumerate(scales, start=2):
    # Add Poisson noise to the image
    noisy_image = add_poisson_noise(original_image, scale)
    
    # Display the noisy image
    plt.subplot(2, 3, i)
    plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
    plt.title(f'(Scale={scale})')

    # Save the noisy image to a new file
    cv2.imwrite(f'Poisson_noisy_image_scale_{scale}.png', noisy_image)

# Denoise the image
denoised_image_2 = cv2.medianBlur(add_poisson_noise(original_image, 1.5), 5)  # The second parameter is the kernel size, which should be an odd integer
denoised_image_1 = cv2.medianBlur(add_poisson_noise(original_image, 1), 5)  # The second parameter is the kernel size, which should be an odd integer
denoised_image_05 = cv2.medianBlur(add_poisson_noise(original_image, 0.5), 5)  # The second parameter is the kernel size, which should be an odd integer
denoised_image_01 = cv2.medianBlur(add_poisson_noise(original_image, 0.1), 5)  # The second parameter is the kernel size, which should be an odd integer

# Save the noisy image to a new file
cv2.imwrite('Poisson_noise_Denoised_median_density_2.png', denoised_image_2)
cv2.imwrite('Poisson_noise_Denoised_median_density_1.png', denoised_image_1)
cv2.imwrite('Poisson_noise_Denoised_median_density_0.5.png', denoised_image_05)
cv2.imwrite('Poisson_noise_Denoised_median_density_0.1.png', denoised_image_01)

# Apply a Gaussian blur filter
blurred_image_2 = cv2.GaussianBlur(add_poisson_noise(original_image, 2), (5, 5), 0)
blurred_image_1 = cv2.GaussianBlur(add_poisson_noise(original_image, 1), (5, 5), 0)
blurred_image_05 = cv2.GaussianBlur(add_poisson_noise(original_image, 0.5), (5, 5), 0)
blurred_image_01 = cv2.GaussianBlur(add_poisson_noise(original_image, 0.1), (5, 5), 0)

# Save the Gaussian blur filter a new file
cv2.imwrite('Poisson_noise_Denoised_blur_filter_density_0.05.png', blurred_image_2)
cv2.imwrite('Poisson_noise_Denoised_blur_filter_density_0.1.png', blurred_image_1)
cv2.imwrite('Poisson_noise_Denoised_blur_filter_density_0.2.png', blurred_image_05)
cv2.imwrite('Poisson_noise_Denoised_blur_filter_density_0.4.png', blurred_image_01)

# Apply a bilateral filter
bilateralFilter_image_2 = cv2.bilateralFilter(add_poisson_noise(original_image, 2), 9, 75, 75)
bilateralFilter_image_1 = cv2.bilateralFilter(add_poisson_noise(original_image, 1), 9, 75, 75)
bilateralFilter_image_05 = cv2.bilateralFilter(add_poisson_noise(original_image, 0.5), 9, 75, 75)
bilateralFilter_image_01 = cv2.bilateralFilter(add_poisson_noise(original_image, 0.1), 9, 75, 75)

# Save the bilateral filter a new file
cv2.imwrite('Poisson_noise_Denoised_bilateral_filter_density_0.05.png', bilateralFilter_image_2)
cv2.imwrite('Poisson_noise_Denoised_bilateral_filter_density_0.1.png', bilateralFilter_image_1)
cv2.imwrite('Poisson_noise_Denoised_bilateral_filter_density_0.2.png', bilateralFilter_image_05)
cv2.imwrite('Poisson_noise_Denoised_bilateral_filter_density_0.4.png', bilateralFilter_image_01)

# Apply sharpening filter (Laplacian) in spatial domain
sharpened_image_2 = cv2.Laplacian(denoised_image_2, cv2.CV_64F)
sharpened_image_2 = np.uint8(np.absolute(sharpened_image_2))

sharpened_image_1 = cv2.Laplacian(denoised_image_1, cv2.CV_64F)
sharpened_image_1 = np.uint8(np.absolute(sharpened_image_1))

sharpened_image_05 = cv2.Laplacian(denoised_image_05, cv2.CV_64F)
sharpened_image_05 = np.uint8(np.absolute(sharpened_image_05))

sharpened_image_01 = cv2.Laplacian(denoised_image_01, cv2.CV_64F)
sharpened_image_01 = np.uint8(np.absolute(sharpened_image_01))

# Save the sharpened images to new files
cv2.imwrite('Poisson_Sharpened_laplacian_filter_density_2.png', sharpened_image_2 + denoised_image_2)
cv2.imwrite('Poisson_Sharpened_laplacian_filter_density_1.png', sharpened_image_1 + denoised_image_1)
cv2.imwrite('Poisson_Sharpened_laplacian_filter_density_0.5.png', sharpened_image_05 + denoised_image_05)
cv2.imwrite('Poisson_Sharpened_laplacian_filter_density_0.1.png', sharpened_image_01 + denoised_image_01)

# Apply Sobel filter as a sharpening filter in spatial domain
sobel_sharpened_image_2_x = cv2.Sobel(denoised_image_2, cv2.CV_64F, 1, 0, ksize=3)
sobel_sharpened_image_2_y = cv2.Sobel(denoised_image_2, cv2.CV_64F, 0, 1, ksize=3)
# sobel_sharpened_image_2 = np.sqrt(sobel_sharpened_image_2_x**2 + sobel_sharpened_image_2_y**2)
sobel_sharpened_image_2 = np.uint8(np.absolute(sobel_sharpened_image_2_x) + np.absolute(sobel_sharpened_image_2_y))

sobel_sharpened_image_1_x = cv2.Sobel(denoised_image_1, cv2.CV_64F, 1, 0, ksize=3)
sobel_sharpened_image_1_y = cv2.Sobel(denoised_image_1, cv2.CV_64F, 0, 1, ksize=3)
# sobel_sharpened_image_1 = np.sqrt(sobel_sharpened_image_1_x**2 + sobel_sharpened_image_1_y**2)
sobel_sharpened_image_1 = np.uint8(np.absolute(sobel_sharpened_image_1_x) + np.absolute(sobel_sharpened_image_1_y))

sobel_sharpened_image_05_x = cv2.Sobel(denoised_image_05, cv2.CV_64F, 1, 0, ksize=3)
sobel_sharpened_image_05_y = cv2.Sobel(denoised_image_05, cv2.CV_64F, 0, 1, ksize=3)
# sobel_sharpened_image_05 = np.sqrt(sobel_sharpened_image_05_x**2 + sobel_sharpened_image_05_y**2)
sobel_sharpened_image_05 = np.uint8(np.absolute(sobel_sharpened_image_05_x) + np.absolute(sobel_sharpened_image_05_y))

sobel_sharpened_image_01_x = cv2.Sobel(denoised_image_01, cv2.CV_64F, 1, 0, ksize=3)
sobel_sharpened_image_01_y = cv2.Sobel(denoised_image_01, cv2.CV_64F, 0, 1, ksize=3)
# sobel_sharpened_image_01 = np.sqrt(sobel_sharpened_image_01_x**2 + sobel_sharpened_image_01_y**2)
sobel_sharpened_image_01 = np.uint8(np.absolute(sobel_sharpened_image_01_x) + np.absolute(sobel_sharpened_image_01_y))

# Save the Sobel sharpened images to new files
cv2.imwrite('Poisson_noise_Sobel_Sharpened_density_2.png', sobel_sharpened_image_2 + denoised_image_2)
cv2.imwrite('Poisson_noise_Sobel_Sharpened_density_1.png', sobel_sharpened_image_1 + denoised_image_1)
cv2.imwrite('Poisson_noise_Sobel_Sharpened_density_0.5.png', sobel_sharpened_image_05 + denoised_image_05)
cv2.imwrite('Poisson_noise_Sobel_Sharpened_density_0.1.png', sobel_sharpened_image_01 + denoised_image_01)

# Apply the Roberts Cross operator
roberts_x_2 = cv2.filter2D(denoised_image_2, -1, np.array([[1, 0], [0, -1]]))
roberts_y_2 = cv2.filter2D(denoised_image_2, -1, np.array([[0, 1], [-1, 0]]))
roberts_x_1 = cv2.filter2D(denoised_image_1, -1, np.array([[1, 0], [0, -1]]))
roberts_y_1 = cv2.filter2D(denoised_image_1, -1, np.array([[0, 1], [-1, 0]]))
roberts_x_01 = cv2.filter2D(denoised_image_01, -1, np.array([[1, 0], [0, -1]]))
roberts_y_01 = cv2.filter2D(denoised_image_01, -1, np.array([[0, 1], [-1, 0]]))
roberts_x_05 = cv2.filter2D(denoised_image_05, -1, np.array([[1, 0], [0, -1]]))
roberts_y_05 = cv2.filter2D(denoised_image_05, -1, np.array([[0, 1], [-1, 0]]))

# Combine the results to get the final edges
roberts_edges_2 = np.sqrt(np.square(roberts_x_2) + np.square(roberts_y_2))
roberts_edges_1 = np.sqrt(np.square(roberts_x_1) + np.square(roberts_y_1))
roberts_edges_01 = np.sqrt(np.square(roberts_x_01) + np.square(roberts_y_01))
roberts_edges_05 = np.sqrt(np.square(roberts_x_05) + np.square(roberts_y_05))

# Save the Sobel sharpened images to new files
cv2.imwrite('Poisson_noise_Robert_Sharpened_density_0.005.png', roberts_edges_2 + denoised_image_2)
cv2.imwrite('Poisson_noise_Robert_Sharpened_density_0.01.png', roberts_edges_1 + denoised_image_1)
cv2.imwrite('Poisson_noise_Robert_Sharpened_density_0.02.png', roberts_edges_01 + denoised_image_01)
cv2.imwrite('Poisson_noise_Robert_Sharpened_density_0.04.png', roberts_edges_05 + denoised_image_05)

# Define a high-pass filter kernel (Laplacian)
kernel = np.array([[0, -1, 0],
                   [-1, 4, -1],
                   [0, -1, 0]])

# Apply the high-pass filter using convolution
high_pass_05 = cv2.filter2D(denoised_image_05, -1, kernel)
high_pass_01 = cv2.filter2D(denoised_image_01, -1, kernel)
high_pass_1 = cv2.filter2D(denoised_image_1, -1, kernel)
high_pass_2 = cv2.filter2D(denoised_image_2, -1, kernel)

# Save the Sobel sharpened images to new files
cv2.imwrite('Poisson_noise_High_pass_Sharpened_density_0.05.png', high_pass_05 + denoised_image_05)
cv2.imwrite('Poisson_noise_High_pass_Sharpened_density_0.01.png', high_pass_01 + denoised_image_01)
cv2.imwrite('Poisson_noise_High_pass_Sharpened_density_0.1.png', high_pass_1 + denoised_image_1)
cv2.imwrite('Poisson_noise_High_pass_Sharpened_density_0.2.png', high_pass_2 + denoised_image_2)

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
cv2.imwrite('Poisson_noise_unsharp_masked_image.png', unsharp_masked_image)

plt.show()
