import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean, std_dev):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, std_dev, (row, col, ch))
    noisy = np.clip(image + gauss, 0, 255)
    return noisy.astype(np.uint8)

# Read the image
image_path = "../../DS/Med/1.png"
original_image = cv2.imread(image_path)

# Display the original image
plt.subplot(3, 4, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Orginal Image')

# Gaussian noise parameters
gaussian_noises = [(0, 40), (4, 30), (10, 0), (10, 25)]

# Apply noise with different parameters
for i, (mean, std_dev) in enumerate(gaussian_noises, start=2):
    # Add Gaussian noise to the image
    noisy_image = add_gaussian_noise(original_image, mean, std_dev)
    
    # Display the noisy image
    plt.subplot(3, 4, i)
    plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
    plt.title(f'(M={mean}, S-D={std_dev})')
    # Save the noisy image to a new file
    cv2.imwrite(f'Gussian_noisy_image_mean_{mean}_and_Standard-deviation_{std_dev}.png', noisy_image)

    # Apply filtering with different filter sizes
    filter_sizes = [3, 5, 7, 9]
    for j, filter_size in enumerate(filter_sizes, start=1):
        # Blur the noisy image with the current filter size
        filtered_image = cv2.GaussianBlur(noisy_image, (filter_size, filter_size), 0)
        
        # Display the filtered image
        plt.subplot(3, 4, i + j)
        plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
        plt.title(f'F-S={filter_size}')
        # Save the noisy image to a new file
        cv2.imwrite(f'Gussian_noisy_image_with_filter_size={filter_size}.png', noisy_image)

# Denoise the image
denoised_image_040 = cv2.medianBlur(add_gaussian_noise(original_image, 0, 40), 5)  # The second parameter is the kernel size, which should be an odd integer
denoised_image_4030 = cv2.medianBlur(add_gaussian_noise(original_image, 40, 30), 5)  # The second parameter is the kernel size, which should be an odd integer
denoised_image_100 = cv2.medianBlur(add_gaussian_noise(original_image, 10, 0), 5)  # The second parameter is the kernel size, which should be an odd integer
denoised_image_1025 = cv2.medianBlur(add_gaussian_noise(original_image, 10, 25), 5)  # The second parameter is the kernel size, which should be an odd integer

# Save the noisy image to a new file
cv2.imwrite('Gaussian_noise_Denoised_median_filter_mean_0_and_Standard-deviation_40.png', denoised_image_040)
cv2.imwrite('Gaussian_noise_Denoised_median_filter_mean_40_and_Standard-deviation_30.png', denoised_image_4030)
cv2.imwrite('Gaussian_noise_Denoised_median_filter_mean_10_and_Standard-deviation_0.png', denoised_image_100)
cv2.imwrite('Gaussian_noise_Denoised_median_filter_mean_10_and_Standard-deviation_25.png', denoised_image_1025)

# Apply a Gaussian blur filter
blurred_image_040 = cv2.GaussianBlur(add_gaussian_noise(original_image, 0, 40), (5, 5), 0)
blurred_image_4030 = cv2.GaussianBlur(add_gaussian_noise(original_image, 40, 30), (5, 5), 0)
blurred_image_100 = cv2.GaussianBlur(add_gaussian_noise(original_image, 10, 0), (5, 5), 0)
blurred_image_1025 = cv2.GaussianBlur(add_gaussian_noise(original_image, 10, 25), (5, 5), 0)

# Save the Gaussian blur filter a new file
cv2.imwrite('Gaussian_noise_Denoised_blur_filter_mean_0_and_Standard-deviation_40.png', blurred_image_040)
cv2.imwrite('Gaussian_noise_Denoised_blur_filter_mean_40_and_Standard-deviation_30.png', blurred_image_4030)
cv2.imwrite('Gaussian_noise_Denoised_blur_filter_mean_10_and_Standard-deviation_0.png', blurred_image_100)
cv2.imwrite('Gaussian_noise_Denoised_blur_filter_mean_10_and_Standard-deviation_25.png', blurred_image_1025)

# Apply a bilateral filter
bilateralFilter_image_040 = cv2.bilateralFilter(add_gaussian_noise(original_image, 0, 40), 9, 75, 75)
bilateralFilter_image_4030 = cv2.bilateralFilter(add_gaussian_noise(original_image, 40, 30), 9, 75, 75)
bilateralFilter_image_100 = cv2.bilateralFilter(add_gaussian_noise(original_image, 10, 0), 9, 75, 75)
bilateralFilter_image_1025 = cv2.bilateralFilter(add_gaussian_noise(original_image, 10, 25), 9, 75, 75)

# Save the bilateral filter a new file
cv2.imwrite('Gaussian_noise_Denoised_bilateral_filter_mean_0_and_Standard-deviation_40.png', bilateralFilter_image_040)
cv2.imwrite('Gaussian_noise_Denoised_bilateral_filter_mean_40_and_Standard-deviation_30.png', bilateralFilter_image_4030)
cv2.imwrite('Gaussian_noise_Denoised_bilateral_filter_mean_10_and_Standard-deviation_0.png', bilateralFilter_image_100)
cv2.imwrite('Gaussian_noise_Denoised_bilateral_filter_mean_10_and_Standard-deviation_25.png', bilateralFilter_image_1025)

# Apply sharpening filter (Laplacian) in spatial domain
sharpened_image_040 = cv2.Laplacian(denoised_image_040, cv2.CV_64F)
sharpened_image_040 = np.uint8(np.absolute(sharpened_image_040))

sharpened_image_4030 = cv2.Laplacian(denoised_image_4030, cv2.CV_64F)
sharpened_image_4030 = np.uint8(np.absolute(sharpened_image_4030))

sharpened_image_100 = cv2.Laplacian(denoised_image_100, cv2.CV_64F)
sharpened_image_100 = np.uint8(np.absolute(sharpened_image_100))

sharpened_image_1025 = cv2.Laplacian(denoised_image_1025, cv2.CV_64F)
sharpened_image_1025 = np.uint8(np.absolute(sharpened_image_1025))

# Save the sharpened images to new files
cv2.imwrite('Gaussian_noise_Denoised_laplacian_sharpened_mean_0_and_Standard-deviation_40.png', sharpened_image_040 + denoised_image_040)
cv2.imwrite('Gaussian_noise_Denoised_laplacian_sharpened_mean_40_and_Standard-deviation_30.png', sharpened_image_4030 + denoised_image_4030)
cv2.imwrite('Gaussian_noise_Denoised_laplacian_sharpened_mean_10_and_Standard-deviation_0.png', sharpened_image_100 + denoised_image_100)
cv2.imwrite('Gaussian_noise_Denoised_laplacian_sharpened_mean_10_and_Standard-deviation_25.png', sharpened_image_1025 + denoised_image_1025)

# Apply Sobel filter as a sharpening filter in spatial domain
sobel_sharpened_image_040_x = cv2.Sobel(denoised_image_040, cv2.CV_64F, 1, 0, ksize=3)
sobel_sharpened_image_040_y = cv2.Sobel(denoised_image_040, cv2.CV_64F, 0, 1, ksize=3)
sobel_sharpened_image_040 = np.sqrt(sobel_sharpened_image_040_x**2 + sobel_sharpened_image_040_y**2)
# sobel_sharpened_image_040 = np.uint8(np.absolute(sobel_sharpened_image_040_x) + np.absolute(sobel_sharpened_image_040_y))

sobel_sharpened_image_4030_x = cv2.Sobel(denoised_image_4030, cv2.CV_64F, 1, 0, ksize=3)
sobel_sharpened_image_4030_y = cv2.Sobel(denoised_image_4030, cv2.CV_64F, 0, 1, ksize=3)
sobel_sharpened_image_4030 = np.sqrt(sobel_sharpened_image_4030_x**2 + sobel_sharpened_image_4030_y**2)
# sobel_sharpened_image_4030 = np.uint8(np.absolute(sobel_sharpened_image_4030_x) + np.absolute(sobel_sharpened_image_4030_y))

sobel_sharpened_image_100_x = cv2.Sobel(denoised_image_100, cv2.CV_64F, 1, 0, ksize=3)
sobel_sharpened_image_100_y = cv2.Sobel(denoised_image_100, cv2.CV_64F, 0, 1, ksize=3)
sobel_sharpened_image_100 = np.sqrt(sobel_sharpened_image_100_x**2 + sobel_sharpened_image_100_y**2)
# sobel_sharpened_image_100 = np.uint8(np.absolute(sobel_sharpened_image_100_x) + np.absolute(sobel_sharpened_image_100_y))

sobel_sharpened_image_1025_x = cv2.Sobel(denoised_image_1025, cv2.CV_64F, 1, 0, ksize=3)
sobel_sharpened_image_1025_y = cv2.Sobel(denoised_image_1025, cv2.CV_64F, 0, 1, ksize=3)
sobel_sharpened_image_1025 = np.sqrt(sobel_sharpened_image_1025_x**2 + sobel_sharpened_image_1025_y**2)
# sobel_sharpened_image_1025 = np.uint8(np.absolute(sobel_sharpened_image_1025_x) + np.absolute(sobel_sharpened_image_1025_y))

# Save the Sobel sharpened images to new files
cv2.imwrite('Poisson_noise_Sobel_Sharpened_density_040.png', sobel_sharpened_image_040 + denoised_image_040)
cv2.imwrite('Poisson_noise_Sobel_Sharpened_density_4030.png', sobel_sharpened_image_4030 + denoised_image_4030)
cv2.imwrite('Poisson_noise_Sobel_Sharpened_density_100.png', sobel_sharpened_image_100 + denoised_image_100)
cv2.imwrite('Poisson_noise_Sobel_Sharpened_density_1025.png', sobel_sharpened_image_1025 + denoised_image_1025)

# Apply the Roberts Cross operator
roberts_x_4030 = cv2.filter2D(denoised_image_4030, -1, np.array([[1, 0], [0, -1]]))
roberts_y_4030 = cv2.filter2D(denoised_image_4030, -1, np.array([[0, 1], [-1, 0]]))
roberts_x_1025 = cv2.filter2D(denoised_image_1025, -1, np.array([[1, 0], [0, -1]]))
roberts_y_1025 = cv2.filter2D(denoised_image_1025, -1, np.array([[0, 1], [-1, 0]]))
roberts_x_100 = cv2.filter2D(denoised_image_100, -1, np.array([[1, 0], [0, -1]]))
roberts_y_100 = cv2.filter2D(denoised_image_100, -1, np.array([[0, 1], [-1, 0]]))
roberts_x_040 = cv2.filter2D(denoised_image_040, -1, np.array([[1, 0], [0, -1]]))
roberts_y_040 = cv2.filter2D(denoised_image_040, -1, np.array([[0, 1], [-1, 0]]))

# Combine the results to get the final edges
roberts_edges_4030 = np.sqrt(np.square(roberts_x_4030) + np.square(roberts_y_4030))
roberts_edges_1025 = np.sqrt(np.square(roberts_x_1025) + np.square(roberts_y_1025))
roberts_edges_100 = np.sqrt(np.square(roberts_x_100) + np.square(roberts_y_100))
roberts_edges_040 = np.sqrt(np.square(roberts_x_040) + np.square(roberts_y_040))

# Save the Sobel sharpened images to new files
cv2.imwrite('gussian_noise_Robert_Sharpened_density_4030.png', roberts_edges_4030 + denoised_image_4030)
cv2.imwrite('gussian_noise_Robert_Sharpened_density_1025.png', roberts_edges_1025 + denoised_image_1025)
cv2.imwrite('gussian_noise_Robert_Sharpened_density_100.png', roberts_edges_100 + denoised_image_100)
cv2.imwrite('gussian_noise_Robert_Sharpened_density_040.png', roberts_edges_040 + denoised_image_040)

# Define a high-pass filter kernel (Laplacian)
kernel = np.array([[0, -1, 0],
                   [-1, 4, -1],
                   [0, -1, 0]])

# Apply the high-pass filter using convolution
high_pass_4030 = cv2.filter2D(denoised_image_4030, -1, kernel)
high_pass_040 = cv2.filter2D(denoised_image_040, -1, kernel)
high_pass_100 = cv2.filter2D(denoised_image_100, -1, kernel)
high_pass_1025 = cv2.filter2D(denoised_image_1025, -1, kernel)

# Save the Sobel sharpened images to new files
cv2.imwrite('Gaussian_noise_High_pass_Sharpened_density_0.05.png', high_pass_4030 + denoised_image_4030)
cv2.imwrite('Gaussian_noise_High_pass_Sharpened_density_0.01.png', high_pass_040 + denoised_image_040)
cv2.imwrite('Gaussian_noise_High_pass_Sharpened_density_0.1.png', high_pass_100 + denoised_image_100)
cv2.imwrite('Gaussian_noise_High_pass_Sharpened_density_0.2.png', high_pass_1025 + denoised_image_1025)

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
cv2.imwrite('Gaussian_noise_unsharp_masked_image.png', unsharp_masked_image)

plt.show()
