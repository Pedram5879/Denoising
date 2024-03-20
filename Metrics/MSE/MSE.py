import cv2
import numpy as np

def mse(image_original, image_compressed):
    # تصاویر را به آرایه‌های numpy تبدیل می‌کنیم
    img_original = np.array(image_original, dtype=np.float64)
    img_compressed = np.array(image_compressed, dtype=np.float64)

    # محاسبه میانگین مربعات اختلافات
    mse_value = np.mean((img_original - img_compressed) ** 2)
    
    return mse_value

# نمونه استفاده از تابع
image_original = cv2.imread('./orginal.png')
image_compressed = cv2.imread('../../Gaussian_noise/Gaussian_noise_Denoised_bilateral_filter_mean_0_and_Standard-deviation_40.png')

mse_result = mse(image_original, image_compressed)
print(f'MSE: {mse_result:.2f}')
