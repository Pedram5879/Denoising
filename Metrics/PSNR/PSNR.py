import cv2
import numpy as np
import math

def psnr(image_original, image_compressed):
    # تصاویر را به آرایه‌های numpy تبدیل می‌کنیم
    img_original = np.array(image_original, dtype=np.float64)
    img_compressed = np.array(image_compressed, dtype=np.float64)

    # محدوده پیکسل‌ها را از 0 تا 255 به [-128, 127] محدود می‌کنیم
    img_original = img_original - 128.0
    img_compressed = img_compressed - 128.0

    # محاسبه میانگین مربعات اختلافات
    mse = np.mean((img_original - img_compressed) ** 2)

    # اگر MSE صفر باشد، PSNR نامعتبر است (تقسیم بر صفر)
    if mse == 0:
        return float('inf')

    # محاسبه PSNR
    max_pixel = 255.0
    psnr_value = 10 * math.log10((max_pixel ** 2) / mse)
    
    return psnr_value

# نمونه استفاده از تابع
image_original = cv2.imread('./orginal.bmp')
image_compressed = cv2.imread('../../Salt_and_Pepper/Salt_and_Pepper_Denoised_blur_filter_density_0.1.png')

psnr_result = psnr(image_original, image_compressed)
print(f'PSNR: {psnr_result:.2f} dB')
