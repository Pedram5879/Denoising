import cv2
import numpy as np

def ssim(image_original, image_compressed):
    # تصاویر را به آرایه‌های numpy تبدیل می‌کنیم
    img_original = np.array(image_original, dtype=np.float64)
    img_compressed = np.array(image_compressed, dtype=np.float64)

    # مقادیر مشخصات
    K1 = 0.01
    K2 = 0.03
    L = 255.0

    # محاسبه میانگین
    mu_x = np.mean(img_original)
    mu_y = np.mean(img_compressed)

    # محاسبه واریانس
    var_x = np.var(img_original)
    var_y = np.var(img_compressed)

    # محاسبه کوواریانس
    cov_xy = np.cov(img_original.reshape(-1), img_compressed.reshape(-1))[0, 1]

    # محاسبه SSIM
    c1 = (K1 * L) ** 2
    c2 = (K2 * L) ** 2
    ssim_value = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))

    return ssim_value

# نمونه استفاده از تابع
image_original = cv2.imread('./orginal.png')
image_compressed = cv2.imread('../../Speckle_Noise/speckel_noise_Denoised_blur_filter_density_0.001.png')

ssim_result = ssim(image_original, image_compressed)
print(f'SSIM: {ssim_result:.4f}')
