"""
Модуль обработки изображений для лабораторной работы №1
Реализует: фильтрацию, построение гистограмм и анализ шума
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import os

# ==========================================
# ФИЛЬТРАЦИЯ ИЗОБРАЖЕНИЙ
# ==========================================

def process_image(input_path, output_path, kernel_size, sigma, filter_type='gaussian', add_timestamp=False):
    """
    Основная функция обработки изображения
    Параметры:
    - kernel_size: размер ядра (нечётное число)
    - sigma: стандартное отклонение для Гаусса
    - filter_type: 'gaussian' или 'bilateral'
    - add_timestamp: добавлять ли дату и время
    """
    # Читаем изображение
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {input_path}")
    
    # Применяем выбранный фильтр
    if filter_type == 'gaussian':
        # Фильтр Гаусса - хорош для общего шумоподавления
        processed = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    elif filter_type == 'bilateral':
        # Билатеральный фильтр - сохраняет края (сигма цвета = сигма пространства)
        processed = cv2.bilateralFilter(img, kernel_size, sigma*5, sigma*5)
    else:
        # По умолчанию - медианный (запасной вариант)
        processed = cv2.medianBlur(img, kernel_size)
    
    # Добавляем временную метку, если нужно
    if add_timestamp:
        processed = add_timestamp_to_image_array(processed)
    
    # Сохраняем результат
    cv2.imwrite(output_path, processed)
    return output_path


def add_timestamp_to_image_array(img_array):
    """
    Накладывает дату/время на изображение (работает с numpy array)
    """
    # Конвертируем BGR в RGB для PIL
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    draw = ImageDraw.Draw(pil_img)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Пытаемся загрузить шрифт
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
        except:
            font = ImageFont.load_default()
    
    # Размер текста
    try:
        bbox = draw.textbbox((0, 0), timestamp, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except:
        text_width = len(timestamp) * 16
        text_height = 30
    
    # Позиция (правый нижний угол)
    margin = 15
    x = pil_img.width - text_width - margin
    y = pil_img.height - text_height - margin
    
    # Полупрозрачный фон
    padding = 10
    draw.rectangle(
        [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
        fill=(0, 0, 0, 180)
    )
    draw.text((x, y), timestamp, fill=(255, 255, 255), font=font)
    
    # Конвертируем обратно в BGR для OpenCV
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return result

# ==========================================
# ПОСТРОЕНИЕ ГИСТОГРАММ (СРАВНИТЕЛЬНЫЙ РЕЖИМ)
# ==========================================

def plot_histograms_comparison(original_path, processed_path, output_path):
    """
    Строит СРАВНИТЕЛЬНЫЙ график гистограмм:
    - Сплошные линии = оригинал
    - Пунктирные линии = после обработки
    Это наглядно показывает изменения в цветовых каналах
    """
    # Загружаем изображения
    img_orig = cv2.imread(original_path)
    img_proc = cv2.imread(processed_path)
    
    # Конвертируем в RGB
    img_orig_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img_proc_rgb = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
    
    # Цвета каналов
    colors = ['red', 'green', 'blue']
    channels = [0, 1, 2]
    
    plt.figure(figsize=(12, 7))
    
    for channel, color in zip(channels, colors):
        # Гистограмма исходного
        hist_orig = cv2.calcHist([img_orig_rgb], [channel], None, [256], [0, 256])
        # Гистограмма обработанного
        hist_proc = cv2.calcHist([img_proc_rgb], [channel], None, [256], [0, 256])
        
        # Рисуем сплошной линией (оригинал) и пунктиром (обработанное)
        plt.plot(hist_orig, color=color, linestyle='-', linewidth=2, 
                 label=f'{color.upper()} (оригинал)')
        plt.plot(hist_proc, color=color, linestyle='--', linewidth=2, 
                 label=f'{color.upper()} (обработано)')
    
    plt.title('Сравнительный анализ гистограмм цветовых каналов', fontsize=14)
    plt.xlabel('Интенсивность пикселей (0-255)', fontsize=12)
    plt.ylabel('Количество пикселей', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

# ==========================================
# ГРАФИК РАСПРЕДЕЛЕНИЯ ШУМА
# ==========================================

def plot_noise_analysis(original_path, processed_path, output_path):
    """
    Анализ шума: разница между изображениями
    Показывает, какие пиксели были изменены больше всего
    """
    img_orig = cv2.imread(original_path)
    img_proc = cv2.imread(processed_path)
    
    # Вычисляем разницу (шум)
    noise = cv2.absdiff(img_orig, img_proc)
    # Переводим в градации серого
    noise_gray = cv2.cvtColor(noise, cv2.COLOR_BGR2GRAY)
    
    # Статистика шума
    mean_noise = np.mean(noise_gray)
    std_noise = np.std(noise_gray)
    
    # Строим гистограмму разницы
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(noise_gray.ravel(), bins=40, range=(0, 255), 
                                color='#FF6B6B', alpha=0.75, edgecolor='white', linewidth=0.5)
    
    # Добавляем вертикальные линии для среднего и медианы
    plt.axvline(mean_noise, color='blue', linestyle='--', linewidth=2, 
                label=f'Среднее: {mean_noise:.2f}')
    plt.axvline(np.median(noise_gray), color='green', linestyle='--', linewidth=2, 
                label=f'Медиана: {np.median(noise_gray):.2f}')
    
    plt.title(f'Распределение шума после фильтрации\n(среднее = {mean_noise:.2f}, σ = {std_noise:.2f})', 
              fontsize=14)
    plt.xlabel('Интенсивность шума', fontsize=12)
    plt.ylabel('Количество пикселей', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
