import cv2
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed


def _analyze_single_well(args):
    """
    Анализ одной лунки. Выполняется в отдельном потоке.
    Возвращает (index, result_dict) для последующей сортировки.
    """
    index, x, y, r, gray, height, width, debug = args

    # Круговая маска (нужна только для статистики в debug — считаем локально)
    # Внутренняя маска (80% радиуса)
    inner_r = int(r * 0.8)
    inner_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(inner_mask, (x, y), inner_r, 255, -1)

    # ROI — только пиксели внутри внутренней маски
    roi = cv2.bitwise_and(gray, gray, mask=inner_mask)
    roi[inner_mask == 0] = 0

    well_pixels = roi[roi > 0]

    if len(well_pixels) == 0:
        has_particle = False
        mean_brightness = 0
        std_brightness = 0
        particle_count = 0
        particle_type = "empty"
    else:
        mean_brightness = float(np.mean(well_pixels))
        std_brightness = float(np.std(well_pixels))

        brightness_threshold = 45
        brightness_upper_threshold = 185
        uniformity_threshold = 57

        if mean_brightness < brightness_threshold:
            has_particle = False
            particle_type = "empty"
            particle_count = 0
        elif mean_brightness > brightness_upper_threshold:
            has_particle = False
            particle_type = "outside"
            particle_count = 0
        elif std_brightness > uniformity_threshold:
            has_particle = False
            particle_type = "debris"
            particle_count = 1
        else:
            has_particle = True
            particle_type = "particle"
            particle_count = 1

    if debug:
        print(f"Лунка {index + 1}: ярк={mean_brightness:.1f}, "
              f"std={std_brightness:.1f} → {particle_type}")

    result = {
        'well_id': index + 1,
        'center': (int(x), int(y)),
        'radius': int(r),
        'inner_r': inner_r,
        'has_particle': has_particle,
        'particle_count': particle_count,
        'mean_brightness': mean_brightness,
        'particle_type': particle_type,
        'std_brightness': std_brightness,
    }
    return index, result

def detect_particles_in_wells(image_path, debug=False, max_workers=None):
    """
    Основная функция для обнаружения частиц в тёмных лунках.
    max_workers: число потоков (None — авто по числу CPU).
    """
    # 1. Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение по пути: {image_path}")

    original_clean = img.copy()
    original_with_marks = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if debug:
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(original_clean, cv2.COLOR_BGR2RGB))
        plt.title('Исходное изображение')
        plt.axis('off')

    # 2. Предобработка
    blurred = cv2.medianBlur(gray, 5)
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    if debug:
        plt.subplot(2, 3, 2)
        plt.imshow(adaptive_thresh, cmap='gray')
        plt.title('После адаптивной бинаризации')
        plt.axis('off')

    # 3. Поиск лунок
    inverted = cv2.bitwise_not(gray)
    blurred_inv = cv2.medianBlur(inverted, 5)

    circles = cv2.HoughCircles(
        blurred_inv,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=20,
        param1=8,
        param2=23,
        minRadius=7,
        maxRadius=12
    )

    results = []
    height, width = gray.shape

    if circles is not None:
        print('Найдено кругов:', len(circles[0]))
        circles = np.uint16(np.around(circles[0]))
        circles = sorted(circles, key=lambda c: (c[1] // 50, c[0]))

        # ---- МНОГОПОТОЧНЫЙ АНАЛИЗ ЛУНОК ----
        tasks = [
            (i, int(x), int(y), int(r), gray, height, width, debug)
            for i, (x, y, r) in enumerate(circles)
        ]

        indexed_results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_analyze_single_well, t) for t in tasks]
            for fut in as_completed(futures):
                idx, res = fut.result()
                indexed_results[idx] = res

        # Восстанавливаем порядок лунок
        results = [indexed_results[i] for i in range(len(circles))]

        # ---- ВИЗУАЛИЗАЦИЯ (в главном потоке, изображение общее) ----
        well_mask = np.zeros((height, width), dtype=np.uint8)
        for r in results:
            x, y = r['center']
            radius = r['radius']
            inner_r = r['inner_r']
            particle_type = r['particle_type']
            i = r['well_id'] - 1

            # Маска лунки (для debug)
            cv2.circle(well_mask, (x, y), radius, 255, -1)

            # Цвет по типу
            if particle_type == "empty":
                color = (0, 0, 255)
            elif particle_type == "outside":
                color = (255, 0, 255)
            elif particle_type == "debris":
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)

            cv2.circle(original_with_marks, (x, y), radius, color, 2)
            cv2.circle(original_with_marks, (x, y), inner_r, (255, 255, 0), 1)
            cv2.circle(original_with_marks, (x, y), 2, color, 3)

            text_x = max(x - 10, 5)
            text_y = max(y - 10, 15)
            cv2.putText(
                original_with_marks, str(i + 1), (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )

        if debug:
            plt.subplot(2, 3, 3)
            plt.imshow(well_mask, cmap='gray')
            plt.title('Маска всех обнаруженных лунок')
            plt.axis('off')
    else:
        print("Внимание: не удалось обнаружить лунки на изображении.")
        print("Попробуйте настроить параметры HoughCircles или улучшить качество изображения.")

    # 5. Статистика
    total_wells = len(results)
    wells_with_particles = sum(1 for r in results if r['has_particle'])
    wells_with_debris = sum(1 for r in results if r.get('particle_type') == 'debris')
    wells_with_outside = sum(1 for r in results if r['particle_type'] == 'outside')
    wells_with_real_particles = sum(1 for r in results if r.get('particle_type') == 'particle')
    total_particles = sum(r['particle_count'] for r in results)

    if debug:
        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(original_with_marks, cv2.COLOR_BGR2RGB))
        plt.title('Результаты анализа (зелёный - есть частица)')
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.axis('off')
        text_str = (
            f"Кол-во лунок: {total_wells}\n"
            f"С частицами: {wells_with_real_particles}\n"
            f"С мусором: {wells_with_debris}\n"
            f"Вне лунки: {wells_with_outside}\n"
            f"Пустых: {total_wells - wells_with_particles}\n"
            f"Всего объектов: {total_particles}"
        )
        plt.text(0.1, 0.5, text_str, fontsize=12,
                 verticalalignment='center',
                 family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        plt.title('Статистика')

        plt.subplot(2, 3, 6)
        brightnesses = [r['mean_brightness'] for r in results]
        plt.hist(brightnesses, bins=20, color='blue', alpha=0.7)
        plt.axvline(x=75, color='red', linestyle='--', label='Порог')
        plt.xlabel('Средняя яркость лунки')
        plt.ylabel('Количество лунок')
        plt.title('Распределение яркости лунок')
        plt.legend()

        plt.tight_layout()
        plt.show()

    return results, original_clean, original_with_marks

def print_results_summary(results):
    """Выводит сводку результатов в консоль"""
    total_wells = len(results)
    wells_with_particles = sum(1 for r in results if r['has_particle'])
    
    print("Результаты анализа лунок")
    print(f"Всего обнаружено лунок: {total_wells}")
    print(f"Лунок с частицами: {wells_with_particles}")
    print(f"Пустых лунок: {total_wells - wells_with_particles}")
    print("\nДетали по лункам:")
    
    for r in results[-10:]:
        status = "Есть частица" if r['has_particle'] else "пусто"
        print(f"Лунка {r['well_id']:2d}: центр {r['center']}, "
              f"радиус {r['radius']:2d} - {status} "
              f"(яркость: {r['mean_brightness']:.1f})")
# Путь к изображению или тест использования на синтетическом
if __name__ == "__main__":
    # Путь к изображению (изменить на актуальный)
    image_path = r"C:\chip1_clean1st.jpg"
    
    # Для тестирования создадим синтетическое изображение,
    # если нет реального файла
    import os
    if not os.path.exists(image_path):
        print(f"Файл {image_path} не найден. Создаю тестовое изображение...")
        
        # Создаём тестовое изображение с лунками
        test_img = np.ones((400, 600, 3), dtype=np.uint8) * 30  # Тёмный фон
        
        # Рисуем несколько лунок
        wells_pos = [(150, 150), (300, 150), (450, 150),
                     (150, 300), (300, 300), (450, 300)]
        
        for i, (x, y) in enumerate(wells_pos):
            # Тёмная лунка
            cv2.circle(test_img, (x, y), 40, (20, 20, 20), -1)
            cv2.circle(test_img, (x, y), 40, (100, 100, 100), 1)
            
            # В некоторые лунки добавляем "частицы" (светлые точки)
            if i % 2 == 0:  # чётные лунки будут с частицами
                cv2.circle(test_img, (x-10, y-10), 5, (220, 220, 220), -1)
                cv2.circle(test_img, (x+15, y+5), 7, (200, 200, 200), -1)
                cv2.circle(test_img, (x-5, y+20), 4, (240, 240, 240), -1)
        
        cv2.imwrite(image_path, test_img)
        print(f"Тестовое изображение сохранено как {image_path}")
    
    # Запускаем анализ
    results, _clean, annotated_img = detect_particles_in_wells(image_path, debug=False)
    print_results_summary(results)

    output_path = "wells_analysis_result.jpg"
    cv2.imwrite(output_path, annotated_img)
    print(f"\nРезультат сохранён в {output_path}")
