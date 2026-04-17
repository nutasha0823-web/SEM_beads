import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_particles_in_wells(image_path, debug=True):
    """
    Основная функция для обнаружения частиц в тёмных лунках.
    """
    
    # 1. ЗАГРУЗКА ИЗОБРАЖЕНИЯ
    # =========================
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение по пути: {image_path}")
    
    # Сохраняем чистое исходное и рабочую копию
    original_clean = img.copy()
    original_with_marks = img.copy()
    # Переводим в оттенки серого для большинства операций OpenCV
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if debug:
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(original_clean, cv2.COLOR_BGR2RGB))
        plt.title('Исходное изображение')
        plt.axis('off')
    
    # 2. ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА (УЛУЧШЕНИЕ КАЧЕСТВА)
    # =================================================
    # Применяем медианный фильтр для уменьшения шума, сохраняя границы
    blurred = cv2.medianBlur(gray, 5)
    
    # Используем адаптивную бинаризацию, чтобы выделить границы объектов
    # Это поможет алгоритму поиска кругов лучше видеть края лунок
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    if debug:
        plt.subplot(2, 3, 2)
        plt.imshow(adaptive_thresh, cmap='gray')
        plt.title('После адаптивной бинаризации')
        plt.axis('off')
    
   # 3. ПОИСК ЛУНОК ПО БЕЛЫМ ОКРУЖНОСТЯМ
   # ====================================

    # Усиливаем белые круги (вычитаем фон или инвертируем)
    # Инвертируем, чтобы белые круги стали тёмными (если нужно для HoughCircles)
    inverted = cv2.bitwise_not(gray)  # белое → чёрное, чёрное → белое

    # ИЛИ более точно: выделяем светлые структуры на тёмном фоне
    # (если фон серый, а круги светлые — оставляем как есть, не инвертируем)

    # Применим медианный фильтр к инвертированному или исходному
    blurred_inv = cv2.medianBlur(inverted, 5)  # или просто blurred, если не инвертируешь

    # Поиск кругов по светлым (в оригинале) объектам
    # 3. ПОИСК ЛУНОК (КРУГЛЫХ ОБЛАСТЕЙ)
    # =================================
    # Используем преобразование Хафа для поиска кругов
    # Параметры подобраны для типичных изображений лунок:
    # - dp: разрешение накопителя (1.2 - хороший баланс)
    # - minDist: минимальное расстояние между центрами лунок
    # - param1: верхний порог для детектора границ Canny
    # - param2: порог центра круга (чем меньше, тем больше ложных срабатываний)
    # - minRadius, maxRadius: ожидаемый размер лунок в пикселях
    circles = cv2.HoughCircles(
        blurred_inv,  # или blurred, если круги и так тёмные
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=8,
        param1=50,
        param2=28,
        minRadius=7,
        maxRadius=12
    )
    
    results = []
    
    if circles is not None:
        # Округляем координаты до целых чисел
        circles = np.uint16(np.around(circles[0]))
        
        # Сортируем лунки по положению (сверху вниз, слева направо)
        circles = sorted(circles, key=lambda c: (c[1] // 50, c[0]))
        
        # Создаём маску для исключения области вокруг найденных лунок
        height, width = gray.shape
        well_mask = np.zeros((height, width), dtype=np.uint8)
        
        # 4. АНАЛИЗ КАЖДОЙ ЛУНКИ
        # ======================
        for i, (x, y, r) in enumerate(circles):
            # Создаём круговую маску для текущей лунки
            circle_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(circle_mask, (x, y), r, 255, -1)
            
            # ===== ЗАЩИТА ОТ ОРЕОЛОВ =====
            # Создаём внутреннюю маску (80% от радиуса) для анализа
            inner_r = int(r * 0.8)
            inner_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(inner_mask, (x, y), inner_r, 255, -1)
            # =============================
            
            # Добавляем в общую маску лунок
            well_mask = cv2.bitwise_or(well_mask, circle_mask)
            
            # 4.1 ВЫДЕЛЕНИЕ ОБЛАСТИ ЛУНКИ
            # Извлекаем область интереса (ROI) - только пиксели внутри внутренней маски
            roi = cv2.bitwise_and(gray, gray, mask=inner_mask)
            roi[inner_mask == 0] = 0
            
            # 4.2 АНАЛИЗ ЧАСТИЦ ВНУТРИ ЛУНКИ
            well_pixels = roi[roi > 0]

            if len(well_pixels) == 0:
             has_particle = False
             mean_brightness = 0
             std_brightness = 0
             particle_count = 0
             particle_type = "empty"
            else:
                mean_brightness = np.mean(well_pixels)
                std_brightness = np.std(well_pixels)

                # ПОРОГИ (подбери под свои картинки)
                brightness_threshold = 56
                brightness_upper_threshold = 160 # верхний порог (частица vs вне лунки)
                uniformity_threshold = 41

                if mean_brightness < brightness_threshold:
                    has_particle = False
                    particle_type = "empty"
                    particle_count = 0
                elif mean_brightness > brightness_upper_threshold:
                    # Слишком ярко - это не частица, а блик/артефакт/фон
                    has_particle = False  # или True, если хочешь их считать
                    particle_type = "outside"  # "вне лунки" / артефакт
                    particle_count = 0
                elif std_brightness > uniformity_threshold:
                    has_particle = False   # или False, если мусор не считаешь
                    particle_type = "debris"
                    particle_count = 1
                else:
                    has_particle = True
                    particle_type = "particle"
                    particle_count = 1

            if debug:
                print(f"Лунка {i+1}: ярк={mean_brightness:.1f}, std={std_brightness:.1f} → {particle_type}")

# Сохраняем результат (СТРОКИ 110-130)
            results.append({
    'well_id': i+1,
    'center': (x, y),
    'radius': r,
    'has_particle': has_particle,
    'particle_count': particle_count,
    'mean_brightness': mean_brightness,
    'particle_type': particle_type,
    'std_brightness': std_brightness
})
            
            # 4.3 ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ НА ИСХОДНОМ ИЗОБРАЖЕНИИ
            # Рисуем границы лунок и отмечаем наличие частиц
           # Выбираем цвет в зависимости от типа
            if particle_type == "empty":
               color = (0, 0, 255)      # красный
            elif particle_type == "outside":
                color = (255, 0, 255)      # малиновый/пурпурный - вне лунки
            elif particle_type == "debris":
                color = (0, 165, 255)    # оранжевый
            else:
                color = (0, 255, 0)      # зелёный

            cv2.circle(original_with_marks, (x, y), r, color, 2)
            cv2.circle(original_with_marks, (x, y), inner_r, (255, 255, 0), 1)
            cv2.circle(original_with_marks, (x, y), 2, color, 3)
            
            # Безопасное размещение текста с проверкой границ
            text_x = max(x-10, 5)
            text_y = max(y-10, 15)
            cv2.putText(
                original_with_marks, str(i+1), (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )
        
        # Используем общую маску для дополнительной статистики
        if debug:
            plt.subplot(2, 3, 3)
            plt.imshow(well_mask, cmap='gray')
            plt.title('Маска всех обнаруженных лунок')
            plt.axis('off')
    
    else:
        print("Внимание: не удалось обнаружить лунки на изображении.")
        print("Попробуйте настроить параметры HoughCircles или улучшить качество изображения.")
    
    # 5. СТАТИСТИКА И ФИНАЛЬНЫЙ ВЫВОД
    # ===============================
    total_wells = len(results)
    wells_with_particles = sum(1 for r in results if r['has_particle'])
    wells_with_debris = sum(1 for r in results if r.get('particle_type') == 'debris')
    wells_with_outside = sum(1 for r in results if r['particle_type'] == 'outside')
    wells_with_real_particles = sum(1 for r in results if r.get('particle_type') == 'particle')
    total_particles = sum(r['particle_count'] for r in results)

    if debug:
        # Результат с обведёнными лунками
        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(original_with_marks, cv2.COLOR_BGR2RGB))
        plt.title('Результаты анализа (зелёный - есть частица)')
        plt.axis('off')
        
        # Статистика (текст)
        plt.subplot(2, 3, 5)
        plt.axis('off')
        text_str = (
            f"СТАТИСТИКА\n"
            f"Лунок: {total_wells}\n"
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
        
        # Гистограмма яркостей
        plt.subplot(2, 3, 6)
        brightnesses = [r['mean_brightness'] for r in results]
        plt.hist(brightnesses, bins=20, color='blue', alpha=0.7)
        plt.axvline(x=56, color='red', linestyle='--', label='Порог')  # Тот же порог, что в коде
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
    
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ АНАЛИЗА ЛУНОК")
    print("="*50)
    print(f"Всего обнаружено лунок: {total_wells}")
    print(f"Лунок с частицами: {wells_with_particles}")
    print(f"Пустых лунок: {total_wells - wells_with_particles}")
    print("\nДетали по лункам:")
    print("-" * 50)
    
    for r in results:
        status = "ЕСТЬ частица" if r['has_particle'] else "пусто"
        print(f"Лунка {r['well_id']:2d}: центр {r['center']}, "
              f"радиус {r['radius']:2d} - {status} "
              f"(яркость: {r['mean_brightness']:.1f})")
# =====================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =====================================================================
if __name__ == "__main__":
    # Путь к вашему изображению (измените на актуальный)
    image_path = r"C:\Users\nutas\chip_waffle_1_151NP_1.jpg"  # Замените на путь к вашему файлу
    
    # Для тестирования создадим синтетическое изображение,
    # если у вас нет реального файла
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
    try:
        results, annotated_img = detect_particles_in_wells(image_path, debug=True)
        print_results_summary(results)
        
        # Сохраняем результат
        output_path = "wells_analysis_result.jpg"
        cv2.imwrite(output_path, annotated_img)
        print(f"\nРезультат сохранён в {output_path}")
        
    except Exception as e:
        print(f"Ошибка при анализе: {e}")
