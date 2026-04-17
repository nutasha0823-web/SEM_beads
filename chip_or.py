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
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.5, 
        minDist=8,
        param1=50, 
        param2=30, 
        minRadius=8, 
        maxRadius=13
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
            # ===============================
            
            # Получаем все пиксели внутри лунки
            well_pixels = roi[roi > 0]
            
            if len(well_pixels) == 0:
                has_particle = False
                mean_brightness = 0
                particle_count = 0
            else:
                # Вычисляем среднюю яркость лунки
                mean_brightness = np.mean(well_pixels)
                
                # ПОРОГ ЯРКОСТИ - НАСТРАИВАЙТЕ ЗДЕСЬ
                brightness_threshold = 70# <--- ИЗМЕНЯЙТЕ ЭТО ЧИСЛО (0-255)
                
                # Если средняя яркость выше порога - в лунке есть частица
                has_particle = mean_brightness > brightness_threshold
                particle_count = 1 if has_particle else 0
                
                # Для отладки - выводим яркость каждой лунки
                if debug:
                    print(f"Лунка {i+1}: средняя яркость = {mean_brightness:.1f} - {'ЕСТЬ' if has_particle else 'НЕТ'} частица")
            
            # Сохраняем результат
            results.append({
                'well_id': i+1,
                'center': (x, y),
                'radius': r,
                'has_particle': has_particle,
                'particle_count': particle_count,
                'mean_brightness': mean_brightness
            })
            
            # 4.3 ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ НА ИСХОДНОМ ИЗОБРАЖЕНИИ
            # Рисуем границы лунок и отмечаем наличие частиц
            color = (0, 255, 0) if has_particle else (0, 0, 255)  # Зелёный если есть частица, красный если пусто
            cv2.circle(original_with_marks, (x, y), r, color, 2)      # Внешняя граница лунки
            cv2.circle(original_with_marks, (x, y), inner_r, (255, 255, 0), 1)  # Внутренняя область (жёлтая)
            cv2.circle(original_with_marks, (x, y), 1, color, 3)      # Центр лунки
            
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
            f"СТАТИСТИКА АНАЛИЗА\n"
            f"{'='*20}\n"
            f"Всего лунок: {total_wells}\n"
            f"Лунок с частицами: {wells_with_particles}\n"
            f"Пустых лунок: {total_wells - wells_with_particles}\n"
            f"Всего частиц: {total_particles}\n"
            f"{'='*20}"
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
        plt.axvline(x=62.5, color='red', linestyle='--', label='Порог')  # Тот же порог, что в коде
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
    image_path = r"C:\Users\nutas\SDS_0.5%_3.leftup.jpg"  # Замените на путь к вашему файлу
    
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
