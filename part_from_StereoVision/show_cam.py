#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Захват видео с двух камер и сохранение кадров для калибровки
Адаптировано из StereoVision/bin/show_webcams
"""

import cv2
import numpy as np
import os
import time
import argparse
from datetime import datetime

class StereoCameraCapture:
    """Класс для захвата видео с двух камер"""
    
    def __init__(self, left_camera_id=0, right_camera_id=1):
        """
        Инициализация двух камер
        
        Args:
            left_camera_id: ID левой камеры (обычно 0)
            right_camera_id: ID правой камеры (обычно 1)
        """
        self.left_camera = cv2.VideoCapture(left_camera_id)
        self.right_camera = cv2.VideoCapture(right_camera_id)
        
        # Проверка подключения
        if not self.left_camera.isOpened():
            raise ValueError(f"Не удалось открыть левую камеру (ID: {left_camera_id})")
        if not self.right_camera.isOpened():
            raise ValueError(f"Не удалось открыть правую камеру (ID: {right_camera_id})")
        
        # Установка одинакового разрешения для обеих камер (опционально)
        self.set_resolution(640, 480)
        
        print(f"✓ Камеры инициализированы: левая (ID:{left_camera_id}), правая (ID:{right_camera_id})")
    
    def set_resolution(self, width, height):
        """Установка разрешения для обеих камер"""
        self.left_camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.left_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.right_camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.right_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Проверка установленного разрешения
        left_width = int(self.left_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        left_height = int(self.left_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  Левая камера: {left_width}x{left_height}")
        
        right_width = int(self.right_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        right_height = int(self.right_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  Правая камера: {right_width}x{right_height}")
    
    def get_frames(self):
        """Получение пары кадров с обеих камер"""
        ret_left, frame_left = self.left_camera.read()
        ret_right, frame_right = self.right_camera.read()
        
        if not ret_left or not ret_right:
            return None, None
        
        return frame_left, frame_right
    
    def show_videos(self, save_folder=None, interval=1.0):
        """
        Отображение видео с возможностью сохранения кадров
        
        Args:
            save_folder: папка для сохранения (если None, только показ)
            interval: интервал между сохранениями в секундах
        """
        if save_folder and not os.path.exists(save_folder):
            os.makedirs(save_folder)
            print(f"✓ Создана папка для сохранения: {save_folder}")
        
        print("\n" + "="*50)
        print("УПРАВЛЕНИЕ:")
        print("  'q' или ESC - выход")
        print("  's' - сохранить текущую пару")
        if save_folder:
            print(f"  Автосохранение каждые {interval} сек")
        print("="*50 + "\n")
        
        frame_count = 0
        last_save_time = time.time()
        
        while True:
            # Захват кадров
            left_frame, right_frame = self.get_frames()
            if left_frame is None or right_frame is None:
                print("Ошибка захвата кадров")
                break
            
            # Создание комбинированного изображения для отображения
            h, w = left_frame.shape[:2]
            combined = np.hstack((left_frame, right_frame))
            
            # Добавление разделительной линии
            cv2.line(combined, (w, 0), (w, h), (255, 255, 255), 2)
            
            # Добавление информации на экран
            cv2.putText(combined, "LEFT", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, "RIGHT", (w + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, f"Frames saved: {frame_count}", (10, h-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Отображение
            cv2.imshow("Stereo Camera Capture", combined)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' или ESC
                break
            elif key == ord('s'):  # 's' - ручное сохранение
                self.save_frames(left_frame, right_frame, frame_count, save_folder)
                frame_count += 1
            
            # Автосохранение по интервалу
            if save_folder and (time.time() - last_save_time) > interval:
                self.save_frames(left_frame, right_frame, frame_count, save_folder)
                frame_count += 1
                last_save_time = time.time()
        
        # Освобождение ресурсов
        self.left_camera.release()
        self.right_camera.release()
        cv2.destroyAllWindows()
        print(f"\n✓ Захват завершен. Сохранено {frame_count} пар")
    
    def save_frames(self, left_frame, right_frame, index, folder):
        """Сохранение пары кадров"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        left_filename = os.path.join(folder, f"left_{timestamp}_{index:04d}.png")
        right_filename = os.path.join(folder, f"right_{timestamp}_{index:04d}.png")
        
        cv2.imwrite(left_filename, left_frame)
        cv2.imwrite(right_filename, right_frame)
        
        print(f"  ✓ Сохранена пара {index}: {left_filename}, {right_filename}")
    
    def __del__(self):
        """Деструктор для освобождения камер"""
        if hasattr(self, 'left_camera'):
            self.left_camera.release()
        if hasattr(self, 'right_camera'):
            self.right_camera.release()


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description="Захват стереопары для калибровки")
    parser.add_argument("--left", type=int, default=0, help="ID левой камеры")
    parser.add_argument("--right", type=int, default=1, help="ID правой камеры")
    parser.add_argument("--save_folder", default="calibration_images", 
                       help="Папка для сохранения кадров")
    parser.add_argument("--interval", type=float, default=2.0,
                       help="Интервал автосохранения (сек)")
    parser.add_argument("--width", type=int, default=640, help="Ширина кадра")
    parser.add_argument("--height", type=int, default=480, help="Высота кадра")
    
    args = parser.parse_args()
    
    try:
        # Инициализация захвата
        capture = StereoCameraCapture(args.left, args.right)
        capture.set_resolution(args.width, args.height)
        
        # Запуск отображения с сохранением
        capture.show_videos(args.save_folder, args.interval)
        
    except ValueError as e:
        print(f"Ошибка: {e}")
        return
    except KeyboardInterrupt:
        print("\nПрервано пользователем")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")


if __name__ == "__main__":
    main()