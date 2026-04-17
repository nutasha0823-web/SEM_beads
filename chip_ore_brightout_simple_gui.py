import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os


def detect_particles_in_wells(image_path, brightness_threshold=56, brightness_upper_threshold=160, 
                             uniformity_threshold=41, min_dist=8, param1=50, param2=28, 
                             min_radius=7, max_radius=12, debug=True):
    """
    Main function to detect particles in dark wells using Hough Circles.
    """
    
    # 1. IMAGE LOADING
    # =========================
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from path: {image_path}")
    
    # Save clean original and working copy
    original_clean = img.copy()
    original_with_marks = img.copy()
    # Convert to grayscale for most OpenCV operations
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if debug:
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(original_clean, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
    
    # 2. PREPROCESSING (QUALITY IMPROVEMENT)
    # =================================================
    # Apply median filter to reduce noise while preserving edges
    blurred = cv2.medianBlur(gray, 5)
    
    # Use adaptive thresholding to highlight object boundaries
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    if debug:
        plt.subplot(2, 3, 2)
        plt.imshow(adaptive_thresh, cmap='gray')
        plt.title('After Adaptive Thresholding')
        plt.axis('off')
    
    # 3. FIND WELLS (CIRCULAR AREAS) USING HOUGH CIRCLES
    # =================================
    # Use Hough Transform to find circles
    # Parameters tuned for typical well images:
    # - dp: accumulator resolution (1.2 - good balance)
    # - minDist: minimum distance between centers
    # - param1: upper threshold for Canny edge detector
    # - param2: center detection threshold (lower = more false positives)
    # - minRadius, maxRadius: expected well size in pixels
    circles = cv2.HoughCircles(
        blurred,  # Use the preprocessed image
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=min_dist,      # Adjustable via GUI
        param1=param1,         # Adjustable via GUI
        param2=param2,         # Adjustable via GUI
        minRadius=min_radius,  # Adjustable via GUI
        maxRadius=max_radius   # Adjustable via GUI
    )
    
    results = []
    
    if circles is not None:
        # Round coordinates to integers
        circles = np.uint16(np.around(circles[0]))
        
        # Sort wells by position (top to bottom, left to right)
        circles = sorted(circles, key=lambda c: (c[1] // 50, c[0]))
        
        # Create mask to exclude area around found wells
        height, width = gray.shape
        well_mask = np.zeros((height, width), dtype=np.uint8)
        
        # 4. ANALYZE EACH WELL
        # ======================
        for i, (x, y, r) in enumerate(circles):
            # Create circular mask for current well
            circle_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(circle_mask, (x, y), r, 255, -1)
            
            # ===== OREO PROTECTION =====
            # Create inner mask (80% of radius) for analysis
            inner_r = int(r * 0.8)
            inner_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(inner_mask, (x, y), inner_r, 255, -1)
            # =============================
            
            # Add to general well mask
            well_mask = cv2.bitwise_or(well_mask, circle_mask)
            
            # 4.1 EXTRACT WELL REGION
            # Extract region of interest (ROI) - only pixels inside inner mask
            roi = cv2.bitwise_and(gray, gray, mask=inner_mask)
            roi[inner_mask == 0] = 0
            
            # 4.2 ANALYZE PARTICLES INSIDE WELL
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

                # THRESHOLDS (now adjustable via parameters)
                if mean_brightness < brightness_threshold:
                    has_particle = False
                    particle_type = "empty"
                    particle_count = 0
                elif mean_brightness > brightness_upper_threshold:
                    # Too bright - not a particle, but glare/artifact/background
                    has_particle = False
                    particle_type = "outside"  # "outside well" / artifact
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
                print(f"Well {i+1}: brightness={mean_brightness:.1f}, std={std_brightness:.1f} → {particle_type}")

            # Store results
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
            
            # 4.3 VISUALIZE RESULTS ON ORIGINAL IMAGE
            # Draw well boundaries and mark presence of particles
            # Choose color based on type
            if particle_type == "empty":
                color = (0, 0, 255)      # red
            elif particle_type == "outside":
                color = (255, 0, 255)      # magenta/purple - outside well
            elif particle_type == "debris":
                color = (0, 165, 255)    # orange
            else:
                color = (0, 255, 0)      # green

            cv2.circle(original_with_marks, (x, y), r, color, 2)
            cv2.circle(original_with_marks, (x, y), inner_r, (255, 255, 0), 1)
            cv2.circle(original_with_marks, (x, y), 2, color, 3)
            
            # Safe text placement with boundary checking
            text_x = max(int(x)-10, 5)
            text_y = max(int(y)-10, 15)
            cv2.putText(
                original_with_marks, str(i+1), (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )
        
        # Use general mask for additional statistics
        if debug:
            plt.subplot(2, 3, 3)
            plt.imshow(well_mask, cmap='gray')
            plt.title('Mask of all detected wells')
            plt.axis('off')
    
    else:
        print("Warning: Could not detect wells in image.")
        print("Try adjusting HoughCircles parameters or improving image quality.")
    
    # 5. STATISTICS AND FINAL OUTPUT
    # ===============================
    total_wells = len(results)
    wells_with_particles = sum(1 for r in results if r['has_particle'])
    wells_with_debris = sum(1 for r in results if r.get('particle_type') == 'debris')
    wells_with_outside = sum(1 for r in results if r['particle_type'] == 'outside')
    wells_with_real_particles = sum(1 for r in results if r.get('particle_type') == 'particle')
    total_particles = sum(r['particle_count'] for r in results)

    if debug:
        # Result with marked wells
        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(original_with_marks, cv2.COLOR_BGR2RGB))
        plt.title('Analysis results (green = has particle)')
        plt.axis('off')
        
        # Statistics (text)
        plt.subplot(2, 3, 5)
        plt.axis('off')
        text_str = (
            f"STATISTICS\n"
            f"Wells: {total_wells}\n"
            f"With particles: {wells_with_real_particles}\n"
            f"With debris: {wells_with_debris}\n"
            f"Outside well: {wells_with_outside}\n"
            f"Empty: {total_wells - wells_with_particles}\n"
            f"Total objects: {total_particles}"
        )
        plt.text(0.1, 0.5, text_str, fontsize=12, 
                verticalalignment='center',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        plt.title('Statistics')
        
        # Brightness histogram
        plt.subplot(2, 3, 6)
        brightnesses = [r['mean_brightness'] for r in results]
        plt.hist(brightnesses, bins=20, color='blue', alpha=0.7)
        plt.axvline(x=brightness_threshold, color='red', linestyle='--', label='Threshold')  # Same threshold as in code
        plt.xlabel('Average well brightness')
        plt.ylabel('Number of wells')
        plt.title('Brightness distribution of wells')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    return results, original_clean, original_with_marks


def print_results_summary(results):
    """Prints a summary of results to console"""
    total_wells = len(results)
    wells_with_particles = sum(1 for r in results if r['has_particle'])
    
    print("\n" + "="*50)
    print("WELL ANALYSIS RESULTS")
    print("="*50)
    print(f"Total wells detected: {total_wells}")
    print(f"Wells with particles: {wells_with_particles}")
    print(f"Empty wells: {total_wells - wells_with_particles}")
    print("\nWell details:")
    print("-" * 50)
    
    for r in results:
        status = "HAS particle" if r['has_particle'] else "empty"
        print(f"Well {r['well_id']:2d}: center {r['center']}, "
              f"radius {r['radius']:2d} - {status} "
              f"(brightness: {r['mean_brightness']:.1f})")


class ParticleDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Particle Detection in Wells - Optimized")
        
        # Default threshold values
        self.brightness_threshold = tk.DoubleVar(value=56)
        self.brightness_upper_threshold = tk.DoubleVar(value=160)
        self.uniformity_threshold = tk.DoubleVar(value=41)
        
        # Hough Circle parameters
        self.min_dist = tk.IntVar(value=8)
        self.param1 = tk.IntVar(value=50)
        self.param2 = tk.IntVar(value=28)
        self.min_radius = tk.IntVar(value=7)
        self.max_radius = tk.IntVar(value=12)
        
        # Load saved thresholds if config file exists
        self.load_config()
        
        # Initialize variables
        self.image_path = None
        self.results = []
        
        # Create GUI elements
        self.create_widgets()
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File selection
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="Image Path:").pack(side=tk.LEFT)
        self.file_entry = ttk.Entry(file_frame, width=50)
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.LEFT)
        
        # Threshold controls
        threshold_frame = ttk.LabelFrame(main_frame, text="Brightness Thresholds", padding="10")
        threshold_frame.pack(fill=tk.X, pady=10)
        
        # Lower brightness threshold
        lower_frame = ttk.Frame(threshold_frame)
        lower_frame.pack(fill=tk.X, pady=2)
        ttk.Label(lower_frame, text="Lower Brightness Threshold:").pack(side=tk.LEFT)
        ttk.Scale(lower_frame, from_=0, to=255, variable=self.brightness_threshold, 
                  orient=tk.HORIZONTAL, command=self.update_labels).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.lower_label = ttk.Label(lower_frame, text=f"{self.brightness_threshold.get()}")
        self.lower_label.pack(side=tk.RIGHT)
        
        # Upper brightness threshold
        upper_frame = ttk.Frame(threshold_frame)
        upper_frame.pack(fill=tk.X, pady=2)
        ttk.Label(upper_frame, text="Upper Brightness Threshold:").pack(side=tk.LEFT)
        ttk.Scale(upper_frame, from_=0, to=255, variable=self.brightness_upper_threshold, 
                  orient=tk.HORIZONTAL, command=self.update_labels).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.upper_label = ttk.Label(upper_frame, text=f"{self.brightness_upper_threshold.get()}")
        self.upper_label.pack(side=tk.RIGHT)
        
        # Uniformity threshold
        uniformity_frame = ttk.Frame(threshold_frame)
        uniformity_frame.pack(fill=tk.X, pady=2)
        ttk.Label(uniformity_frame, text="Uniformity Threshold:").pack(side=tk.LEFT)
        ttk.Scale(uniformity_frame, from_=0, to=100, variable=self.uniformity_threshold, 
                  orient=tk.HORIZONTAL, command=self.update_labels).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.uniformity_label = ttk.Label(uniformity_frame, text=f"{self.uniformity_threshold.get()}")
        self.uniformity_label.pack(side=tk.RIGHT)
        
        # Hough Circle parameters
        hough_frame = ttk.LabelFrame(main_frame, text="Hough Circle Parameters", padding="10")
        hough_frame.pack(fill=tk.X, pady=10)
        
        # Grid for Hough parameters
        ttk.Label(hough_frame, text="Min Distance:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Spinbox(hough_frame, from_=1, to=50, textvariable=self.min_dist, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(hough_frame, text="Param1 (Canny upper threshold):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Spinbox(hough_frame, from_=1, to=200, textvariable=self.param1, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(hough_frame, text="Param2 (Center detection threshold):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Spinbox(hough_frame, from_=1, to=100, textvariable=self.param2, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(hough_frame, text="Min Radius:").grid(row=0, column=2, sticky=tk.W, padx=20, pady=2)
        ttk.Spinbox(hough_frame, from_=1, to=50, textvariable=self.min_radius, width=10).grid(row=0, column=3, padx=5, pady=2)
        
        ttk.Label(hough_frame, text="Max Radius:").grid(row=1, column=2, sticky=tk.W, padx=20, pady=2)
        ttk.Spinbox(hough_frame, from_=1, to=100, textvariable=self.max_radius, width=10).grid(row=1, column=3, padx=5, pady=2)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Analyze Image", command=self.analyze_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Show Results", command=self.show_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Config", command=self.load_config_dialog).pack(side=tk.LEFT, padx=5)
        
        # Results text area
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.results_text = tk.Text(text_frame, height=15, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            self.image_path = file_path
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)
    
    def update_labels(self, value):
        # Update the labels to show current values
        self.lower_label.config(text=f"{int(self.brightness_threshold.get())}")
        self.upper_label.config(text=f"{int(self.brightness_upper_threshold.get())}")
        self.uniformity_label.config(text=f"{int(self.uniformity_threshold.get())}")
    
    def save_config(self):
        config = {
            'brightness_threshold': self.brightness_threshold.get(),
            'brightness_upper_threshold': self.brightness_upper_threshold.get(),
            'uniformity_threshold': self.uniformity_threshold.get(),
            'hough_params': {
                'min_dist': self.min_dist.get(),
                'param1': self.param1.get(),
                'param2': self.param2.get(),
                'min_radius': self.min_radius.get(),
                'max_radius': self.max_radius.get()
            }
        }
        
        with open('detection_config.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        messagebox.showinfo("Config Saved", "Configuration saved to detection_config.json")
    
    def load_config(self):
        config_path = 'detection_config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.brightness_threshold.set(config.get('brightness_threshold', 56))
            self.brightness_upper_threshold.set(config.get('brightness_upper_threshold', 160))
            self.uniformity_threshold.set(config.get('uniformity_threshold', 41))
            
            hough_params = config.get('hough_params', {})
            self.min_dist.set(hough_params.get('min_dist', 8))
            self.param1.set(hough_params.get('param1', 50))
            self.param2.set(hough_params.get('param2', 28))
            self.min_radius.set(hough_params.get('min_radius', 7))
            self.max_radius.set(hough_params.get('max_radius', 12))
    
    def load_config_dialog(self):
        self.load_config()
        self.update_labels(None)
        messagebox.showinfo("Config Loaded", "Configuration loaded from detection_config.json")
    
    def analyze_image(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image file first!")
            return
        
        try:
            # Get current parameter values
            brightness_thresh = self.brightness_threshold.get()
            upper_thresh = self.brightness_upper_threshold.get()
            uniformity_thresh = self.uniformity_threshold.get()
            
            hough_params = {
                'min_dist': self.min_dist.get(),
                'param1': self.param1.get(),
                'param2': self.param2.get(),
                'min_radius': self.min_radius.get(),
                'max_radius': self.max_radius.get()
            }
            
            # Run the detection with current parameters
            self.results, _, _ = detect_particles_in_wells(
                self.image_path,
                brightness_threshold=brightness_thresh,
                brightness_upper_threshold=upper_thresh,
                uniformity_threshold=uniformity_thresh,
                min_dist=hough_params['min_dist'],
                param1=hough_params['param1'],
                param2=hough_params['param2'],
                min_radius=hough_params['min_radius'],
                max_radius=hough_params['max_radius'],
                debug=False  # Don't show plots during GUI operation
            )
            
            # Display summary in text area
            self.display_results_summary()
            
            messagebox.showinfo("Success", f"Analysis completed!\nFound {len(self.results)} wells.")
        except Exception as e:
            messagebox.showerror("Error", f"Error during analysis: {str(e)}")
    
    def display_results_summary(self):
        self.results_text.delete(1.0, tk.END)
        
        if not self.results:
            self.results_text.insert(tk.END, "No results to display.\n")
            return
        
        total_wells = len(self.results)
        wells_with_particles = sum(1 for r in self.results if r['has_particle'])
        wells_with_debris = sum(1 for r in self.results if r.get('particle_type') == 'debris')
        wells_with_outside = sum(1 for r in self.results if r['particle_type'] == 'outside')
        wells_with_real_particles = sum(1 for r in self.results if r.get('particle_type') == 'particle')
        
        summary = f"""{'='*60}
PARTICLE DETECTION RESULTS
{'='*60}
Total wells detected: {total_wells}
Wells with particles: {wells_with_real_particles}
Wells with debris: {wells_with_debris}
Wells with outside artifacts: {wells_with_outside}
Empty wells: {total_wells - wells_with_particles}

Thresholds used:
- Lower brightness threshold: {self.brightness_threshold.get()}
- Upper brightness threshold: {self.brightness_upper_threshold.get()}
- Uniformity threshold: {self.uniformity_threshold.get()}

Detailed well analysis:
{'-'*60}
"""
        
        self.results_text.insert(tk.END, summary)
        
        for r in self.results:
            status = "HAS particle" if r['has_particle'] else "empty"
            details = f"Well {r['well_id']:2d}: center {r['center']}, radius {r['radius']:2d} - {status} (brightness: {r['mean_brightness']:.1f}, type: {r['particle_type']})\n"
            self.results_text.insert(tk.END, details)
    
    def show_results(self):
        if self.results:
            # Re-run the detection with debug=True to show the plot
            brightness_thresh = self.brightness_threshold.get()
            upper_thresh = self.brightness_upper_threshold.get()
            uniformity_thresh = self.uniformity_threshold.get()
            
            hough_params = {
                'min_dist': self.min_dist.get(),
                'param1': self.param1.get(),
                'param2': self.param2.get(),
                'min_radius': self.min_radius.get(),
                'max_radius': self.max_radius.get()
            }
            
            # Run the detection with debug=True to show plots
            detect_particles_in_wells(
                self.image_path,
                brightness_threshold=brightness_thresh,
                brightness_upper_threshold=upper_thresh,
                uniformity_threshold=uniformity_thresh,
                min_dist=hough_params['min_dist'],
                param1=hough_params['param1'],
                param2=hough_params['param2'],
                min_radius=hough_params['min_radius'],
                max_radius=hough_params['max_radius'],
                debug=True
            )
        else:
            messagebox.showwarning("Warning", "No results to display. Please analyze an image first.")


def main():
    root = tk.Tk()
    app = ParticleDetectionGUI(root)
    root.geometry("800x700")
    root.mainloop()


if __name__ == "__main__":
    main()