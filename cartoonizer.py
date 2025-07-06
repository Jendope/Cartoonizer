import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import threading
from tkinter import font as tkfont

class CartoonizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Cartoonizer Pro")
        self.root.geometry("1100x750")
        self.root.minsize(1000, 700)
        
        # Custom colors
        self.bg_color = "#f0f0f0"
        self.primary_color = "#4a6fa5"
        self.secondary_color = "#6c757d"
        self.accent_color = "#ff7e5f"
        
        # Configure styles first
        self.configure_styles()
        
        # Variables
        self.image_path = None
        self.original_image = None
        self.cartoon_image = None
        self.k_value = tk.IntVar(value=6)
        self.show_edges = tk.BooleanVar(value=False)
        self.edge_thickness = tk.IntVar(value=2)
        
        # Create UI
        self.create_widgets()
        
    def configure_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('.', background=self.bg_color)
        style.configure('TFrame', background=self.bg_color)
        style.configure('TLabel', background=self.bg_color)
        style.configure('TButton', background=self.secondary_color, foreground='white')
        style.configure('Accent.TButton', background=self.accent_color, foreground='white')
        style.configure('TLabelFrame', background=self.bg_color)
        style.configure('TLabelFrame.Label', background=self.bg_color)
        style.configure('TScale', background=self.bg_color)
        style.configure('TCheckbutton', background=self.bg_color)
        
        # Custom styles
        style.map('Accent.TButton',
                 background=[('active', self.accent_color), ('pressed', '#e06c4d')])
        style.map('TButton',
                 background=[('active', '#5a6268'), ('pressed', '#495057')])
        
    def create_widgets(self):
        # Set main window background
        self.root.configure(bg=self.bg_color)
        
        # Header frame
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        # App title
        title_font = tkfont.Font(family='Helvetica', size=20, weight='bold')
        ttk.Label(header_frame, text="IMAGE CARTOONIZER PRO", 
                 font=title_font, foreground=self.primary_color).pack(side=tk.LEFT)
        
        # Version label
        ttk.Label(header_frame, text="v1.0", foreground=self.secondary_color).pack(side=tk.RIGHT)
        
        # Main content frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="CONTROLS", padding=(15, 10))
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # File selection
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Button(file_frame, text="📁 Select Image", command=self.load_image, 
                  style='Accent.TButton').pack(fill=tk.X)
        
        # Parameters frame
        params_frame = ttk.Frame(control_frame)
        params_frame.pack(fill=tk.X, pady=5)
        
        # K-means parameter
        k_frame = ttk.Frame(params_frame)
        k_frame.pack(fill=tk.X, pady=5)
        ttk.Label(k_frame, text="Color Regions (K):").pack(anchor=tk.W)
        ttk.Scale(k_frame, from_=3, to=10, variable=self.k_value, 
                 command=lambda v: self.update_k_label()).pack(fill=tk.X)
        self.k_label = ttk.Label(k_frame, text=f"K: {self.k_value.get()}")
        self.k_label.pack(anchor=tk.W)
        
        # Edge thickness parameter
        edge_frame = ttk.Frame(params_frame)
        edge_frame.pack(fill=tk.X, pady=5)
        ttk.Label(edge_frame, text="Edge Thickness:").pack(anchor=tk.W)
        ttk.Scale(edge_frame, from_=1, to=5, variable=self.edge_thickness).pack(fill=tk.X)
        
        # Edge display checkbox
        ttk.Checkbutton(params_frame, text="Show Edge Detection", variable=self.show_edges,
                       command=self.toggle_edge_display).pack(pady=10, anchor=tk.W)
        
        # Process button
        ttk.Button(control_frame, text="🎨 Cartoonize!", command=self.process_image_threaded, 
                  style="Accent.TButton").pack(fill=tk.X, pady=10)
        
        # Save button
        ttk.Button(control_frame, text="💾 Save Result", command=self.save_image).pack(fill=tk.X)
        
        # Right panel - Image display
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Original image frame
        orig_img_frame = ttk.LabelFrame(image_frame, text="ORIGINAL IMAGE", padding=10)
        orig_img_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.original_label = ttk.Label(orig_img_frame, 
                                      text="No image selected\n\nClick 'Select Image' to begin",
                                      anchor=tk.CENTER)
        self.original_label.pack(fill=tk.BOTH, expand=True)
        
        # Cartoon image frame
        cartoon_img_frame = ttk.LabelFrame(image_frame, text="CARTOONIZED RESULT", padding=10)
        cartoon_img_frame.pack(fill=tk.BOTH, expand=True)
        
        self.cartoon_label = ttk.Label(cartoon_img_frame, 
                                      text="Cartoonized result will appear here",
                                      anchor=tk.CENTER)
        self.cartoon_label.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, padding=5)
        self.status.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(0, 10))
        
        # Add some padding to all widgets
        for child in control_frame.winfo_children():
            if isinstance(child, ttk.Frame):
                for subchild in child.winfo_children():
                    subchild.pack_configure(padx=5, pady=2)
    
    def update_k_label(self):
        self.k_label.config(text=f"K: {self.k_value.get()}")
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.display_image(self.original_image, self.original_label)
            self.status.config(text=f"Loaded: {file_path.split('/')[-1]}")
    
    def display_image(self, image, label_widget):
        if image is None:
            return
            
        # Convert to PIL format
        max_size = 450
        height, width = image.shape[:2]
        
        if height > width:
            new_height = max_size
            new_width = int(width * (max_size / height))
        else:
            new_width = max_size
            new_height = int(height * (max_size / width))
            
        image = cv2.resize(image, (new_width, new_height))
        img_pil = Image.fromarray(image)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # Update the label
        label_widget.config(image=img_tk)
        label_widget.image = img_tk
    
    def process_image_threaded(self):
        if self.image_path is None:
            messagebox.showerror("Error", "Please select an image first!")
            return
            
        self.status.config(text="Processing... (this may take a moment)")
        threading.Thread(target=self.process_image, daemon=True).start()
    
    def process_image(self):
        if self.original_image is None:
            self.root.after(0, lambda: messagebox.showerror("Error", "No image loaded!"))
            self.root.after(0, lambda: self.status.config(text="No image loaded"))
            return
            
        try:
            # Edge detection
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
            gray = cv2.medianBlur(gray, 9)

            # Combine adaptive threshold with Canny edges
            edges_adaptive = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11, 2
            )

            edges_canny = cv2.Canny(gray, 30, 100)
            edges = cv2.bitwise_or(edges_adaptive, edges_canny)

            # Fix broken edges
            kernel = np.ones((self.edge_thickness.get(), self.edge_thickness.get()), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            if self.show_edges.get():
                self.display_image(edges, self.cartoon_label)
                self.root.after(0, lambda: self.status.config(text="Showing edge detection"))
                return

            # Flatten color region
            k = self.k_value.get()
            data = self.original_image.reshape(-1, 3)
            kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
            image_reduced = kmeans.cluster_centers_[kmeans.labels_]
            image_reduced = image_reduced.reshape(self.original_image.shape).astype(np.uint8)

            # Smoothing with edge preservation (using fixed blur amount)
            blurred = cv2.bilateralFilter(image_reduced, d=15, sigmaColor=80, sigmaSpace=80)

            # Cartoon Styling
            cartoon = blurred.copy()
            cartoon[edges != 0] = [0, 0, 0]

            # Texture
            #for i in range(0, cartoon.shape[0], 8):
                #cv2.line(cartoon, (0, i), (cartoon.shape[1], i), (200,200,200), 1, cv2.LINE_AA)

            self.cartoon_image = cartoon
            
            # Update UI
            self.root.after(0, self.update_result_display)
            self.root.after(0, lambda: self.status.config(text="Processing complete!"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, lambda: self.status.config(text="Error occurred"))
    
    def update_result_display(self):
        self.display_image(self.original_image, self.original_label)
        self.display_image(self.cartoon_image, self.cartoon_label)
    
    def toggle_edge_display(self):
        if self.show_edges.get() and self.image_path:
            self.process_image_threaded()
        elif self.cartoon_image is not None:
            self.display_image(self.cartoon_image, self.cartoon_label)
    
    def save_image(self):
        if self.cartoon_image is None:
            messagebox.showerror("Error", "No cartoon image to save!")
            return
            
        save_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")],
            title="Save Cartoonized Image"
        )
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(self.cartoon_image, cv2.COLOR_RGB2BGR))
            self.status.config(text=f"Image saved to {save_path.split('/')[-1]}")
            messagebox.showinfo("Success", "Image saved successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    app = CartoonizerApp(root)
    root.mainloop()