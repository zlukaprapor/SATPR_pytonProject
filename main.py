import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageFilter
import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

class ImageLoaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Loader")

        # Ініціалізація змінних
        self.num_classes = 0
        self.images = []
        self.image_matrices = []
        self.labels = []
        self.model = None
        self.class_names = []

        # Створення віджетів
        self.class_label = tk.Label(root, text="Enter number of classes:")
        self.class_label.pack()

        self.class_entry = tk.Entry(root)
        self.class_entry.pack()

        self.load_button = tk.Button(root, text="Load Images", command=self.load_images)
        self.load_button.pack()

        self.process_button = tk.Button(root, text="Process Images", command=self.process_images)
        self.process_button.pack()

        self.save_button = tk.Button(root, text="Save Matrices", command=self.save_matrices)
        self.save_button.pack()

        self.train_button = tk.Button(root, text="Train Model", command=self.train_model)
        self.train_button.pack()

        self.test_button = tk.Button(root, text="Test Model", command=self.test_model)
        self.test_button.pack()

        self.display_button = tk.Button(root, text="Display Images", command=self.display_images)
        self.display_button.pack()

        self.image_frame = tk.Frame(root)
        self.image_frame.pack()

        self.info_label = tk.Label(root, text="")
        self.info_label.pack()

        self.visualize_button = tk.Button(root, text="Visualize Results", command=self.visualize_results)
        self.visualize_button.pack()

    def load_images(self):
        try:
            self.num_classes = int(self.class_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid number for classes.")
            return

        if self.num_classes <= 0:
            messagebox.showerror("Input Error", "Number of classes must be greater than 0.")
            return

        self.images = []
        self.image_matrices = []
        self.labels = []
        self.class_names = []

        for i in range(self.num_classes):
            class_name = simpledialog.askstring("Class Name", f"Enter name for class {i + 1}:")
            self.class_names.append(class_name)
            file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.bmp;*.jpg;*.jpeg")])
            if file_path:
                img = Image.open(file_path).convert("L")  # Convert to grayscale
                self.images.append(img)
                img_array = np.array(img)
                self.image_matrices.append(img_array)
                # Кількість пікселів в одному зображенні
                num_pixels = img.size[0] * img.size[1]
                # Додайте мітки для кожного пікселя
                self.labels.extend([i] * (img_array.size // num_pixels))
            else:
                messagebox.showwarning("Warning", "Image selection cancelled.")
                return

        if len(self.images) != self.num_classes:
            messagebox.showerror("Input Error", "Number of loaded images does not match number of classes.")
            return

        # Перевірка розмірів зображень
        self.check_image_sizes()

    def check_image_sizes(self):
        sizes = [img.size for img in self.images]
        width, height = sizes[0]
        consistent = all(size == (width, height) for size in sizes)
        if not consistent:
            messagebox.showwarning("Warning", "Not all images have the same dimensions.")
        else:
            self.info_label.config(text=f"All images have the same dimensions: {width}x{height}")

    def process_images(self):
        self.image_matrices = []
        for img in self.images:
            # Перетворити зображення в масив
            img_array = np.array(img)
            # Нормалізувати зображення
            normalized_img_array = img_array / 255.0
            # Фільтрувати зображення (опціонально)
            img_filtered = img.filter(ImageFilter.GaussianBlur(1))
            self.images[self.images.index(img)] = img_filtered
            # Сплющити зображення в один вектор
            self.image_matrices.append(normalized_img_array.flatten())

        self.info_label.config(text="Images processed (normalized and filtered).")

    def save_matrices(self):
        if not self.image_matrices:
            messagebox.showwarning("Warning", "No matrices to save.")
            return

        for i, matrix in enumerate(self.image_matrices):
            file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                   filetypes=[("CSV Files", "*.csv")],
                                                   initialfile=f"matrix_{i+1}.csv")
            if file_path:
                np.savetxt(file_path, matrix.reshape(1, -1), delimiter=",")
                messagebox.showinfo("Success", f"Matrix {i+1} saved to {file_path}.")

    def train_model(self):
        if not self.image_matrices:
            messagebox.showwarning("Warning", "No matrices to train on.")
            return

        X = np.array(self.image_matrices)
        y = np.array(self.labels)

        # Перевірте кількість зразків і виберіть відповідну кількість сусідів
        n_neighbors = min(3, len(X))  # Використовуємо мінімум з 3 або кількість зразків
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model.fit(X, y)
        self.info_label.config(text="Model trained using k-NN.")

    def test_model(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Model is not trained.")
            return

        # Simulate testing with same data for demonstration
        X_test = np.array(self.image_matrices)
        y_test = np.array(self.labels)
        y_pred = self.model.predict(X_test)

        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, target_names=self.class_names)

        self.info_label.config(text="Testing completed. Check console for results.")
        print("Confusion Matrix:\n", conf_matrix)
        print("Classification Report:\n", class_report)

    def visualize_results(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Model is not trained.")
            return

        X_test = np.array(self.image_matrices)
        y_test = np.array(self.labels)
        y_pred = self.model.predict(X_test)

        conf_matrix = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(10, 7))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names, rotation=45)
        plt.yticks(tick_marks, self.class_names)

        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    def display_images(self):
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        if not self.images:
            messagebox.showwarning("Warning", "No images loaded.")
            return

        for img in self.images:
            img_tk = ImageTk.PhotoImage(img.resize((250, 250)))  # Resize for display
            label = tk.Label(self.image_frame, image=img_tk)
            label.image = img_tk  # Keep a reference to avoid garbage collection
            label.pack(side=tk.LEFT)

        # Display matrices
        for i, matrix in enumerate(self.image_matrices):
            print(f"Matrix for image {i+1}:")
            print(matrix)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLoaderApp(root)
    root.mainloop()
