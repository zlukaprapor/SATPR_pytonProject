import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, scrolledtext
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt

class ImageLoaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Loader")

        # Ініціалізація змінних
        self.num_classes = 0
        self.images = []
        self.image_matrices = []
        self.class_names = []
        self.base_class_index = None
        self.binary_matrices = []
        self.delta = 10  # Значення за замовчуванням для кроку дельта

        # Створення віджетів
        self.class_label = tk.Label(root, text="Enter number of classes:")
        self.class_label.pack()

        self.class_entry = tk.Entry(root)
        self.class_entry.pack()

        self.load_button = tk.Button(root, text="step 1 -> Load Images", command=self.load_images)
        self.load_button.pack()

        self.check_button = tk.Button(root, text="step 2 -> Check Image Sizes", command=self.check_image_sizes, state=tk.DISABLED)
        self.check_button.pack()

        self.select_base_class_button = tk.Button(root, text="step 3 -> Select Base Class",command=self.select_base_class, state=tk.DISABLED)
        self.select_base_class_button.pack()

        self.show_matrix_button = tk.Button(root, text="step 4 -> Show Matrices", command=self.show_matrices, state=tk.DISABLED)
        self.show_matrix_button.pack()



        self.binary_image_button = tk.Button(root, text="step 5 -> Show Binary Images", command=self.show_binary_images, state=tk.DISABLED)
        self.binary_image_button.pack()

        self.plot_button = tk.Button(root, text="step 6 -> Show Vector", command=self.plot_expectation_vector, state=tk.DISABLED)
        self.plot_button.pack()

        self.image_frame = tk.Frame(root)
        self.image_frame.pack()

        self.info_label = tk.Label(root, text="")
        self.info_label.pack()

        # Віджет для відображення матриць
        self.matrix_window = scrolledtext.ScrolledText(root, width=60, height=20, wrap=tk.WORD)
        self.matrix_window.pack()

    def load_images(self):
        try:
            self.num_classes = int(self.class_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid number of classes.")
            return

        if self.num_classes <= 0:
            messagebox.showerror("Input Error", "Number of classes must be greater than 0.")
            return

        self.images = []
        self.image_matrices = []
        self.class_names = []

        for i in range(self.num_classes):
            class_name = simpledialog.askstring("Class Name", f"Enter name for class {i + 1}:")
            if not class_name:
                messagebox.showerror("Input Error", "Class name cannot be empty.")
                return

            self.class_names.append(class_name)
            file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.bmp;*.jpg;*.jpeg")])
            if file_path:
                img = Image.open(file_path).convert("L")  # Перетворення на відтінки сірого
                self.images.append(img)
                img_array = np.array(img)
                self.image_matrices.append(img_array)
            else:
                messagebox.showwarning("Warning", "Image selection cancelled.")
                return

        if len(self.images) != self.num_classes:
            messagebox.showerror("Input Error", "Number of loaded images does not match the number of classes.")
            return

        # Увімкнення кнопок вибору базового класу та перевірки розміру зображень
        self.select_base_class_button.config(state=tk.NORMAL)
        self.check_button.config(state=tk.NORMAL)
        self.show_matrix_button.config(state=tk.NORMAL)
        self.binary_image_button.config(state=tk.NORMAL)
        self.plot_button.config(state=tk.NORMAL)

        # Відображення завантажених зображень
        self.display_images()

    def select_base_class(self):
        if not self.class_names:
            messagebox.showwarning("Warning", "No classes available.")
            return

        base_class_index = simpledialog.askinteger(
            "Select Base Class",
            f"Enter the index of the base class (1-{self.num_classes}):"
        )

        if base_class_index is None or not (1 <= base_class_index <= self.num_classes):
            messagebox.showerror("Input Error", f"Please enter a valid index between 1 and {self.num_classes}.")
            return

        self.base_class_index = base_class_index - 1  # Збереження індексу базового класу
        self.info_label.config(text=f"Base class selected: {self.class_names[self.base_class_index]}")

        # Побудова бінарних матриць на основі базового класу
        self.build_binary_matrices()

    def build_binary_matrices(self):
        if self.base_class_index is None:
            messagebox.showwarning("Warning", "Please select a base class first.")
            return

        base_matrix = self.image_matrices[self.base_class_index]
        base_mean = np.mean(base_matrix)

        self.binary_matrices = []
        for matrix in self.image_matrices:
            binary_matrix = np.where(matrix >= base_mean, 1, 0)
            self.binary_matrices.append(binary_matrix)

        messagebox.showinfo("Success", "Binary matrices have been built based on the base class.")

    def check_image_sizes(self):
        if not self.images:
            messagebox.showwarning("Warning", "No images loaded.")
            return

        sizes = [img.size for img in self.images]
        width, height = sizes[0]
        consistent = all(size == (width, height) for size in sizes)

        if not consistent:
            messagebox.showwarning("Warning", "Not all images have the same dimensions.")
        else:
            self.info_label.config(text=f"All images have the same dimensions: {width}x{height}")

    def display_images(self):
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        if not self.images:
            messagebox.showwarning("Warning", "No images loaded.")
            return

        for img in self.images:
            img_tk = ImageTk.PhotoImage(img.resize((150, 150)))  # Зміна розміру для відображення
            label = tk.Label(self.image_frame, image=img_tk)
            label.image = img_tk  # Збереження посилання для уникнення видалення
            label.pack(side=tk.LEFT)

    def show_matrices(self):
        if not self.image_matrices:
            messagebox.showwarning("Warning", "No image matrices available.")
            return

        if self.base_class_index is None:
            messagebox.showwarning("Warning", "Please select a base class first.")
            return

        matrix_shapes = [matrix.shape for matrix in self.image_matrices]
        width, height = matrix_shapes[0]
        consistent = all(shape == (width, height) for shape in matrix_shapes)

        if consistent:
            self.info_label.config(text="All matrices have the same dimensions.")
        else:
            messagebox.showwarning("Warning", "Matrices have different dimensions.")

        # Очищення текстового поля
        self.matrix_window.delete(1.0, tk.END)

        # Виведення матриць у вікно
        for i, matrix in enumerate(self.image_matrices):
            self.matrix_window.insert(tk.END, f"Matrix for image {i + 1} ({self.class_names[i]}):\n")
            self.matrix_window.insert(tk.END, f"{matrix}\n\n")

    def show_binary_images(self):
        if not self.binary_matrices:
            messagebox.showwarning("Warning", "No binary matrices available.")
            return

        for widget in self.image_frame.winfo_children():
            widget.destroy()

        for i, binary_matrix in enumerate(self.binary_matrices):
            binary_img = Image.fromarray(np.uint8(binary_matrix * 255))  # Перетворення на зображення
            img_tk = ImageTk.PhotoImage(binary_img.resize((150, 150)))
            label = tk.Label(self.image_frame, image=img_tk)
            label.image = img_tk  # Збереження посилання для уникнення видалення
            label.pack(side=tk.LEFT)
        # Виведення бінарних матриць у текстовому вигляді
        self.show_binary_matrices_in_text()

    def show_binary_matrices_in_text(self):
        if not self.binary_matrices:
            messagebox.showwarning("Warning", "No binary matrices available.")
            return

        # Очищення текстового поля
        self.matrix_window.delete(1.0, tk.END)

        # Виведення бінарних матриць у вікно
        for i, binary_matrix in enumerate(self.binary_matrices):
            self.matrix_window.insert(tk.END, f"Binary matrix for image {i + 1} ({self.class_names[i]}):\n")
            self.matrix_window.insert(tk.END, f"{binary_matrix}\n\n")

    def plot_expectation_vector(self):
        if self.base_class_index is None:
            messagebox.showwarning("Warning", "Please select a base class first.")
            return

        # Обчислення еталонного вектора
        base_matrix = self.image_matrices[self.base_class_index]
        expectation_vector = np.mean(base_matrix, axis=0)

        # Введення кроку дельта (обираємо або використовуємо значення за замовчуванням)
        self.delta = simpledialog.askinteger("Delta", "Enter delta step value (0-255):", initialvalue=self.delta)

        if self.delta is None or not (0 <= self.delta <= 255):
            messagebox.showerror("Input Error", "Please enter a valid delta value between 0 and 255.")
            return

        # Побудова векторів верхнього та нижнього допусків
        lower_bound = expectation_vector - self.delta
        upper_bound = expectation_vector + self.delta

        # Побудова графіка
        x = np.arange(len(expectation_vector))

        plt.figure(figsize=(10, 5))
        plt.plot(x, expectation_vector, label="Expectation (mean) vector", color='blue', marker='o')
        plt.fill_between(x, lower_bound, upper_bound, color='gray', alpha=0.3, label="Tolerance corridor")

        plt.xlabel("Feature index")
        plt.ylabel("Value")
        plt.title("Expectation Vector and Tolerance Corridor")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLoaderApp(root)
    root.mainloop()
