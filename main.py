import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, scrolledtext
from PIL import Image, ImageTk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import math
from scipy.spatial.distance import euclidean

class ImageLoaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Loader")

        # Фіксований розмір вікна
        self.root.geometry('1368x768')
        self.root.resizable(False, False)  # Забороняємо змінювати розмір вікна

        # Ініціалізація змінних
        self.num_classes = 0
        self.images = []
        self.image_matrices = []
        self.class_names = []
        self.base_class_index = None
        self.binary_matrices = []
        self.delta = 10  # Значення за замовчуванням для кроку дельта
        self.SK = []  # Ініціалізація порожніх масивів
        self.SK_PARA = []  # Ініціалізація порожніх масивів

        # Створення фреймів для кращої організації
        self.top_frame = tk.Frame(root, pady=10)
        self.top_frame.pack(side=tk.TOP, fill=tk.X)

        self.middle_frame = tk.Frame(root, pady=10)
        self.middle_frame.pack(side=tk.TOP, fill=tk.X)

        self.bottom_frame = tk.Frame(root, pady=10)
        self.bottom_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Верхній фрейм (введення класів і кнопки)
        self.class_label = tk.Label(self.top_frame, text="Enter number of classes:")
        self.class_label.pack(side=tk.LEFT, padx=5)

        self.class_entry = tk.Entry(self.top_frame)
        self.class_entry.pack(side=tk.LEFT, padx=5)

        self.load_button = tk.Button(self.top_frame, text="Step 1 -> Load Images", command=self.load_images)
        self.load_button.pack(side=tk.LEFT, padx=5)

        # Середній фрейм (кнопки дій)
        self.check_button = tk.Button(self.middle_frame, text="Step 2 -> Check Image Sizes",
                                      command=self.check_image_sizes, state=tk.DISABLED)
        self.check_button.pack(side=tk.LEFT, padx=5)

        self.select_base_class_button = tk.Button(self.middle_frame, text="Step 3 -> Select Base Class",
                                                  command=self.select_base_class, state=tk.DISABLED)
        self.select_base_class_button.pack(side=tk.LEFT, padx=5)

        self.show_matrix_button = tk.Button(self.middle_frame, text="Step 4 -> Show Matrices",
                                            command=self.show_matrices, state=tk.DISABLED)
        self.show_matrix_button.pack(side=tk.LEFT, padx=5)

        self.binary_image_button = tk.Button(self.middle_frame, text="Step 5 -> Show Binary Images",
                                             command=self.show_binary_images, state=tk.DISABLED)
        self.binary_image_button.pack(side=tk.LEFT, padx=5)

        self.plot_button = tk.Button(self.middle_frame, text="Step 6 -> Show Vector",
                                     command=self.plot_expectation_vector, state=tk.DISABLED)
        self.plot_button.pack(side=tk.LEFT, padx=5)

        self.tolerance_button = tk.Button(self.middle_frame, text="Step 7 -> Show Tolerance System",
                                          command=self.plot_tolerance_system, state=tk.DISABLED)
        self.tolerance_button.pack(side=tk.LEFT, padx=5)

        self.sk_map_button = tk.Button(self.middle_frame, text="Step 8 -> Show SK_MAP", command=self.plot_sk_map,
                                       state=tk.DISABLED)
        self.sk_map_button.pack(side=tk.LEFT, padx=5)

        self.distance_button = tk.Button(self.middle_frame, text="Step 9 -> Compute Distances",
                                         command=self.compute_distances_from_reference, state=tk.DISABLED)
        self.distance_button.pack(side=tk.LEFT, padx=5)

        # Нижній фрейм (відображення зображень та текстові поля)
        self.image_frame = tk.Frame(self.bottom_frame)
        self.image_frame.pack(side=tk.LEFT, padx=10)

        self.info_label = tk.Label(self.bottom_frame, text="")
        self.info_label.pack(side=tk.TOP, padx=5)

        # Віджет для відображення матриць
        self.matrix_window = scrolledtext.ScrolledText(self.bottom_frame, width=60, height=20, wrap=tk.WORD)
        self.matrix_window.pack(side=tk.LEFT, padx=10)

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
        self.tolerance_button.config(state=tk.NORMAL)
        self.sk_map_button.config(state=tk.NORMAL)
        self.distance_button.config(state=tk.NORMAL)

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

        # Обчислення кодових відстаней
        self.compute_coding_distances()

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
            self.matrix_window.insert(tk.END, f"Binary Matrix for image {i + 1} ({self.class_names[i]}):\n")
            self.matrix_window.insert(tk.END, f"{binary_matrix}\n\n")

    def clear_canvas(self):
        # Очищення віджетів, що були додані раніше до контейнера
        for widget in self.image_frame.winfo_children():
            widget.destroy()

    def plot_expectation_vector(self):
        self.clear_canvas()  # Очищення фрейму перед побудовою графіка

        if not self.image_matrices:
            messagebox.showwarning("Warning", "No image matrices available.")
            return

        # Обчислення середнього та стандартного відхилення для кожного класу
        avg_values = [np.mean(matrix) for matrix in self.image_matrices]
        std_values = [np.std(matrix) for matrix in self.image_matrices]

        # Створення фігури matplotlib
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)

        # Позиції на осі X
        x = np.arange(self.num_classes)

        # Побудова лінійного графіку із помилками (error bars)
        ax.errorbar(x, avg_values, yerr=std_values, fmt='-o', color='blue', ecolor='red', capsize=5,
                    label='Avg. values with Std. Dev.')

        # Додавання підписів значень
        for i, avg in enumerate(avg_values):
            ax.text(i, avg + 0.05, round(avg, 2), ha='center', va='bottom', fontsize=10)

        # Налаштування осей
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Average Pixel Value', fontsize=12)
        ax.set_title('Average Pixel Values with Standard Deviation by Class', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, fontsize=10)

        # Додавання сітки
        ax.grid(True, linestyle='--', alpha=0.6)

        # Додавання легенди
        ax.legend()

        # Відображення графіка у tkinter через FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=self.image_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def plot_tolerance_system(self):
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

        # Побудова графіка системи контрольних допусків
        x = np.arange(len(expectation_vector))

        # Створення фігури matplotlib для контрольних допусків
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(x, lower_bound, label="Lower Tolerance", color='red', linestyle='--')
        ax.plot(x, upper_bound, label="Upper Tolerance", color='green', linestyle='--')
        ax.plot(x, expectation_vector, label="Expectation (mean) vector", color='blue', marker='o')

        ax.set_xlabel("Feature index")
        ax.set_ylabel("Tolerance Value")
        ax.set_title("Tolerance System")
        ax.legend()
        ax.grid(True)

        # Очищення попереднього графіка, якщо такий існує
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        # Вбудовування графіка у tkinter через FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=self.image_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def compute_class_centers(self):
        """Обчислює геометричні центри для кожного класу (середнє значення пікселів)."""
        self.class_centers = [np.mean(matrix) for matrix in self.image_matrices]

    def find_nearest_neighbor(self, current_class_idx):
        """Знаходить найближчий клас-сусід на основі відстаней між центрами класів."""
        current_center = self.class_centers[current_class_idx]
        min_distance = float('inf')
        nearest_neighbor_idx = None

        for i, center in enumerate(self.class_centers):
            if i != current_class_idx:  # не порівнюємо клас сам із собою
                distance = abs(current_center - center)
                if distance < min_distance:
                    min_distance = distance
                    nearest_neighbor_idx = i

        return nearest_neighbor_idx

    def compute_coding_distances(self):
        """Розраховує кодові відстані та формує масиви SK і SK_PARA."""
        self.compute_class_centers()  # Спочатку обчислюємо центри класів

        self.SK = []
        self.SK_PARA = []

        for class_idx in range(self.num_classes):
            # Масив для кодових відстаней поточного класу
            current_class_matrix = self.image_matrices[class_idx]
            current_center = self.class_centers[class_idx]

            # Знаходимо найближчого сусіда
            nearest_neighbor_idx = self.find_nearest_neighbor(class_idx)
            nearest_neighbor_matrix = self.image_matrices[nearest_neighbor_idx]
            nearest_center = self.class_centers[nearest_neighbor_idx]

            # Обчислюємо відстані між геометричним центром поточного класу та реалізаціями цього класу
            distances_to_current_class = [abs(current_center - np.mean(row)) for row in current_class_matrix]

            # Обчислюємо відстані між геометричним центром поточного класу та реалізаціями найближчого сусіда
            distances_to_nearest_neighbor = [abs(current_center - np.mean(row)) for row in nearest_neighbor_matrix]

            # Формуємо масив SK (кодові відстані для поточного класу)
            self.SK.append([distances_to_current_class, distances_to_nearest_neighbor])

            # Формуємо масив SK_PARA (кодові відстані для сусіднього класу)
            distances_to_nearest_neighbor_class = [abs(nearest_center - np.mean(row)) for row in current_class_matrix]
            distances_to_nearest_neighbor_to_self = [abs(nearest_center - np.mean(row)) for row in nearest_neighbor_matrix]
            self.SK_PARA.append([distances_to_nearest_neighbor_class, distances_to_nearest_neighbor_to_self])

        messagebox.showinfo("Success", "Coding distances (SK and SK_PARA) have been calculated.")


    def plot_sk_map(self):

        """Відображення розподілу реалізацій між поточним класом і його найближчим сусідом."""
        if not self.SK or not self.SK_PARA:
            messagebox.showwarning("Warning", "Please calculate coding distances first.")
            return

        self.clear_canvas()  # Очищуємо попередній графік

        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)

        # Масиви для X-координат (реалізації розташовуються по горизонтальній осі)
        # Припускаємо, що всі класи мають однакову кількість реалізацій
        num_realizations_current = len(self.SK[0][0])
        num_realizations_neighbor = len(self.SK_PARA[0][0])

        x_current_class = np.arange(num_realizations_current)
        x_nearest_class = np.arange(num_realizations_neighbor)

        # Відображення реалізацій поточного класу та їх відстаней
       # ax.scatter(x_current_class, self.SK[0][0], color='blue', label='Current Class Realizations')
       # ax.scatter(x_nearest_class, self.SK[1][0], color='green', label='Nearest Neighbor Realizations')

        # Відображення реалізацій сусіднього класу
        ax.scatter(x_current_class, self.SK_PARA[0][0], color='red', label='SK_PARA (Current to Neighbor)')
        ax.scatter(x_nearest_class, self.SK_PARA[1][0], color='purple', label='SK_PARA (Neighbor to Current)')

        # Налаштування графіку
        ax.set_xlabel("Realization Index")
        ax.set_ylabel("Coding Distance")
        ax.set_title("Distribution of Realizations between Current Class and Neighbor")
        ax.legend()
        ax.grid(True)

        # Вбудовуємо графік у tkinter через FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=self.image_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def compute_distances_from_reference(self):
        """Обчислити відстані від кожного зображення до його центру класу на основі бінарних матриць."""
        if not self.binary_matrices:
            messagebox.showwarning("Увага", "Спочатку обчисліть бінарні матриці.")
            return

        distances = []
        for i in range(self.num_classes):
            current_class_matrix = self.binary_matrices[i]

            # Обчислення центру класу на основі бінарної матриці
            class_center = np.mean(current_class_matrix, axis=0)

            if class_center.ndim > 1:
                class_center = class_center.flatten()

            # Обчисліть відстані для кожного рядка в бінарній матриці поточного класу
            class_distances = []
            for row in current_class_matrix:
                row_flat = row.flatten()  # Перетворення рядка в 1-D
                if row_flat.ndim == 1 and class_center.ndim == 1:
                    distance = euclidean(row_flat, class_center)
                    class_distances.append(distance)
                else:
                    print(f"Неправильний формат: row_shape={row_flat.shape}, class_center_shape={class_center.shape}")

            distances.append(class_distances)

        # Виведення відстаней у вікно матриці
        self.matrix_window.delete(1.0, tk.END)
        for i, dist_list in enumerate(distances):
            self.matrix_window.insert(tk.END,
                                      f"Відстані від класу {self.class_names[i]} до його центру (на основі бінарних матриць):\n")
            # Округлення кожного значення до десятої при виведенні
            formatted_dist_list = ', '.join(f"{d:.1f}" for d in dist_list)
            self.matrix_window.insert(tk.END, f"{formatted_dist_list}\n\n")

        messagebox.showinfo("Успіх", "Відстані були обчислені та відображені.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLoaderApp(root)
    root.mainloop()
