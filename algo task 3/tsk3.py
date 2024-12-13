import cv2
import numpy as np
from UI import Ui_MainWindow
import cmath
from PyQt5.QtWidgets import *
import wfdb
from PyQt5.QtWidgets import QFileDialog, QLabel
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import numpy as np
import pandas as pd  
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.inputframe = self.ui.inputImageFrame
        self.outputFrame = self.ui.outputImageFrame
        self.ui.actionImport.triggered.connect(self.ImportImg)
        self.ui.generateButton.clicked.connect(self.Generate)
        self.ui.progressBar.setValue(0)  # Initialize progress bar

    def ImportImg(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Image Files (*.jpeg *.jpg *.png *.bmp *.gif)")
        file_dialog.setWindowTitle("Open Image")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            print(file_path)
            self.original_image = cv2.imread(file_path)  # Load the original image
            self.FrameImg(self.inputframe, file_path)    # Display the original image
            self.ui.oldSizeLabel.setText(f"Old Size: {self.original_image.shape[1]} x {self.original_image.shape[0]}")  # Set old size label

    def Generate(self):
        target_width = self.original_image.shape[1] - 50
        resized_image = self.seam_carve_with_progress(self.original_image, target_width)
        self.ui.newSizeLabel.setText(f"New Size: {resized_image.shape[1]} x {resized_image.shape[0]}")

    def FrameImg(self, frame, file_path):
        pixmap = QPixmap(file_path)
        
        # Ensure that each frame gets its own QLabel
        if frame == self.inputframe:
            if not hasattr(self, 'input_image_label'):
                self.input_image_label = QLabel(frame)
                self.input_image_label.setAlignment(Qt.AlignCenter)
            label = self.input_image_label
        elif frame == self.outputFrame:
            if not hasattr(self, 'output_image_label'):
                self.output_image_label = QLabel(frame)
                self.output_image_label.setAlignment(Qt.AlignCenter)
            label = self.output_image_label

        # Scale the pixmap to fit the frame without affecting the original image
        scaled_pixmap = pixmap.scaled(frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)
        label.setGeometry(0, 0, frame.width(), frame.height())
        label.show()


    def calculate_energy(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        energy = np.abs(sobel_x) + np.abs(sobel_y)
        return energy

    def find_vertical_seam(self,energy):
        rows, cols = len(energy), len(energy[0])
        dp = [[float('inf')] * cols for _ in range(rows)]
        
        # Initialize the first row
        for j in range(cols):
            dp[0][j] = energy[0][j]
        for i in range(1, rows):
            for j in range(cols):
                dp[i][j] = energy[i][j] + min(
                dp[i-1][j-1] if j-1 >= 0 else float('inf'),
                dp[i-1][j],
                dp[i-1][j+1] if j+1 < cols else float('inf'))

    # Find the end of the lowest-energy seam
        min_energy = float('inf')
        self.end_col = -1
        for j in range(cols):
            if dp[rows-1][j] < min_energy:
                min_energy = dp[rows-1][j]
                self.end_col = j

        # Backtrack to find the seam
        seam = []
        current_col = self.end_col
        for i in range(rows-1, -1, -1):
            seam.append((i, current_col))
            # Move to the previous row
            if i > 0:
                if current_col > 0 and dp[i-1][current_col-1] == dp[i][current_col] - energy[i][current_col]:
                    current_col -= 1
                elif current_col < cols-1 and dp[i-1][current_col+1] == dp[i][current_col] - energy[i][current_col]:
                    current_col += 1

        seam.reverse()
        return seam

    def remove_vertical_seam(self, image, seam):
        rows, cols, channels = image.shape
        new_image = np.zeros((rows, cols - 1, channels), dtype=image.dtype)  # Adjust the width

        for i in range(rows):
           col = seam[i][1]  # Extract the column index from the seam tuple
           # Copy pixels before the seam
           new_image[i, :col, :] = image[i, :col, :]
           # Copy pixels after the seam
           new_image[i, col:, :] = image[i, col+1:, :]

        return new_image

    def seam_carve_with_progress(self, image, target_width):
        total_steps = image.shape[1] - target_width
        for step in range(total_steps):
            energy = self.calculate_energy(image)
            seam = self.find_vertical_seam(energy)
            image = self.remove_vertical_seam(image, seam)
            progress = int((step + 1) / total_steps * 100)
            self.ui.progressBar.setValue(progress)
            QApplication.processEvents()  # Update the UI

        output_file_path = "resized_image.jpg"
        cv2.imwrite(output_file_path, image)
        self.FrameImg(self.outputFrame, output_file_path)
        self.ui.progressBar.setValue(100)  # Ensure progress bar is full at the end
        return image

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())