import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QVBoxLayout, QWidget, QFileDialog, QSpinBox, QComboBox,
                             QCheckBox, QHBoxLayout)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class FireSmokeDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        self.image_files = []
        self.current_index = 0
        
        self.gmm_a = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=16, detectShadows=False)
        self.gmm_b = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=16, detectShadows=False)
        self.mog2 = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=16, detectShadows=False)
        self.prev_frame = None
    
    def initUI(self):
        self.setWindowTitle("Fire and Smoke Detection")
        
        # Input and output image labels
        self.input_image_label = QLabel(self)
        self.output_image_label = QLabel(self)
        
        # Frame skip spinbox
        self.frame_skip_label = QLabel("Frame Skip:", self)
        self.frame_skip_spinbox = QSpinBox(self)
        self.frame_skip_spinbox.setRange(0, 100)
        self.frame_skip_spinbox.setValue(0)
        
        # Detection type combobox
        self.detection_type_label = QLabel("Detection Type:", self)
        self.detection_type_combobox = QComboBox(self)
        self.detection_type_combobox.addItems(["Smoke", "Fire", "Combined"])
        
        # Output overlay checkbox
        self.output_overlay_checkbox = QCheckBox("Output Overlay", self)
        
        # Select directory button
        self.select_dir_button = QPushButton("Select Image Directory", self)
        self.select_dir_button.clicked.connect(self.select_directory)
        
        # Layouts
        hbox = QHBoxLayout()
        hbox.addWidget(self.input_image_label)
        hbox.addWidget(self.output_image_label)
        
        options_layout = QVBoxLayout()
        options_layout.addWidget(self.frame_skip_label)
        options_layout.addWidget(self.frame_skip_spinbox)
        options_layout.addWidget(self.detection_type_label)
        options_layout.addWidget(self.detection_type_combobox)
        options_layout.addWidget(self.output_overlay_checkbox)
        options_layout.addWidget(self.select_dir_button)
        
        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addLayout(options_layout)
        
        container = QWidget()
        container.setLayout(vbox)
        self.setCentralWidget(container)
        
        self.resize(1200, 600)
    
    def select_directory(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if dir_name:
            self.load_images_from_directory(dir_name)
    
    def load_images_from_directory(self, dir_name):
        self.image_files = sorted([os.path.join(dir_name, f) for f in os.listdir(dir_name) if f.endswith(('jpg', 'png', 'jpeg'))])
        self.current_index = 0
        self.prev_frame = None
        self.process_and_display_images()
    
    def process_and_display_images(self):
        if not self.image_files:
            return
        
        frame_skip = self.frame_skip_spinbox.value()
        detection_type = self.detection_type_combobox.currentText().lower()
        overlay_output = self.output_overlay_checkbox.isChecked()
        
        while self.current_index < len(self.image_files):
            frame = cv2.imread(self.image_files[self.current_index])
            
            if frame is None:
                self.current_index += 1 + frame_skip
                continue
            
            if detection_type == "fire":
                output_map = self.create_fire_map(frame)
            elif detection_type == "smoke":
                if self.prev_frame is not None:
                    output_map = self.create_smoke_map(self.prev_frame, frame)
                else:
                    output_map = np.zeros_like(frame)
            else:
                output_map = self.combined_detection(frame)
            
            if overlay_output:
                output_map_colored = self.apply_overlay(frame, output_map, detection_type)
            else:
                output_map_colored = cv2.cvtColor(output_map, cv2.COLOR_GRAY2BGR)
            
            self.display_image(self.input_image_label, frame)
            self.display_image(self.output_image_label, output_map_colored)
            
            self.prev_frame = frame.copy()
            self.current_index += 1 + frame_skip
    
    def display_image(self, label, image):
        # Convert the image to QImage and display
        qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimage)
        label.setPixmap(pixmap)
    
    def create_fire_map(self, frame):
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab_frame)
        
        fg_mask_a = self.gmm_a.apply(a)
        fg_mask_b = self.gmm_b.apply(b)
        
        fire_map = cv2.bitwise_and(fg_mask_a, fg_mask_b)
        return fire_map
    
    def create_smoke_map(self, prev_frame, frame):
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        smoke_map = cv2.absdiff(prev_gray, curr_gray)
        _, smoke_map = cv2.threshold(smoke_map, 50, 255, cv2.THRESH_BINARY)
        
        return smoke_map
    
    def combined_detection(self, frame):
        fg_mask = self.mog2.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        return fg_mask
    
    def apply_overlay(self, input_image, output_map, detection_type):
        output_colored = np.zeros_like(input_image)
        
        if detection_type == "fire":
            color = (0, 0, 255)  # Red
        elif detection_type == "smoke":
            color = (255, 0, 0)  # Blue
        else:
            color = (0, 255, 0)  # Green
        
        contours, _ = cv2.findContours(output_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output_colored, (x, y), (x+w, y+h), color, 2)
        
        overlayed_image = cv2.addWeighted(input_image, 0.7, output_colored, 0.3, 0)
        return overlayed_image

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = FireSmokeDetectionApp()
    ex.show()
    sys.exit(app.exec_())
