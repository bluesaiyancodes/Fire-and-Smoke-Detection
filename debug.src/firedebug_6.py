import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QVBoxLayout, QWidget, QFileDialog, QSpinBox, QComboBox,
                             QCheckBox, QHBoxLayout, QGridLayout, QFrame, QSlider, QGroupBox)
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
        self.setFixedSize(1200, 800)  # Adjusting the window size for more space

        # Input and output image labels
        self.input_image_label = QLabel(self)
        self.output_image_label = QLabel(self)
        self.input_image_label.setFixedSize(580, 450)  # Adjusting image label size
        self.output_image_label.setFixedSize(580, 450)  # Adjusting image label size
        self.input_image_label.setScaledContents(True)
        self.output_image_label.setScaledContents(True)

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

        # Alpha slider for overlay transparency
        self.alpha_slider_label = QLabel("Overlay Alpha:", self)
        self.alpha_slider = QSlider(Qt.Horizontal, self)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(50)
        self.alpha_slider.setTickPosition(QSlider.TicksBelow)
        self.alpha_slider.setTickInterval(10)

        # Select directory button
        self.select_dir_button = QPushButton("Select Image Directory", self)
        self.select_dir_button.clicked.connect(self.select_directory)
        
        # Next and Previous buttons
        self.next_button = QPushButton("Next", self)
        self.next_button.clicked.connect(self.show_next_image)
        self.prev_button = QPushButton("Previous", self)
        self.prev_button.clicked.connect(self.show_prev_image)
        
        # Layouts for image display and controls
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.input_image_label)
        image_layout.addWidget(self.output_image_label)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)

        # Divider line
        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        hline.setFrameShadow(QFrame.Sunken)

        # Controls layout in a group box for better organization
        controls_groupbox = QGroupBox("Controls")
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.frame_skip_label)
        controls_layout.addWidget(self.frame_skip_spinbox)
        controls_layout.addWidget(self.detection_type_label)
        controls_layout.addWidget(self.detection_type_combobox)
        controls_layout.addWidget(self.output_overlay_checkbox)
        controls_layout.addWidget(self.alpha_slider_label)
        controls_layout.addWidget(self.alpha_slider)
        controls_layout.addWidget(self.select_dir_button)
        controls_groupbox.setLayout(controls_layout)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(image_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(hline)
        main_layout.addWidget(controls_groupbox)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Set focus policies to handle arrow key events
        self.setFocusPolicy(Qt.StrongFocus)
        self.input_image_label.setFocusPolicy(Qt.NoFocus)
        self.output_image_label.setFocusPolicy(Qt.NoFocus)
        self.frame_skip_spinbox.setFocusPolicy(Qt.ClickFocus)
        self.detection_type_combobox.setFocusPolicy(Qt.ClickFocus)
        self.output_overlay_checkbox.setFocusPolicy(Qt.ClickFocus)
        self.alpha_slider.setFocusPolicy(Qt.ClickFocus)
        self.select_dir_button.setFocusPolicy(Qt.ClickFocus)
        self.next_button.setFocusPolicy(Qt.NoFocus)
        self.prev_button.setFocusPolicy(Qt.NoFocus)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            self.show_next_image()
        elif event.key() == Qt.Key_Left:
            self.show_prev_image()

    def select_directory(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if dir_name:
            self.load_images_from_directory(dir_name)
    
    def load_images_from_directory(self, dir_name):
        self.image_files = sorted([os.path.join(dir_name, f) for f in os.listdir(dir_name) if f.endswith(('jpg', 'png', 'jpeg'))])
        self.current_index = 0
        self.prev_frame = None
        if self.image_files:
            self.process_and_display_current_image()
    
    def process_and_display_current_image(self):
        if not self.image_files:
            return
        
        current_image_path = self.image_files[self.current_index]
        frame = cv2.imread(current_image_path)
        
        if frame is None:
            return
        
        detection_type = self.detection_type_combobox.currentText().lower()
        overlay_output = self.output_overlay_checkbox.isChecked()
        alpha = self.alpha_slider.value() / 100.0  # Get alpha value from slider

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
            output_map_colored = self.apply_overlay(frame, output_map, detection_type, alpha)
        else:
            output_map_colored = cv2.cvtColor(output_map, cv2.COLOR_GRAY2BGR)
        
        self.show_image(frame, self.input_image_label)
        self.show_image(output_map_colored, self.output_image_label)
        
        self.prev_frame = frame.copy()
    
    def show_image(self, image, label):
        resized_image = cv2.resize(image, (label.width(), label.height()), interpolation=cv2.INTER_AREA)
        qimage = QImage(resized_image.data, resized_image.shape[1], resized_image.shape[0], resized_image.strides[0], QImage.Format_BGR888)
        label.setPixmap(QPixmap.fromImage(qimage))
    
    def show_next_image(self):
        frame_skip = self.frame_skip_spinbox.value()
        self.current_index = min(self.current_index + 1 + frame_skip, len(self.image_files) - 1)
        self.process_and_display_current_image()
    
    def show_prev_image(self):
        frame_skip = self.frame_skip_spinbox.value()
        self.current_index = max(self.current_index - 1 - frame_skip, 0)
        self.process_and_display_current_image()
    
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
    
    def apply_overlay(self, input_image, output_map, detection_type, alpha):
        overlay = np.zeros_like(input_image, dtype=np.uint8)
        
        if detection_type == "fire":
            color = (0, 0, 255)  # Red
        elif detection_type == "smoke":
            color = (255, 0, 0)  # Blue
        else:
            color = (0, 255, 0)  # Green
        
        overlay[output_map > 0] = color
        
        blended = cv2.addWeighted(input_image, 1 - alpha, overlay, alpha, 0)
        return blended

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = FireSmokeDetectionApp()
    ex.show()
    sys.exit(app.exec_())
