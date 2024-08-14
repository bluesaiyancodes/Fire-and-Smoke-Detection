import sys
import os
import cv2
import numpy as np
import torch
from sklearn.decomposition import PCA
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QVBoxLayout, QWidget, QFileDialog, QSpinBox, QComboBox,
                             QCheckBox, QHBoxLayout, QGridLayout, QFrame, QSlider, 
                             QGroupBox, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class FireSmokeDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.show_instructions()

        self.image_files = []
        self.current_index = 0
        self.is_paused = False
        self.current_frame = None
        
        # Initialize GMMs with short history for better responsiveness
        self.gmm_a = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=16, detectShadows=False)
        self.gmm_b = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=16, detectShadows=False)

        # Load the DINOv2 model
        self.dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.dinov2_model.eval()  # Set the model to evaluation mode
    
    def initUI(self):
        self.setWindowTitle("Fire and Smoke Detection with PCA Visualization")
        self.setFixedSize(1800, 800)  # Increased window size for additional output

        # Input, output, and PCA image labels
        self.input_image_label = QLabel(self)
        self.output_image_label = QLabel(self)
        self.pca_image_label = QLabel(self)
        self.input_image_label.setFixedSize(580, 450)
        self.output_image_label.setFixedSize(580, 450)
        self.pca_image_label.setFixedSize(580, 450)
        self.input_image_label.setScaledContents(True)
        self.output_image_label.setScaledContents(True)
        self.pca_image_label.setScaledContents(True)

        # Frame skip spinbox
        self.frame_skip_label = QLabel("Frame Skip:", self)
        self.frame_skip_spinbox = QSpinBox(self)
        self.frame_skip_spinbox.setRange(0, 100)
        self.frame_skip_spinbox.setValue(0)
        
        # Detection type combobox
        self.detection_type_label = QLabel("Detection Type:", self)
        self.detection_type_combobox = QComboBox(self)
        self.detection_type_combobox.addItems(["Smoke", "Fire", "Combined"])
        self.detection_type_combobox.setCurrentIndex(1)  # Default to "Fire"

        # Output overlay checkbox - Set to checked by default
        self.output_overlay_checkbox = QCheckBox("Output Overlay", self)
        self.output_overlay_checkbox.setChecked(True)

        # Alpha slider for overlay transparency
        self.alpha_slider_label = QLabel("Overlay Alpha:", self)
        self.alpha_slider = QSlider(Qt.Horizontal, self)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(60)  # Default to 60% transparency
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

        # Button to select ROI and update GMM
        self.select_roi_button = QPushButton("Select ROI", self)
        self.select_roi_button.clicked.connect(self.update_gmm_with_selection)

        # Checkbox to enable/disable real-time GMM updating
        self.update_gmm_checkbox = QCheckBox("Update GMM with ROI", self)

        # Layouts for image display and controls
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.input_image_label)
        image_layout.addWidget(self.output_image_label)
        image_layout.addWidget(self.pca_image_label)

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
        controls_layout.addWidget(self.select_roi_button)
        controls_layout.addWidget(self.update_gmm_checkbox)
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
        self.pca_image_label.setFocusPolicy(Qt.NoFocus)
        self.frame_skip_spinbox.setFocusPolicy(Qt.ClickFocus)
        self.detection_type_combobox.setFocusPolicy(Qt.ClickFocus)
        self.output_overlay_checkbox.setFocusPolicy(Qt.ClickFocus)
        self.alpha_slider.setFocusPolicy(Qt.ClickFocus)
        self.select_dir_button.setFocusPolicy(Qt.ClickFocus)
        self.next_button.setFocusPolicy(Qt.NoFocus)
        self.prev_button.setFocusPolicy(Qt.NoFocus)
        self.select_roi_button.setFocusPolicy(Qt.ClickFocus)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            self.show_next_image()
        elif event.key() == Qt.Key_Left:
            self.show_prev_image()

    def show_instructions(self):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Instructions")
        msg.setText("Welcome to the Fire and Smoke Detection Application!\n\n"
                    "Instructions:\n"
                    "1. Use 'Select Image Directory' to load a sequence of images.\n"
                    "2. Navigate through images using the 'Next' and 'Previous' buttons or the left and right arrow keys.\n"
                    "3. Use the 'Frame Skip' option to skip images when navigating.\n"
                    "4. Choose a detection type: Smoke, Fire, or Combined.\n"
                    "5. Enable 'Output Overlay' to overlay the detection result on the image.\n"
                    "6. Adjust the 'Overlay Alpha' slider to set the transparency of the overlay.\n"
                    "7. Use 'Select ROI' to pause and manually select a region to update the GMM.\n"
                    "8. View the PCA output of the DINOv2-processed image in the third panel.")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

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
        if not self.image_files or self.is_paused:
            return
        
        current_image_path = self.image_files[self.current_index]
        frame = cv2.imread(current_image_path)
        self.current_frame = frame  # Store the current frame for ROI selection
        
        if frame is None:
            return
        
        detection_type = self.detection_type_combobox.currentText().lower()
        overlay_output = self.output_overlay_checkbox.isChecked()
        alpha = self.alpha_slider.value() / 100.0  # Get alpha value from slider

        if detection_type == "fire":
            output_map = self.create_fire_map(frame)
        elif detection_type == "smoke":
            output_map = self.create_smoke_map(self.prev_frame, frame)
        else:
            fire_map = self.create_fire_map(frame)
            smoke_map = self.create_smoke_map(self.prev_frame, frame)
            output_map = cv2.bitwise_or(fire_map, smoke_map)
        
        if overlay_output:
            output_map_colored = cv2.addWeighted(frame, 1 - alpha, output_map, alpha, 0)
        else:
            output_map_colored = output_map

        # Process PCA with DINOv2
        pca_output = self.process_pca_with_dino(frame)
        
        self.show_image(frame, self.input_image_label)
        self.show_image(output_map_colored, self.output_image_label)
        self.show_image(pca_output, self.pca_image_label)
        
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
        # Convert the frame to Lab color space
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab_frame)
        
        # Apply Gaussian Mixture Model to a and b channels
        fg_mask_a = self.gmm_a.apply(a)
        fg_mask_b = self.gmm_b.apply(b)
        
        # Combine the two masks by averaging them
        fire_map = cv2.addWeighted(fg_mask_a, 0.5, fg_mask_b, 0.5, 0)

        # Convert fire_map to a heatmap or color map for visualization
        fire_map_colored = cv2.applyColorMap(fire_map, cv2.COLORMAP_JET)

        return fire_map_colored

    def create_smoke_map(self, prev_frame, frame):
        if prev_frame is None:
            prev_frame = frame

        # Convert the current frame to grayscale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize the BackgroundSubtractorMOG2 with adjusted parameters if not already initialized
        if not hasattr(self, 'mog2_smoke'):
            self.mog2_smoke = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=8, detectShadows=True)
        
        # Apply the MOG2 background subtractor to the current frame
        fg_mask = self.mog2_smoke.apply(curr_gray)

        # Apply thresholding to the foreground mask to obtain a binary smoke map
        _, smoke_map = cv2.threshold(fg_mask, 150, 255, cv2.THRESH_BINARY)
        
        return smoke_map

    def update_gmm_with_selection(self):
        # Pause the current frame
        self.is_paused = True
        
        # Let the user select a region of interest (ROI)
        roi = cv2.selectROI("Select ROI", self.current_frame, showCrosshair=True, fromCenter=False)
        
        # Check if a valid ROI was selected
        if roi[2] > 0 and roi[3] > 0:
            x, y, w, h = roi
            selected_region = self.current_frame[y:y+h, x:x+w]
            
            # Convert the selected region to Lab color space
            lab_region = cv2.cvtColor(selected_region, cv2.COLOR_BGR2Lab)
            l, a, b = cv2.split(lab_region)
            
            # Update the GMM models with the new region
            self.gmm_a.apply(a)
            self.gmm_b.apply(b)
        
        # Close the ROI selection window
        cv2.destroyWindow("Select ROI")
        
        # Resume the processing
        self.is_paused = False
        self.process_and_display_current_image()

    def extract_feature_map(self, frame):
        # Preprocess the frame for DINOv2 (resize, normalize, etc.)
        frame_resized = cv2.resize(frame, (224, 224))
        frame_resized = frame_resized.astype(np.float32) / 255.0
        frame_resized = torch.tensor(frame_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        # Hook to capture intermediate features
        def hook_fn(module, input, output):
            self.feature_map = output

        # Attach the hook to the layer where you want to extract features
        handle = self.dinov2_model.blocks[-1].register_forward_hook(hook_fn)

        # Run the forward pass to get the feature map
        with torch.no_grad():
            _ = self.dinov2_model(frame_resized)

        # Remove the hook
        handle.remove()

        # The feature map now contains the 2D feature map
        feature_map_np = self.feature_map.cpu().numpy()[0]  # Shape: (257, 384)

        # Exclude the class token (first token)
        feature_map_np = feature_map_np[1:]  # Now shape: (256, 384)

        # Reshape the feature map to a 2D grid (16x16 patches)
        feature_map_np = feature_map_np.reshape(16, 16, -1)  # Shape: (16, 16, 384)

        return feature_map_np

    def process_pca_with_dino(self, frame):
        # Extract the 2D feature map
        feature_map = self.extract_feature_map(frame)

        # Apply PCA on the feature map
        n_patches = feature_map.shape[0]  # This should be 16 for a 16x16 grid
        pca = PCA(n_components=3)
        pca_result = np.zeros((n_patches, n_patches, 3))

        for i in range(n_patches):
            pca_result[i] = pca.fit_transform(feature_map[i])

        # Normalize and convert to RGB image
        pca_result -= pca_result.min()
        pca_result /= (pca_result.max() + 1e-8)  # Avoid division by zero
        pca_result *= 255
        pca_image = pca_result.astype(np.uint8)

        # Resize for display
        pca_image_resized = cv2.resize(pca_image, (self.pca_image_label.width(), self.pca_image_label.height()), interpolation=cv2.INTER_AREA)

        return pca_image_resized

    






if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = FireSmokeDetectionApp()
    ex.show()
    sys.exit(app.exec_())
