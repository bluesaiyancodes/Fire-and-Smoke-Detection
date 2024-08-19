import sys
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QWidget, QSlider, QGroupBox, QHBoxLayout, QButtonGroup)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from torchvision import transforms as pth_transforms
from PIL import Image


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setText('\n\n Drop Image Here \n\n')
        self.setStyleSheet('''
            QLabel{
                border: 4px dashed #aaa
            }
        ''')

    def setPixmap(self, image):
        super().setPixmap(image)


class FireSmokeDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.current_frame = None
        self.prev_frame = None
        self.image_path = None
        self.is_paused = False
        self.first_image_processed = False  # To track if the first image has been processed
        self.gmm_initialized = False  # Track GMM initialization status

        # Track previous detected regions (up to 5 frames)
        self.prev_detected_regions = []

        # Initialize GMMs with short history for better responsiveness
        self.gmm_a = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=16, detectShadows=False)
        self.gmm_b = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=16, detectShadows=False)

        # Initialize BackgroundSubtractorMOG2 for frame difference
        self.mog2 = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=16, detectShadows=False)

        # Load the DINOv2 model
        self.dinov2_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
        #self.dinov2_model = torch.hub.load("mhamilton723/FeatUp", 'dino16') -> Needs Cuda, not working on my machine
        self.dinov2_model.eval()  # Set the model to evaluation mode

    def initUI(self):
        self.setWindowTitle("Fire and Smoke Detection")
        self.setFixedSize(1300, 700)
        self.setAcceptDrops(True)

        # Input and output image labels
        self.input_image_label = ImageLabel()
        self.output_image_label = QLabel(self)
        self.output_image_label.setFixedSize(580, 450)
        self.output_image_label.setAlignment(Qt.AlignCenter)
        self.output_image_label.setScaledContents(True)

        # Buttons for selecting GMM, Frame Diff, Dino, and Contrast
        self.gmm_button = QPushButton("GMM", self)
        self.gmm_button.clicked.connect(self.set_gmm_mode)

        self.frame_diff_button = QPushButton("Frame Diff", self)
        self.frame_diff_button.clicked.connect(self.set_frame_diff_mode)

        self.dino_button = QPushButton("DINO", self)
        self.dino_button.clicked.connect(self.set_dino_mode)

        self.contrast_button = QPushButton("Contrast", self)
        self.contrast_button.clicked.connect(self.set_contrast_mode)

        # Group the buttons so only one can be selected at a time
        self.button_group = QButtonGroup()
        self.button_group.addButton(self.gmm_button)
        self.button_group.addButton(self.frame_diff_button)
        self.button_group.addButton(self.dino_button)
        self.button_group.addButton(self.contrast_button)

        # Default to GMM mode
        self.set_gmm_mode()

        # Alpha slider for overlay transparency
        self.alpha_slider_label = QLabel("Overlay Alpha:", self)
        self.alpha_slider = QSlider(Qt.Horizontal, self)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(60)
        self.alpha_slider.setTickPosition(QSlider.TicksBelow)
        self.alpha_slider.setTickInterval(10)

        # Process button
        self.process_button = QPushButton("Process", self)
        self.process_button.clicked.connect(self.process_current_image)

        # Update GMM button
        self.update_gmm_button = QPushButton("Update", self)
        self.update_gmm_button.clicked.connect(self.handle_update)

        # Layouts for image display and controls
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.input_image_label)
        image_layout.addWidget(self.output_image_label)

        # Controls layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.gmm_button)
        button_layout.addWidget(self.frame_diff_button)
        button_layout.addWidget(self.dino_button)
        button_layout.addWidget(self.contrast_button)
        button_layout.addWidget(self.process_button)
        button_layout.addWidget(self.update_gmm_button)

        controls_layout = QVBoxLayout()
        controls_layout.addLayout(button_layout)

        # Adjusted position for alpha slider and label
        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(self.alpha_slider_label)
        alpha_layout.addWidget(self.alpha_slider)

        controls_layout.addLayout(alpha_layout)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(image_layout)
        main_layout.addLayout(controls_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.load_image(file_path)
            event.acceptProposedAction()
        else:
            event.ignore()

    def load_image(self, file_path):
        # Save the current frame as the previous frame if it's the first load
        if self.current_frame is not None:
            self.prev_frame = self.current_frame.copy()

        # Load the new image
        self.image_path = file_path
        frame = cv2.imread(file_path)
        if frame is not None:
            self.current_frame = frame
            self.show_image(frame, self.input_image_label)

        # Automatically update GMM and process the first image
        if not self.first_image_processed:
            self.update_gmm_with_selection()
            self.first_image_processed = True
            self.process_current_image()

    def set_gmm_mode(self):
        self.selected_mode = "gmm"
        self.gmm_button.setStyleSheet("background-color: lightblue")
        self.frame_diff_button.setStyleSheet("")
        self.dino_button.setStyleSheet("")
        self.contrast_button.setStyleSheet("")

    def set_frame_diff_mode(self):
        self.selected_mode = "frame_diff"
        self.gmm_button.setStyleSheet("")
        self.frame_diff_button.setStyleSheet("background-color: lightblue")
        self.dino_button.setStyleSheet("")
        self.contrast_button.setStyleSheet("")

    def set_dino_mode(self):
        self.selected_mode = "dino"
        self.gmm_button.setStyleSheet("")
        self.frame_diff_button.setStyleSheet("")
        self.dino_button.setStyleSheet("background-color: lightblue")
        self.contrast_button.setStyleSheet("")

    def set_contrast_mode(self):
        self.selected_mode = "contrast"
        self.gmm_button.setStyleSheet("")
        self.frame_diff_button.setStyleSheet("")
        self.dino_button.setStyleSheet("")
        self.contrast_button.setStyleSheet("background-color: lightblue")

    def process_current_image(self):
        if self.current_frame is None:
            return

        if self.selected_mode == "gmm":
            output_image = self.create_fire_map(self.current_frame)
        elif self.selected_mode == "frame_diff":
            output_image = self.create_frame_diff_map(self.current_frame)
        elif self.selected_mode == "dino":
            output_image = self.process_attention_with_dino(self.current_frame)
        elif self.selected_mode == "contrast":
            output_image = self.create_contrast_map(self.current_frame)

        self.show_image(output_image, self.output_image_label)

    def show_image(self, image, label):
        resized_image = cv2.resize(image, (label.width(), label.height()), interpolation=cv2.INTER_AREA)
        qimage = QImage(resized_image.data, resized_image.shape[1], resized_image.shape[0], resized_image.strides[0], QImage.Format_BGR888)
        label.setPixmap(QPixmap.fromImage(qimage))

    def create_fire_map(self, frame):
        # Convert the frame to Lab color space
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab_frame)

        # Apply Gaussian Mixture Model to a and b channels
        fg_mask_a = self.gmm_a.apply(a)
        fg_mask_b = self.gmm_b.apply(b)

        # Combine the two masks using bitwise AND
        fire_map = cv2.addWeighted(fg_mask_a, 0.5, fg_mask_b, 0.5, 0)

        # Convert fire_map to a heatmap or color map for visualization
        fire_map_colored = cv2.applyColorMap(fire_map, cv2.COLORMAP_JET)

        # Apply alpha blending to overlay the fire map on the input image
        alpha = self.alpha_slider.value() / 100.0
        overlayed_image = cv2.addWeighted(frame, 1 - alpha, fire_map_colored, alpha, 0)

        return overlayed_image

    def create_frame_diff_map(self, frame):
        # Apply MOG2 to the current frame
        fg_mask = self.mog2.apply(frame)

        # Apply thresholding to the foreground mask to obtain a binary difference map
        _, diff_map = cv2.threshold(fg_mask, 150, 255, cv2.THRESH_BINARY)

        # Apply alpha blending to overlay the difference map on the input image
        alpha = self.alpha_slider.value() / 100.0
        diff_map_colored = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)
        overlayed_image = cv2.addWeighted(frame, 1 - alpha, diff_map_colored, alpha, 0)

        return overlayed_image

    # contrast inspiired from : https://stackoverflow.com/questions/66373003/computing-a-contrast-map
    def create_contrast_map(self, frame):
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            return frame  # No previous frame to compare with

        # Convert to grayscale for brightness and contrast calculations
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)

        h, w = current_gray.shape
        grid_lines_x = self.generate_fisheye_grid_lines(w, num_grids=10)
        grid_lines_y = self.generate_fisheye_grid_lines(h, num_grids=10)

        current_detected_regions = np.zeros_like(frame)

        for i in range(len(grid_lines_y) - 1):
            for j in range(len(grid_lines_x) - 1):
                x_start = grid_lines_x[j]
                y_start = grid_lines_y[i]
                x_end = grid_lines_x[j + 1]
                y_end = grid_lines_y[i + 1]

                if x_end <= x_start or y_end <= y_start:
                    continue

                # Extract patches from the current and previous frames
                patch_current = current_gray[y_start:y_end, x_start:x_end]
                patch_prev = prev_gray[y_start:y_end, x_start:x_end]

                if patch_current.size == 0 or patch_prev.size == 0:
                    continue

                # Calculate brightness and contrast differences
                brightness_diff = np.abs(np.mean(patch_current) - np.mean(patch_prev))
                contrast_diff = np.abs(np.std(patch_current) - np.std(patch_prev))

                # Dynamic thresholds based on overall brightness
                brightness_threshold = max(15, np.mean(current_gray) * 0.1)
                contrast_threshold = max(10, np.std(current_gray) * 0.1)

                if brightness_diff > brightness_threshold or contrast_diff > contrast_threshold:
                    # Highlight detected regions
                    current_detected_regions[y_start:y_end, x_start:x_end] = (0, 255, 0)

                # Draw faded grid lines
                faded_gray = (150, 150, 150)
                cv2.rectangle(current_detected_regions, (x_start, y_start), (x_end, y_end), faded_gray, 1)

        # Track previous detected regions (up to 5 frames)
        if len(self.prev_detected_regions) >= 5:
            self.prev_detected_regions.pop(0)
        self.prev_detected_regions.append(current_detected_regions.copy())

        # Fade out previous frames' detected regions
        overlayed_image = frame.copy()
        for idx, prev_regions in enumerate(reversed(self.prev_detected_regions)):
            alpha = (idx + 1) / len(self.prev_detected_regions) * 0.5
            overlayed_image = cv2.addWeighted(overlayed_image, 1 - alpha, prev_regions, alpha, 0)

        # Detect upward movement pattern and draw the arrow
        movement_direction = self.detect_movement_direction()
        #if movement_direction == "up":
            #cv2.arrowedLine(overlayed_image, (w // 2, h - 30), (w // 2, 30), (255, 255, 255), 5)  # White arrow pointing upwards

        self.prev_frame = frame.copy()
        return overlayed_image

    def detect_movement_direction(self):
        if len(self.prev_detected_regions) < 2:
            return None

        movement_detected = False
        previous_centers = []

        for detected_regions in self.prev_detected_regions:
            moments = cv2.moments(cv2.cvtColor(detected_regions, cv2.COLOR_BGR2GRAY))
            if moments['m00'] != 0:
                center_y = int(moments['m01'] / moments['m00'])
                previous_centers.append(center_y)

        # Check if centers are moving upwards
        if len(previous_centers) >= 2:
            upward_movement = all(previous_centers[i] > previous_centers[i + 1] for i in range(len(previous_centers) - 1))
            if upward_movement:
                return "up"

        return None


    def generate_fisheye_grid_lines(self, size, num_grids):
        # Generate non-uniform grid lines, dense at the edges and sparse in the center
        mid_point = size // 2
        grid_lines = [0]

        for i in range(1, num_grids + 1):
            # Symmetrically generate grid positions from the center outward
            position_start = int(mid_point * (i / num_grids) ** 2)
            position_end = size - position_start

            # Avoid adding duplicate points
            if position_start not in grid_lines:
                grid_lines.append(position_start)
            if position_end not in grid_lines:
                grid_lines.append(position_end)

        grid_lines = sorted(set(grid_lines))  # Sort and remove duplicates
        return grid_lines

    def handle_update(self):
        if self.selected_mode == "gmm":
            self.update_gmm_with_selection()
        elif self.selected_mode == "frame_diff":
            self.reset_frame_diff()
        elif self.selected_mode == "contrast":
            self.reset_contrast()

    def update_gmm_with_selection(self):
        if self.current_frame is None:
            return

        self.is_paused = True
        
        # Step 1: Select the region of interest (ROI)
        roi = cv2.selectROI("Select ROI", self.current_frame, showCrosshair=True, fromCenter=False)
        if roi[2] > 0 and roi[3] > 0:
            x, y, w, h = roi
            selected_region = self.current_frame[y:y+h, x:x+w]
            lab_region = cv2.cvtColor(selected_region, cv2.COLOR_BGR2Lab)
            l, a, b = cv2.split(lab_region)

            # Step 2: Apply GMM for the first time
            fg_mask_a = self.gmm_a.apply(a)
            fg_mask_b = self.gmm_b.apply(b)

            # Combine the masks for the first application
            initial_fire_map = cv2.addWeighted(fg_mask_a, 0.5, fg_mask_b, 0.5, 0)

            # Step 3: Apply GMM again to refine the detection
            # Use the initial_fire_map as a mask to focus on the identified regions
            refined_fg_mask_a = self.gmm_a.apply(initial_fire_map)
            refined_fg_mask_b = self.gmm_b.apply(initial_fire_map)

            # Combine the masks for the second application
            refined_fire_map = cv2.addWeighted(refined_fg_mask_a, 0.5, refined_fg_mask_b, 0.5, 0)

            # Update the current frame's GMM models with the refined detection
            self.gmm_a.apply(a)
            self.gmm_b.apply(b)

        cv2.destroyWindow("Select ROI")
        self.is_paused = False

        # Ensure GMM remains initialized after update
        self.gmm_initialized = True

        # Process the current image after updating GMM
        self.process_current_image()

    def reset_frame_difference(self):
        # Reset the previous frame used for frame difference
        self.prev_frame = None
        print("Frame difference reset.")

    def reset_contrast(self):
        # Reset the previous detected regions for contrast detection
        self.prev_detected_regions.clear()
        print("Contrast detection reset.")


    def process_attention_with_dino(self, frame):
        transform = pth_transforms.Compose([
            pth_transforms.Resize((224, 224)),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        frame_resized = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        frame_resized = frame_resized.to(device)

        patch_size = self.dinov2_model.patch_embed.patch_size

        with torch.no_grad():
            attentions = self.dinov2_model.get_last_selfattention(frame_resized)

        nh = attentions.shape[1]
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
        w_featmap = frame_resized.shape[-2] // patch_size
        h_featmap = frame_resized.shape[-1] // patch_size
        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
        attention_map = attentions.mean(axis=0)

        attention_map -= attention_map.min()
        attention_map /= attention_map.max()
        attention_map *= 255
        attention_map = attention_map.astype(np.uint8)

        heatmap = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)
        heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

        alpha = self.alpha_slider.value() / 100.0
        overlayed_image = cv2.addWeighted(frame, 1 - alpha, heatmap_resized, alpha, 0)

        return overlayed_image


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = FireSmokeDetectionApp()
    ex.show()
    sys.exit(app.exec_())
