


# Fire and Smoke Detection

💡 This project demonstrates a demo application for detecting fire and smoke in images. The application allows users to select a directory of images, process them for fire and smoke detection, and visualize the results using a GUI built with PyQt. Additionally, the project integrates DINO-based attention maps for deeper insights into the regions the model is focusing on.

## Video Demonstration

Check out the video demonstration of the application in action:
![sample video](firesample.mov)


## Usage

### 1. Clone the Repository

Clone the repository to your local machine using the following command:

```bash
git clone https://github.com/bluesaiyancodes/Fire-and-Smoke-Detection.git
```

### 2. Create a New Environment and Install Required Packages

Navigate to the project directory and install the required packages by running:

```bash
pip install -r requirements.txt
```

### 3. Run the Application

You can run the application by executing the following command:

```bash
python fire_main.py
```

This will launch the GUI where you can select image directories, process images for fire and smoke detection, and view the results.

### 4. Build a Windows Application

To build a standalone Windows application, use the following command:

```bash
pyinstaller fire_main.py
```

After the build process, the application will be available in the `dist/` directory. Ensure that the `_internal` directory is also included when distributing the application. For more details on packaging with PyInstaller, refer to the [PyInstaller documentation](https://pyinstaller.org/en/stable/).

## Dataset

The sample dataset provided with this project is part of a larger fire and smoke dataset (화재발생예측(2024)) from [AI HUB Korea](https://www.aihub.or.kr/). Sample images of smoke and fire are included to test the application's capabilities.

## Additional Information

This project leverages several key technologies and models:

- **PyQt**: Used for building the graphical user interface (GUI) that allows users to interact with the application.
- **OpenCV**: Handles image processing tasks such as fire and smoke detection using Gaussian Mixture Models (GMM) and Background Subtractor MOG2.
- **DINO**: A vision transformer model from Facebook AI Research (FAIR) used to generate attention maps, highlighting areas in images that the model deems important for prediction.
  - I have used Dinov1 instead of v2 as it was easier to get the attention maps from v1 following the guides from the [repo code](https://github.com/facebookresearch/dino/blob/main/visualize_attention.py).
- **PyInstaller**: A tool used to bundle the Python application into standalone executables, making it easier to distribute.

The application is designed for demonstration purposes and can be expanded or customized for more extensive fire and smoke detection projects.
