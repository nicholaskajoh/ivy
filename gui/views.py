import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QHBoxLayout, QLineEdit, QCheckBox, QComboBox, QSpinBox, QLabel, QFormLayout, QVBoxLayout
from pyqtgraph import ImageView


class MainWindow(QMainWindow):
    def __init__(self, camera=None):
        super().__init__()
        self.central_widget = QWidget()
        self.parent_layout = QVBoxLayout()
        self.central_widget.setLayout(self.parent_layout)

        self.toolbar_layout = QHBoxLayout()
        
        self.content_layout = QHBoxLayout(self.central_widget)
        self.config_widget = ConfigurationsWidget()


        self.camera = camera
        self.enable_webcam_button = QPushButton('Enable webcam', self.central_widget)
        self.disable_webcam_button = QPushButton('Disable webcam', self.central_widget)
        self.disable_webcam_button.setEnabled(False)

        self.enable_webcam_button.clicked.connect(self.enable_webcam_button_pressed)
        self.disable_webcam_button.clicked.connect(self.disable_webcam_button_pressed)

        self.enable_streaming_button = QPushButton('Enable streaming', self.central_widget)
        self.disable_streaming_button = QPushButton('Disable streaming', self.central_widget)

        self.enable_streaming_button.clicked.connect(self.enable_streaming_button_pressed)
        self.disable_streaming_button.clicked.connect(self.disable_streaming_button_pressed)
        self.disable_streaming_button.setEnabled(False)

        self.image_view = ImageView()
        self.content_layout.addWidget(self.config_widget)
        self.content_layout.addWidget(self.image_view)

        self.droi = []
        self.is_webcam_enabled = False;
        self.is_streaming_enabled = False;

        self.toolbar = QWidget()
        self.toolbar.setLayout(self.toolbar_layout)
        self.toolbar_layout.addWidget(self.enable_webcam_button)
        self.toolbar_layout.addWidget(self.disable_webcam_button)
        self.toolbar_layout.addWidget(self.enable_streaming_button)
        self.toolbar_layout.addWidget(self.disable_streaming_button)

        self.parent_layout.ad

        self.setCentralWidget(self.central_widget)

    def update_image(self):
        frame = self.camera.get_frame()
        self.image_view.setImage(frame.T)

    def update_video(self):
        self.image_view.setImage(self.camera.last_frame.T)

    def update_brightness(self, value):
        value /= 10
        self.camera.set_brightness(value)

    def start_video_playback(self):
        self.video_thread = VideoThread(self.camera)
        self.video_thread.start()
        self.update_timer.start(30)
    
    def enable_webcam_button_pressed(self):
        self.is_webcam_enabled = True     
        self.enable_webcam_button.setEnabled(False)
        self.disable_webcam_button.setEnabled(True)   
        print('Webcam enabled')

    def disable_webcam_button_pressed(self):
        self.is_webcam_enabled = False
        self.disable_webcam_button.setEnabled(False)
        self.enable_webcam_button.setEnabled(True)    
        self.cap.release()
        print('Webcam disabled')

    def enable_streaming_button_pressed(self):
        self.enable_streaming_button.setEnabled(False)
        self.disable_streaming_button.setEnabled(True)
        print('Streaming enabled')

    def disable_streaming_button_pressed(self):
        self.disable_streaming_button.setEnabled(False)
        self.enable_streaming_button.setEnabled(True)
        print('Streaming disabled')

class VideoThread(QThread):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera

    def run(self):
        self.camera.acquire_video(200)

class ConfigurationsWidget(QWidget):

    def __init__(self):
        super().__init__()
        self.initialize_ui()

    def initialize_ui(self):
        self.layout = QFormLayout(self.central_widget)
        
        self.name_label = QLabel()
        self.name_text = QLineEdit()
        self.layout.addRow(self.name_label, self.name_text)

        self.show_roi_lb = 'Show DROI'
        self.show_roi_cb = QCheckBox()
        self.layout.addRow(self.show_roi_lb, self.show_roi_lb)

        self.mctf_lb = 'Maximum Executive Tracking Failure'
        self.mctf_sb = QSpinBox()
        self.mctf_sb.setValue(3)
        self.mctf_sb.setMinimum(2)
        self.mctf_sb.setMaximum(5)
        self.layout.addRow(self.mctf_lb, self.mctf_sb)

        self.detection_interval_lb = 'Detection Interval'
        self.detection_interval_sb = QSpinBox()
        self.detection_interval_sb.setValue(10)
        self.layout.addRow(self.detection_interval_lb, self.detection_interval_lb)

        self.detector_lb = 'Detector algorithm'
        self.detector_cmb = QComboBox()
        self.detector_cmb.addItems(['YOLO', 'HAARC','BGSUB', 'SSD'])
        self.layout.addRow(self.detector_lb, self.detector_cmb)

        self.tracker_lb = 'Tracker algorithm'
        self.tracker_cmb = QComboBox
        self.tracker_cmb.addItems(['CSRT', 'KCF','CAMSHIFT'])
        self.layout.addRow(self.tracker_lb, self.tracker_cmb)

        self.records_lb = 'Keep records'
        self.records_cb = QCheckBox()
        self.layout.addRow(self.records_lb, self.records_cb)

        self.cl_position_lb = 'Counting line position'
        self.cl_position_cmb = QComboBox()
        self.cl_position_cmb.addItems(['TOP', 'BOTTOM', 'LEFT','BOTTOM'])
        self.layout.addRow(self.cl_position_lb, self.cl_position_cmb)

