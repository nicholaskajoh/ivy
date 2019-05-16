import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QHBoxLayout

def enable_webcam_button_pressed():
    enable_webcam_button.setEnabled(False)
    disable_webcam_button.setEnabled(True)
    print('Webcam enabled')

    ret, frame = cap.read()
    print(np.min(frame))

def disable_webcam_button_pressed():
    enable_webcam_button.setEnabled(False)
    disable_webcam_button.setEnabled(True)    
    print('Webcam disabled')

    ret, frame = cap.read()
    print(np.max(frame))

def enable_streaming_button_pressed():
    enable_streaming_button.setEnabled(False)
    disable_streaming_button.setEnabled(True)
    print('Streaming enabled')

def disable_streaming_button_pressed():
    disable_streaming_button.setEnabled(False)
    enable_streaming_button.setEnabled(True)
    print('Streaming disabled')

app = QApplication([])
main_win = QMainWindow()
central_widget = QWidget()
toolbar_layout = QHBoxLayout(central_widget)
cap = cv2.VideoCapture(0)

enable_webcam_button = QPushButton('Enable webcam', central_widget)
disable_webcam_button = QPushButton('Disable webcam', central_widget)
disable_webcam_button.setEnabled(False)

enable_webcam_button.clicked.connect(enable_webcam_button_pressed)
disable_webcam_button.clicked.connect(disable_webcam_button_pressed)

enable_streaming_button = QPushButton('Enable streaming', central_widget)
disable_streaming_button = QPushButton('Disable streaming', central_widget)

enable_streaming_button.clicked.connect(enable_streaming_button_pressed)
disable_streaming_button.clicked.connect(disable_streaming_button_pressed)
disable_streaming_button.setEnabled(False)

toolbar_layout.addWidget(enable_webcam_button)
toolbar_layout.addWidget(disable_webcam_button)
toolbar_layout.addWidget(enable_streaming_button)
toolbar_layout.addWidget(disable_streaming_button)

main_win.setCentralWidget(central_widget)
main_win.show()
app.exit(app.exec_())
cap.release()
