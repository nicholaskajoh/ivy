from PyQt5.QtWidgets import QApplication
from gui.models import Camera
from gui.views import MainWindow

camera = Camera(0)
camera.initialize()

app = QApplication([])
app.setStyle('Fusion')
main_window = MainWindow(camera)
main_window.show()
app.exit(app.exec_())