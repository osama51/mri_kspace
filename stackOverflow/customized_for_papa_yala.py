import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class ProcessRunnable(QRunnable):
    def __init__(self, target, args):
        QRunnable.__init__(self)
        self.target = target
        self.args = args

    def run(self):
        self.target(*self.args)

    def start(self):
        QThreadPool.globalInstance().start(self)

def display_kspace():
    # Your implementation of display_kspace
    # Update UI element here

    print("Displaying k-space")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Background Task Example')
        self.resize(350, 250)

        self.start_button = QPushButton('Start', self)
        self.start_button.clicked.connect(self.start_background_task)

        layout = QVBoxLayout()
        layout.addWidget(self.start_button)
        self.setLayout(layout)

    def start_background_task(self):
        self.start_button.setEnabled(False)

        # Create and start the background task
        task = ProcessRunnable(target=display_kspace, args=())
        task.start()

        # Enable the button after 5 seconds (just for demonstration)
        QTimer.singleShot(5000, self.enable_button)

    def enable_button(self):
        self.start_button.setEnabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
