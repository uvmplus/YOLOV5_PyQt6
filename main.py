
"""
Main entry point for the YOLOv5 Detector application.
This script initializes and runs the application.
"""
import sys
import os

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
print(f'sys.path = {sys.path}')

from PyQt6.QtWidgets import QApplication
from gui.main_windows import MainWindow
def main():
    """Initialize and run the application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()