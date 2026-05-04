#!/usr/bin/env python3
import sys
from PySide6.QtWidgets import QApplication

# High DPI scaling is enabled by default in PySide6 (Qt6), 
# so we do not need to set AA_EnableHighDpiScaling anymore.

from main_window import SoundEffectsApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    window = SoundEffectsApp()
    window.show()
    sys.exit(app.exec())