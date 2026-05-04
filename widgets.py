from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtCore import Qt, QThread, Signal, QMutex, QMutexLocker
from PySide6.QtGui import QImage, QPixmap, QPainter
import numpy as np

class ImageWidget(QWidget):
    """Custom widget to display numpy array visualizations efficiently."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = QPixmap()
        self._image_data = None  # Prevent GC of underlying bytes
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 150)

    def set_image(self, image_array: np.ndarray):
        if image_array is None:
            return
        
        # Ensure it's a numpy array on CPU
        if hasattr(image_array, 'get'): # CuPy
            image_array = image_array.get()
            
        # Clip and convert
        img_uint8 = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
        h, w, ch = img_uint8.shape
        
        # Ensure contiguous array for QImage
        if not img_uint8.flags['C_CONTIGUOUS']:
            img_uint8 = np.ascontiguousarray(img_uint8)

        bytes_per_line = ch * w
        
        # Create QImage referencing the numpy buffer
        # Note: img_uint8 must persist as long as QImage exists
        q_img = QImage(img_uint8.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to QPixmap (copies data, safe to reuse img_uint8 buffer next frame)
        self._pixmap = QPixmap.fromImage(q_img)
        
        # Keep reference to array just in case, though QPixmap is now independent
        self._image_data = img_uint8
        
        self.update()

    def paintEvent(self, event):
        if not self._pixmap.isNull():
            painter = QPainter(self)
            # Scale pixmap to fit widget size smoothly
            scaled_pixmap = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            # Center it
            x = (self.width() - scaled_pixmap.width()) // 2
            y = (self.height() - scaled_pixmap.height()) // 2
            painter.drawPixmap(x, y, scaled_pixmap)
            painter.end()

    def sizeHint(self):
        return self._pixmap.size() if not self._pixmap.isNull() else super().sizeHint()


class PlaybackWorker(QThread):
    """Worker thread for non-blocking audio playback."""
    finished = Signal()
    error = Signal(str)

    def __init__(self, audio_data, sample_rate):
        super().__init__()
        self.audio_data = audio_data.copy()  # Copy to prevent race conditions
        self.sample_rate = sample_rate
        self._stop_flag = False
        self._mutex = QMutex()

    def run(self):
        # Try sounddevice first
        try:
            import sounddevice as sd
            sd.play(self.audio_data, self.sample_rate)
            stream = sd.get_stream()
            while stream is not None and stream.active:
                {
                    QMutexLocker(self._mutex)
                }
                if self._stop_flag:
                    break
                QThread.msleep(50)
            sd.stop()
        except ImportError:
            # Fallback to pygame
            try:
                import pygame
                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=1)
                
                audio_int = (self.audio_data * 32767).astype(np.int16)
                sound = pygame.sndarray.make_sound(audio_int)
                sound.play()
                
                while pygame.mixer.get_busy():
                    {
                        QMutexLocker(self._mutex)
                    }
                    if self._stop_flag:
                        pygame.mixer.stop()
                        break
                    QThread.msleep(50)
                    
            except Exception as e:
                self.error.emit(f"Playback failed (install sounddevice or pygame):\n{str(e)}")
                return
        except Exception as e:
            self.error.emit(f"Playback error:\n{str(e)}")
            return
            
        self.finished.emit()

    def stop_playback(self):
        locker = QMutexLocker(self._mutex)
        self._stop_flag = True