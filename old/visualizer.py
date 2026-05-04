"""
Sound Visualization Module
==========================
This module provides comprehensive audio visualization capabilities including:
- Waveform display
- Frequency spectrum (FFT) analysis
- Spectrogram visualization
- Real-time animation support
"""

import numpy as np
from typing import Tuple, Optional, List
from scipy import signal
from scipy.fft import fft, fftfreq
import colorsys


class AudioVisualizer:
    """
    Main visualization class for audio analysis and display.
    """
    
    def __init__(self, sample_rate: int = 44100):
        """Initialize the visualizer."""
        self.sample_rate = sample_rate
        
    def get_waveform_data(self, audio_data: np.ndarray, num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Get waveform data for visualization."""
        if len(audio_data) > num_points:
            indices = np.linspace(0, len(audio_data) - 1, num_points, dtype=int)
            waveform = audio_data[indices]
        else:
            waveform = audio_data
            
        duration = len(audio_data) / self.sample_rate
        time_axis = np.linspace(0, duration, len(waveform))
        
        return time_axis, waveform
    
    def get_spectrum_data(self, audio_data: np.ndarray, fft_size: int = None, window: str = "hann") -> Tuple[np.ndarray, np.ndarray]:
        """Get frequency spectrum data using FFT."""
        if fft_size is None:
            fft_size = 2 ** int(np.ceil(np.log2(len(audio_data))))
        
        if window == "hann":
            window_func = np.hanning(len(audio_data))
        elif window == "hamming":
            window_func = np.hamming(len(audio_data))
        elif window == "blackman":
            window_func = np.blackman(len(audio_data))
        else:
            window_func = np.ones(len(audio_data))
        
        windowed_data = audio_data * window_func
        
        if len(windowed_data) < fft_size:
            padded = np.zeros(fft_size)
            padded[:len(windowed_data)] = windowed_data
            windowed_data = padded
        
        fft_result = fft(windowed_data[:fft_size])
        frequencies = fftfreq(fft_size, 1.0 / self.sample_rate)
        
        positive_freq_idx = frequencies >= 0
        frequencies = frequencies[positive_freq_idx]
        magnitudes = np.abs(fft_result[positive_freq_idx])
        
        magnitudes_db = 20 * np.log10(magnitudes + 1e-10)
        
        audible_idx = frequencies <= 20000
        return frequencies[audible_idx], magnitudes_db[audible_idx]
    
    def get_spectrogram_data(self, audio_data: np.ndarray, window_size: int = 1024, hop_size: int = 512, window: str = "hann") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get spectrogram data for visualization."""
        frequencies, times, spectrogram = signal.spectrogram(
            audio_data,
            fs=self.sample_rate,
            window=window,
            nperseg=window_size,
            noverlap=window_size - hop_size,
            scaling='spectrum'
        )
        
        spectrogram_db = 20 * np.log10(spectrogram + 1e-10)
        
        freq_mask = frequencies <= 20000
        frequencies = frequencies[freq_mask]
        spectrogram_db = spectrogram_db[freq_mask, :]
        
        return frequencies, times, spectrogram_db


class WaveformRenderer:
    """
    Renderer for drawing waveform visualizations.
    """
    
    def __init__(self, width: int = 800, height: int = 300):
        """Initialize the renderer."""
        self.width = width
        self.height = height
        
    def render_waveform(self, audio_data: np.ndarray, color: Tuple[float, float, float] = (0.2, 0.6, 1.0), background: Tuple[float, float, float] = (0.1, 0.1, 0.15), grid_color: Tuple[float, float, float] = (0.3, 0.3, 0.35), show_grid: bool = True) -> np.ndarray:
        """Render waveform as an image array."""
        image = np.zeros((self.height, self.width, 3), dtype=np.float32)
        image[:, :] = background
        
        if show_grid:
            for i in range(5):
                y = int(self.height * (i + 1) / 6)
                image[y, :] = grid_color
            
            for i in range(9):
                x = int(self.width * (i + 1) / 10)
                image[:, x] = grid_color
        
        center_y = self.height // 2
        image[center_y, :] = (grid_color[0] * 1.2, grid_color[1] * 1.2, grid_color[2] * 1.2)
        
        if len(audio_data) > self.width:
            indices = np.linspace(0, len(audio_data) - 1, self.width, dtype=int)
            waveform = audio_data[indices]
        else:
            waveform = np.interp(
                np.linspace(0, len(audio_data) - 1, self.width),
                np.arange(len(audio_data)),
                audio_data
            )
        
        for x in range(self.width):
            amplitude = waveform[x]
            y_offset = int(amplitude * self.height / 2)
            y_start = center_y - y_offset
            y_end = center_y + y_offset
            
            y_start = max(0, min(self.height - 1, y_start))
            y_end = max(0, min(self.height - 1, y_end))
            
            for y in range(min(y_start, y_end), max(y_start, y_end) + 1):
                image[y, x] = color
        
        return image
    
    def render_spectrum(self, frequencies: np.ndarray, magnitudes: np.ndarray, color: Tuple[float, float, float] = (0.2, 0.8, 0.4), background: Tuple[float, float, float] = (0.1, 0.1, 0.15), grid_color: Tuple[float, float, float] = (0.3, 0.3, 0.35), show_grid: bool = True, log_freq: bool = True, min_db: float = -80, max_db: float = 0) -> np.ndarray:
        """Render frequency spectrum as an image array."""
        image = np.zeros((self.height, self.width, 3), dtype=np.float32)
        image[:, :] = background
        
        if show_grid:
            for db in range(int(min_db), int(max_db) + 1, 20):
                y = int(self.height * (max_db - db) / (max_db - min_db))
                y = max(0, min(self.height - 1, y))
                image[y, :] = grid_color
            
            freq_markers = [100, 500, 1000, 5000, 10000]
            for freq in freq_markers:
                if log_freq:
                    x = int(self.width * (np.log10(freq / 20) / np.log10(20000 / 20)))
                else:
                    x = int(self.width * freq / 20000)
                x = max(0, min(self.width - 1, x))
                image[:, x] = grid_color
        
        if log_freq:
            freq_range = np.logspace(np.log10(20), np.log10(20000), self.width)
            interp_mags = np.interp(freq_range, frequencies, magnitudes)
        else:
            interp_mags = np.interp(np.linspace(20, 20000, self.width), frequencies, magnitudes)
        
        for x in range(self.width):
            mag = interp_mags[x]
            normalized = (mag - min_db) / (max_db - min_db)
            normalized = max(0, min(1, normalized))
            
            bar_height = int(normalized * self.height)
            
            for y in range(self.height - bar_height, self.height):
                if 0 <= y < self.height:
                    gradient = 1 - (self.height - 1 - y) / self.height
                    r = color[0] * (0.5 + 0.5 * gradient)
                    g = color[1] * (0.5 + 0.5 * gradient)
                    b = color[2] * (0.5 + 0.5 * gradient)
                    image[y, x] = (r, g, b)
        
        return image
    
    def render_spectrogram(self, frequencies: np.ndarray, times: np.ndarray, spectrogram: np.ndarray, colormap: str = "viridis", background: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
        """Render spectrogram as an image array."""
        from scipy.ndimage import zoom
        
        image = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        freq_zoom = self.height / len(frequencies)
        time_zoom = self.width / len(times)
        
        resized = zoom(spectrogram, (freq_zoom, time_zoom), order=1)
        
        min_val = np.min(resized)
        max_val = np.max(resized)
        
        if max_val - min_val > 0:
            normalized = (resized - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(resized)
        
        colors = self._get_colormap_array(normalized.flatten(), colormap)
        colors = colors.reshape(self.height, self.width, 3)
        
        image = np.flip(colors, axis=0)
        
        return image
    
    def _get_colormap_array(self, values: np.ndarray, colormap: str) -> np.ndarray:
        """Apply colormap to normalized values."""
        colors = np.zeros((len(values), 3))
        
        for i, val in enumerate(values):
            val = max(0, min(1, val))
            
            if colormap == "viridis":
                r = 0.267 + 0.329 * val + 0.386 * val ** 2
                g = 0.004 + 1.376 * val - 0.624 * val ** 2
                b = 0.329 + 0.996 * val - 0.542 * val ** 2
            elif colormap == "magma":
                r = 0.001 + 1.446 * val - 0.447 * val ** 2
                g = 0.002 + 0.864 * val + 0.134 * val ** 2
                b = 0.014 + 2.127 * val - 1.141 * val ** 2
            elif colormap == "rainbow":
                h = 0.7 - val * 0.7
                r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            elif colormap == "grayscale":
                r = g = b = val
            else:
                r = val
                g = 0.3
                b = 1 - val
            
            colors[i] = [min(1, max(0, r)), min(1, max(0, g)), min(1, max(0, b))]
        
        return colors
    
    def render_waveform_and_spectrum(self, audio_data: np.ndarray, visualizer: AudioVisualizer) -> Tuple[np.ndarray, np.ndarray]:
        """Render both waveform and spectrum visualizations."""
        time_axis, waveform = visualizer.get_waveform_data(audio_data, self.width)
        frequencies, magnitudes = visualizer.get_spectrum_data(audio_data)
        
        waveform_image = self.render_waveform(waveform)
        spectrum_image = self.render_spectrum(frequencies, magnitudes)
        
        return waveform_image, spectrum_image