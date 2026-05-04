"""
Sound Visualization Module - GPU Accelerated Rendering (CuPy + Numba)
"""
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

try:
    import cupy as cp
    from cupyx.scipy.ndimage import zoom as cp_zoom
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    from scipy.ndimage import zoom as cp_zoom
    GPU_AVAILABLE = False

try:
    from numba import cuda
    NUMBA_CUDA_AVAILABLE = True
except ImportError:
    NUMBA_CUDA_AVAILABLE = False

def to_cpu(arr):
    if GPU_AVAILABLE and hasattr(arr, 'get'):
        return arr.get()
    return arr

def to_gpu(arr):
    if GPU_AVAILABLE and not isinstance(arr, cp.ndarray):
        return cp.asarray(arr)
    return arr

if NUMBA_CUDA_AVAILABLE:
    @cuda.jit
    def _cuda_render_waveform_kernel(image, waveform, height, width, color, grid_color, center_y):
        x = cuda.grid(1)
        if x < width:
            # Draw center line
            image[center_y, x, 0] = grid_color[0]
            image[center_y, x, 1] = grid_color[1]
            image[center_y, x, 2] = grid_color[2]
            
            amplitude = waveform[x]
            y_offset = int(amplitude * height / 2)
            y_start = max(0, min(height - 1, center_y - y_offset))
            y_end = max(0, min(height - 1, center_y + y_offset))
            
            for y in range(min(y_start, y_end), max(y_start, y_end) + 1):
                image[y, x, 0] = color[0]
                image[y, x, 1] = color[1]
                image[y, x, 2] = color[2]

    @cuda.jit
    def _cuda_render_spectrum_kernel(image, interp_mags, height, width, color, min_db, max_db):
        x = cuda.grid(1)
        if x < width:
            normalized = max(0.0, min(1.0, (interp_mags[x] - min_db) / (max_db - min_db)))
            bar_height = int(normalized * height)
            bar_top = height - bar_height
            for y in range(height):
                if y >= bar_top:
                    gradient = 0.5 + 0.5 * (y - bar_top) / height
                    image[y, x, 0] = color[0] * gradient
                    image[y, x, 1] = color[1] * gradient
                    image[y, x, 2] = color[2] * gradient

class AudioVisualizer:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def get_waveform_data(self, audio_data, num_points=1000):
        audio_cp = to_gpu(audio_data)
        if len(audio_cp) == 0: return cp.zeros(num_points)
        if len(audio_cp) > num_points:
            indices = cp.linspace(0, len(audio_cp) - 1, num_points, dtype=int)
            waveform = audio_cp[indices]
        else:
            waveform = audio_cp
        return waveform

    def get_spectrum_data(self, audio_data, fft_size=None, window="hann"):
        audio_cp = to_gpu(audio_data)
        if len(audio_cp) == 0: return cp.zeros(10), cp.zeros(10)
        
        if fft_size is None: fft_size = 2 ** int(cp.ceil(cp.log2(len(audio_cp))))
        window_func = cp.hanning(len(audio_cp)) if window == "hann" else cp.ones(len(audio_cp))
        
        windowed_data = audio_cp * window_func
        if len(windowed_data) < fft_size:
            padded = cp.zeros(fft_size); padded[:len(windowed_data)] = windowed_data; windowed_data = padded
            
        fft_result = cp.fft.fft(windowed_data[:fft_size])
        frequencies = cp.fft.fftfreq(fft_size, 1.0 / self.sample_rate)
        
        positive = frequencies >= 0; frequencies = frequencies[positive]; magnitudes = cp.abs(fft_result[positive])
        magnitudes_db = 20 * cp.log10(magnitudes + 1e-10); audible = frequencies <= 20000
        return frequencies[audible], magnitudes_db[audible]

    def get_spectrogram_data(self, audio_data, window_size=1024, hop_size=512, window="hann"):
        # Fallback to CPU for STFT
        audio_np = to_cpu(audio_data)
        if len(audio_np) < window_size:
            return np.zeros(10), np.zeros(10), np.zeros((10,10))
            
        frequencies, times, spectrogram = signal.spectrogram(
            audio_np, fs=self.sample_rate, window=window,
            nperseg=window_size, noverlap=window_size - hop_size, scaling='spectrum')
        spectrogram_db = 20 * np.log10(spectrogram + 1e-10); freq_mask = frequencies <= 20000
        return frequencies[freq_mask], times, to_gpu(spectrogram_db[freq_mask, :])


class WaveformRenderer:
    def __init__(self, width=800, height=300):
        self.width = max(1, width); self.height = max(1, height)

    def render_waveform(self, audio_data, color=(0.2, 0.6, 1.0), background=(0.1, 0.1, 0.15), grid_color=(0.3, 0.3, 0.35), show_grid=True):
        image_np = np.full((self.height, self.width, 3), background, dtype=np.float32)
        
        if show_grid:
            for i in range(5): y = int(self.height * (i + 1) / 6); image_np[y, :] = grid_color
            for i in range(9): x = int(self.width * (i + 1) / 10); image_np[:, x] = grid_color
        center_y = self.height // 2
        image_np[center_y, :] = tuple(min(c * 1.2, 1.0) for c in grid_color)
        
        if len(audio_data) == 0: return image_np
        audio_cp = to_gpu(audio_data)
        
        if len(audio_cp) > self.width:
            indices = cp.linspace(0, len(audio_cp) - 1, self.width, dtype=int); waveform = audio_cp[indices]
        else:
            waveform = cp.interp(cp.linspace(0, len(audio_cp) - 1, self.width), cp.arange(len(audio_cp)), audio_cp)
        
        image_cp = to_gpu(image_np)
        
        if NUMBA_CUDA_AVAILABLE:
            threads_per_block = 256; blocks_per_grid = (self.width + threads_per_block - 1) // threads_per_block
            _cuda_render_waveform_kernel[blocks_per_grid, threads_per_block](
                image_cp, waveform, self.height, self.width, color, grid_color, center_y)
        else:
            # Vectorized CPU Fallback
            y_offsets = (waveform * self.height / 2).astype(int)
            y_starts = np.clip(center_y - np.abs(y_offsets), 0, self.height - 1)
            y_ends = np.clip(center_y + np.abs(y_offsets), 0, self.height - 1)
            image_np = to_cpu(image_cp) # Use CPU array
            for x in range(self.width):
                ys = min(y_starts[x], y_ends[x]); ye = max(y_starts[x], y_ends[x])
                image_np[ys:ye + 1, x] = color
            return image_np
                
        return to_cpu(image_cp)

    def render_spectrum(self, frequencies, magnitudes, color=(0.2, 0.8, 0.4), background=(0.1, 0.1, 0.15), grid_color=(0.3, 0.3, 0.35), show_grid=True, log_freq=True, min_db=-80, max_db=0):
        image_np = np.full((self.height, self.width, 3), background, dtype=np.float32)
        
        if show_grid:
            for db in range(int(min_db), int(max_db) + 1, 20):
                y = max(0, min(self.height - 1, int(self.height * (max_db - db) / (max_db - min_db)))); image_np[y, :] = grid_color
        
        image_cp = to_gpu(image_np)
        frequencies_cp = to_gpu(frequencies); magnitudes_cp = to_gpu(magnitudes)
        
        if len(frequencies_cp) == 0: return to_cpu(image_cp)
        
        freq_range = cp.logspace(cp.log10(max(20, frequencies_cp.min())), cp.log10(min(20000, frequencies_cp.max())), self.width) if log_freq else cp.linspace(20, 20000, self.width)
        interp_mags = cp.interp(freq_range, frequencies_cp, magnitudes_cp)
        
        if NUMBA_CUDA_AVAILABLE:
            threads_per_block = 256; blocks_per_grid = (self.width + threads_per_block - 1) // threads_per_block
            _cuda_render_spectrum_kernel[blocks_per_grid, threads_per_block](
                image_cp, interp_mags, self.height, self.width, color, float(min_db), float(max_db))
        else:
            normalized = cp.clip((interp_mags - min_db) / (max_db - min_db), 0, 1)
            bar_heights = (normalized * self.height).astype(int)
            y_coords = np.arange(self.height)[:, np.newaxis]
            bar_tops = (self.height - bar_heights)[np.newaxis, :]
            bar_mask = y_coords >= bar_tops
            gradient = np.clip(0.5 + 0.5 * (y_coords - bar_tops) / self.height, 0, 1)
            
            image_np = to_cpu(image_cp)
            for c in range(3): 
                image_np[:, :, c] = np.where(bar_mask, color[c] * gradient, image_np[:, :, c])
                
        return to_cpu(image_cp)

    def render_spectrogram(self, frequencies, times, spectrogram, colormap="viridis", background=(0.0, 0.0, 0.0)):
        spec_cp = to_gpu(spectrogram)
        if spec_cp.size == 0: return np.full((self.height, self.width, 3), background, dtype=np.float32)
        
        freq_zoom = self.height / max(1, len(frequencies)); time_zoom = self.width / max(1, len(times))
        resized = cp_zoom(spec_cp, (freq_zoom, time_zoom), order=1)
        
        if resized.shape[0] > self.height: resized = resized[:self.height, :]
        if resized.shape[1] > self.width: resized = resized[:, :self.width]
        
        if resized.shape[0] < self.height or resized.shape[1] < self.width:
            padded = cp.full((self.height, self.width), cp.min(resized)); padded[:resized.shape[0], :resized.shape[1]] = resized; resized = padded
            
        min_val, max_val = cp.min(resized), cp.max(resized)
        rng = max_val - min_val
        normalized = (resized - min_val) / rng if rng > 0 else cp.zeros_like(resized)
        
        colors = self._get_colormap_array(normalized.flatten(), colormap).reshape(self.height, self.width, 3)
        return np.flip(to_cpu(colors), axis=0)

    def _get_colormap_array(self, values_cp, colormap):
        values_cp = cp.clip(values_cp, 0, 1); n = len(values_cp); colors = cp.zeros((n, 3))
        
        if colormap == "viridis":
            colors[:, 0] = cp.clip(0.267 + 0.329 * values_cp + 0.386 * values_cp ** 2, 0, 1)
            colors[:, 1] = cp.clip(0.004 + 1.376 * values_cp - 0.624 * values_cp ** 2, 0, 1)
            colors[:, 2] = cp.clip(0.329 + 0.996 * values_cp - 0.542 * values_cp ** 2, 0, 1)
        elif colormap == "magma":
            colors[:, 0] = cp.clip(0.001 + 1.446 * values_cp - 0.447 * values_cp ** 2, 0, 1)
            colors[:, 1] = cp.clip(0.002 + 0.864 * values_cp + 0.134 * values_cp ** 2, 0, 1)
            colors[:, 2] = cp.clip(0.014 + 2.127 * values_cp - 1.141 * values_cp ** 2, 0, 1)
        else: # Grayscale fallback
            colors[:, 0] = colors[:, 1] = colors[:, 2] = values_cp
        return colors