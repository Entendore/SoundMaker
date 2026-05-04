"""
Procedural Sound Effects Generator Module - GPU Accelerated (CuPy + Numba)
"""
import numpy as np
from enum import Enum
from scipy import signal as sp_signal

# CuPy / GPU Fallbacks
try:
    import cupy as cp
    from cupyx.scipy.signal import lfilter as cp_lfilter
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp  # Fallback to numpy
    from scipy.signal import lfilter as cp_lfilter
    GPU_AVAILABLE = False

# Numba CUDA
try:
    from numba import cuda
    import numba
    NUMBA_CUDA_AVAILABLE = True
except ImportError:
    NUMBA_CUDA_AVAILABLE = False

def to_cpu(arr):
    """Moves array to CPU (numpy)."""
    if GPU_AVAILABLE and hasattr(arr, 'get'): # CuPy array check
        return arr.get()
    return arr

def to_gpu(arr):
    """Moves array to GPU (cupy) if available."""
    if GPU_AVAILABLE and not isinstance(arr, cp.ndarray):
        return cp.asarray(arr)
    return arr

# ------------------------------------------------------------------
#  Numba CUDA Kernels for Stateful DSP
# ------------------------------------------------------------------
if NUMBA_CUDA_AVAILABLE:
    @cuda.jit
    def _cuda_phaser_kernel(signal, c_coeffs, output, feedback_val, num_samples, stages):
        # Single thread for sequential state processing
        if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
            x_prev = cuda.local.array(8, dtype=numba.float64)
            y_prev = cuda.local.array(8, dtype=numba.float64)
            for s in range(stages):
                x_prev[s] = 0.0
                y_prev[s] = 0.0
            fb_sample = 0.0
            for i in range(num_samples):
                sample = signal[i] + fb_sample * feedback_val
                for s in range(stages):
                    c = c_coeffs[s, i]
                    # All-pass filter difference equation: y = -c*x + x_prev + c*y_prev
                    y = -c * sample + x_prev[s] + c * y_prev[s]
                    x_prev[s] = sample
                    y_prev[s] = y
                    sample = y
                output[i] = sample
                fb_sample = sample

    @cuda.jit
    def _cuda_compressor_envelope_kernel(abs_signal, envelope, attack_coef, release_coef, num_samples):
        if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
            envelope[0] = abs_signal[0]
            for i in range(1, num_samples):
                if abs_signal[i] > envelope[i-1]:
                    envelope[i] = attack_coef * envelope[i-1] + (1 - attack_coef) * abs_signal[i]
                else:
                    envelope[i] = release_coef * envelope[i-1] + (1 - release_coef) * abs_signal[i]


class WaveformType(Enum):
    SINE = "sine"
    SQUARE = "square"
    SAWTOOTH = "sawtooth"
    TRIANGLE = "triangle"
    NOISE_WHITE = "noise_white"
    NOISE_PINK = "noise_pink"
    NOISE_BROWN = "noise_brown"
    PULSE = "pulse"
    HARMONIC = "harmonic"
    FM_BELL = "fm_bell"
    FM_METAL = "fm_metal"
    FM_BRASS = "fm_brass"
    FM_STRINGS = "fm_strings"


class FMSynthesis:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def generate_fm(self, carrier_freq, modulator_freq, duration, modulation_index=2.0, amplitude=0.8):
        num_samples = int(self.sample_rate * duration)
        if num_samples <= 0: return cp.array([])
        t = cp.linspace(0, duration, num_samples, endpoint=False, dtype=cp.float64)
        modulator = cp.sin(2 * cp.pi * modulator_freq * t)
        phase = 2 * cp.pi * carrier_freq * t + modulation_index * modulator
        return cp.sin(phase) * amplitude

    def generate_bell(self, frequency, duration, amplitude=0.7):
        num_samples = int(self.sample_rate * duration)
        if num_samples <= 0: return cp.array([])
        t = cp.linspace(0, duration, num_samples, endpoint=False, dtype=cp.float64)
        # FM Bell: Carrier freq, Modulator freq ratio ~2.4, exponential decay
        modulator = cp.sin(2 * cp.pi * frequency * 2.4 * t) * cp.exp(-t * 3) * 3.0
        carrier = cp.sin(2 * cp.pi * frequency * t + modulator)
        return carrier * cp.exp(-t * 2) * amplitude

    def generate_metallic(self, frequency, duration, amplitude=0.7, brightness=0.5):
        num_samples = int(self.sample_rate * duration)
        if num_samples <= 0: return cp.array([])
        t = cp.linspace(0, duration, num_samples, endpoint=False, dtype=cp.float64)
        phase = 2 * cp.pi * frequency * t
        # Add inharmonic components
        for freq_ratio, mod_idx, decay in [(1.0, 1.4, 2.5), (2.7, 1.8, 3.5), (5.0, 0.8, 6.0)]:
            phase += (cp.sin(2 * cp.pi * frequency * freq_ratio * t)
                      * cp.exp(-t * decay * (1 + brightness)) * mod_idx)
        return cp.sin(phase) * cp.exp(-t * (2 - brightness)) * amplitude

    def generate_brass(self, frequency, duration, amplitude=0.7):
        num_samples = int(self.sample_rate * duration)
        if num_samples <= 0: return cp.array([])
        t = cp.linspace(0, duration, num_samples, endpoint=False, dtype=cp.float64)
        attack_samples = int(0.08 * self.sample_rate)
        mod_index_env = cp.ones(num_samples)
        if attack_samples > 0:
            mod_index_env[:attack_samples] = cp.linspace(0.5, 4.0, attack_samples)
            mod_index_env[attack_samples:] = 3.5
        
        modulator = cp.sin(2 * cp.pi * frequency * t)
        carrier = cp.sin(2 * cp.pi * frequency * t + modulator * mod_index_env)
        
        env = cp.ones_like(t)
        a_mask = t < 0.08
        r_mask = t >= (duration - 0.15)
        env[a_mask] = t[a_mask] / 0.08
        if cp.any(r_mask):
            env[r_mask] = 1 - (t[r_mask] - (duration - 0.15)) / 0.15
        return carrier * env * amplitude

    def generate_strings(self, frequency, duration, amplitude=0.6):
        num_samples = int(self.sample_rate * duration)
        if num_samples <= 0: return cp.array([])
        t = cp.linspace(0, duration, num_samples, endpoint=False, dtype=cp.float64)
        vibrato = 3.0 * cp.sin(2 * cp.pi * 5.0 * t)
        
        modulator = cp.sin(2 * cp.pi * frequency * 0.5 * t)
        carrier = cp.sin(2 * cp.pi * (frequency + vibrato) * t + modulator * 0.5)
        # Slight detuning for richness
        carrier2 = cp.sin(2 * cp.pi * (frequency * 1.002 + vibrato) * t + modulator * 0.5) * 0.3
        sound = carrier + carrier2
        
        env = cp.ones_like(t)
        a_mask = t < 0.15
        r_mask = t >= (duration - 0.2)
        env[a_mask] = cp.sin(cp.pi * t[a_mask] / 0.3) # S-curve like attack
        if cp.any(r_mask):
            env[r_mask] = cp.cos(cp.pi * (t[r_mask] - (duration - 0.2)) / 0.4)
            
        sound = sound * env * amplitude
        peak = cp.max(cp.abs(sound))
        return sound / peak if peak > 0 else sound


def cp_filtfilt(b, a, x):
    """Zero-phase filtering on GPU via forward-backward IIR."""
    # Ensure inputs are on correct device
    b = to_gpu(b)
    a = to_gpu(a)
    x = to_gpu(x)
    
    y = cp_lfilter(b, a, x)
    y = y[::-1]
    y = cp_lfilter(b, a, y)
    return y[::-1]


class SoundGenerator:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.default_amplitude = 0.8
        self.fm = FMSynthesis(sample_rate)

    @staticmethod
    def _normalize_if_clipping(signal: cp.ndarray) -> cp.ndarray:
        peak = cp.max(cp.abs(signal))
        return signal / peak if peak > 1.0 else signal

    def generate_waveform(self, waveform_type, frequency, duration, amplitude=None,
                          phase=0.0, pulse_width=0.5, harmonics=5):
        if amplitude is None: amplitude = self.default_amplitude
        num_samples = int(self.sample_rate * duration)
        if num_samples <= 0: return cp.array([])
        
        t = cp.linspace(0, duration, num_samples, endpoint=False, dtype=cp.float64)

        if waveform_type == WaveformType.FM_BELL: return self.fm.generate_bell(frequency, duration, amplitude)
        elif waveform_type == WaveformType.FM_METAL: return self.fm.generate_metallic(frequency, duration, amplitude)
        elif waveform_type == WaveformType.FM_BRASS: return self.fm.generate_brass(frequency, duration, amplitude)
        elif waveform_type == WaveformType.FM_STRINGS: return self.fm.generate_strings(frequency, duration, amplitude)
        elif waveform_type == WaveformType.SINE: wave = cp.sin(2 * cp.pi * frequency * t + phase)
        elif waveform_type == WaveformType.SQUARE: wave = cp.sign(cp.sin(2 * cp.pi * frequency * t + phase))
        elif waveform_type == WaveformType.SAWTOOTH: wave = 2 * (frequency * t + phase / (2 * cp.pi)) % 1 - 1
        elif waveform_type == WaveformType.TRIANGLE: wave = 2 * cp.abs(2 * (frequency * t + phase / (2 * cp.pi)) % 1 - 1) - 1
        elif waveform_type == WaveformType.NOISE_WHITE: wave = cp.random.uniform(-1, 1, num_samples)
        elif waveform_type == WaveformType.NOISE_PINK:
            white = cp.random.randn(num_samples)
            # Paul Kellet's pink noise filter approximation
            b = cp.asarray([0.049922035, -0.095993537, 0.050612699, -0.004408786])
            a = cp.asarray([1.0, -2.494956002, 2.017265875, -0.522189400])
            pink = cp_lfilter(b, a, white)
            peak = cp.max(cp.abs(pink)); wave = pink / peak if peak > 0 else pink
        elif waveform_type == WaveformType.NOISE_BROWN:
            white = cp.random.randn(num_samples)
            b = cp.asarray([0.04]); a = cp.asarray([1.0, -0.96])
            brown = cp_lfilter(b, a, white)
            peak = cp.max(cp.abs(brown)); wave = brown / peak if peak > 0 else brown
        elif waveform_type == WaveformType.PULSE: wave = cp.where((frequency * t + phase / (2 * cp.pi)) % 1 < pulse_width, 1.0, -1.0)
        elif waveform_type == WaveformType.HARMONIC:
            wave = cp.zeros_like(t)
            for n in range(1, harmonics + 1): wave += (1.0 / n) * cp.sin(2 * cp.pi * n * frequency * t + phase)
            peak = cp.max(cp.abs(wave)); wave = wave / peak if peak > 0 else wave
        else: wave = cp.sin(2 * cp.pi * frequency * t + phase)
        return wave * amplitude

    def apply_adsr_envelope(self, signal, attack, decay, sustain, release, sustain_level=0.7):
        num_samples = len(signal)
        if num_samples == 0: return signal
        
        a_s = int(attack * self.sample_rate); d_s = int(decay * self.sample_rate)
        s_s = int(sustain * self.sample_rate); r_s = int(release * self.sample_rate)
        
        total = a_s + d_s + s_s + r_s
        if total > num_samples:
            scale = num_samples / total
            a_s = int(a_s * scale); d_s = int(d_s * scale); s_s = int(s_s * scale)
            r_s = num_samples - a_s - d_s - s_s # Ensure fit
            
        envelope = cp.ones(num_samples)
        if a_s > 0: envelope[:a_s] = cp.linspace(0, 1, a_s)
        if d_s > 0: envelope[a_s:a_s + d_s] = cp.linspace(1, sustain_level, d_s)
        if s_s > 0: envelope[a_s + d_s:a_s + d_s + s_s] = sustain_level
        
        remaining = num_samples - a_s - d_s - s_s
        if remaining > 0: envelope[a_s + d_s + s_s:] = cp.linspace(sustain_level, 0, remaining)
        return signal * envelope

    def apply_frequency_sweep(self, waveform_type, start_freq, end_freq, duration, amplitude=None, sweep_type="linear"):
        if amplitude is None: amplitude = self.default_amplitude
        num_samples = int(self.sample_rate * duration)
        if num_samples <= 0: return cp.array([])
        
        t = cp.linspace(0, duration, num_samples, endpoint=False, dtype=cp.float64)
        
        if sweep_type == "linear":
            # Phase is integral of linear frequency change: f(t) = f0 + (f1-f0)/T * t
            phase = 2 * cp.pi * (start_freq * t + (end_freq - start_freq) * t ** 2 / (2 * duration))
        else: # exponential
            if start_freq <= 0: start_freq = 1.0 # Avoid log(0)
            k = (end_freq / start_freq) ** (1 / duration)
            # Phase is integral of exponential frequency
            phase = 2 * cp.pi * start_freq * (k ** t - 1) / cp.log(k)
            
        if waveform_type == WaveformType.SINE: wave = cp.sin(phase)
        elif waveform_type == WaveformType.SQUARE: wave = cp.sign(cp.sin(phase))
        elif waveform_type == WaveformType.SAWTOOTH: wave = 2 * (phase / (2 * cp.pi) % 1) - 1
        elif waveform_type == WaveformType.TRIANGLE: wave = 2 * cp.abs(2 * (phase / (2 * cp.pi) % 1 - 1) - 1)
        else: wave = cp.sin(phase)
        return wave * amplitude

    def apply_reverb(self, signal, room_size=0.5, damping=0.5, wet_level=0.3):
        sr = self.sample_rate
        # Simplified Freeverb-like structure
        comb_delays_s = [0.0297, 0.0371, 0.0411, 0.0437]
        comb_delays = [max(1, int(d * sr * (0.5 + room_size))) for d in comb_delays_s]
        allpass_delays_s = [0.005, 0.0017]
        allpass_delays = [max(1, int(d * sr * (0.5 + room_size))) for d in allpass_delays_s]
        
        feedback = 0.7 * (1 - damping * 0.5)
        wet = cp.zeros_like(signal)
        
        # Parallel comb filters
        for delay in comb_delays:
            b = cp.zeros(delay + 1); b[0] = 1.0; a = cp.zeros(delay + 1); a[0] = 1.0; a[delay] = -feedback
            wet += cp_lfilter(b, a, signal)
        wet /= len(comb_delays)
        
        # Series allpass filters
        for delay in allpass_delays:
            g = 0.5; b = cp.zeros(delay + 1); b[0] = -g; b[delay] = 1.0; a = cp.zeros(delay + 1); a[0] = 1.0; a[delay] = g
            wet = cp_lfilter(b, a, wet)
            
        output = signal * (1 - wet_level) + wet * wet_level
        return self._normalize_if_clipping(output)

    def apply_delay(self, signal, delay_time=0.3, feedback=0.4, mix=0.5):
        delay_samples = int(delay_time * self.sample_rate)
        if delay_samples <= 0 or delay_samples >= len(signal): return signal
        b = cp.zeros(delay_samples + 1); b[0] = 1.0; a = cp.zeros(delay_samples + 1); a[0] = 1.0; a[delay_samples] = -feedback
        delayed = cp_lfilter(b, a, signal)
        
        fade_len = min(int(0.01 * self.sample_rate), len(delayed) // 4)
        if fade_len > 0: delayed[-fade_len:] *= cp.linspace(1, 0, fade_len)
        
        output = signal * (1 - mix) + delayed * mix
        return self._normalize_if_clipping(output)

    def apply_distortion(self, signal, drive=0.5, type_="soft"):
        driven = signal * (1 + drive * 10)
        if type_ == "soft": output = cp.tanh(driven)
        elif type_ == "hard": output = cp.clip(driven, -1, 1)
        elif type_ == "fuzz": output = cp.sign(cp.tanh(driven)) * (1 - cp.exp(-cp.abs(cp.tanh(driven)) * 2))
        else: output = cp.tanh(driven)
        return self._normalize_if_clipping(output)

    def apply_lowpass_filter(self, signal, cutoff=1000.0):
        nyq = self.sample_rate / 2; Wn = min(cutoff / nyq, 0.99)
        b, a = sp_signal.butter(2, Wn, btype='low')
        return cp_filtfilt(b, a, signal)

    def apply_highpass_filter(self, signal, cutoff=100.0):
        nyq = self.sample_rate / 2; Wn = max(cutoff / nyq, 0.01)
        b, a = sp_signal.butter(2, Wn, btype='high')
        return cp_filtfilt(b, a, signal)

    def apply_bitcrusher(self, signal, bit_depth=8, sample_rate_reduction=1):
        levels = 2 ** bit_depth
        quantized = cp.round((signal + 1) * (levels / 2)) / (levels / 2) - 1
        if sample_rate_reduction > 1:
            indices = cp.arange(0, len(quantized), sample_rate_reduction)
            hold_indices = cp.repeat(indices, sample_rate_reduction)[:len(quantized)]
            return quantized[hold_indices]
        return quantized

    def apply_chorus(self, signal, rate=1.5, depth=0.5, voices=3, mix=0.5):
        num_samples = len(signal); t = cp.arange(num_samples, dtype=cp.float64) / self.sample_rate
        base_delay_s = 0.02; max_mod_s = 0.005 * depth; chorus_wet = cp.zeros(num_samples, dtype=cp.float64)
        
        for voice in range(voices):
            lfo = cp.sin(2 * cp.pi * rate * (1 + voice * 0.1) * t + voice * cp.pi / voices)
            delay_samples = (base_delay_s + lfo * max_mod_s) * self.sample_rate
            sample_indices = cp.arange(num_samples, dtype=cp.float64)
            read_idx = sample_indices - delay_samples
            
            valid = (read_idx >= 0) & (read_idx < num_samples - 1)
            int_idx = cp.floor(read_idx).astype(int); frac = read_idx - int_idx
            safe_idx = cp.clip(int_idx, 0, num_samples - 2); delayed = cp.zeros(num_samples)
            
            valid_idx = cp.where(valid)[0]
            if len(valid_idx) > 0:
                safe_v = safe_idx[valid_idx]; frac_v = frac[valid_idx]
                delayed[valid_idx] = signal[safe_v] * (1 - frac_v) + signal[safe_v + 1] * frac_v
            
            chorus_wet += delayed * (0.7 / (voice + 1))
            
        output = signal * (1 - mix) + chorus_wet * mix
        return self._normalize_if_clipping(output)

    def apply_phaser(self, signal, rate=0.5, depth=0.7, stages=4, mix=0.5, feedback=0.7):
        signal_cp = to_gpu(signal); num_samples = len(signal_cp)
        if num_samples == 0: return signal_cp
        
        t = cp.arange(num_samples, dtype=cp.float64) / self.sample_rate
        lfo = 0.5 + 0.5 * cp.sin(2 * cp.pi * rate * t)
        c_coeffs = cp.zeros((stages, num_samples), dtype=cp.float64)
        
        # Calculate time-varying all-pass filter coefficients
        for s in range(stages):
            freq = 300 + lfo * depth * 2000 * (1 + s * 0.2)
            tan_val = cp.tan(cp.pi * freq / self.sample_rate)
            c_coeffs[s] = (tan_val - 1) / (tan_val + 1)
        
        output = cp.zeros(num_samples, dtype=cp.float64)
        
        if NUMBA_CUDA_AVAILABLE:
            _cuda_phaser_kernel[1, 1](signal_cp, c_coeffs, output, float(feedback), num_samples, int(stages))
        else:
            # CPU Fallback (Sequential Loop)
            signal_np = to_cpu(signal_cp); c_coeffs_np = to_cpu(c_coeffs); output_np = np.zeros(num_samples)
            x_prev = np.zeros(stages); y_prev = np.zeros(stages); fb_sample = 0.0
            for i in range(num_samples):
                sample = signal_np[i] + fb_sample * feedback
                for s in range(stages):
                    c = c_coeffs_np[s, i]; y = -c * sample + x_prev[s] + c * y_prev[s]
                    x_prev[s] = sample; y_prev[s] = y; sample = y
                output_np[i] = sample; fb_sample = sample
            output = to_gpu(output_np)
            
        result = signal_cp * (1 - mix) + output * mix
        return self._normalize_if_clipping(result)

    def apply_compressor(self, signal, threshold=-20.0, ratio=4.0, attack=0.01, release=0.1, makeup_gain=0.0):
        signal_cp = to_gpu(signal); threshold_lin = 10 ** (threshold / 20)
        abs_signal = cp.abs(signal_cp); num_samples = len(signal_cp)
        
        attack_coef = float(cp.exp(-1 / (attack * self.sample_rate)))
        release_coef = float(cp.exp(-1 / (release * self.sample_rate)))
        envelope = cp.zeros(num_samples, dtype=cp.float64)
        
        if NUMBA_CUDA_AVAILABLE:
            _cuda_compressor_envelope_kernel[1, 1](abs_signal, envelope, attack_coef, release_coef, num_samples)
        else:
            # CPU Fallback
            envelope[0] = abs_signal[0]
            for i in range(1, num_samples):
                if abs_signal[i] > envelope[i-1]: envelope[i] = attack_coef * envelope[i-1] + (1 - attack_coef) * abs_signal[i]
                else: envelope[i] = release_coef * envelope[i-1] + (1 - release_coef) * abs_signal[i]
                
        gain = cp.ones_like(envelope); above = envelope > threshold_lin
        gain[above] = (threshold_lin * (envelope[above] / threshold_lin) ** (1 / ratio) / envelope[above])
        
        makeup_lin = 10 ** (makeup_gain / 20)
        # Soft clip to safe output
        return cp.tanh(signal_cp * gain * makeup_lin * 1.5) / 1.5

    def mix_signals(self, signals, amplitudes=None):
        if amplitudes is None: amplitudes = [1.0 / len(signals)] * len(signals)
        max_len = max(len(s) for s in signals); mixed = cp.zeros(max_len)
        for sig, amp in zip(signals, amplitudes):
            padded = cp.zeros(max_len); padded[:len(sig)] = sig; mixed += padded * amp
        peak = cp.max(cp.abs(mixed)); return mixed / peak if peak > 0 else mixed


class SoundPresetGenerator:
    def __init__(self, generator: SoundGenerator): self.generator = generator

    def generate_explosion(self, duration=1.0, intensity=0.8):
        sweep = self.generator.apply_frequency_sweep(WaveformType.SINE, 150, 30, duration, 0.5, "exponential")
        noise = self.generator.generate_waveform(WaveformType.NOISE_BROWN, 0, duration, 0.7)
        sound = self.generator.mix_signals([sweep, noise], [0.6, 0.4])
        sound = self.generator.apply_adsr_envelope(sound, 0.01, 0.1, 0.3, max(0.01, duration - 0.41), 0.5)
        return self.generator.apply_lowpass_filter(sound, 800 * intensity) * intensity
        
    def generate_laser(self, duration=0.3, frequency=800, sweep_range=600):
        sound = self.generator.apply_frequency_sweep(WaveformType.SAWTOOTH, frequency + sweep_range, frequency - sweep_range, duration, 0.6, "exponential")
        return self.generator.apply_distortion(self.generator.apply_adsr_envelope(sound, 0.01, 0.05, 0.15, 0.09, 0.3), 0.2, "soft")
        
    def generate_coin(self, duration=0.15, base_freq=988):
        t1 = self.generator.generate_waveform(WaveformType.SINE, base_freq, duration * 0.5, 0.5)
        t2 = self.generator.generate_waveform(WaveformType.SINE, base_freq * 1.5, duration * 0.7, 0.4)
        return self.generator.apply_adsr_envelope(self.generator.mix_signals([t1, t2], [0.6, 0.4]), 0.005, 0.03, 0.05, max(0.01, duration - 0.085), 0.3)

    def generate_jump(self, duration=0.25, base_freq=300):
        return self.generator.apply_lowpass_filter(self.generator.apply_adsr_envelope(self.generator.apply_frequency_sweep(WaveformType.SQUARE, base_freq, base_freq * 3, duration, 0.4), 0.02, 0.1, 0.05, 0.08, 0.3), 2000)

    def generate_powerup(self, duration=0.8, base_freq=200):
        s1 = self.generator.apply_frequency_sweep(WaveformType.SINE, base_freq, base_freq * 4, duration, 0.4)
        s2 = self.generator.apply_frequency_sweep(WaveformType.SINE, base_freq * 1.5, base_freq * 6, duration, 0.3)
        return self.generator.apply_adsr_envelope(self.generator.mix_signals([s1, s2], [0.6, 0.4]), 0.05, 0.15, 0.4, 0.2, 0.6)

    def generate_footstep(self, duration=0.15, surface="default"):
        noise = self.generator.generate_waveform(WaveformType.NOISE_BROWN, 0, duration, 0.5)
        thump = self.generator.generate_waveform(WaveformType.SINE, 80, duration, 0.3)
        return self.generator.apply_adsr_envelope(self.generator.apply_lowpass_filter(self.generator.mix_signals([noise, thump], [0.4, 0.6]), 600), 0.005, 0.03, 0.05, max(0.01, duration - 0.085), 0.2)

    def generate_alarm(self, duration=1.0, frequency=800, pattern_rate=4.0):
        num_samples = int(self.generator.sample_rate * duration); sound = cp.zeros(num_samples); beep_dur = 0.5 / pattern_rate
        beep_s = int(beep_dur * self.generator.sample_rate)
        for i in range(int(duration * pattern_rate * 2)):
            si = int(i * beep_s * 2)
            if si + beep_s > num_samples: break
            sound[si:si + beep_s] = self.generator.generate_waveform(WaveformType.SQUARE, frequency, beep_dur, 0.5)
        return sound

    def generate_hit(self, duration=0.2, intensity=0.7):
        thump = self.generator.generate_waveform(WaveformType.SINE, 80, duration, 0.6); noise = self.generator.generate_waveform(WaveformType.NOISE_WHITE, 0, duration * 0.5, 0.4)
        return self.generator.apply_lowpass_filter(self.generator.apply_adsr_envelope(self.generator.mix_signals([thump, noise], [0.7, 0.3]), 0.005, 0.05, 0.05, max(0.01, duration - 0.105), 0.2), 1500 * intensity) * intensity

    def generate_bell(self, frequency=440.0, duration=1.5): return self.generator.fm.generate_bell(frequency, duration, 0.7)
    def generate_gong(self, frequency=150.0, duration=3.0): return self.generator.fm.generate_metallic(frequency, duration, 0.8, brightness=0.3)
    
    def generate_scifi_beep(self, frequency=800.0, duration=0.2):
        sound = self.generator.fm.generate_fm(carrier_freq=frequency, modulator_freq=frequency * 1.5, duration=duration, modulation_index=2.0, amplitude=0.6)
        return self.generator.apply_adsr_envelope(sound, 0.01, 0.05, 0.08, 0.06, 0.5)