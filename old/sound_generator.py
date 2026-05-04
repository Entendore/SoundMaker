"""
Procedural Sound Effects Generator Module
==========================================
This module provides comprehensive sound synthesis capabilities including:
- Multiple waveform types (sine, square, sawtooth, triangle, noise)
- FM (Frequency Modulation) synthesis for bell/metallic sounds
- ADSR envelope shaping
- Various audio effects (reverb, delay, distortion, filters, chorus, phaser, compressor)
- Sound preset generation for common effects
"""

import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any
from enum import Enum
import math
from scipy import signal as sp_signal


class WaveformType(Enum):
    """Enumeration of available waveform types for sound synthesis."""
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
    """
    FM (Frequency Modulation) Synthesis engine.
    
    FM synthesis creates complex timbres by modulating the frequency of a
    carrier wave with another wave (modulator). This produces rich harmonic
    content perfect for bells, metallic sounds, and synthesized instruments.
    """
    
    def __init__(self, sample_rate: int = 44100):
        """Initialize FM synthesizer."""
        self.sample_rate = sample_rate
    
    def generate_fm(
        self,
        carrier_freq: float,
        modulator_freq: float,
        duration: float,
        modulation_index: float = 2.0,
        modulator_waveform: str = "sine",
        carrier_waveform: str = "sine",
        amplitude: float = 0.8,
        modulator_envelope: tuple = None,
        carrier_envelope: tuple = None
    ) -> np.ndarray:
        """Generate FM synthesized sound."""
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, dtype=np.float64)
        
        modulator = self._generate_wave(modulator_waveform, modulator_freq, t)
        
        if modulator_envelope:
            modulator = self._apply_envelope(modulator, *modulator_envelope)
        
        modulated_phase = 2 * np.pi * carrier_freq * t + modulation_index * modulator
        carrier = self._generate_wave_with_phase(carrier_waveform, modulated_phase)
        
        if carrier_envelope:
            carrier = self._apply_envelope(carrier, *carrier_envelope)
        
        return carrier * amplitude
    
    def generate_bell(self, frequency: float, duration: float, amplitude: float = 0.7) -> np.ndarray:
        """Generate a bell-like sound using FM synthesis."""
        carrier_freq = frequency
        modulator_freq = frequency * 2.4
        modulation_index = 3.0
        
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, dtype=np.float64)
        
        modulator = np.sin(2 * np.pi * modulator_freq * t)
        mod_env = np.exp(-t * 3)
        modulator *= mod_env * modulation_index
        
        phase = 2 * np.pi * carrier_freq * t + modulator
        carrier = np.sin(phase)
        
        envelope = np.exp(-t * 2)
        sound = carrier * envelope * amplitude
        
        return sound
    
    def generate_metallic(self, frequency: float, duration: float, amplitude: float = 0.7, brightness: float = 0.5) -> np.ndarray:
        """Generate metallic/gong-like sound using multiple FM operators."""
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, dtype=np.float64)
        
        modulators = [
            (1.0, 1.4, 2.5),
            (2.7, 1.8, 3.5),
            (5.0, 0.8, 6.0),
        ]
        
        phase = 2 * np.pi * frequency * t
        
        for freq_ratio, mod_idx, decay in modulators:
            mod_freq = frequency * freq_ratio
            modulator = np.sin(2 * np.pi * mod_freq * t)
            mod_env = np.exp(-t * decay * (1 + brightness))
            phase += modulator * mod_env * mod_idx
        
        carrier = np.sin(phase)
        envelope = np.exp(-t * (2 - brightness))
        sound = carrier * envelope * amplitude
        
        return sound
    
    def generate_brass(self, frequency: float, duration: float, amplitude: float = 0.7) -> np.ndarray:
        """Generate brass-like sound using FM synthesis."""
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, dtype=np.float64)
        
        carrier_freq = frequency
        modulator_freq = frequency
        
        attack_time = 0.08
        attack_samples = int(attack_time * self.sample_rate)
        mod_index_env = np.ones(num_samples)
        mod_index_env[:attack_samples] = np.linspace(0.5, 4.0, attack_samples)
        mod_index_env[attack_samples:] = 3.5
        
        modulator = np.sin(2 * np.pi * modulator_freq * t)
        phase = 2 * np.pi * carrier_freq * t + modulator * mod_index_env
        carrier = np.sin(phase)
        
        envelope = self._brass_envelope(t, duration)
        sound = carrier * envelope * amplitude
        
        return sound
    
    def generate_strings(self, frequency: float, duration: float, amplitude: float = 0.6) -> np.ndarray:
        """Generate string-like sound using FM synthesis."""
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, dtype=np.float64)
        
        carrier_freq = frequency
        modulator_freq = frequency * 0.5
        modulation_index = 0.5
        
        vibrato_freq = 5.0
        vibrato_depth = 3.0
        
        modulator = np.sin(2 * np.pi * modulator_freq * t)
        vibrato = vibrato_depth * np.sin(2 * np.pi * vibrato_freq * t)
        
        phase = 2 * np.pi * (carrier_freq + vibrato) * t + modulator * modulation_index
        carrier = np.sin(phase)
        
        phase2 = 2 * np.pi * (carrier_freq * 1.002 + vibrato) * t + modulator * modulation_index
        carrier2 = np.sin(phase2) * 0.3
        
        sound = carrier + carrier2
        envelope = self._string_envelope(t, duration)
        sound = sound * envelope * amplitude
        
        max_val = np.max(np.abs(sound))
        if max_val > 0:
            sound = sound / max_val
        
        return sound
    
    def _generate_wave(self, waveform: str, frequency: float, t: np.ndarray) -> np.ndarray:
        """Generate a waveform."""
        if waveform == "sine":
            return np.sin(2 * np.pi * frequency * t)
        elif waveform == "square":
            return np.sign(np.sin(2 * np.pi * frequency * t))
        elif waveform == "sawtooth":
            return 2 * (frequency * t % 1) - 1
        elif waveform == "triangle":
            return 2 * np.abs(2 * (frequency * t % 1) - 1) - 1
        return np.sin(2 * np.pi * frequency * t)
    
    def _generate_wave_with_phase(self, waveform: str, phase: np.ndarray) -> np.ndarray:
        """Generate a waveform with given phase array."""
        if waveform == "sine":
            return np.sin(phase)
        elif waveform == "square":
            return np.sign(np.sin(phase))
        elif waveform == "sawtooth":
            return 2 * (phase / (2 * np.pi) % 1) - 1
        elif waveform == "triangle":
            return 2 * np.abs(2 * (phase / (2 * np.pi) % 1) - 1) - 1
        return np.sin(phase)
    
    def _apply_envelope(self, signal: np.ndarray, attack: float, decay: float, 
                        sustain: float, release: float, sustain_level: float) -> np.ndarray:
        """Apply ADSR envelope."""
        num_samples = len(signal)
        envelope = np.ones(num_samples)
        
        attack_samples = int(attack * self.sample_rate)
        decay_samples = int(decay * self.sample_rate)
        sustain_samples = int(sustain * self.sample_rate)
        release_samples = int(release * self.sample_rate)
        
        total = attack_samples + decay_samples + sustain_samples + release_samples
        if total > num_samples:
            scale = num_samples / total
            attack_samples = int(attack_samples * scale)
            decay_samples = int(decay_samples * scale)
            sustain_samples = int(sustain_samples * scale)
            release_samples = num_samples - attack_samples - decay_samples - sustain_samples
        
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        if decay_samples > 0:
            start = attack_samples
            end = start + decay_samples
            envelope[start:end] = np.linspace(1, sustain_level, decay_samples)
        if sustain_samples > 0:
            start = attack_samples + decay_samples
            end = start + sustain_samples
            envelope[start:end] = sustain_level
        if release_samples > 0:
            start = attack_samples + decay_samples + sustain_samples
            envelope[start:] = np.linspace(sustain_level, 0, num_samples - start)
        
        return signal * envelope
    
    def _brass_envelope(self, t: np.ndarray, duration: float) -> np.ndarray:
        """Create brass-like amplitude envelope."""
        attack_time = 0.08
        release_time = 0.15
        
        envelope = np.ones_like(t)
        
        attack_mask = t < attack_time
        envelope[attack_mask] = t[attack_mask] / attack_time
        
        release_start = duration - release_time
        release_mask = t >= release_start
        envelope[release_mask] = 1 - (t[release_mask] - release_start) / release_time
        
        sustain_mask = ~attack_mask & ~release_mask
        envelope[sustain_mask] = 0.95 - 0.05 * (t[sustain_mask] - attack_time) / (release_start - attack_time)
        
        return envelope
    
    def _string_envelope(self, t: np.ndarray, duration: float) -> np.ndarray:
        """Create string-like amplitude envelope."""
        attack_time = 0.15
        release_time = 0.2
        
        envelope = np.ones_like(t)
        
        attack_mask = t < attack_time
        envelope[attack_mask] = np.sin(np.pi * t[attack_mask] / (2 * attack_time))
        
        release_start = duration - release_time
        release_mask = t >= release_start
        envelope[release_mask] = np.cos(np.pi * (t[release_mask] - release_start) / (2 * release_time))
        
        sustain_mask = ~attack_mask & ~release_mask
        envelope[sustain_mask] = 0.9 + 0.1 * np.sin(2 * np.pi * 0.5 * t[sustain_mask])
        
        return envelope


class SoundGenerator:
    """
    Main sound generator class for procedural audio synthesis.
    """
    
    def __init__(self, sample_rate: int = 44100):
        """Initialize the sound generator."""
        self.sample_rate = sample_rate
        self.default_amplitude = 0.8
        self.fm = FMSynthesis(sample_rate)
        
    def generate_waveform(
        self,
        waveform_type: WaveformType,
        frequency: float,
        duration: float,
        amplitude: float = None,
        phase: float = 0.0,
        pulse_width: float = 0.5,
        harmonics: int = 5
    ) -> np.ndarray:
        """Generate a waveform of the specified type."""
        if amplitude is None:
            amplitude = self.default_amplitude
            
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, dtype=np.float64)
        
        if waveform_type == WaveformType.FM_BELL:
            return self.fm.generate_bell(frequency, duration, amplitude)
        elif waveform_type == WaveformType.FM_METAL:
            return self.fm.generate_metallic(frequency, duration, amplitude)
        elif waveform_type == WaveformType.FM_BRASS:
            return self.fm.generate_brass(frequency, duration, amplitude)
        elif waveform_type == WaveformType.FM_STRINGS:
            return self.fm.generate_strings(frequency, duration, amplitude)
        
        if waveform_type == WaveformType.SINE:
            wave = np.sin(2 * np.pi * frequency * t + phase)
        elif waveform_type == WaveformType.SQUARE:
            wave = np.sign(np.sin(2 * np.pi * frequency * t + phase)).astype(np.float64)
        elif waveform_type == WaveformType.SAWTOOTH:
            wave = 2 * (frequency * t + phase / (2 * np.pi)) % 1 - 1
        elif waveform_type == WaveformType.TRIANGLE:
            wave = 2 * np.abs(2 * (frequency * t + phase / (2 * np.pi)) % 1 - 1) - 1
        elif waveform_type == WaveformType.NOISE_WHITE:
            wave = np.random.uniform(-1, 1, num_samples)
        elif waveform_type == WaveformType.NOISE_PINK:
            white = np.random.uniform(-1, 1, num_samples)
            b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
            a = [1, -2.494956002, 2.017265875, -0.522189400]
            pink = sp_signal.lfilter(b, a, white)
            wave = pink / np.max(np.abs(pink)) if np.max(np.abs(pink)) > 0 else pink
        elif waveform_type == WaveformType.NOISE_BROWN:
            white = np.random.uniform(-1, 1, num_samples)
            brown = np.cumsum(white)
            wave = brown / np.max(np.abs(brown)) if np.max(np.abs(brown)) > 0 else brown
        elif waveform_type == WaveformType.PULSE:
            saw = (frequency * t + phase / (2 * np.pi)) % 1
            wave = np.where(saw < pulse_width, 1.0, -1.0)
        elif waveform_type == WaveformType.HARMONIC:
            wave = np.zeros_like(t)
            for n in range(1, harmonics + 1):
                wave += (1.0 / n) * np.sin(2 * np.pi * n * frequency * t + phase)
            wave = wave / np.max(np.abs(wave)) if np.max(np.abs(wave)) > 0 else wave
        else:
            wave = np.sin(2 * np.pi * frequency * t + phase)
            
        return wave * amplitude
    
    def apply_adsr_envelope(
        self,
        signal: np.ndarray,
        attack: float,
        decay: float,
        sustain: float,
        release: float,
        sustain_level: float = 0.7
    ) -> np.ndarray:
        """Apply ADSR envelope to a signal."""
        num_samples = len(signal)
        
        attack_samples = int(attack * self.sample_rate)
        decay_samples = int(decay * self.sample_rate)
        sustain_samples = int(sustain * self.sample_rate)
        release_samples = int(release * self.sample_rate)
        
        total_envelope_samples = attack_samples + decay_samples + sustain_samples + release_samples
        if total_envelope_samples > num_samples:
            scale = num_samples / total_envelope_samples
            attack_samples = int(attack_samples * scale)
            decay_samples = int(decay_samples * scale)
            sustain_samples = int(sustain_samples * scale)
            release_samples = num_samples - attack_samples - decay_samples - sustain_samples
        
        envelope = np.ones(num_samples)
        
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        if decay_samples > 0:
            envelope[attack_samples:attack_samples + decay_samples] = np.linspace(1, sustain_level, decay_samples)
        if sustain_samples > 0:
            envelope[attack_samples + decay_samples:attack_samples + decay_samples + sustain_samples] = sustain_level
        if release_samples > 0:
            envelope[attack_samples + decay_samples + sustain_samples:] = np.linspace(sustain_level, 0, num_samples - attack_samples - decay_samples - sustain_samples)
        
        return signal * envelope
    
    def apply_frequency_sweep(
        self,
        waveform_type: WaveformType,
        start_freq: float,
        end_freq: float,
        duration: float,
        amplitude: float = None,
        sweep_type: str = "linear"
    ) -> np.ndarray:
        """Generate a frequency sweep (chirp) signal."""
        if amplitude is None:
            amplitude = self.default_amplitude
            
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, dtype=np.float64)
        
        if sweep_type == "linear":
            phase = 2 * np.pi * (start_freq * t + (end_freq - start_freq) * t**2 / (2 * duration))
        else:
            k = (end_freq / start_freq) ** (1 / duration)
            phase = 2 * np.pi * start_freq * (k**t - 1) / np.log(k)
        
        if waveform_type == WaveformType.SINE:
            wave = np.sin(phase)
        elif waveform_type == WaveformType.SQUARE:
            wave = np.sign(np.sin(phase))
        elif waveform_type == WaveformType.SAWTOOTH:
            wave = 2 * (phase / (2 * np.pi) % 1) - 1
        elif waveform_type == WaveformType.TRIANGLE:
            wave = 2 * np.abs(2 * (phase / (2 * np.pi) % 1 - 1)) - 1
        else:
            wave = np.sin(phase)
            
        return wave * amplitude
    
    def apply_reverb(self, signal: np.ndarray, room_size: float = 0.5, damping: float = 0.5, wet_level: float = 0.3) -> np.ndarray:
        """Apply simple reverb effect using multiple delay lines."""
        delay_times = [0.03, 0.05, 0.07, 0.11]
        delay_samples = [int(dt * self.sample_rate * room_size * 2) for dt in delay_times]
        
        wet_signal = np.zeros_like(signal)
        
        for i, delay in enumerate(delay_samples):
            if delay < len(signal):
                delayed = np.zeros_like(signal)
                delayed[delay:] = signal[:-delay] if delay > 0 else signal
                decay = (1 - damping) * (0.7 ** i)
                wet_signal += delayed * decay
        
        max_wet = np.max(np.abs(wet_signal))
        if max_wet > 0:
            wet_signal = wet_signal / max_wet
        
        output = signal * (1 - wet_level) + wet_signal * wet_level
        
        max_output = np.max(np.abs(output))
        if max_output > 1:
            output = output / max_output
            
        return output
    
    def apply_delay(self, signal: np.ndarray, delay_time: float = 0.3, feedback: float = 0.4, mix: float = 0.5) -> np.ndarray:
        """Apply delay/echo effect."""
        delay_samples = int(delay_time * self.sample_rate)
        output = signal.copy()
        
        iterations = int(1 / (1 - feedback)) if feedback < 1 else 10
        current_delay = delay_samples
        current_gain = feedback
        
        for _ in range(iterations):
            if current_delay >= len(signal):
                break
            delayed = np.zeros_like(signal)
            delayed[current_delay:] = signal[:-current_delay] * current_gain
            output += delayed
            current_delay += delay_samples
            current_gain *= feedback
        
        output = signal * (1 - mix) + output * mix
        
        max_val = np.max(np.abs(output))
        if max_val > 1:
            output = output / max_val
            
        return output
    
    def apply_distortion(self, signal: np.ndarray, drive: float = 0.5, type_: str = "soft") -> np.ndarray:
        """Apply distortion effect."""
        gain = 1 + drive * 10
        driven = signal * gain
        
        if type_ == "soft":
            output = np.tanh(driven)
        elif type_ == "hard":
            output = np.clip(driven, -1, 1)
        elif type_ == "fuzz":
            output = np.tanh(driven)
            output = np.sign(output) * (1 - np.exp(-np.abs(output) * 2))
        else:
            output = np.tanh(driven)
        
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val
            
        return output
    
    def apply_lowpass_filter(self, signal: np.ndarray, cutoff: float = 1000.0, resonance: float = 0.5) -> np.ndarray:
        """Apply a lowpass filter."""
        nyquist = self.sample_rate / 2
        normalized_cutoff = min(cutoff / nyquist, 0.99)
        
        b, a = sp_signal.butter(2, normalized_cutoff, btype='low')
        filtered = sp_signal.filtfilt(b, a, signal)
        
        return filtered
    
    def apply_highpass_filter(self, signal: np.ndarray, cutoff: float = 100.0, resonance: float = 0.5) -> np.ndarray:
        """Apply a highpass filter."""
        nyquist = self.sample_rate / 2
        normalized_cutoff = max(cutoff / nyquist, 0.01)
        
        b, a = sp_signal.butter(2, normalized_cutoff, btype='high')
        filtered = sp_signal.filtfilt(b, a, signal)
        
        return filtered
    
    def apply_bitcrusher(self, signal: np.ndarray, bit_depth: int = 8, sample_rate_reduction: int = 1) -> np.ndarray:
        """Apply bitcrushing effect for lo-fi sound."""
        levels = 2 ** bit_depth
        quantized = np.round((signal + 1) * (levels / 2)) / (levels / 2) - 1
        
        if sample_rate_reduction > 1:
            hold_length = sample_rate_reduction
            crushed = np.zeros_like(quantized)
            for i in range(0, len(quantized), hold_length):
                end_idx = min(i + hold_length, len(quantized))
                crushed[i:end_idx] = quantized[i]
            return crushed
        
        return quantized
    
    def apply_chorus(self, signal: np.ndarray, rate: float = 1.5, depth: float = 0.5, voices: int = 3, mix: float = 0.5) -> np.ndarray:
        """Apply chorus effect for thick, lush sounds."""
        output = signal.copy()
        num_samples = len(signal)
        t = np.arange(num_samples) / self.sample_rate
        
        base_delay = int(0.02 * self.sample_rate)
        max_modulation = int(0.005 * self.sample_rate * depth)
        
        for voice in range(voices):
            voice_rate = rate * (1 + voice * 0.1)
            voice_phase = voice * np.pi / voices
            
            lfo = np.sin(2 * np.pi * voice_rate * t + voice_phase)
            modulated_delay = base_delay + (lfo * max_modulation).astype(int)
            
            delayed = np.zeros_like(signal)
            for i in range(num_samples):
                delay_idx = i - modulated_delay[i]
                if 0 <= delay_idx < num_samples - 1:
                    idx_floor = int(delay_idx)
                    idx_ceil = idx_floor + 1
                    frac = delay_idx - idx_floor
                    delayed[i] = signal[idx_floor] * (1 - frac) + signal[idx_ceil] * frac
            
            output += delayed * (0.7 / (voice + 1))
        
        output = signal * (1 - mix) + output * mix / (voices + 1) * 2
        
        max_val = np.max(np.abs(output))
        if max_val > 1:
            output = output / max_val
            
        return output
    
    def apply_phaser(self, signal: np.ndarray, rate: float = 0.5, depth: float = 0.7, stages: int = 4, mix: float = 0.5, feedback: float = 0.7) -> np.ndarray:
        """Apply phaser effect for swirling, sweeping sounds."""
        num_samples = len(signal)
        t = np.arange(num_samples) / self.sample_rate
        
        lfo = 0.5 + 0.5 * np.sin(2 * np.pi * rate * t)
        
        output = np.zeros_like(signal)
        delayed = 0
        
        for i in range(num_samples):
            sample = signal[i] + delayed * feedback
            
            for stage in range(stages):
                freq = 300 + lfo[i] * depth * 2000
                c = (np.tan(np.pi * freq / self.sample_rate) - 1) / (np.tan(np.pi * freq / self.sample_rate) + 1)
                
                if i > 0:
                    sample = -c * sample + output[i-1] + c * output[i-1]
            
            output[i] = sample
            delayed = sample
        
        output = signal * (1 - mix) + output * mix
        
        max_val = np.max(np.abs(output))
        if max_val > 1:
            output = output / max_val
            
        return output
    
    def apply_compressor(self, signal: np.ndarray, threshold: float = -20.0, ratio: float = 4.0, attack: float = 0.01, release: float = 0.1, makeup_gain: float = 0.0) -> np.ndarray:
        """Apply dynamic range compression."""
        threshold_lin = 10 ** (threshold / 20)
        abs_signal = np.abs(signal)
        
        attack_coef = np.exp(-1 / (attack * self.sample_rate))
        release_coef = np.exp(-1 / (release * self.sample_rate))
        
        envelope = np.zeros_like(signal)
        envelope[0] = abs_signal[0]
        
        for i in range(1, len(signal)):
            if abs_signal[i] > envelope[i-1]:
                envelope[i] = attack_coef * envelope[i-1] + (1 - attack_coef) * abs_signal[i]
            else:
                envelope[i] = release_coef * envelope[i-1] + (1 - release_coef) * abs_signal[i]
        
        gain = np.ones_like(envelope)
        above_threshold = envelope > threshold_lin
        
        gain[above_threshold] = threshold_lin * (envelope[above_threshold] / threshold_lin) ** (1 / ratio) / envelope[above_threshold]
        
        makeup_lin = 10 ** (makeup_gain / 20)
        output = signal * gain * makeup_lin
        output = np.tanh(output * 1.5) / 1.5
        
        return output
    
    def apply_tremolo(self, signal: np.ndarray, rate: float = 5.0, depth: float = 0.5, waveform: str = "sine") -> np.ndarray:
        """Apply tremolo (amplitude modulation) effect."""
        num_samples = len(signal)
        t = np.arange(num_samples) / self.sample_rate
        
        if waveform == "sine":
            lfo = np.sin(2 * np.pi * rate * t)
        elif waveform == "square":
            lfo = np.sign(np.sin(2 * np.pi * rate * t))
        elif waveform == "triangle":
            lfo = 2 * np.abs(2 * (rate * t % 1) - 1) - 1
        else:
            lfo = np.sin(2 * np.pi * rate * t)
        
        lfo = 1 - depth * (0.5 + 0.5 * lfo)
        
        return signal * lfo
    
    def apply_vibrato(self, signal: np.ndarray, rate: float = 5.0, depth: float = 0.5) -> np.ndarray:
        """Apply vibrato (pitch modulation) effect."""
        num_samples = len(signal)
        t = np.arange(num_samples) / self.sample_rate
        
        max_delay = int(0.02 * self.sample_rate)
        lfo = np.sin(2 * np.pi * rate * t)
        mod_delay = max_delay * 0.5 * (1 + depth * lfo)
        
        output = np.zeros_like(signal)
        
        for i in range(num_samples):
            delay_idx = i - mod_delay[i]
            if 0 <= delay_idx < num_samples - 1:
                idx_floor = int(delay_idx)
                idx_ceil = idx_floor + 1
                frac = delay_idx - idx_floor
                output[i] = signal[idx_floor] * (1 - frac) + signal[idx_ceil] * frac
        
        return output
    
    def mix_signals(self, signals: list, amplitudes: list = None) -> np.ndarray:
        """Mix multiple signals together."""
        if amplitudes is None:
            amplitudes = [1.0 / len(signals)] * len(signals)
        
        max_len = max(len(s) for s in signals)
        padded_signals = []
        for s in signals:
            if len(s) < max_len:
                padded = np.zeros(max_len)
                padded[:len(s)] = s
                padded_signals.append(padded)
            else:
                padded_signals.append(s)
        
        mixed = np.zeros(max_len)
        for signal, amp in zip(padded_signals, amplitudes):
            mixed += signal * amp
        
        max_val = np.max(np.abs(mixed))
        if max_val > 0:
            mixed = mixed / max_val
            
        return mixed


class SoundPresetGenerator:
    """Generator for common sound effect presets."""
    
    def __init__(self, generator: SoundGenerator):
        self.generator = generator
    
    def generate_explosion(self, duration: float = 1.0, intensity: float = 0.8) -> np.ndarray:
        sweep = self.generator.apply_frequency_sweep(WaveformType.SINE, 150, 30, duration, 0.5, "exponential")
        noise = self.generator.generate_waveform(WaveformType.NOISE_BROWN, 0, duration, 0.7)
        sound = self.generator.mix_signals([sweep, noise], [0.6, 0.4])
        sound = self.generator.apply_adsr_envelope(sound, 0.01, 0.1, 0.3, duration - 0.41, 0.5)
        sound = self.generator.apply_lowpass_filter(sound, 800 * intensity)
        return sound * intensity
    
    def generate_laser(self, duration: float = 0.3, frequency: float = 800, sweep_range: float = 600) -> np.ndarray:
        sound = self.generator.apply_frequency_sweep(WaveformType.SAWTOOTH, frequency + sweep_range, frequency - sweep_range, duration, 0.6, "exponential")
        sound = self.generator.apply_adsr_envelope(sound, 0.01, 0.05, 0.15, 0.09, 0.3)
        sound = self.generator.apply_distortion(sound, 0.2, "soft")
        return sound
    
    def generate_coin(self, duration: float = 0.15, base_freq: float = 988) -> np.ndarray:
        tone1 = self.generator.generate_waveform(WaveformType.SINE, base_freq, duration * 0.5, 0.5)
        tone2 = self.generator.generate_waveform(WaveformType.SINE, base_freq * 1.5, duration * 0.7, 0.4)
        
        max_len = max(len(tone1), len(tone2))
        padded1, padded2 = np.zeros(max_len), np.zeros(max_len)
        padded1[:len(tone1)], padded2[:len(tone2)] = tone1, tone2
        
        sound = padded1 + padded2
        sound = self.generator.apply_adsr_envelope(sound, 0.005, 0.03, 0.05, duration - 0.085, 0.3)
        return sound
    
    def generate_jump(self, duration: float = 0.25, base_freq: float = 300) -> np.ndarray:
        sound = self.generator.apply_frequency_sweep(WaveformType.SQUARE, base_freq, base_freq * 3, duration, 0.4)
        sound = self.generator.apply_adsr_envelope(sound, 0.02, 0.1, 0.05, 0.08, 0.3)
        sound = self.generator.apply_lowpass_filter(sound, 2000)
        return sound
    
    def generate_powerup(self, duration: float = 0.8, base_freq: float = 200) -> np.ndarray:
        sound1 = self.generator.apply_frequency_sweep(WaveformType.SINE, base_freq, base_freq * 4, duration, 0.4)
        sound2 = self.generator.apply_frequency_sweep(WaveformType.SINE, base_freq * 1.5, base_freq * 6, duration, 0.3)
        sound = self.generator.mix_signals([sound1, sound2], [0.6, 0.4])
        sound = self.generator.apply_adsr_envelope(sound, 0.05, 0.15, 0.4, 0.2, 0.6)
        return sound
    
    def generate_footstep(self, duration: float = 0.15, surface: str = "default") -> np.ndarray:
        if surface == "grass":
            noise = self.generator.generate_waveform(WaveformType.NOISE_PINK, 0, duration, 0.5)
            sound = self.generator.apply_lowpass_filter(noise, 500)
        elif surface == "metal":
            sound = self.generator.generate_waveform(WaveformType.SQUARE, 200, duration, 0.4)
            sound = self.generator.apply_delay(sound, 0.02, 0.3, 0.3)
        elif surface == "water":
            noise = self.generator.generate_waveform(WaveformType.NOISE_WHITE, 0, duration, 0.4)
            sound = self.generator.apply_lowpass_filter(noise, 1500)
        else:
            noise = self.generator.generate_waveform(WaveformType.NOISE_BROWN, 0, duration, 0.5)
            thump = self.generator.generate_waveform(WaveformType.SINE, 80, duration, 0.3)
            sound = self.generator.mix_signals([noise, thump], [0.4, 0.6])
        
        sound = self.generator.apply_adsr_envelope(sound, 0.005, 0.03, 0.05, duration - 0.085, 0.2)
        return sound
    
    def generate_alarm(self, duration: float = 1.0, frequency: float = 800, pattern_rate: float = 4.0) -> np.ndarray:
        num_samples = int(self.generator.sample_rate * duration)
        sound = np.zeros(num_samples)
        beep_duration = 0.5 / pattern_rate
        beep_samples = int(beep_duration * self.generator.sample_rate)
        num_beeps = int(duration * pattern_rate * 2)
        
        for i in range(num_beeps):
            start_idx = int(i * beep_samples * 2)
            if start_idx + beep_samples > num_samples:
                break
            beep = self.generator.generate_waveform(WaveformType.SQUARE, frequency, beep_duration, 0.5)
            sound[start_idx:start_idx + len(beep)] = beep
        
        return sound
    
    def generate_hit(self, duration: float = 0.2, intensity: float = 0.7) -> np.ndarray:
        thump = self.generator.generate_waveform(WaveformType.SINE, 80, duration, 0.6)
        noise = self.generator.generate_waveform(WaveformType.NOISE_WHITE, 0, duration * 0.5, 0.4)
        sound = self.generator.mix_signals([thump, noise], [0.7, 0.3])
        sound = self.generator.apply_adsr_envelope(sound, 0.005, 0.05, 0.05, duration - 0.105, 0.2)
        sound = self.generator.apply_lowpass_filter(sound, 1500 * intensity)
        return sound * intensity
    
    def generate_bell(self, frequency: float = 440.0, duration: float = 1.5) -> np.ndarray:
        return self.generator.fm.generate_bell(frequency, duration, 0.7)
    
    def generate_gong(self, frequency: float = 150.0, duration: float = 3.0) -> np.ndarray:
        return self.generator.fm.generate_metallic(frequency, duration, 0.8, brightness=0.3)
    
    def generate_scifi_beep(self, frequency: float = 800.0, duration: float = 0.2) -> np.ndarray:
        sound = self.generator.fm.generate_fm(carrier_freq=frequency, modulator_freq=frequency * 1.5, duration=duration, modulation_index=2.0, amplitude=0.6)
        sound = self.generator.apply_adsr_envelope(sound, 0.01, 0.05, 0.08, 0.06, 0.5)
        return sound