#!/usr/bin/env python3
"""
Procedural Sound Effects Generator
===================================
A comprehensive application for generating and visualizing procedural sound effects.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import threading
import os
import wave
import json
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

from sound_generator import SoundGenerator, WaveformType, SoundPresetGenerator
from visualizer import AudioVisualizer, WaveformRenderer


class PresetManager:
    """Manager for saving and loading sound presets."""
    
    def __init__(self, presets_dir: str = None):
        if presets_dir is None:
            presets_dir = os.path.join(os.path.dirname(__file__), "presets")
        self.presets_dir = presets_dir
        os.makedirs(presets_dir, exist_ok=True)
    
    def save_preset(self, name: str, params: Dict[str, Any]) -> str:
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in name)
        filepath = os.path.join(self.presets_dir, f"{safe_name}.json")
        
        preset_data = {
            "name": name,
            "version": "2.0",
            "created": datetime.now().isoformat(),
            "parameters": params
        }
        
        with open(filepath, 'w') as f:
            json.dump(preset_data, f, indent=2)
        
        return filepath
    
    def load_preset(self, filepath: str) -> Dict[str, Any]:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data.get("parameters", data)
    
    def list_presets(self) -> list:
        presets = []
        if os.path.exists(self.presets_dir):
            for filename in os.listdir(self.presets_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.presets_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        name = data.get("name", filename[:-5])
                        presets.append((name, filepath))
                    except:
                        presets.append((filename[:-5], filepath))
        return sorted(presets, key=lambda x: x[0])


class RealtimeVisualizer:
    """Real-time audio visualization during playback."""
    
    def __init__(self, canvas: tk.Canvas, visualizer: AudioVisualizer, renderer: WaveformRenderer, mode: str = "waveform"):
        self.canvas = canvas
        self.visualizer = visualizer
        self.renderer = renderer
        self.mode = mode
        self.is_running = False
        self.audio_data = None
        self.sample_rate = 44100
        self.position = 0
        self.animation_id = None
        
    def start(self, audio_data: np.ndarray, sample_rate: int = 44100):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.position = 0
        self.is_running = True
        self._animate()
    
    def stop(self):
        self.is_running = False
        if self.animation_id:
            self.canvas.after_cancel(self.animation_id)
            self.animation_id = None
    
    def _animate(self):
        if not self.is_running or self.audio_data is None:
            return
        
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width < 10 or height < 10:
            self.animation_id = self.canvas.after(50, self._animate)
            return
        
        window_size = int(0.05 * self.sample_rate)
        start = self.position
        end = min(start + window_size, len(self.audio_data))
        
        if start >= len(self.audio_data):
            self.stop()
            return
        
        audio_window = self.audio_data[start:end]
        
        if len(audio_window) < window_size:
            padded = np.zeros(window_size)
            padded[:len(audio_window)] = audio_window
            audio_window = padded
        
        renderer = WaveformRenderer(width, height)
        
        if self.mode == "spectrum":
            freqs, mags = self.visualizer.get_spectrum_data(audio_window)
            image = renderer.render_spectrum(freqs, mags)
        else:
            image = renderer.render_waveform(audio_window)
        
        self._display_image(image)
        
        self.position += int(0.05 * self.sample_rate)
        self.animation_id = self.canvas.after(50, self._animate)
    
    def _display_image(self, image_array: np.ndarray):
        image_uint8 = (image_array * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_uint8, mode='RGB')
        photo = ImageTk.PhotoImage(pil_image)
        self.canvas.image = photo
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)


class SoundEffectsApp:
    """Main application class for the Sound Effects Generator."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Procedural Sound Effects Generator v2.0")
        self.root.geometry("1450x950")
        self.root.minsize(1200, 800)
        
        self.sample_rate = 44100
        self.generator = SoundGenerator(self.sample_rate)
        self.preset_generator = SoundPresetGenerator(self.generator)
        self.visualizer = AudioVisualizer(self.sample_rate)
        self.preset_manager = PresetManager()
        
        self.current_audio: Optional[np.ndarray] = None
        self.is_playing = False
        self.playback_thread: Optional[threading.Thread] = None
        self.stop_playback_flag = False
        
        self.rt_waveform_viz = None
        self.rt_spectrum_viz = None
        
        self._setup_styles()
        self._create_menu()
        self._create_main_layout()
        self._create_control_panels()
        self._create_visualization_panel()
        self._create_status_bar()
        
        self.generate_sound()
        
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Helvetica', 12, 'bold'))
        style.configure('Section.TLabelframe.Label', font=('Helvetica', 10, 'bold'))
        style.configure('Big.TButton', font=('Helvetica', 11, 'bold'), padding=8)
        style.configure('Preset.TButton', font=('Helvetica', 9), padding=3)
        
    def _create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Sound", command=self.reset_parameters, accelerator="Ctrl+N")
        file_menu.add_command(label="Export WAV...", command=self.export_wav, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Load Preset...", command=self.load_preset_dialog, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Preset...", command=self.save_preset_dialog, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Reset All Parameters", command=self.reset_parameters)
        edit_menu.add_command(label="Randomize Parameters", command=self.randomize_parameters)
        
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        self.realtime_viz_var = tk.BooleanVar(value=True)
        view_menu.add_checkbutton(label="Real-time Visualization", variable=self.realtime_viz_var)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
        self.root.bind('<Control-n>', lambda e: self.reset_parameters())
        self.root.bind('<Control-s>', lambda e: self.export_wav())
        self.root.bind('<Control-o>', lambda e: self.load_preset_dialog())
        self.root.bind('<Control-Shift-s>', lambda e: self.save_preset_dialog())
        self.root.bind('<space>', lambda e: self.toggle_playback())
        
    def _create_main_layout(self):
        self.main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.control_panel = ttk.Frame(self.main_container, width=480)
        self.main_container.add(self.control_panel, weight=1)
        
        self.viz_panel = ttk.Frame(self.main_container)
        self.main_container.add(self.viz_panel, weight=2)
        
    def _create_control_panels(self):
        canvas = tk.Canvas(self.control_panel, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.control_panel, orient="vertical", command=canvas.yview)
        self.controls_frame = ttk.Frame(canvas)
        
        self.controls_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=self.controls_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        self._create_waveform_section()
        self._create_frequency_section()
        self._create_envelope_section()
        self._create_effects_section()
        self._create_new_effects_section()
        self._create_presets_section()
        self._create_preset_manager_section()
        self._create_action_buttons()
        
    def _create_waveform_section(self):
        frame = ttk.LabelFrame(self.controls_frame, text="Waveform", style='Section.TLabelframe')
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(frame, text="Type:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.waveform_var = tk.StringVar(value=WaveformType.SINE.value)
        ttk.Combobox(frame, textvariable=self.waveform_var, values=[w.value for w in WaveformType], state='readonly', width=22).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(frame, text="Amplitude:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.amplitude_var = tk.DoubleVar(value=0.8)
        ttk.Scale(frame, from_=0.0, to=1.0, variable=self.amplitude_var, orient=tk.HORIZONTAL, length=160).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(frame, text="Harmonics:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.harmonics_var = tk.IntVar(value=5)
        ttk.Spinbox(frame, from_=1, to=16, textvariable=self.harmonics_var, width=20).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(frame, text="Pulse Width:").grid(row=3, column=0, sticky='w', padx=5, pady=2)
        self.pulse_width_var = tk.DoubleVar(value=0.5)
        ttk.Scale(frame, from_=0.1, to=0.9, variable=self.pulse_width_var, orient=tk.HORIZONTAL, length=160).grid(row=3, column=1, padx=5, pady=2)
        
    def _create_frequency_section(self):
        frame = ttk.LabelFrame(self.controls_frame, text="Frequency & Duration", style='Section.TLabelframe')
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(frame, text="Frequency (Hz):").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.frequency_var = tk.DoubleVar(value=440.0)
        ttk.Spinbox(frame, from_=20, to=20000, textvariable=self.frequency_var, width=20).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(frame, text="Duration (s):").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.duration_var = tk.DoubleVar(value=0.5)
        ttk.Spinbox(frame, from_=0.01, to=10.0, increment=0.1, textvariable=self.duration_var, width=20).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(row=2, column=0, columnspan=2, sticky='ew', pady=5)
        
        self.sweep_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Enable Frequency Sweep", variable=self.sweep_enabled_var).grid(row=3, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        
        ttk.Label(frame, text="End Freq (Hz):").grid(row=4, column=0, sticky='w', padx=5, pady=2)
        self.end_freq_var = tk.DoubleVar(value=880.0)
        ttk.Spinbox(frame, from_=20, to=20000, textvariable=self.end_freq_var, width=20).grid(row=4, column=1, padx=5, pady=2)
        
        ttk.Label(frame, text="Sweep Type:").grid(row=5, column=0, sticky='w', padx=5, pady=2)
        self.sweep_type_var = tk.StringVar(value="linear")
        ttk.Combobox(frame, textvariable=self.sweep_type_var, values=["linear", "exponential"], state='readonly', width=20).grid(row=5, column=1, padx=5, pady=2)
        
    def _create_envelope_section(self):
        frame = ttk.LabelFrame(self.controls_frame, text="ADSR Envelope", style='Section.TLabelframe')
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.envelope_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Enable ADSR Envelope", variable=self.envelope_enabled_var).grid(row=0, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        
        ttk.Label(frame, text="Attack (s):").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.attack_var = tk.DoubleVar(value=0.01)
        ttk.Spinbox(frame, from_=0.0, to=2.0, increment=0.01, textvariable=self.attack_var, width=20).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(frame, text="Decay (s):").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.decay_var = tk.DoubleVar(value=0.1)
        ttk.Spinbox(frame, from_=0.0, to=2.0, increment=0.01, textvariable=self.decay_var, width=20).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(frame, text="Sustain (s):").grid(row=3, column=0, sticky='w', padx=5, pady=2)
        self.sustain_var = tk.DoubleVar(value=0.2)
        ttk.Spinbox(frame, from_=0.0, to=2.0, increment=0.01, textvariable=self.sustain_var, width=20).grid(row=3, column=1, padx=5, pady=2)
        
        ttk.Label(frame, text="Release (s):").grid(row=4, column=0, sticky='w', padx=5, pady=2)
        self.release_var = tk.DoubleVar(value=0.1)
        ttk.Spinbox(frame, from_=0.0, to=2.0, increment=0.01, textvariable=self.release_var, width=20).grid(row=4, column=1, padx=5, pady=2)
        
        ttk.Label(frame, text="Sustain Level:").grid(row=5, column=0, sticky='w', padx=5, pady=2)
        self.sustain_level_var = tk.DoubleVar(value=0.7)
        ttk.Scale(frame, from_=0.0, to=1.0, variable=self.sustain_level_var, orient=tk.HORIZONTAL, length=160).grid(row=5, column=1, padx=5, pady=2)
        
    def _create_effects_section(self):
        frame = ttk.LabelFrame(self.controls_frame, text="Effects (Basic)", style='Section.TLabelframe')
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.reverb_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Reverb", variable=self.reverb_enabled_var).grid(row=0, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        
        ttk.Label(frame, text="Room Size:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.reverb_room_var = tk.DoubleVar(value=0.5)
        ttk.Scale(frame, from_=0.1, to=1.0, variable=self.reverb_room_var, orient=tk.HORIZONTAL, length=120).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(frame, text="Wet Level:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.reverb_wet_var = tk.DoubleVar(value=0.3)
        ttk.Scale(frame, from_=0.0, to=1.0, variable=self.reverb_wet_var, orient=tk.HORIZONTAL, length=120).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(row=3, column=0, columnspan=2, sticky='ew', pady=3)
        
        self.delay_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Delay/Echo", variable=self.delay_enabled_var).grid(row=4, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        
        ttk.Label(frame, text="Delay Time (s):").grid(row=5, column=0, sticky='w', padx=5, pady=2)
        self.delay_time_var = tk.DoubleVar(value=0.3)
        ttk.Spinbox(frame, from_=0.01, to=2.0, increment=0.01, textvariable=self.delay_time_var, width=15).grid(row=5, column=1, padx=5, pady=2)
        
        ttk.Label(frame, text="Feedback:").grid(row=6, column=0, sticky='w', padx=5, pady=2)
        self.delay_feedback_var = tk.DoubleVar(value=0.4)
        ttk.Scale(frame, from_=0.0, to=0.9, variable=self.delay_feedback_var, orient=tk.HORIZONTAL, length=120).grid(row=6, column=1, padx=5, pady=2)
        
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(row=7, column=0, columnspan=2, sticky='ew', pady=3)
        
        self.distortion_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Distortion", variable=self.distortion_enabled_var).grid(row=8, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        
        ttk.Label(frame, text="Drive:").grid(row=9, column=0, sticky='w', padx=5, pady=2)
        self.distortion_drive_var = tk.DoubleVar(value=0.5)
        ttk.Scale(frame, from_=0.0, to=1.0, variable=self.distortion_drive_var, orient=tk.HORIZONTAL, length=120).grid(row=9, column=1, padx=5, pady=2)
        
        ttk.Label(frame, text="Type:").grid(row=10, column=0, sticky='w', padx=5, pady=2)
        self.distortion_type_var = tk.StringVar(value="soft")
        ttk.Combobox(frame, textvariable=self.distortion_type_var, values=["soft", "hard", "fuzz"], state='readonly', width=15).grid(row=10, column=1, padx=5, pady=2)
        
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(row=11, column=0, columnspan=2, sticky='ew', pady=3)
        
        self.lowpass_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Lowpass Filter", variable=self.lowpass_enabled_var).grid(row=12, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        
        ttk.Label(frame, text="Cutoff (Hz):").grid(row=13, column=0, sticky='w', padx=5, pady=2)
        self.lowpass_cutoff_var = tk.DoubleVar(value=1000.0)
        ttk.Spinbox(frame, from_=100, to=20000, increment=100, textvariable=self.lowpass_cutoff_var, width=15).grid(row=13, column=1, padx=5, pady=2)
        
        self.highpass_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Highpass Filter", variable=self.highpass_enabled_var).grid(row=14, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        
        ttk.Label(frame, text="Cutoff (Hz):").grid(row=15, column=0, sticky='w', padx=5, pady=2)
        self.highpass_cutoff_var = tk.DoubleVar(value=100.0)
        ttk.Spinbox(frame, from_=20, to=5000, increment=50, textvariable=self.highpass_cutoff_var, width=15).grid(row=15, column=1, padx=5, pady=2)
        
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(row=16, column=0, columnspan=2, sticky='ew', pady=3)
        
        self.bitcrush_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Bitcrusher", variable=self.bitcrush_enabled_var).grid(row=17, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        
        ttk.Label(frame, text="Bit Depth:").grid(row=18, column=0, sticky='w', padx=5, pady=2)
        self.bitcrush_depth_var = tk.IntVar(value=8)
        ttk.Spinbox(frame, from_=1, to=16, textvariable=self.bitcrush_depth_var, width=15).grid(row=18, column=1, padx=5, pady=2)
        
    def _create_new_effects_section(self):
        frame = ttk.LabelFrame(self.controls_frame, text="Effects (Advanced)", style='Section.TLabelframe')
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.chorus_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Chorus", variable=self.chorus_enabled_var).grid(row=0, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        
        ttk.Label(frame, text="Rate (Hz):").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.chorus_rate_var = tk.DoubleVar(value=1.5)
        ttk.Spinbox(frame, from_=0.1, to=10.0, increment=0.1, textvariable=self.chorus_rate_var, width=15).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(frame, text="Depth:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.chorus_depth_var = tk.DoubleVar(value=0.5)
        ttk.Scale(frame, from_=0.0, to=1.0, variable=self.chorus_depth_var, orient=tk.HORIZONTAL, length=120).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(frame, text="Voices:").grid(row=3, column=0, sticky='w', padx=5, pady=2)
        self.chorus_voices_var = tk.IntVar(value=3)
        ttk.Spinbox(frame, from_=1, to=6, textvariable=self.chorus_voices_var, width=15).grid(row=3, column=1, padx=5, pady=2)
        
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(row=4, column=0, columnspan=2, sticky='ew', pady=3)
        
        self.phaser_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Phaser", variable=self.phaser_enabled_var).grid(row=5, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        
        ttk.Label(frame, text="Rate (Hz):").grid(row=6, column=0, sticky='w', padx=5, pady=2)
        self.phaser_rate_var = tk.DoubleVar(value=0.5)
        ttk.Spinbox(frame, from_=0.1, to=5.0, increment=0.1, textvariable=self.phaser_rate_var, width=15).grid(row=6, column=1, padx=5, pady=2)
        
        ttk.Label(frame, text="Depth:").grid(row=7, column=0, sticky='w', padx=5, pady=2)
        self.phaser_depth_var = tk.DoubleVar(value=0.7)
        ttk.Scale(frame, from_=0.0, to=1.0, variable=self.phaser_depth_var, orient=tk.HORIZONTAL, length=120).grid(row=7, column=1, padx=5, pady=2)
        
        ttk.Label(frame, text="Stages:").grid(row=8, column=0, sticky='w', padx=5, pady=2)
        self.phaser_stages_var = tk.IntVar(value=4)
        ttk.Spinbox(frame, from_=2, to=8, textvariable=self.phaser_stages_var, width=15).grid(row=8, column=1, padx=5, pady=2)
        
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(row=9, column=0, columnspan=2, sticky='ew', pady=3)
        
        self.compressor_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Compressor", variable=self.compressor_enabled_var).grid(row=10, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        
        ttk.Label(frame, text="Threshold (dB):").grid(row=11, column=0, sticky='w', padx=5, pady=2)
        self.compressor_threshold_var = tk.DoubleVar(value=-20.0)
        ttk.Spinbox(frame, from_=-60, to=0, increment=1, textvariable=self.compressor_threshold_var, width=15).grid(row=11, column=1, padx=5, pady=2)
        
        ttk.Label(frame, text="Ratio:").grid(row=12, column=0, sticky='w', padx=5, pady=2)
        self.compressor_ratio_var = tk.DoubleVar(value=4.0)
        ttk.Spinbox(frame, from_=1.0, to=20.0, increment=0.5, textvariable=self.compressor_ratio_var, width=15).grid(row=12, column=1, padx=5, pady=2)
        
        ttk.Label(frame, text="Makeup Gain (dB):").grid(row=13, column=0, sticky='w', padx=5, pady=2)
        self.compressor_makeup_var = tk.DoubleVar(value=0.0)
        ttk.Spinbox(frame, from_=0, to=24, increment=1, textvariable=self.compressor_makeup_var, width=15).grid(row=13, column=1, padx=5, pady=2)
        
    def _create_presets_section(self):
        frame = ttk.LabelFrame(self.controls_frame, text="Quick Presets", style='Section.TLabelframe')
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        presets = [
            ("Explosion", self.preset_explosion),
            ("Laser", self.preset_laser),
            ("Coin", self.preset_coin),
            ("Jump", self.preset_jump),
            ("Power-up", self.preset_powerup),
            ("Hit", self.preset_hit),
            ("Alarm", self.preset_alarm),
            ("Footstep", self.preset_footstep),
            ("Bell (FM)", self.preset_bell),
            ("Gong (FM)", self.preset_gong),
            ("Sci-Fi Beep", self.preset_scifi),
        ]
        
        row, col = 0, 0
        for name, command in presets:
            btn = ttk.Button(frame, text=name, command=command, width=10, style='Preset.TButton')
            btn.grid(row=row, column=col, padx=2, pady=2)
            col += 1
            if col > 2:
                col = 0
                row += 1
                
    def _create_preset_manager_section(self):
        frame = ttk.LabelFrame(self.controls_frame, text="Preset Manager", style='Section.TLabelframe')
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Save Preset...", command=self.save_preset_dialog, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Load Preset...", command=self.load_preset_dialog, width=15).pack(side=tk.LEFT, padx=2)
        
        self.preset_listbox = tk.Listbox(frame, height=4, selectmode=tk.SINGLE)
        self.preset_listbox.pack(fill=tk.X, padx=5, pady=5)
        self.preset_listbox.bind('<Double-1>', lambda e: self.load_selected_preset())
        
        ttk.Button(frame, text="Load Selected", command=self.load_selected_preset, width=15).pack(pady=2)
        
        self._refresh_preset_list()
        
    def _create_action_buttons(self):
        frame = ttk.Frame(self.controls_frame)
        frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(frame, text="Generate Sound", command=self.generate_sound, style='Big.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(frame, text="Play Sound", command=self.play_sound, style='Big.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(frame, text="Stop Playback", command=self.stop_sound, style='Big.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(frame, text="Export WAV...", command=self.export_wav, style='Big.TButton').pack(fill=tk.X, pady=3)
        
    def _create_visualization_panel(self):
        self.viz_notebook = ttk.Notebook(self.viz_panel)
        self.viz_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        waveform_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(waveform_frame, text="Waveform")
        self.waveform_canvas = tk.Canvas(waveform_frame, bg='#1a1a24', highlightthickness=0)
        self.waveform_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        spectrum_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(spectrum_frame, text="Spectrum")
        self.spectrum_canvas = tk.Canvas(spectrum_frame, bg='#1a1a24', highlightthickness=0)
        self.spectrum_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        spectrogram_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(spectrogram_frame, text="Spectrogram")
        self.spectrogram_canvas = tk.Canvas(spectrogram_frame, bg='#1a1a24', highlightthickness=0)
        self.spectrogram_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        combined_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(combined_frame, text="Combined")
        self.combined_waveform_canvas = tk.Canvas(combined_frame, bg='#1a1a24', highlightthickness=0, height=200)
        self.combined_waveform_canvas.pack(fill=tk.X, padx=5, pady=2)
        self.combined_spectrum_canvas = tk.Canvas(combined_frame, bg='#1a1a24', highlightthickness=0)
        self.combined_spectrum_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
        
        info_frame = ttk.LabelFrame(self.viz_panel, text="Sound Info")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        self.info_label = ttk.Label(info_frame, text="No sound generated yet")
        self.info_label.pack(padx=10, pady=5)
        
        self.waveform_canvas.bind('<Configure>', self._on_resize)
        self.spectrum_canvas.bind('<Configure>', self._on_resize)
        self.spectrogram_canvas.bind('<Configure>', self._on_resize)
        
    def _create_status_bar(self):
        self.status_var = tk.StringVar(value="Ready - Press Space to play/stop")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def _on_resize(self, event):
        if self.current_audio is not None and not self.is_playing:
            self.update_visualizations()
            
    def _refresh_preset_list(self):
        self.preset_listbox.delete(0, tk.END)
        for name, filepath in self.preset_manager.list_presets():
            self.preset_listbox.insert(tk.END, name)
            
    def get_all_parameters(self) -> Dict[str, Any]:
        return {
            "waveform_type": self.waveform_var.get(),
            "amplitude": self.amplitude_var.get(),
            "harmonics": self.harmonics_var.get(),
            "pulse_width": self.pulse_width_var.get(),
            "frequency": self.frequency_var.get(),
            "duration": self.duration_var.get(),
            "sweep_enabled": self.sweep_enabled_var.get(),
            "end_frequency": self.end_freq_var.get(),
            "sweep_type": self.sweep_type_var.get(),
            "envelope_enabled": self.envelope_enabled_var.get(),
            "attack": self.attack_var.get(),
            "decay": self.decay_var.get(),
            "sustain": self.sustain_var.get(),
            "release": self.release_var.get(),
            "sustain_level": self.sustain_level_var.get(),
            "reverb_enabled": self.reverb_enabled_var.get(),
            "reverb_room": self.reverb_room_var.get(),
            "reverb_wet": self.reverb_wet_var.get(),
            "delay_enabled": self.delay_enabled_var.get(),
            "delay_time": self.delay_time_var.get(),
            "delay_feedback": self.delay_feedback_var.get(),
            "distortion_enabled": self.distortion_enabled_var.get(),
            "distortion_drive": self.distortion_drive_var.get(),
            "distortion_type": self.distortion_type_var.get(),
            "lowpass_enabled": self.lowpass_enabled_var.get(),
            "lowpass_cutoff": self.lowpass_cutoff_var.get(),
            "highpass_enabled": self.highpass_enabled_var.get(),
            "highpass_cutoff": self.highpass_cutoff_var.get(),
            "bitcrush_enabled": self.bitcrush_enabled_var.get(),
            "bitcrush_depth": self.bitcrush_depth_var.get(),
            "chorus_enabled": self.chorus_enabled_var.get(),
            "chorus_rate": self.chorus_rate_var.get(),
            "chorus_depth": self.chorus_depth_var.get(),
            "chorus_voices": self.chorus_voices_var.get(),
            "phaser_enabled": self.phaser_enabled_var.get(),
            "phaser_rate": self.phaser_rate_var.get(),
            "phaser_depth": self.phaser_depth_var.get(),
            "phaser_stages": self.phaser_stages_var.get(),
            "compressor_enabled": self.compressor_enabled_var.get(),
            "compressor_threshold": self.compressor_threshold_var.get(),
            "compressor_ratio": self.compressor_ratio_var.get(),
            "compressor_makeup": self.compressor_makeup_var.get(),
        }
        
    def set_all_parameters(self, params: Dict[str, Any]):
        for key, var in [
            ("waveform_type", self.waveform_var),
            ("amplitude", self.amplitude_var),
            ("harmonics", self.harmonics_var),
            ("pulse_width", self.pulse_width_var),
            ("frequency", self.frequency_var),
            ("duration", self.duration_var),
            ("sweep_enabled", self.sweep_enabled_var),
            ("end_frequency", self.end_freq_var),
            ("sweep_type", self.sweep_type_var),
            ("envelope_enabled", self.envelope_enabled_var),
            ("attack", self.attack_var),
            ("decay", self.decay_var),
            ("sustain", self.sustain_var),
            ("release", self.release_var),
            ("sustain_level", self.sustain_level_var),
            ("reverb_enabled", self.reverb_enabled_var),
            ("reverb_room", self.reverb_room_var),
            ("reverb_wet", self.reverb_wet_var),
            ("delay_enabled", self.delay_enabled_var),
            ("delay_time", self.delay_time_var),
            ("delay_feedback", self.delay_feedback_var),
            ("distortion_enabled", self.distortion_enabled_var),
            ("distortion_drive", self.distortion_drive_var),
            ("distortion_type", self.distortion_type_var),
            ("lowpass_enabled", self.lowpass_enabled_var),
            ("lowpass_cutoff", self.lowpass_cutoff_var),
            ("highpass_enabled", self.highpass_enabled_var),
            ("highpass_cutoff", self.highpass_cutoff_var),
            ("bitcrush_enabled", self.bitcrush_enabled_var),
            ("bitcrush_depth", self.bitcrush_depth_var),
            ("chorus_enabled", self.chorus_enabled_var),
            ("chorus_rate", self.chorus_rate_var),
            ("chorus_depth", self.chorus_depth_var),
            ("chorus_voices", self.chorus_voices_var),
            ("phaser_enabled", self.phaser_enabled_var),
            ("phaser_rate", self.phaser_rate_var),
            ("phaser_depth", self.phaser_depth_var),
            ("phaser_stages", self.phaser_stages_var),
            ("compressor_enabled", self.compressor_enabled_var),
            ("compressor_threshold", self.compressor_threshold_var),
            ("compressor_ratio", self.compressor_ratio_var),
            ("compressor_makeup", self.compressor_makeup_var),
        ]:
            if key in params:
                var.set(params[key])
    
    def generate_sound(self):
        try:
            self.status_var.set("Generating sound...")
            self.root.update()
            
            waveform_type = WaveformType(self.waveform_var.get())
            frequency = self.frequency_var.get()
            duration = self.duration_var.get()
            amplitude = self.amplitude_var.get()
            
            if self.sweep_enabled_var.get():
                audio = self.generator.apply_frequency_sweep(
                    waveform_type, frequency, self.end_freq_var.get(),
                    duration, amplitude, self.sweep_type_var.get()
                )
            else:
                audio = self.generator.generate_waveform(
                    waveform_type, frequency, duration, amplitude,
                    pulse_width=self.pulse_width_var.get(),
                    harmonics=self.harmonics_var.get()
                )
            
            if self.envelope_enabled_var.get():
                audio = self.generator.apply_adsr_envelope(
                    audio, self.attack_var.get(), self.decay_var.get(),
                    self.sustain_var.get(), self.release_var.get(),
                    self.sustain_level_var.get()
                )
            
            audio = self._apply_effects(audio)
            
            self.current_audio = audio
            self.update_visualizations()
            self.update_info()
            self.status_var.set("Sound generated - Press Space to play")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate sound: {str(e)}")
            self.status_var.set("Error generating sound")
            
    def _apply_effects(self, audio: np.ndarray) -> np.ndarray:
        result = audio.copy()
        
        if self.reverb_enabled_var.get():
            result = self.generator.apply_reverb(result, self.reverb_room_var.get(), 0.5, self.reverb_wet_var.get())
        
        if self.delay_enabled_var.get():
            result = self.generator.apply_delay(result, self.delay_time_var.get(), self.delay_feedback_var.get(), 0.5)
        
        if self.distortion_enabled_var.get():
            result = self.generator.apply_distortion(result, self.distortion_drive_var.get(), self.distortion_type_var.get())
        
        if self.lowpass_enabled_var.get():
            result = self.generator.apply_lowpass_filter(result, self.lowpass_cutoff_var.get())
        
        if self.highpass_enabled_var.get():
            result = self.generator.apply_highpass_filter(result, self.highpass_cutoff_var.get())
        
        if self.bitcrush_enabled_var.get():
            result = self.generator.apply_bitcrusher(result, self.bitcrush_depth_var.get())
        
        if self.chorus_enabled_var.get():
            result = self.generator.apply_chorus(
                result, self.chorus_rate_var.get(), self.chorus_depth_var.get(),
                self.chorus_voices_var.get(), 0.5
            )
        
        if self.phaser_enabled_var.get():
            result = self.generator.apply_phaser(
                result, self.phaser_rate_var.get(), self.phaser_depth_var.get(),
                self.phaser_stages_var.get(), 0.5, 0.7
            )
        
        if self.compressor_enabled_var.get():
            result = self.generator.apply_compressor(
                result, self.compressor_threshold_var.get(), self.compressor_ratio_var.get(),
                0.01, 0.1, self.compressor_makeup_var.get()
            )
        
        return result
        
    def update_visualizations(self):
        if self.current_audio is None:
            return
        self.root.update()
        self._update_waveform_display()
        self._update_spectrum_display()
        self._update_spectrogram_display()
        self._update_combined_display()
        
    def _update_waveform_display(self):
        canvas = self.waveform_canvas
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 10 or height < 10:
            return
        renderer = WaveformRenderer(width, height)
        time_axis, waveform = self.visualizer.get_waveform_data(self.current_audio, width)
        image = renderer.render_waveform(waveform)
        self._display_image(canvas, image)
        
    def _update_spectrum_display(self):
        canvas = self.spectrum_canvas
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 10 or height < 10:
            return
        renderer = WaveformRenderer(width, height)
        frequencies, magnitudes = self.visualizer.get_spectrum_data(self.current_audio)
        image = renderer.render_spectrum(frequencies, magnitudes)
        self._display_image(canvas, image)
        
    def _update_spectrogram_display(self):
        canvas = self.spectrogram_canvas
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 10 or height < 10:
            return
        renderer = WaveformRenderer(width, height)
        frequencies, times, spectrogram = self.visualizer.get_spectrogram_data(self.current_audio)
        image = renderer.render_spectrogram(frequencies, times, spectrogram)
        self._display_image(canvas, image)
        
    def _update_combined_display(self):
        canvas1 = self.combined_waveform_canvas
        width1, height1 = canvas1.winfo_width(), canvas1.winfo_height()
        if width1 > 10 and height1 > 10:
            renderer1 = WaveformRenderer(width1, height1)
            time_axis, waveform = self.visualizer.get_waveform_data(self.current_audio, width1)
            image1 = renderer1.render_waveform(waveform)
            self._display_image(canvas1, image1)
        
        canvas2 = self.combined_spectrum_canvas
        width2, height2 = canvas2.winfo_width(), canvas2.winfo_height()
        if width2 > 10 and height2 > 10:
            renderer2 = WaveformRenderer(width2, height2)
            frequencies, magnitudes = self.visualizer.get_spectrum_data(self.current_audio)
            image2 = renderer2.render_spectrum(frequencies, magnitudes)
            self._display_image(canvas2, image2)
            
    def _display_image(self, canvas: tk.Canvas, image_array: np.ndarray):
        image_uint8 = (image_array * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_uint8, mode='RGB')
        photo = ImageTk.PhotoImage(pil_image)
        canvas.image = photo
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        
    def update_info(self):
        if self.current_audio is None:
            return
        duration = len(self.current_audio) / self.sample_rate
        max_amp = np.max(np.abs(self.current_audio))
        rms = np.sqrt(np.mean(self.current_audio ** 2))
        info_text = f"Duration: {duration:.3f}s | Samples: {len(self.current_audio):,} | Peak: {max_amp:.3f} | RMS: {rms:.3f} | SR: {self.sample_rate} Hz"
        self.info_label.config(text=info_text)
        
    def play_sound(self):
        if self.current_audio is None:
            messagebox.showwarning("Warning", "No sound to play. Generate a sound first.")
            return
        if self.is_playing:
            return
            
        self.is_playing = True
        self.stop_playback_flag = False
        self.status_var.set("Playing...")
        
        if self.realtime_viz_var.get():
            self._start_realtime_viz()
        
        def playback_thread():
            try:
                import sounddevice as sd
                sd.play(self.current_audio, self.sample_rate)
                sd.wait()
            except ImportError:
                try:
                    import pygame
                    pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=1)
                    audio_int = (self.current_audio * 32767).astype(np.int16)
                    sound = pygame.sndarray.make_sound(audio_int)
                    sound.play()
                    while pygame.mixer.get_busy() and not self.stop_playback_flag:
                        pygame.time.Clock().tick(60)
                    pygame.mixer.quit()
                except ImportError:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Install sounddevice or pygame"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Playback failed: {str(e)}"))
            finally:
                self.is_playing = False
                self.root.after(0, self._stop_realtime_viz)
                self.root.after(0, lambda: self.status_var.set("Ready - Press Space to play"))
                
        self.playback_thread = threading.Thread(target=playback_thread, daemon=True)
        self.playback_thread.start()
        
    def _start_realtime_viz(self):
        self.rt_waveform_viz = RealtimeVisualizer(self.waveform_canvas, self.visualizer, WaveformRenderer(100, 100), "waveform")
        self.rt_waveform_viz.start(self.current_audio, self.sample_rate)
        
        self.rt_spectrum_viz = RealtimeVisualizer(self.spectrum_canvas, self.visualizer, WaveformRenderer(100, 100), "spectrum")
        self.rt_spectrum_viz.start(self.current_audio, self.sample_rate)
        
    def _stop_realtime_viz(self):
        if self.rt_waveform_viz:
            self.rt_waveform_viz.stop()
        if self.rt_spectrum_viz:
            self.rt_spectrum_viz.stop()
        self.update_visualizations()
        
    def stop_sound(self):
        self.stop_playback_flag = True
        self.is_playing = False
        
        try:
            import sounddevice as sd
            sd.stop()
        except:
            pass
        try:
            import pygame
            if pygame.mixer.get_init():
                pygame.mixer.stop()
        except:
            pass
            
        self._stop_realtime_viz()
        self.status_var.set("Playback stopped")
        
    def toggle_playback(self):
        if self.is_playing:
            self.stop_sound()
        else:
            self.play_sound()
        
    def export_wav(self):
        if self.current_audio is None:
            messagebox.showwarning("Warning", "No sound to export.")
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            title="Export Sound"
        )
        
        if not filepath:
            return
            
        try:
            audio_int = (self.current_audio * 32767).astype(np.int16)
            with wave.open(filepath, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int.tobytes())
            messagebox.showinfo("Success", f"Sound exported to:\n{filepath}")
            self.status_var.set(f"Exported to {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {str(e)}")
            
    def save_preset_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Save Preset")
        dialog.geometry("300x120")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Preset Name:").pack(padx=10, pady=5)
        name_var = tk.StringVar(value="My Preset")
        entry = ttk.Entry(dialog, textvariable=name_var, width=40)
        entry.pack(padx=10, pady=5)
        entry.select_range(0, tk.END)
        entry.focus()
        
        def do_save():
            name = name_var.get().strip()
            if not name:
                messagebox.showwarning("Warning", "Please enter a name.")
                return
            try:
                params = self.get_all_parameters()
                filepath = self.preset_manager.save_preset(name, params)
                messagebox.showinfo("Success", f"Preset saved to:\n{filepath}")
                self._refresh_preset_list()
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save preset: {str(e)}")
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Save", command=do_save, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy, width=10).pack(side=tk.LEFT, padx=5)
        
    def load_preset_dialog(self):
        presets = self.preset_manager.list_presets()
        if not presets:
            messagebox.showinfo("Info", "No saved presets found.")
            return
            
        dialog = tk.Toplevel(self.root)
        dialog.title("Load Preset")
        dialog.geometry("350x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Select a preset:").pack(padx=10, pady=5)
        
        listbox = tk.Listbox(dialog, height=10)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        for name, filepath in presets:
            listbox.insert(tk.END, name)
        
        def do_load():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("Warning", "Please select a preset.")
                return
            try:
                filepath = presets[selection[0]][1]
                params = self.preset_manager.load_preset(filepath)
                self.set_all_parameters(params)
                self.generate_sound()
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load preset: {str(e)}")
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Load", command=do_load, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy, width=10).pack(side=tk.LEFT, padx=5)
        
        listbox.bind('<Double-1>', lambda e: do_load())
        
    def load_selected_preset(self):
        selection = self.preset_listbox.curselection()
        if not selection:
            return
        presets = self.preset_manager.list_presets()
        filepath = presets[selection[0]][1]
        try:
            params = self.preset_manager.load_preset(filepath)
            self.set_all_parameters(params)
            self.generate_sound()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load preset: {str(e)}")
    
    def randomize_parameters(self):
        import random
        
        waveforms = [w.value for w in WaveformType]
        self.waveform_var.set(random.choice(waveforms))
        self.amplitude_var.set(random.uniform(0.5, 1.0))
        self.harmonics_var.set(random.randint(2, 10))
        self.pulse_width_var.set(random.uniform(0.2, 0.8))
        self.frequency_var.set(random.uniform(100, 2000))
        self.duration_var.set(random.uniform(0.1, 2.0))
        self.sweep_enabled_var.set(random.choice([True, False]))
        self.end_freq_var.set(random.uniform(100, 3000))
        self.envelope_enabled_var.set(random.choice([True, False]))
        self.attack_var.set(random.uniform(0.0, 0.2))
        self.decay_var.set(random.uniform(0.0, 0.3))
        self.sustain_var.set(random.uniform(0.0, 0.5))
        self.release_var.set(random.uniform(0.0, 0.3))
        
        for var in [self.reverb_enabled_var, self.delay_enabled_var, self.distortion_enabled_var,
                    self.chorus_enabled_var, self.phaser_enabled_var]:
            var.set(random.random() < 0.3)
        
        self.generate_sound()
        self.status_var.set("Parameters randomized")
        
    def _reset_envelope(self):
        self.envelope_enabled_var.set(False)
        
    def _reset_effects(self):
        for var in [self.reverb_enabled_var, self.delay_enabled_var, self.distortion_enabled_var,
                    self.lowpass_enabled_var, self.highpass_enabled_var, self.bitcrush_enabled_var,
                    self.chorus_enabled_var, self.phaser_enabled_var, self.compressor_enabled_var]:
            var.set(False)
        
    def preset_explosion(self):
        self._reset_envelope()
        self._reset_effects()
        self.current_audio = self.preset_generator.generate_explosion()
        self.update_visualizations()
        self.update_info()
        self.status_var.set("Generated: Explosion")
        
    def preset_laser(self):
        self._reset_envelope()
        self._reset_effects()
        self.current_audio = self.preset_generator.generate_laser()
        self.update_visualizations()
        self.update_info()
        self.status_var.set("Generated: Laser")
        
    def preset_coin(self):
        self._reset_envelope()
        self._reset_effects()
        self.current_audio = self.preset_generator.generate_coin()
        self.update_visualizations()
        self.update_info()
        self.status_var.set("Generated: Coin")
        
    def preset_jump(self):
        self._reset_envelope()
        self._reset_effects()
        self.current_audio = self.preset_generator.generate_jump()
        self.update_visualizations()
        self.update_info()
        self.status_var.set("Generated: Jump")
        
    def preset_powerup(self):
        self._reset_envelope()
        self._reset_effects()
        self.current_audio = self.preset_generator.generate_powerup()
        self.update_visualizations()
        self.update_info()
        self.status_var.set("Generated: Power-up")
        
    def preset_hit(self):
        self._reset_envelope()
        self._reset_effects()
        self.current_audio = self.preset_generator.generate_hit()
        self.update_visualizations()
        self.update_info()
        self.status_var.set("Generated: Hit")
        
    def preset_alarm(self):
        self._reset_envelope()
        self._reset_effects()
        self.current_audio = self.preset_generator.generate_alarm()
        self.update_visualizations()
        self.update_info()
        self.status_var.set("Generated: Alarm")
        
    def preset_footstep(self):
        self._reset_envelope()
        self._reset_effects()
        self.current_audio = self.preset_generator.generate_footstep()
        self.update_visualizations()
        self.update_info()
        self.status_var.set("Generated: Footstep")
        
    def preset_bell(self):
        self._reset_envelope()
        self._reset_effects()
        freq = self.frequency_var.get()
        self.current_audio = self.preset_generator.generate_bell(freq)
        self.update_visualizations()
        self.update_info()
        self.status_var.set("Generated: FM Bell")
        
    def preset_gong(self):
        self._reset_envelope()
        self._reset_effects()
        freq = self.frequency_var.get()
        self.current_audio = self.preset_generator.generate_gong(freq)
        self.update_visualizations()
        self.update_info()
        self.status_var.set("Generated: FM Gong")
        
    def preset_scifi(self):
        self._reset_envelope()
        self._reset_effects()
        freq = self.frequency_var.get()
        self.current_audio = self.preset_generator.generate_scifi_beep(freq)
        self.update_visualizations()
        self.update_info()
        self.status_var.set("Generated: Sci-Fi Beep")
        
    def reset_parameters(self):
        self.waveform_var.set(WaveformType.SINE.value)
        self.harmonics_var.set(5)
        self.pulse_width_var.set(0.5)
        self.amplitude_var.set(0.8)
        self.frequency_var.set(440.0)
        self.duration_var.set(0.5)
        self.sweep_enabled_var.set(False)
        self.end_freq_var.set(880.0)
        self.sweep_type_var.set("linear")
        
        self._reset_envelope()
        self._reset_effects()
        
        self.attack_var.set(0.01)
        self.decay_var.set(0.1)
        self.sustain_var.set(0.2)
        self.release_var.set(0.1)
        self.sustain_level_var.set(0.7)
        
        self.generate_sound()
        
    def show_about(self):
        about_text = """Procedural Sound Effects Generator v2.0
        
A comprehensive tool for generating and visualizing 
procedurally generated sound effects.

Features:
• Multiple waveform types including FM synthesis
• ADSR envelope shaping  
• Audio effects (reverb, delay, distortion, chorus, phaser, compressor)
• Real-time visualization during playback
• Preset save/load system
• WAV export

Keyboard Shortcuts:
• Space: Play/Stop
• Ctrl+N: New Sound
• Ctrl+S: Export WAV
• Ctrl+O: Load Preset
• Ctrl+Shift+S: Save Preset

Created with Python, NumPy, SciPy, and Tkinter
"""
        messagebox.showinfo("About", about_text)


def main():
    """Main entry point."""
    root = tk.Tk()
    app = SoundEffectsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()