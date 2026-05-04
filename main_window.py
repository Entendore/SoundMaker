from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QComboBox, QCheckBox, QPushButton,
    QSlider, QSpinBox, QDoubleSpinBox, QSplitter, QTabWidget,
    QListWidget, QScrollArea, QMessageBox, QStatusBar,
    QDialog, QLineEdit, QDialogButtonBox, QFormLayout,
    QSizePolicy, QGridLayout, QFileDialog, QApplication, QFrame
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QCloseEvent, QShortcut
import numpy as np
import random

from sound_generator import SoundGenerator, WaveformType, SoundPresetGenerator, to_cpu, to_gpu
from visualizer import AudioVisualizer, WaveformRenderer
from preset_manager import PresetManager
from widgets import ImageWidget, PlaybackWorker


class SoundEffectsApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procedural Sound FX Gen v2.2")
        self.resize(1500, 950)
        self.setMinimumSize(1200, 800)

        self.sample_rate = 44100
        self.generator = SoundGenerator(self.sample_rate)
        self.preset_generator = SoundPresetGenerator(self.generator)
        self.visualizer = AudioVisualizer(self.sample_rate)
        self.preset_manager = PresetManager()

        self.current_audio = None
        self.playback_worker = None
        self.is_playing = False
        self.realtime_enabled = True

        # Timers
        self.rt_timer = QTimer(self)
        self.rt_timer.setInterval(50)
        self.rt_timer.timeout.connect(self._update_realtime_viz)
        self.rt_position = 0

        self._regen_timer = QTimer(self)
        self._regen_timer.setSingleShot(True)
        self._regen_timer.setInterval(300)
        self._regen_timer.timeout.connect(self.generate_sound)

        self.params = {}

        self._setup_menubar()
        self._setup_ui()
        self._setup_statusbar()
        self._apply_stylesheet()
        self._refresh_preset_list()
        
        QTimer.singleShot(100, self.generate_sound)

    def closeEvent(self, event: QCloseEvent):
        if self.playback_worker and self.playback_worker.isRunning():
            self.stop_sound()
            if not self.playback_worker.wait(1000):
                self.playback_worker.terminate()
        event.accept()

    def _setup_menubar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        
        export_action = QAction("Export WAV...", self)
        export_action.setShortcut(QKeySequence("Ctrl+S"))
        export_action.triggered.connect(self.export_wav)
        file_menu.addAction(export_action)

        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction(QAction("Reset All", self, triggered=self.reset_parameters))
        edit_menu.addAction(QAction("Randomize", self, triggered=self.randomize_parameters))

        # Global Shortcut
        space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        space_shortcut.activated.connect(self.toggle_playback)

    def _setup_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — Press Space to play/stop")

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # === LEFT PANEL (Controls) ===
        left_panel = QWidget()
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel_layout.setContentsMargins(0, 0, 0, 0)
        left_panel_layout.setSpacing(0)

        # -- Control Tabs --
        self.control_tabs = QTabWidget()
        self.control_tabs.setObjectName("ControlTabs")
        
        # Tab 1: Synth
        synth_tab = QWidget()
        synth_scroll = QScrollArea()
        synth_scroll.setWidgetResizable(True)
        synth_scroll.setFrameShape(QFrame.NoFrame)
        synth_content = QWidget()
        self.synth_layout = QVBoxLayout(synth_content)
        self.synth_layout.setSpacing(15)
        synth_scroll.setWidget(synth_content)
        QVBoxLayout(synth_tab).addWidget(synth_scroll)
        self.control_tabs.addTab(synth_tab, "Synth")

        # Tab 2: Effects
        fx_tab = QWidget()
        fx_scroll = QScrollArea()
        fx_scroll.setWidgetResizable(True)
        fx_scroll.setFrameShape(QFrame.NoFrame)
        fx_content = QWidget()
        self.fx_layout = QVBoxLayout(fx_content)
        self.fx_layout.setSpacing(15)
        fx_scroll.setWidget(fx_content)
        QVBoxLayout(fx_tab).addWidget(fx_scroll)
        self.control_tabs.addTab(fx_tab, "Effects")

        # Tab 3: Library
        lib_tab = QWidget()
        lib_scroll = QScrollArea()
        lib_scroll.setWidgetResizable(True)
        lib_scroll.setFrameShape(QFrame.NoFrame)
        lib_content = QWidget()
        self.lib_layout = QVBoxLayout(lib_content)
        self.lib_layout.setSpacing(15)
        lib_scroll.setWidget(lib_content)
        QVBoxLayout(lib_tab).addWidget(lib_scroll)
        self.control_tabs.addTab(lib_tab, "Library")

        left_panel_layout.addWidget(self.control_tabs)

        # -- Transport Footer --
        footer = QFrame()
        footer.setObjectName("TransportFooter")
        footer_layout = QVBoxLayout(footer)
        self._create_transport_controls(footer_layout)
        left_panel_layout.addWidget(footer)

        left_panel.setMaximumWidth(450)
        left_panel.setMinimumWidth(350)
        splitter.addWidget(left_panel)

        # === RIGHT PANEL (Visualization) ===
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        viz_layout.setContentsMargins(0, 0, 0, 0)

        self.tab_widget = QTabWidget()
        viz_layout.addWidget(self.tab_widget)

        # Waveform Tab
        self.waveform_viz = ImageWidget()
        self.tab_widget.addTab(self.waveform_viz, "Waveform")

        # Spectrum Tab
        self.spectrum_viz = ImageWidget()
        self.tab_widget.addTab(self.spectrum_viz, "Spectrum")

        # Spectrogram Tab
        self.spectrogram_viz = ImageWidget()
        self.tab_widget.addTab(self.spectrogram_viz, "Spectrogram")

        # Combined Tab
        combined_widget = QWidget()
        combined_layout = QVBoxLayout(combined_widget)
        combined_layout.setContentsMargins(0, 0, 0, 0)
        self.combined_wave_viz = ImageWidget()
        self.combined_wave_viz.setMaximumHeight(250)
        self.combined_spec_viz = ImageWidget()
        combined_layout.addWidget(self.combined_wave_viz)
        combined_layout.addWidget(self.combined_spec_viz)
        self.tab_widget.addTab(combined_widget, "Combined")

        self.info_label = QLabel("Initializing...")
        self.info_label.setObjectName("InfoLabel")
        self.info_label.setWordWrap(True)
        viz_layout.addWidget(self.info_label)

        splitter.addWidget(viz_widget)
        splitter.setSizes([400, 1100])

        # Populate Tabs
        self._create_waveform_section()
        self._create_frequency_section()
        self._create_envelope_section()
        self.synth_layout.addStretch()

        self._create_effects_basic_section()
        self._create_effects_advanced_section()
        self.fx_layout.addStretch()

        self._create_presets_section()
        self._create_preset_manager_section()
        self.lib_layout.addStretch()

    def _create_transport_controls(self, parent_layout):
        # Top row: Generate
        gen_btn = QPushButton("🔊  Generate Sound")
        gen_btn.setObjectName("GenBtn")
        gen_btn.clicked.connect(self.generate_sound)
        parent_layout.addWidget(gen_btn)

        # Middle row: Play/Stop
        btn_row = QHBoxLayout()
        play_btn = QPushButton("▶  Play")
        play_btn.setObjectName("PlayBtn")
        play_btn.clicked.connect(self.play_sound)
        
        stop_btn = QPushButton("⏹  Stop")
        stop_btn.setObjectName("StopBtn")
        stop_btn.clicked.connect(self.stop_sound)
        
        btn_row.addWidget(play_btn)
        btn_row.addWidget(stop_btn)
        parent_layout.addLayout(btn_row)

        # Bottom row: Export
        exp_btn = QPushButton("💾  Export WAV...")
        exp_btn.clicked.connect(self.export_wav)
        parent_layout.addWidget(exp_btn)

    # --- Widget Creation Helpers ---
    def _add_param(self, key, label, widget_type, min_val, max_val, default, step=1, decimals=2, parent_layout=None):
        layout = parent_layout if parent_layout else self.synth_layout
        row_layout = QHBoxLayout()
        lbl = QLabel(f"{label}:")
        lbl.setMinimumWidth(100)
        row_layout.addWidget(lbl)
        widget = None

        if widget_type == 'combo':
            widget = QComboBox()
            widget.addItems([w.value for w in WaveformType])
            widget.setCurrentText(str(default))
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row_layout.addWidget(widget)
            widget.currentTextChanged.connect(self._schedule_regen)

        elif widget_type == 'check':
            widget = QCheckBox(label)
            widget.setChecked(bool(default))
            lbl.hide()
            row_layout.insertWidget(0, widget)
            row_layout.addStretch()
            widget.stateChanged.connect(self._schedule_regen)

        elif widget_type == 'slider_float':
            actual_step = step if 0 < step < 1 else 0.01
            scale = int(round(1.0 / actual_step))
            slider = QSlider(Qt.Horizontal)
            slider.setRange(int(min_val * scale), int(max_val * scale))
            slider.setValue(int(default * scale))
            spin = QDoubleSpinBox()
            spin.setRange(min_val, max_val)
            spin.setValue(default)
            spin.setSingleStep(actual_step)
            spin.setDecimals(decimals)
            spin.setMaximumWidth(70)
            
            slider.valueChanged.connect(lambda v, s=scale, sp=spin: sp.setValue(v / s))
            spin.valueChanged.connect(lambda v, s=scale, sl=slider: sl.blockSignals(True) or sl.setValue(int(v * s)) or sl.blockSignals(False))
            
            slider.valueChanged.connect(self._schedule_regen)
            spin.valueChanged.connect(self._schedule_regen)
            
            row_layout.addWidget(slider)
            row_layout.addWidget(spin)
            widget = spin

        elif widget_type == 'spin_int':
            widget = QSpinBox()
            widget.setRange(int(min_val), int(max_val))
            widget.setValue(int(default))
            widget.setSingleStep(int(step))
            row_layout.addWidget(widget)
            widget.valueChanged.connect(self._schedule_regen)

        elif widget_type == 'spin_float':
            widget = QDoubleSpinBox()
            widget.setRange(min_val, max_val)
            widget.setValue(default)
            widget.setSingleStep(step)
            widget.setDecimals(decimals)
            row_layout.addWidget(widget)
            widget.valueChanged.connect(self._schedule_regen)

        self.params[key] = widget
        layout.addLayout(row_layout)
        return widget

    def _schedule_regen(self):
        self._regen_timer.start()

    # --- Sections ---
    def _create_waveform_section(self):
        group = QGroupBox("Oscillator")
        layout = QVBoxLayout(group)
        self._add_param('waveform_type', 'Type', 'combo', 0, 0, "sine", parent_layout=layout)
        self._add_param('amplitude', 'Amplitude', 'slider_float', 0.0, 1.0, 0.8, parent_layout=layout)
        self._add_param('harmonics', 'Harmonics', 'spin_int', 1, 16, 5, parent_layout=layout)
        self._add_param('pulse_width', 'Pulse Width', 'slider_float', 0.1, 0.9, 0.5, parent_layout=layout)
        self.synth_layout.addWidget(group)

    def _create_frequency_section(self):
        group = QGroupBox("Pitch & Time")
        layout = QVBoxLayout(group)
        self._add_param('frequency', 'Frequency (Hz)', 'spin_float', 20, 20000, 440.0, 10, 1, parent_layout=layout)
        self._add_param('duration', 'Duration (s)', 'spin_float', 0.01, 10.0, 0.5, 0.1, 2, parent_layout=layout)
        
        # Sweep layout
        sweep_group = QGroupBox("Frequency Sweep")
        sweep_layout = QVBoxLayout(sweep_group)
        self._add_param('sweep_enabled', 'Enabled', 'check', 0, 0, False, parent_layout=sweep_layout)
        self._add_param('end_frequency', 'End Freq (Hz)', 'spin_float', 20, 20000, 880.0, 10, 1, parent_layout=sweep_layout)
        sweep_type = QComboBox()
        sweep_type.addItems(["linear", "exponential"])
        sweep_type.currentTextChanged.connect(self._schedule_regen)
        row = QHBoxLayout()
        row.addWidget(QLabel("Sweep Type:"))
        row.addWidget(sweep_type)
        sweep_layout.addLayout(row)
        self.params['sweep_type'] = sweep_type
        
        layout.addWidget(sweep_group)
        self.synth_layout.addWidget(group)

    def _create_envelope_section(self):
        group = QGroupBox("ADSR Envelope")
        layout = QVBoxLayout(group)
        self._add_param('envelope_enabled', 'Enabled', 'check', 0, 0, False, parent_layout=layout)
        self._add_param('attack', 'Attack (s)', 'spin_float', 0.0, 2.0, 0.01, 0.01, 3, parent_layout=layout)
        self._add_param('decay', 'Decay (s)', 'spin_float', 0.0, 2.0, 0.1, 0.01, 3, parent_layout=layout)
        self._add_param('sustain', 'Sustain (s)', 'spin_float', 0.0, 2.0, 0.2, 0.01, 3, parent_layout=layout)
        self._add_param('release', 'Release (s)', 'spin_float', 0.0, 2.0, 0.1, 0.01, 3, parent_layout=layout)
        self._add_param('sustain_level', 'Sustain Level', 'slider_float', 0.0, 1.0, 0.7, parent_layout=layout)
        self.synth_layout.addWidget(group)

    def _create_effects_basic_section(self):
        group = QGroupBox("Standard Effects")
        layout = QVBoxLayout(group)
        
        # Reverb
        self._add_param('reverb_enabled', 'Reverb', 'check', 0, 0, False, parent_layout=layout)
        self._add_param('reverb_room', 'Room Size', 'slider_float', 0.1, 1.0, 0.5, parent_layout=layout)
        self._add_param('reverb_wet', 'Wet Level', 'slider_float', 0.0, 1.0, 0.3, parent_layout=layout)
        layout.addWidget(self._create_h_separator())

        # Delay
        self._add_param('delay_enabled', 'Delay/Echo', 'check', 0, 0, False, parent_layout=layout)
        self._add_param('delay_time', 'Time (s)', 'spin_float', 0.01, 2.0, 0.3, 0.01, 2, parent_layout=layout)
        self._add_param('delay_feedback', 'Feedback', 'slider_float', 0.0, 0.9, 0.4, parent_layout=layout)
        layout.addWidget(self._create_h_separator())

        # Distortion
        self._add_param('distortion_enabled', 'Distortion', 'check', 0, 0, False, parent_layout=layout)
        self._add_param('distortion_drive', 'Drive', 'slider_float', 0.0, 1.0, 0.5, parent_layout=layout)
        dist_type = QComboBox()
        dist_type.addItems(["soft", "hard", "fuzz"])
        dist_type.currentTextChanged.connect(self._schedule_regen)
        row = QHBoxLayout(); row.addWidget(QLabel("Type:")); row.addWidget(dist_type)
        layout.addLayout(row)
        self.params['distortion_type'] = dist_type
        
        self.fx_layout.addWidget(group)

    def _create_effects_advanced_section(self):
        group = QGroupBox("Filters & Modulation")
        layout = QVBoxLayout(group)

        # Filters
        self._add_param('lowpass_enabled', 'Lowpass', 'check', 0, 0, False, parent_layout=layout)
        self._add_param('lowpass_cutoff', 'Cutoff (Hz)', 'spin_float', 100, 20000, 1000, 100, 0, parent_layout=layout)
        layout.addWidget(self._create_h_separator())

        self._add_param('highpass_enabled', 'Highpass', 'check', 0, 0, False, parent_layout=layout)
        self._add_param('highpass_cutoff', 'Cutoff (Hz)', 'spin_float', 20, 5000, 100, 50, 0, parent_layout=layout)
        layout.addWidget(self._create_h_separator())

        self._add_param('bitcrush_enabled', 'Bitcrusher', 'check', 0, 0, False, parent_layout=layout)
        self._add_param('bitcrush_depth', 'Bit Depth', 'spin_int', 1, 16, 8, parent_layout=layout)
        layout.addWidget(self._create_h_separator())

        # Modulation
        self._add_param('chorus_enabled', 'Chorus', 'check', 0, 0, False, parent_layout=layout)
        self._add_param('chorus_rate', 'Rate', 'spin_float', 0.1, 10.0, 1.5, 0.1, 1, parent_layout=layout)
        
        self._add_param('phaser_enabled', 'Phaser', 'check', 0, 0, False, parent_layout=layout)
        self._add_param('phaser_rate', 'Rate', 'spin_float', 0.1, 5.0, 0.5, 0.1, 1, parent_layout=layout)
        
        self._add_param('compressor_enabled', 'Compressor', 'check', 0, 0, False, parent_layout=layout)
        self._add_param('compressor_threshold', 'Threshold (dB)', 'spin_float', -60, 0, -20.0, 1, 1, parent_layout=layout)
        
        self.fx_layout.addWidget(group)

    def _create_presets_section(self):
        group = QGroupBox("Quick Presets")
        layout = QGridLayout(group)
        presets = [
            ("Explosion", self.preset_explosion), ("Laser", self.preset_laser),
            ("Coin", self.preset_coin), ("Jump", self.preset_jump),
            ("Power-up", self.preset_powerup), ("Hit", self.preset_hit),
            ("Alarm", self.preset_alarm), ("Footstep", self.preset_footstep),
            ("Bell", self.preset_bell), ("Gong", self.preset_gong),
            ("Sci-Fi", self.preset_scifi)
        ]
        for i, (name, cmd) in enumerate(presets):
            btn = QPushButton(name)
            btn.clicked.connect(cmd)
            layout.addWidget(btn, i // 2, i % 2)
        self.lib_layout.addWidget(group)

    def _create_preset_manager_section(self):
        group = QGroupBox("Preset Library")
        layout = QVBoxLayout(group)
        
        self.preset_list = QListWidget()
        self.preset_list.itemDoubleClicked.connect(self.load_selected_preset)
        layout.addWidget(self.preset_list)
        
        btn_row = QHBoxLayout()
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self.load_selected_preset)
        save_btn = QPushButton("Save...")
        save_btn.clicked.connect(self.save_preset_dialog)
        btn_row.addWidget(load_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)
        
        self.lib_layout.addWidget(group)

    def _create_h_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    # --- Parameter Handling ---
    def _get_val(self, key):
        w = self.params.get(key)
        if not w: return None
        if isinstance(w, QComboBox): return w.currentText()
        elif isinstance(w, QCheckBox): return w.isChecked()
        elif isinstance(w, (QDoubleSpinBox, QSpinBox)): return w.value()
        return None

    def _set_val(self, key, val):
        w = self.params.get(key)
        if not w: return
        w.blockSignals(True)
        try:
            if isinstance(w, QComboBox): w.setCurrentText(str(val))
            elif isinstance(w, QCheckBox): w.setChecked(bool(val))
            elif isinstance(w, QDoubleSpinBox): w.setValue(float(val))
            elif isinstance(w, QSpinBox): w.setValue(int(val))
        finally:
            w.blockSignals(False)

    def get_all_parameters(self): return {k: self._get_val(k) for k in self.params}
    def set_all_parameters(self, params):
        for k, v in params.items():
            if k in self.params: self._set_val(k, v)

    def _reset_effects(self):
        for k, w in self.params.items():
            if k.endswith('_enabled') and isinstance(w, QCheckBox): w.setChecked(False)

    # --- Core Logic ---
    def generate_sound(self):
        self._regen_timer.stop()
        self.status_bar.showMessage("Generating...")
        QApplication.processEvents()

        try:
            wt = WaveformType(self._get_val('waveform_type'))
            freq, dur, amp = self._get_val('frequency'), self._get_val('duration'), self._get_val('amplitude')
            
            if self._get_val('sweep_enabled'):
                audio = self.generator.apply_frequency_sweep(wt, freq, self._get_val('end_frequency'), dur, amp, self._get_val('sweep_type'))
            else:
                audio = self.generator.generate_waveform(wt, freq, dur, amp, pulse_width=self._get_val('pulse_width'), harmonics=self._get_val('harmonics'))
            
            if self._get_val('envelope_enabled'):
                audio = self.generator.apply_adsr_envelope(audio, self._get_val('attack'), self._get_val('decay'), self._get_val('sustain'), self._get_val('release'), self._get_val('sustain_level'))
            
            # Effects Chain
            if self._get_val('reverb_enabled'): audio = self.generator.apply_reverb(audio, self._get_val('reverb_room'), 0.5, self._get_val('reverb_wet'))
            if self._get_val('delay_enabled'): audio = self.generator.apply_delay(audio, self._get_val('delay_time'), self._get_val('delay_feedback'), 0.5)
            if self._get_val('distortion_enabled'): audio = self.generator.apply_distortion(audio, self._get_val('distortion_drive'), self._get_val('distortion_type'))
            if self._get_val('lowpass_enabled'): audio = self.generator.apply_lowpass_filter(audio, self._get_val('lowpass_cutoff'))
            if self._get_val('highpass_enabled'): audio = self.generator.apply_highpass_filter(audio, self._get_val('highpass_cutoff'))
            if self._get_val('bitcrush_enabled'): audio = self.generator.apply_bitcrusher(audio, self._get_val('bitcrush_depth'))
            if self._get_val('chorus_enabled'): audio = self.generator.apply_chorus(audio, self._get_val('chorus_rate'), self._get_val('chorus_depth'), self._get_val('chorus_voices'), 0.5)
            if self._get_val('phaser_enabled'): audio = self.generator.apply_phaser(audio, self._get_val('phaser_rate'), self._get_val('phaser_depth'), self._get_val('phaser_stages'), 0.5, 0.7)
            if self._get_val('compressor_enabled'): audio = self.generator.apply_compressor(audio, self._get_val('compressor_threshold'), self._get_val('compressor_ratio'), 0.01, 0.1, self._get_val('compressor_makeup'))
            
            self.current_audio = to_cpu(audio)
            self.rt_position = 0
            self.update_visualizations()
            self.update_info()
            self.status_bar.showMessage("Ready")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to generate sound:\n{str(e)}")

    def update_visualizations(self):
        if self.current_audio is None: return
        w, h = self.waveform_viz.width(), self.waveform_viz.height()
        if w < 10 or h < 10: return

        audio_gpu = to_gpu(self.current_audio)
        
        r = WaveformRenderer(w, h)
        wf = self.visualizer.get_waveform_data(audio_gpu, r.width)
        self.waveform_viz.set_image(r.render_waveform(wf))

        r_spec = WaveformRenderer(self.spectrum_viz.width(), max(1, self.spectrum_viz.height()))
        freqs, mags = self.visualizer.get_spectrum_data(audio_gpu)
        self.spectrum_viz.set_image(r_spec.render_spectrum(freqs, mags))

        r_sg = WaveformRenderer(self.spectrogram_viz.width(), max(1, self.spectrogram_viz.height()))
        sg_freqs, sg_times, spec = self.visualizer.get_spectrogram_data(audio_gpu)
        self.spectrogram_viz.set_image(r_sg.render_spectrogram(sg_freqs, sg_times, spec))

        r_cw = WaveformRenderer(self.combined_wave_viz.width(), max(1, self.combined_wave_viz.height()))
        wf_c = self.visualizer.get_waveform_data(audio_gpu, r_cw.width)
        self.combined_wave_viz.set_image(r_cw.render_waveform(wf_c))

        r_cs = WaveformRenderer(self.combined_spec_viz.width(), max(1, self.combined_spec_viz.height()))
        self.combined_spec_viz.set_image(r_cs.render_spectrum(freqs, mags))

    def update_info(self):
        if self.current_audio is None: return
        dur = len(self.current_audio) / self.sample_rate
        peak = np.max(np.abs(self.current_audio))
        rms = np.sqrt(np.mean(self.current_audio ** 2))
        self.info_label.setText(f"Duration: {dur:.3f}s | Samples: {len(self.current_audio):,} | Peak: {peak:.3f} | RMS: {rms:.3f}")

    def play_sound(self):
        if self.current_audio is None or self.is_playing: return
        self.is_playing = True
        self.rt_position = 0
        self.status_bar.showMessage("Playing...")
        self.playback_worker = PlaybackWorker(self.current_audio, self.sample_rate)
        self.playback_worker.finished.connect(self._on_playback_finished)
        self.playback_worker.error.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self.playback_worker.start()
        if self.realtime_enabled: self.rt_timer.start()

    def _on_playback_finished(self):
        self.is_playing = False
        self.rt_timer.stop()
        self.rt_position = 0
        self.status_bar.showMessage("Ready")
        self.update_visualizations()

    def stop_sound(self):
        if self.playback_worker and self.is_playing:
            self.playback_worker.stop_playback()
        self.is_playing = False
        self.rt_timer.stop()
        self.rt_position = 0
        self.status_bar.showMessage("Stopped")
        self.update_visualizations()

    def toggle_playback(self):
        if self.is_playing: self.stop_sound()
        else: self.play_sound()

    def _update_realtime_viz(self):
        if not self.is_playing or self.current_audio is None: self.rt_timer.stop(); return
        window_size = int(0.05 * self.sample_rate)
        start = self.rt_position
        end = min(start + window_size, len(self.current_audio))
        if start >= len(self.current_audio): self.rt_timer.stop(); return
        
        audio_window = to_gpu(self.current_audio[start:end])
        r1 = WaveformRenderer(self.waveform_viz.width(), max(1, self.waveform_viz.height()))
        self.waveform_viz.set_image(r1.render_waveform(audio_window))
        r2 = WaveformRenderer(self.spectrum_viz.width(), max(1, self.spectrum_viz.height()))
        freqs, mags = self.visualizer.get_spectrum_data(audio_window)
        self.spectrum_viz.set_image(r2.render_spectrum(freqs, mags))
        self.rt_position += window_size

    def export_wav(self):
        if self.current_audio is None: return
        import wave
        filepath, _ = QFileDialog.getSaveFileName(self, "Export Sound", "", "WAV files (*.wav)")
        if not filepath: return
        try:
            audio_int = (np.clip(self.current_audio, -1, 1) * 32767).astype(np.int16)
            with wave.open(filepath, 'w') as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(self.sample_rate)
                f.writeframes(audio_int.tobytes())
            self.status_bar.showMessage(f"Exported to {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export:\n{str(e)}")

    # --- Preset Logic ---
    def save_preset_dialog(self):
        dialog = QDialog(self); dialog.setWindowTitle("Save Preset")
        layout = QFormLayout(dialog)
        name_edit = QLineEdit("My Preset"); layout.addRow("Name:", name_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.Accepted and name_edit.text().strip():
            self.preset_manager.save_preset(name_edit.text().strip(), self.get_all_parameters())
            self._refresh_preset_list()

    def load_selected_preset(self):
        row = self.preset_list.currentRow()
        if row < 0: return
        presets = self.preset_manager.list_presets()
        if row >= len(presets): return
        params = self.preset_manager.load_preset(presets[row][1])
        self.set_all_parameters(params)
        self.generate_sound()

    def _refresh_preset_list(self):
        self.preset_list.clear()
        for name, _ in self.preset_manager.list_presets(): self.preset_list.addItem(name)

    def randomize_parameters(self):
        self._set_val('waveform_type', random.choice([w.value for w in WaveformType]))
        self._set_val('frequency', random.uniform(100, 2000))
        for k in ['reverb_enabled', 'delay_enabled', 'distortion_enabled']: self._set_val(k, random.random() < 0.3)
        self.generate_sound()

    def reset_parameters(self):
        defaults = {
            'waveform_type': 'sine', 'amplitude': 0.8, 'harmonics': 5, 'pulse_width': 0.5,
            'frequency': 440.0, 'duration': 0.5, 'sweep_enabled': False, 'end_frequency': 880.0, 'sweep_type': 'linear',
            'envelope_enabled': False, 'attack': 0.01, 'decay': 0.1, 'sustain': 0.2, 'release': 0.1, 'sustain_level': 0.7,
            'reverb_enabled': False, 'reverb_room': 0.5, 'reverb_wet': 0.3, 'delay_enabled': False, 'delay_time': 0.3, 'delay_feedback': 0.4,
            'distortion_enabled': False, 'distortion_drive': 0.5, 'distortion_type': 'soft', 'lowpass_enabled': False, 'lowpass_cutoff': 1000.0,
            'highpass_enabled': False, 'highpass_cutoff': 100.0, 'bitcrush_enabled': False, 'bitcrush_depth': 8,
            'chorus_enabled': False, 'chorus_rate': 1.5, 'chorus_depth': 0.5, 'chorus_voices': 3,
            'phaser_enabled': False, 'phaser_rate': 0.5, 'phaser_depth': 0.7, 'phaser_stages': 4,
            'compressor_enabled': False, 'compressor_threshold': -20.0, 'compressor_ratio': 4.0, 'compressor_makeup': 0.0
        }
        self.set_all_parameters(defaults)
        self.generate_sound()

    # --- Quick Presets ---
    def preset_explosion(self):
        self._reset_effects(); self.set_all_parameters({'waveform_type':'noise_brown','frequency':440,'duration':1.0,'amplitude':0.8,'sweep_enabled':True,'end_frequency':30,'sweep_type':'exponential','envelope_enabled':True,'attack':0.01,'decay':0.1,'sustain':0.3,'release':0.5,'sustain_level':0.5,'lowpass_enabled':True,'lowpass_cutoff':800,'reverb_enabled':True,'reverb_room':0.7,'reverb_wet':0.4}); self.generate_sound()
    def preset_laser(self):
        self._reset_effects(); self.set_all_parameters({'waveform_type':'sawtooth','frequency':1400,'duration':0.3,'amplitude':0.6,'sweep_enabled':True,'end_frequency':200,'sweep_type':'exponential','envelope_enabled':True,'attack':0.01,'decay':0.05,'sustain':0.15,'release':0.09,'sustain_level':0.3,'distortion_enabled':True,'distortion_drive':0.2,'distortion_type':'soft'}); self.generate_sound()
    def preset_coin(self):
        self._reset_effects(); self.set_all_parameters({'waveform_type':'sine','frequency':988,'duration':0.15,'amplitude':0.5,'sweep_enabled':False,'envelope_enabled':True,'attack':0.005,'decay':0.03,'sustain':0.05,'release':0.065,'sustain_level':0.3}); self.generate_sound()
    def preset_jump(self):
        self._reset_effects(); self.set_all_parameters({'waveform_type':'square','frequency':300,'duration':0.25,'amplitude':0.4,'sweep_enabled':True,'end_frequency':900,'sweep_type':'linear','envelope_enabled':True,'attack':0.02,'decay':0.1,'sustain':0.05,'release':0.08,'sustain_level':0.3,'lowpass_enabled':True,'lowpass_cutoff':2000}); self.generate_sound()
    def preset_powerup(self):
        self._reset_effects(); self.set_all_parameters({'waveform_type':'sine','frequency':200,'duration':0.8,'amplitude':0.4,'sweep_enabled':True,'end_frequency':800,'sweep_type':'exponential','envelope_enabled':True,'attack':0.05,'decay':0.15,'sustain':0.4,'release':0.2,'sustain_level':0.6,'chorus_enabled':True,'chorus_rate':2.0,'chorus_depth':0.4,'chorus_voices':3}); self.generate_sound()
    def preset_hit(self):
        self._reset_effects(); self.set_all_parameters({'waveform_type':'noise_white','frequency':80,'duration':0.2,'amplitude':0.7,'sweep_enabled':False,'envelope_enabled':True,'attack':0.005,'decay':0.05,'sustain':0.05,'release':0.09,'sustain_level':0.2,'lowpass_enabled':True,'lowpass_cutoff':1500}); self.generate_sound()
    def preset_alarm(self):
        self._reset_effects(); self.set_all_parameters({'waveform_type':'square','frequency':800,'duration':1.0,'amplitude':0.5,'sweep_enabled':False,'envelope_enabled':False,'phaser_enabled':True,'phaser_rate':4.0,'phaser_depth':0.8,'phaser_stages':4}); self.generate_sound()
    def preset_footstep(self):
        self._reset_effects(); self.set_all_parameters({'waveform_type':'noise_brown','frequency':80,'duration':0.15,'amplitude':0.5,'sweep_enabled':False,'envelope_enabled':True,'attack':0.005,'decay':0.03,'sustain':0.05,'release':0.065,'sustain_level':0.2,'lowpass_enabled':True,'lowpass_cutoff':600}); self.generate_sound()
    def preset_bell(self):
        self._reset_effects(); self.set_all_parameters({'waveform_type':'fm_bell','frequency':440,'duration':1.5,'amplitude':0.7,'sweep_enabled':False,'envelope_enabled':False,'reverb_enabled':True,'reverb_room':0.6,'reverb_wet':0.35}); self.generate_sound()
    def preset_gong(self):
        self._reset_effects(); self.set_all_parameters({'waveform_type':'fm_metal','frequency':150,'duration':3.0,'amplitude':0.8,'sweep_enabled':False,'envelope_enabled':False,'reverb_enabled':True,'reverb_room':0.8,'reverb_wet':0.5}); self.generate_sound()
    def preset_scifi(self):
        self._reset_effects(); self.set_all_parameters({'waveform_type':'fm_bell','frequency':800,'duration':0.2,'amplitude':0.6,'sweep_enabled':False,'envelope_enabled':True,'attack':0.01,'decay':0.05,'sustain':0.08,'release':0.06,'sustain_level':0.5,'phaser_enabled':True,'phaser_rate':3.0,'phaser_depth':0.6,'phaser_stages':3}); self.generate_sound()

    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QSplitter::handle { background-color: #555; width: 2px; }
            
            /* Tabs */
            QTabWidget::pane { border: 1px solid #555; background-color: #333; border-radius: 4px; }
            QTabBar::tab { background-color: #444; color: #aaa; padding: 8px 20px; border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px; }
            QTabBar::tab:selected { background-color: #333; color: #fff; border-bottom: 2px solid #1e90ff; }
            
            QGroupBox { 
                font-weight: bold; border: 1px solid #555; border-radius: 4px; 
                margin-top: 1.5em; padding-top: 10px; color: #ddd; background-color: #353535;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #1e90ff; }
            
            QLabel { color: #ccc; }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { 
                background-color: #404040; color: #eee; border: 1px solid #555; border-radius: 3px; padding: 4px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox::down-arrow { image: none; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 5px solid #aaa; }
            
            QSlider::groove:horizontal { border: 1px solid #555; height: 4px; background: #404040; border-radius: 2px; }
            QSlider::handle:horizontal { background: #1e90ff; width: 14px; margin: -5px 0; border-radius: 7px; }
            
            QPushButton { 
                background-color: #404040; color: #eee; border: 1px solid #555; border-radius: 4px; 
                padding: 6px 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #505050; border: 1px solid #777; }
            QPushButton:pressed { background-color: #333; }
            
            /* Transport Footer */
            #TransportFooter { background-color: #2b2b2b; border-top: 1px solid #555; padding: 10px; }
            QPushButton#GenBtn { background-color: #2d5a27; border: none; padding: 10px; font-size: 14px; }
            QPushButton#GenBtn:hover { background-color: #3a7a33; }
            QPushButton#PlayBtn { background-color: #1e3a5f; border: none; padding: 10px; }
            QPushButton#PlayBtn:hover { background-color: #2a5080; }
            QPushButton#StopBtn { background-color: #5a1e1e; border: none; padding: 10px; }
            QPushButton#StopBtn:hover { background-color: #7a2a2a; }
            
            QListWidget { background-color: #404040; color: #eee; border: 1px solid #555; border-radius: 4px; }
            QListWidget::item:selected { background-color: #1e90ff; }
            
            QScrollArea { border: none; background-color: transparent; }
            QWidget { background-color: transparent; } /* General cleanup */
            
            #InfoLabel { color: #1e90ff; font-family: Consolas, monospace; font-size: 13px; padding: 5px; background-color: #222; border-radius: 3px; }
        """)