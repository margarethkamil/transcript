import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue
import threading
import time
import argparse
import os
import traceback
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QLabel, QTextEdit, QScrollArea, QHBoxLayout, 
                            QSlider, QComboBox, QCheckBox, QProgressBar, QGroupBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor, QTextCursor
import re
import difflib
from collections import Counter

class TranscriptionSignals(QObject):
    """Custom signals for thread communication"""
    text_signal = pyqtSignal(str)
    audio_level_signal = pyqtSignal(float)  # Signal for audio input level
    status_signal = pyqtSignal(str)  # Signal for transcription status

class TranscriberWindow(QMainWindow):
    """PyQt-based GUI for displaying transcriptions"""
    
    def __init__(self, model_type="base"):
        super().__init__()
        self.setWindowTitle("Real-time Transcriber")
        self.setGeometry(100, 100, 900, 700)  # Increased size to accommodate new controls
        
        # Store model type that was passed from command line
        self.model_type = model_type
        
        self.setup_ui()
        
        # Keep window on top
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        
        # Init timer for progress bar updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_indicators)
        self.update_timer.start(100)  # Update every 100ms
        
        # Default settings - these will be linked to UI controls
        self.chunk_size = 2.5
        self.noise_reduction = True
        
        # Store audio level for display
        self.current_audio_level = 0.0
        self.transcription_status = "Ready"
        
    def setup_ui(self):
        """Set up the user interface"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header label with gradient background
        header = QLabel("REAL-TIME SPEECH TRANSCRIPTION")
        header.setFont(QFont("Segoe UI", 18, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                          stop:0 #2c3e50, stop:1 #3498db);
                color: white;
                padding: 15px;
                border-radius: 8px;
            }
        """)
        main_layout.addWidget(header)
        
        # === NEW: Progress indicators section ===
        indicators_group = QGroupBox("Status Indicators")
        indicators_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 1ex;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        indicators_layout = QVBoxLayout(indicators_group)
        
        # Add audio level indicator
        level_layout = QHBoxLayout()
        level_label = QLabel("Microphone Level:")
        level_label.setFixedWidth(120)
        self.audio_level_bar = QProgressBar()
        self.audio_level_bar.setRange(0, 100)
        self.audio_level_bar.setValue(0)
        self.audio_level_bar.setTextVisible(True)
        self.audio_level_bar.setFormat("%v%")
        self.audio_level_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bbb;
                border-radius: 4px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                stop:0 #3498db, stop:1 #2ecc71);
                border-radius: 3px;
            }
        """)
        level_layout.addWidget(level_label)
        level_layout.addWidget(self.audio_level_bar)
        indicators_layout.addLayout(level_layout)
        
        # Add transcription status indicator
        status_layout = QHBoxLayout()
        status_label = QLabel("Status:")
        status_label.setFixedWidth(120)
        self.status_indicator = QLabel("Ready")
        self.status_indicator.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
            }
        """)
        status_layout.addWidget(status_label)
        status_layout.addWidget(self.status_indicator)
        indicators_layout.addLayout(status_layout)
        
        main_layout.addWidget(indicators_group)
        
        # === NEW: Customizable settings section ===
        settings_group = QGroupBox("Transcription Settings")
        settings_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 1ex;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        settings_layout = QVBoxLayout(settings_group)
        
        # Chunk size slider
        chunk_layout = QHBoxLayout()
        chunk_label = QLabel("Chunk Size (sec):")
        chunk_label.setFixedWidth(120)
        self.chunk_slider = QSlider(Qt.Horizontal)
        self.chunk_slider.setRange(10, 50)  # 1.0 to 5.0 seconds (x10 for precision)
        self.chunk_slider.setValue(25)      # Default 2.5 seconds
        self.chunk_slider.setTickPosition(QSlider.TicksBelow)
        self.chunk_slider.setTickInterval(5)
        self.chunk_value_label = QLabel("2.5")
        self.chunk_value_label.setFixedWidth(30)
        self.chunk_slider.valueChanged.connect(self.update_chunk_size)
        chunk_layout.addWidget(chunk_label)
        chunk_layout.addWidget(self.chunk_slider)
        chunk_layout.addWidget(self.chunk_value_label)
        settings_layout.addLayout(chunk_layout)
        
        # Noise reduction checkbox
        noise_layout = QHBoxLayout()
        noise_label = QLabel("Noise Reduction:")
        noise_label.setFixedWidth(120)
        self.noise_checkbox = QCheckBox()
        self.noise_checkbox.setChecked(True)
        self.noise_checkbox.stateChanged.connect(self.update_noise_reduction)
        noise_layout.addWidget(noise_label)
        noise_layout.addWidget(self.noise_checkbox)
        noise_layout.addStretch()
        settings_layout.addLayout(noise_layout)
        
        # Model type dropdown
        model_layout = QHBoxLayout()
        model_label = QLabel("Model Type:")
        model_label.setFixedWidth(120)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setCurrentText(self.model_type)  # Use the model type passed from command line
        self.model_combo.currentTextChanged.connect(self.update_model_type)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        settings_layout.addLayout(model_layout)
        
        main_layout.addWidget(settings_group)
        
        # Text display for transcriptions
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setFont(QFont("Segoe UI", 12))
        self.text_display.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                color: #212529;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        main_layout.addWidget(self.text_display)
        
        # Add initial text as a single HTML block
        initial_text = (
            "<div style='text-align: center;'>"
            "<h2 style='color: #2c3e50;'>Real-time Speech Transcription</h2>"
            "<p style='color: #666; font-size: 14px;'>Speak into your microphone and your words will appear here.</p>"
            "<hr style='border: 1px solid #dee2e6;'>"
            "</div>"
        )
        self.text_display.setHtml(initial_text)
        
    def update_chunk_size(self):
        """Update chunk size from slider"""
        value = self.chunk_slider.value() / 10.0  # Convert from int to float (1-5 seconds)
        self.chunk_value_label.setText(f"{value:.1f}")
        self.chunk_size = value
        
    def update_noise_reduction(self, state):
        """Update noise reduction setting"""
        self.noise_reduction = (state == Qt.Checked)
        
    def update_model_type(self, model_name):
        """Update model type setting"""
        self.model_type = model_name
        
    def update_indicators(self):
        """Update the UI indicators with current values"""
        # Update audio level indicator
        self.audio_level_bar.setValue(int(self.current_audio_level * 100))
        
        # Update status with different colors based on status
        if self.transcription_status == "Transcribing":
            self.status_indicator.setText("Transcribing")
            self.status_indicator.setStyleSheet("background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px; padding: 5px;")
        elif self.transcription_status == "Listening":
            self.status_indicator.setText("Listening")
            self.status_indicator.setStyleSheet("background-color: #fff3cd; border: 1px solid #ffeeba; border-radius: 4px; padding: 5px;")
        elif self.transcription_status == "Error":
            self.status_indicator.setText("Error")
            self.status_indicator.setStyleSheet("background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 4px; padding: 5px;")
        else:
            self.status_indicator.setText("Ready")
            self.status_indicator.setStyleSheet("background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 4px; padding: 5px;")
        
    def add_transcription(self, text):
        """Add transcription text to the display"""
        if text:
            # Add timestamp
            timestamp = time.strftime("%H:%M:%S")
            formatted = (
                "<div style='margin: 8px 0;'>"
                f"<span style='color: #3498db; font-weight: bold;'>[{timestamp}]</span> "
                f"<span style='color: #2c3e50;'>{text}</span>"
                "</div>"
            )
            
            # Add to display
            self.text_display.append(formatted)
            
            # Scroll to bottom
            cursor = self.text_display.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.text_display.setTextCursor(cursor)
            
            # Debug print
            print(f"Added to GUI: [{timestamp}] {text}")
    
    def set_audio_level(self, level):
        """Set the current audio input level (0.0-1.0)"""
        self.current_audio_level = min(max(level, 0.0), 1.0)  # Clamp between 0 and 1
        
    def set_status(self, status):
        """Set the current transcription status"""
        self.transcription_status = status

class RealtimeTranscriber:
    def __init__(self, model_size="base", device="cpu", compute_type="int8", use_gui=False):
        # Basic configuration
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.use_gui = use_gui
        self.model = None
        
        # Audio settings
        self.sample_rate = 16000  # Whisper expects 16kHz audio
        self.channels = 1
        self.dtype = np.float32
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.transcription_buffer = []
        self.audio_devices = []
        self.input_device = None
        
        # Context window for connected transcriptions
        self.context_window = []
        self.context_window_size = 8  # Increased from 5 to 8 for better context
        self.last_output_time = time.time()
        self.output_delay = 3.0  # Increased from 2.0 to 3.0 seconds
        self.current_context = ""  # Current buffered text
        self.last_segment_time = 0
        self.segment_timeout = 3.0  # 3 seconds
        
        # Audio overlap settings
        self.overlap_ratio = 0.15  # Reduced from 0.25 to 0.15 (15% overlap between audio chunks)
        self.previous_audio = None  # Store previous audio for overlap
        
        # User settings (can be modified through UI)
        self.chunk_duration = 2.5  # seconds
        self.noise_reduction_enabled = True
        
        # GUI reference and signal communication
        self.app = None
        self.window = None
        self.signals = TranscriptionSignals()
        
        print("Loading Whisper model...")
        try:
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            print("Model loaded successfully!")
            self.init_audio_devices()
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def init_audio_devices(self):
        """Initialize audio devices"""
        try:
            self.audio_devices = sd.query_devices()
            default_device = sd.default.device[0]
            self.input_device = default_device
            
            device_info_str = "Available microphones:\n"
            for i, dev in enumerate(self.audio_devices):
                if dev.get('max_input_channels', 0) > 0:
                    if i == default_device:
                        device_info_str += f"* {i}: {dev['name']} (default)\n"
                    else:
                        device_info_str += f"  {i}: {dev['name']}\n"
            print(device_info_str)
            
        except Exception as e:
            print(f"Error querying audio devices: {e}")
            traceback.print_exc()
            self.audio_devices = []
    
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio capture"""
        if status:
            print(f"Status: {status}")
        
        # Add audio data to queue
        self.audio_queue.put(indata.copy())
        
        # Calculate audio level for UI indicators
        if self.use_gui and self.signals:
            # Calculate RMS of the audio data
            rms = np.sqrt(np.mean(indata**2))
            # Scale and apply some smoothing for better visualization
            normalized_level = min(1.0, rms * 10)  # Scale up by 10x for better visibility
            self.signals.audio_level_signal.emit(normalized_level)
    
    def should_flush_context(self):
        """Determine if we should output the current context - improved version"""
        current_time = time.time()
        
        # Flush if we have content and exceeded the delay time since last output
        if self.current_context and (current_time - self.last_output_time) > self.output_delay:
            return True
            
        # Flush if we have content and exceeded the silence timeout (reduced from 3.0 to 2.0)
        if self.current_context and (current_time - self.last_segment_time) > 2.0:
            return True
            
        # Flush shorter segments after a brief silence (reduced from 1.5 to 1.0 seconds)
        if self.current_context and (current_time - self.last_segment_time) > 1.0:
            return True
            
        # Flush if we have a complete sentence with proper ending and reduced word count threshold
        if (self.current_context and 
            re.search(r'[.!?]$', self.current_context) and 
            len(self.current_context.split()) > 6):  # Reduced from 10 to 6
            return True
            
        # NEW: Flush if context exceeds a certain length regardless of other conditions
        if self.current_context and len(self.current_context.split()) > 20:
            return True
            
        return False
    
    def remove_duplicates(self, text):
        """Enhanced duplicate removal with better fuzzy matching"""
        if not text:
            return text
        
        # First handle exact duplicates of phrases
        for phrase_len in range(5, 1, -1):
            pattern = r'\b(\w+(?:\s+\w+){' + str(phrase_len-1) + r'})\s+\1\b'
            text = re.sub(pattern, r'\1', text)
        
        # Handle single word duplicates
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)
        
        # Process for fuzzy matching
        words = text.split()
        i = 0
        
        # Better fuzzy matching with adaptive comparison
        while i < len(words) - 2:  # Need at least 3 words for a meaningful phrase
            if i + 3 > len(words):
                break
                
            # Get current phrase (2-3 words)
            phrase_len = min(3, len(words) - i)
            current_phrase = ' '.join(words[i:i+phrase_len]).lower()
            
            # Look ahead for similar phrases
            max_look_ahead = min(15, len(words) - i - phrase_len)  # Limit look-ahead distance
            
            for j in range(i + phrase_len, i + phrase_len + max_look_ahead):
                if j + phrase_len > len(words):
                    break
                    
                compare_phrase = ' '.join(words[j:j+phrase_len]).lower()
                
                # Use sequence matcher for fuzzy comparison
                similarity = difflib.SequenceMatcher(None, current_phrase, compare_phrase).ratio()
                
                # If high similarity, remove the later occurrence
                if similarity > 0.75:  # Threshold for "similar enough"
                    # Remove the duplicate segment
                    words = words[:j] + words[j+phrase_len:]
                    continue
            
            i += 1
        
        # Reassemble and clean up
        text = ' '.join(words)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def post_process_transcription(self, text):
        """Apply generalized post-processing to improve transcription quality"""
        if not text:
            return text
            
        # First apply duplicate removal (includes fuzzy matching)
        text = self.remove_duplicates(text)
        
        # Apply generic orthographic corrections
        text = self._apply_orthographic_corrections(text)
        
        # Fix speech pattern artifacts
        text = self._fix_speech_patterns(text)
        
        # Improve sentence structure
        text = self._improve_sentence_structure(text)
        
        # Ensure proper capitalization
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
            
        # Ensure proper sentence ending
        if text and not re.search(r'[.!?]$', text):
            if re.search(r'\b(who|what|where|when|why|how|which)\b', text.lower()):
                text += '?'
            else:
                text += '.'
                
        return text
    
    def _apply_orthographic_corrections(self, text):
        """Apply general orthographic corrections to common speech-to-text errors"""
        # Common orthographic patterns in speech-to-text systems
        corrections = [
            # Fix spacing around punctuation
            (r'\s+([.,;:!?])', r'\1'),
            
            # Fix repeated punctuation
            (r'([.,;:!?]){2,}', r'\1'),
            
            # Fix capitalization after sentence endings
            (r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper()),
            
            # Ensure "I" is always capitalized
            (r'\b(i)(\s+|\'|$)', lambda m: 'I' + m.group(2)),
            
            # Fix common article errors
            (r'\b(a)\s+([aeiou][a-z]+)', r'an \2'),
            (r'\b(an)\s+([bcdfghjklmnpqrstvwxyz][a-z]+)', r'a \2'),
            
            # Fix double articles
            (r'\b(a|an|the)\s+(a|an|the)\b', r'\1'),
            
            # Common contractions
            (r'\b(cant)\b', r"can't"),
            (r'\b(dont)\b', r"don't"),
            (r'\b(wont)\b', r"won't"),
            (r'\b(im)\b', r"I'm"),
            (r'\b(youre)\b', r"you're"),
            (r'\b(theyre)\b', r"they're"),
        ]
        
        # Apply all corrections
        for pattern, replacement in corrections:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            
        return text
    
    def _fix_speech_patterns(self, text):
        """Clean up common speech patterns and hesitations"""
        # Remove filler words and hesitations
        fillers = [
            r'\b(um+|uh+|er+|ah+)\b',
            r'\b(like|you know|i mean)\b(?! (to|that|the|a|an|how|what))',
            r'\b(so+)\b(?! (that|much|many|far|good|bad|well|it|i|we|they))',
            r'\b(basically|literally|actually)\b(?! (is|are|was|were|have|had|will|would|could|should))'
        ]
        
        for filler in fillers:
            text = re.sub(filler, '', text, flags=re.IGNORECASE)
            
        # Fix repeated words (beyond what remove_duplicates catches)
        text = re.sub(r'\b(\w+)(\s+\1){1,}\b', r'\1', text, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _improve_sentence_structure(self, text):
        """Improve sentence structure with generic rules"""
        # Fix sentence fragments
        if len(text.split()) <= 3 and not re.search(r'[.!?]$', text):
            # Too short to be a complete sentence, likely a fragment
            return text  # Keep as is, will be merged with next segment
            
        # Improve punctuation for lists
        text = re.sub(r'(\w+)( \w+)? and (\w+)', r'\1\2, and \3', text)
        
        # Balance parentheses, quotes, etc.
        if text.count('(') > text.count(')'):
            text += ')'
        if text.count('"') % 2 != 0:
            text += '"'
            
        return text
    
    def add_to_context(self, text):
        """Simplified version that adds text to context with minimal processing"""
        if not text.strip():
            return
        
        # Update timing information
        self.last_segment_time = time.time()
        
        # Basic cleanup - remove punctuation at beginning and trailing/leading spaces
        text = text.strip()
        text = re.sub(r'^[,.;:!?]\s*', '', text)
        
        # Only do minimal duplication check with current context
        if self.current_context:
            # Check if this segment is an exact duplicate or subset of current context
            if text.lower() in self.current_context.lower():
                return
            
            # Add space if needed between existing context and new text
            if not self.current_context.endswith(' '):
                self.current_context += ' '
        
        # Add text to context
        self.current_context += text
        
        # Add to context window history (keep for context length management)
        self.context_window.append(text)
        if len(self.context_window) > self.context_window_size:
            self.context_window.pop(0)
        
        # Check if we should output the context
        if self.should_flush_context():
            self.flush_context()
    
    def flush_context(self):
        """Output the current context and reset with improved context management"""
        if not self.current_context:
            return
            
        # First apply basic processing to current context
        processed_text = self.post_process_transcription(self.current_context)
        
        # Context continuity check
        if self.transcription_buffer:
            previous_text = self.transcription_buffer[-1]
            
            # Check for potential sentence continuation
            if self._appears_to_continue(previous_text, processed_text):
                # Merge with previous segment
                merged_text = self._merge_segments(previous_text, processed_text)
                self.transcription_buffer[-1] = merged_text
                
                # Update GUI/output with merged text
                if self.use_gui and self.signals:
                    self.signals.text_signal.emit(merged_text)
                else:
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] {merged_text}")
                
                self.current_context = ""
                self.last_output_time = time.time()
                return
        
        # Standard processing for new segments
        self.transcription_buffer.append(processed_text)
        
        # Update GUI if available
        if self.use_gui and self.signals:
            self.signals.text_signal.emit(processed_text)
        else:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {processed_text}")
            
        # Reset context
        self.current_context = ""
        self.last_output_time = time.time()
    
    def _appears_to_continue(self, previous, current):
        """Determine if current segment likely continues the previous one"""
        # If previous ends with sentence-ending punctuation, likely not continuing
        if re.search(r'[.!?]$', previous):
            return False
            
        # If current starts with lowercase and no sentence markers, likely continuing
        if current and current[0].islower() and not re.match(r'^(however|but|and|or|so|yet|therefore|thus|moreover|furthermore)', current.lower()):
            return True
            
        # Check for partial sentences in previous segment
        prev_words = len(previous.split())
        if prev_words < 5 and not re.search(r'[.!?]$', previous):
            return True
            
        return False
    
    def _merge_segments(self, previous, current):
        """Intelligently merge two text segments"""
        # Remove redundant words between segments
        prev_words = previous.split()
        curr_words = current.split()
        
        # Check for overlapping consecutive words (2-3 word overlap)
        overlap_len = 0
        for i in range(1, 4):  # Check for overlaps of 1-3 words
            if len(prev_words) >= i and len(curr_words) >= i:
                prev_end = ' '.join(prev_words[-i:]).lower()
                curr_start = ' '.join(curr_words[:i]).lower()
                if prev_end == curr_start:
                    overlap_len = i
                    break
        
        # If we found overlap, remove duplicate words
        if overlap_len > 0:
            merged = previous + ' ' + ' '.join(curr_words[overlap_len:])
        else:
            # Ensure proper spacing
            if previous.endswith(('.', ',', '?', '!', ':', ';')):
                merged = previous + ' ' + current
            else:
                merged = previous + ' ' + current
                
        return merged
    
    def process_audio(self):
        """Process audio chunks and perform transcription"""
        print("Audio processing thread started")
        
        # Audio processing parameters
        chunk_size = int(self.chunk_duration * self.sample_rate / 1024)
        overlap_size = int(chunk_size * self.overlap_ratio)
        
        self.previous_audio = None
        audio_buffer = []
        
        # For checking periodic flush
        last_check_time = time.time()
        check_interval = 1.0  # Check every second
        
        while self.is_running:
            # Update status to "Listening"
            if self.use_gui and self.signals:
                self.signals.status_signal.emit("Listening")
                
            # Periodic check for unflushed content
            current_time = time.time()
            if current_time - last_check_time > check_interval:
                last_check_time = current_time
                # If we have content and no recent updates (1.5s silence), flush it
                if self.current_context and (current_time - self.last_segment_time > 1.5):
                    print(f"Forcing flush of content after silence: {self.current_context}")
                    self.flush_context()
            
            # Collect audio chunks
            audio_data = []
            
            # Number of chunks to collect (adjusted for potential overlap)
            chunks_to_collect = chunk_size
            
            for _ in range(chunks_to_collect):
                if not self.is_running:
                    break
                try:
                    audio_data.append(self.audio_queue.get(timeout=1))
                except queue.Empty:
                    continue
            
            if audio_data and self.is_running:
                # Process collected audio
                current_audio = np.concatenate(audio_data)
                
                # Apply audio preprocessing
                if current_audio.dtype != np.float32:
                    current_audio = current_audio.astype(np.float32)
                
                if current_audio.ndim > 1:
                    current_audio = current_audio.mean(axis=1)
                
                # Normalize audio
                if np.abs(current_audio).max() > 0:
                    current_audio = current_audio / np.abs(current_audio).max()
                
                # Apply noise reduction if enabled
                if self.noise_reduction_enabled:
                    # Simple noise gate - zero out low amplitude signals
                    noise_threshold = 0.02  # Adjust as needed
                    current_audio[np.abs(current_audio) < noise_threshold] = 0
                
                # Create overlapping audio by combining with previous chunk
                if self.previous_audio is not None:
                    # Create overlapping audio segment
                    overlap_audio = np.concatenate([
                        self.previous_audio[-int(len(self.previous_audio) * self.overlap_ratio):],
                        current_audio
                    ])
                else:
                    overlap_audio = current_audio
                
                # Save current audio for next iteration
                self.previous_audio = current_audio
                
                # Skip if silence
                mean_amplitude = np.abs(overlap_audio).mean()
                if mean_amplitude < 0.005:
                    # Check if we need to flush due to silence
                    silence_duration = time.time() - self.last_segment_time
                    
                    # Use a shorter timeout (1.5 seconds) for handling silence if there's content
                    if self.current_context and silence_duration > 1.5:  # Shorter timeout for silence
                        self.flush_context()
                    continue
                
                # Update status to "Transcribing"
                if self.use_gui and self.signals:
                    self.signals.status_signal.emit("Transcribing")
                
                # Transcribe
                try:
                    segments, _ = self.model.transcribe(
                        overlap_audio,
                        beam_size=5,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500), # Reduced from 700ms to 500ms
                        language="en"  # Force English language for better accuracy
                    )
                    
                    # Process results
                    segments = list(segments)
                    for segment in segments:
                        transcript = segment.text.strip()
                        if transcript:
                            print(f"Raw transcription segment: {transcript}")
                            # Add to context window instead of directly outputting
                            self.add_to_context(transcript)
                    
                except Exception as e:
                    error_msg = f"Transcription error: {e}"
                    print(error_msg)
                    traceback.print_exc()
                    if self.use_gui and self.signals:
                        self.signals.status_signal.emit("Error")
                        self.signals.text_signal.emit(f"ERROR: {error_msg}")
        
        # Make sure to flush any remaining context when stopping
        self.flush_context()
        print("Audio processing thread stopped")
    
    def start_transcription(self):
        """Start the transcription process"""
        if self.is_running:
            return
            
        self.is_running = True
        
        # Start audio processing thread
        print("Starting audio processing thread")
        process_thread = threading.Thread(target=self.process_audio, daemon=True)
        process_thread.start()
        
        # Start audio capture thread
        print("Starting audio capture thread")
        audio_thread = threading.Thread(target=self._start_audio_capture, daemon=True)
        audio_thread.start()
    
    def _start_audio_capture(self):
        """Start audio capture in a separate thread"""
        try:
            # Use the selected input device
            with sd.InputStream(samplerate=self.sample_rate,
                              channels=self.channels,
                              dtype=self.dtype,
                              blocksize=1024,
                              device=self.input_device,
                              callback=self.audio_callback):
                print(f"Recording started on device {self.input_device}")
                
                # Keep thread alive
                while self.is_running:
                    time.sleep(0.1)
                    
        except Exception as e:
            error_msg = f"Error in audio stream: {e}"
            print(error_msg)
            traceback.print_exc()
            if self.use_gui and self.signals:
                self.signals.status_signal.emit("Error")
                self.signals.text_signal.emit(f"ERROR: {error_msg}")
            self.is_running = False
    
    def run_with_gui(self):
        """Run in GUI mode using PyQt"""
        # Create Qt application
        self.app = QApplication(sys.argv)
        
        # Create main window - pass the actual model size being used
        self.window = TranscriberWindow(model_type=self.model_size)
        
        # Connect signals for thread-safe updates
        self.signals.text_signal.connect(self.window.add_transcription)
        self.signals.audio_level_signal.connect(self.window.set_audio_level)
        self.signals.status_signal.connect(self.window.set_status)
        
        # Connect settings changes to transcriber
        self.window.chunk_slider.valueChanged.connect(
            lambda: self.update_settings(chunk_size=self.window.chunk_size))
        self.window.noise_checkbox.stateChanged.connect(
            lambda: self.update_settings(noise_reduction=self.window.noise_reduction))
        self.window.model_combo.currentTextChanged.connect(
            lambda: self.update_settings(model_type=self.window.model_type))
        
        # Show window
        self.window.show()
        
        # Start transcription in background
        self.start_transcription()
        
        # Start Qt event loop
        print("Starting PyQt application")
        exit_code = self.app.exec_()
        
        # App exited, clean up
        self.stop()
        
        # Return exit code
        return exit_code
    
    def run_cli(self):
        """Run in CLI mode"""
        # Start transcription
        self.start_transcription()
        
        try:
            # Keep main thread alive
            print("Recording started. Press Ctrl+C to stop and save the transcript.")
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the transcription process and save results"""
        if not self.is_running:
            return  # Already stopped
            
        print("Stopping transcription...")
        self.is_running = False
        
        # Flush any remaining context
        self.flush_context()
        
        # Save transcription
        if self.transcription_buffer:
            with open("transcript.txt", "w") as f:
                f.write("\n".join(self.transcription_buffer))
            save_msg = "Transcript saved to transcript.txt"
            print(save_msg)
        else:
            print("No transcription to save.")
        
        print("Transcription stopped.")
    
    def update_settings(self, chunk_size=None, noise_reduction=None, model_type=None):
        """Update transcriber settings from UI controls"""
        if chunk_size is not None:
            self.chunk_duration = chunk_size
            print(f"Chunk duration updated to {chunk_size} seconds")
            
        if noise_reduction is not None:
            self.noise_reduction_enabled = noise_reduction
            print(f"Noise reduction {'enabled' if noise_reduction else 'disabled'}")
            
        if model_type is not None and model_type != self.model_size:
            print(f"Model type change requested from {self.model_size} to {model_type}")
            print("Note: Model change requires restarting the application")
            # Model type changes require reloading the model, which is complex
            # We'll just notify the user but not actually change it during runtime


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Real-time audio transcription using Whisper")
    parser.add_argument("--model", type=str, default="base", 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to run inference on")
    parser.add_argument("--compute_type", type=str, default="int8", 
                        choices=["int8", "float16", "float32"],
                        help="Computation type for inference")
    parser.add_argument("--gui", action="store_true", help="Use GUI mode")
    
    args = parser.parse_args()
    
    try:
        # Create transcriber
        transcriber = RealtimeTranscriber(
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            use_gui=args.gui
        )
        
        # Run in appropriate mode
        if args.gui:
            sys.exit(transcriber.run_with_gui())
        else:
            transcriber.run_cli()
            
    except Exception as e:
        print(f"Error starting application: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 