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
                            QLabel, QTextEdit, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor, QTextCursor
import re

class TranscriptionSignals(QObject):
    """Custom signals for thread communication"""
    text_signal = pyqtSignal(str)

class TranscriberWindow(QMainWindow):
    """PyQt-based GUI for displaying transcriptions"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Transcriber")
        self.setGeometry(100, 100, 800, 600)
        self.setup_ui()
        
        # Keep window on top
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        
    def setup_ui(self):
        """Set up the user interface"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)  # Add spacing between widgets
        main_layout.setContentsMargins(20, 20, 20, 20)  # Add margins
        
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
        self.overlap_ratio = 0.25  # 25% overlap between audio chunks
        self.previous_audio = None  # Store previous audio for overlap
        
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
    
    def should_flush_context(self):
        """Determine if we should output the current context"""
        current_time = time.time()
        
        # Flush if we have content and exceeded the delay time since last output
        if self.current_context and (current_time - self.last_output_time) > self.output_delay:
            return True
            
        # Flush if we have content and exceeded the silence timeout
        if self.current_context and (current_time - self.last_segment_time) > self.segment_timeout:
            return True
            
        # NEW: Flush shorter segments after a brief silence (1.5 seconds)
        if self.current_context and (current_time - self.last_segment_time) > 1.5:
            return True
            
        # Flush if we have a complete sentence with proper ending and significant length
        if (self.current_context and 
            re.search(r'[.!?]$', self.current_context) and 
            len(self.current_context.split()) > 10):
            return True
            
        return False
    
    def remove_duplicates(self, text):
        """Remove duplicate words and phrases from text"""
        if not text:
            return text
            
        # First check for exact repetition of 2-4 word phrases
        for phrase_len in range(4, 1, -1):
            pattern = r'\b(\w+(?:\s+\w+){' + str(phrase_len-1) + r'})\s+\1\b'
            text = re.sub(pattern, r'\1', text)
        
        # Remove immediate word repetitions (like "the the" or "and and")
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)
        
        # Check for repetition across sentence boundaries
        text = re.sub(r'(\w+)[\.\?\!]\s+\1\b', r'\1.', text, flags=re.IGNORECASE)
        
        # Detect partial phrase repetitions
        words = text.split()
        
        # NEW: Detect abandoned and restarted phrases
        i = 0
        while i < len(words) - 3:
            # Look for restart markers that often indicate abandoned phrases
            restart_markers = ["alright", "well", "so", "anyway", "um", "uh", "like", "okay", "right"]
            
            # Check if we have a restart indicator
            has_restart = False
            restart_idx = -1
            
            # Look for restart markers or punctuation that might indicate a restart
            for j in range(i, min(i + 4, len(words))):
                if j < len(words) and (words[j].lower() in restart_markers or 
                                        words[j] in [",", ".", "?", "..."]):
                    has_restart = True
                    restart_idx = j
                    break
            
            if has_restart and restart_idx < len(words) - 2:
                # Check if words after the restart repeat earlier words
                before_restart = words[max(0, i-3):restart_idx]
                after_restart = words[restart_idx+1:restart_idx+5]
                
                # Count matching words
                matches = 0
                match_indices = []
                
                for k, word1 in enumerate(before_restart):
                    for l, word2 in enumerate(after_restart):
                        if word1.lower() == word2.lower():
                            matches += 1
                            match_indices.append((k, l))
                
                # If we found multiple matches or sequence matches, it's likely a restart
                if matches >= 2 or (matches == 1 and len(before_restart) >= 2 and len(after_restart) >= 2):
                    # Found a likely restart - remove the abandoned phrase and the restart marker
                    words = words[:i] + words[restart_idx+1:]
                    continue
            
            # Special case for "and to" -> "and today" type restarts (common speech error)
            if i < len(words) - 4 and words[i].lower() in ["my", "the", "this", "that", "and", "to", "with"]:
                if words[i+2].lower() in ["and", "alright", "well", "so"] and words[i].lower() == words[i+3].lower():
                    # Pattern like "my name and my" - remove abandoned phrase
                    words = words[:i] + words[i+2:]
                    continue
                
            i += 1
        
        # NEW: Detect partial phrase repetitions (3-word sequences that repeat with variations)
        i = 0
        while i < len(words) - 5:
            # Check if a 2-3 word sequence appears again within the next few words
            for seq_len in range(3, 1, -1):
                if i + seq_len + 3 >= len(words):  # Ensure we have enough words left
                    continue
                    
                # Get the sequence we're checking
                seq = words[i:i+seq_len]
                seq_text = ' '.join(seq).lower()
                
                # Look for this sequence in the next few words
                next_text = ' '.join(words[i+seq_len:i+seq_len+6]).lower()
                
                # If 2+ words of the sequence repeat, consider it a partial repetition
                # Check each word in the sequence
                matches = 0
                for word in seq:
                    if word.lower() in next_text.split():
                        matches += 1
                
                # If most words match (proportional to sequence length)
                if matches >= max(2, seq_len-1):
                    # Find where repetition seems to end
                    end_idx = i + seq_len
                    for j in range(i+seq_len, min(i+seq_len+6, len(words))):
                        if j < len(words) and words[j].lower() in [w.lower() for w in seq]:
                            end_idx = j + 1  # Include this word in what we'll remove
                    
                    # Remove the repetition portion (keep the first occurrence)
                    words = words[:i+seq_len] + words[end_idx:]
                    break  # We modified the list, so restart the outer loop
            i += 1
        
        # NEW: Handle phrases like "to make it to make the"
        i = 0
        while i < len(words) - 4:  # Need at least 4 words
            if i + 3 < len(words):
                # Look for a pattern like "X Y Z ... X Y something"
                if (words[i].lower() == words[i+3].lower() and
                    i + 4 < len(words) and words[i+1].lower() == words[i+4].lower()):
                    # Found a partial repeat with pattern "X Y ... X Y"
                    words = words[:i+3] + words[i+5:]  # Remove the repetition
                    continue  # Check the same position again
            i += 1
        
        # Remove stutters (same word with slight variations)
        i = 0
        while i < len(words) - 1:
            if words[i].lower() == words[i+1].lower():
                words.pop(i+1)
                continue
            i += 1
        
        text = ' '.join(words)
        
        return text
    
    def add_to_context(self, text):
        """Add a new segment to the context window and determine if it should be output"""
        if not text.strip():
            return
            
        # Update timing information
        self.last_segment_time = time.time()
        
        # Remove punctuation at the beginning that might have been added by Whisper
        text = text.strip()
        text = re.sub(r'^[,.;:!?]\s*', '', text)
        
        # Apply duplicate removal
        text = self.remove_duplicates(text)
        
        # Check for duplicates with the end of the current context
        if self.current_context:
            # Create combined text and check for repeated content
            combined = self.current_context + " " + text
            combined = self.remove_duplicates(combined)
            
            # If the deduped combined text is shorter than the separate texts,
            # there was a duplication that was removed
            if len(combined.split()) < len(self.current_context.split()) + len(text.split()):
                # Replace context and text with deduplicated version
                current_words = self.current_context.split()
                text_words = text.split()
                combined_words = combined.split()
                
                # If the combined text is shorter, find how words to keep from the current context
                if len(combined_words) > len(current_words):
                    # Some duplication, but not all of text was duplicate
                    self.current_context = ' '.join(current_words)
                    remaining_words = combined_words[len(current_words):]
                    text = ' '.join(remaining_words)
                else:
                    # Complete duplication, just keep current context
                    return
        
        # Improved sentence boundary detection
        # Check for explicit sentence boundaries with improved pattern matching
        is_new_sentence = False
        
        # Look for period/question/exclamation + space + capital letter pattern in combined text
        if self.current_context:
            combined = self.current_context + " " + text
            if re.search(r'[.!?]\s+[A-Z]', combined):
                # Find the last sentence ending punctuation
                match = list(re.finditer(r'[.!?]\s+[A-Z]', combined))
                if match:
                    last_match = match[-1]
                    split_pos = last_match.start() + 1  # Include the punctuation
                    
                    # Split at this position
                    first_part = combined[:split_pos].strip()
                    second_part = combined[split_pos:].strip()
                    
                    # If first part is substantive, flush it
                    if first_part and len(first_part.split()) > 3:
                        self.current_context = first_part
                        self.flush_context()
                        self.current_context = second_part
                        return
        
        # Also check if text starts with clear sentence beginning indicators
        if (self.current_context and 
            ((re.search(r'[.!?]$', self.current_context) and re.match(r'[A-Z]', text)) or
             re.match(r'^(However|Nevertheless|Therefore|Furthermore|Moreover|In addition|Thus|Also|Besides|Indeed|Still|Anyway)', text))):
            is_new_sentence = True
        
        # If it's clearly a new sentence and we have content, flush current context first
        if is_new_sentence and self.current_context:
            self.flush_context()
        
        # Add space if needed between existing context and new text
        if self.current_context and not self.current_context.endswith(' '):
            self.current_context += ' '
            
        # Add the new text to the current context
        self.current_context += text
        
        # Remove duplicate spaces
        self.current_context = re.sub(r'\s+', ' ', self.current_context)
        
        # Add to context window history
        self.context_window.append(text)
        if len(self.context_window) > self.context_window_size:
            self.context_window.pop(0)
            
        # Check if we should output the context
        if self.should_flush_context():
            self.flush_context()
    
    def flush_context(self):
        """Output the current context and reset"""
        if not self.current_context:
            return
            
        # Final check for any duplicates before output
        self.current_context = self.remove_duplicates(self.current_context)
        
        # Capitalize the first letter of the context if needed
        if self.current_context and self.current_context[0].islower():
            self.current_context = self.current_context[0].upper() + self.current_context[1:]
        
        # Ensure the context ends with proper punctuation
        if not re.search(r'[.!?]$', self.current_context):
            # Choose appropriate punctuation based on content
            if re.search(r'\b(who|what|where|when|why|how|which)\b', self.current_context.lower()):
                self.current_context += '?'
            else:
                self.current_context += '.'
            
        # Add to final transcription buffer
        self.transcription_buffer.append(self.current_context)
        
        # Update GUI if available
        if self.use_gui and self.signals:
            self.signals.text_signal.emit(self.current_context)
        else:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {self.current_context}")
            
        # Reset context
        self.current_context = ""
        self.last_output_time = time.time()
    
    def process_audio(self):
        """Process audio chunks and perform transcription"""
        print("Audio processing thread started")
        
        # Audio processing parameters
        chunk_duration = 3.0  # seconds
        chunk_size = int(chunk_duration * self.sample_rate / 1024)
        overlap_size = int(chunk_size * self.overlap_ratio)
        
        self.previous_audio = None
        audio_buffer = []
        
        # For checking periodic flush
        last_check_time = time.time()
        check_interval = 1.0  # Check every second
        
        while self.is_running:
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
                
                # Transcribe
                try:
                    segments, _ = self.model.transcribe(
                        overlap_audio,
                        beam_size=5,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=700),
                        #language="en"  # Force English language for better accuracy
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
                self.signals.text_signal.emit(f"ERROR: {error_msg}")
            self.is_running = False
    
    def run_with_gui(self):
        """Run in GUI mode using PyQt"""
        # Create Qt application
        self.app = QApplication(sys.argv)
        
        # Create main window
        self.window = TranscriberWindow()
        
        # Connect signals for thread-safe updates
        self.signals.text_signal.connect(self.window.add_transcription)
        
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