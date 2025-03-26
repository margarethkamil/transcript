import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue
import threading
import time
import argparse

class RealtimeTranscriber:
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        """
        Initialize the transcriber with the specified Whisper model
        
        Args:
            model_size: Size of the Whisper model ("tiny", "base", "small", "medium", "large")
            device: Device to run the model on ("cpu" or "cuda")
            compute_type: Computation type ("int8", "float16", "float32")
        """
        print(f"Loading Whisper model ({model_size})...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Model loaded!")
        
        # Audio parameters
        self.sample_rate = 16000  # Whisper expects 16kHz audio
        self.channels = 1
        self.dtype = np.float32
        
        # Buffer for audio data
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.transcription_buffer = []
        
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio capture"""
        if status:
            print(f"Status: {status}")
        # Add audio data to queue
        self.audio_queue.put(indata.copy())
    
    def process_audio(self):
        """Process audio chunks and perform transcription"""
        while self.is_running:
            # Collect 3 seconds of audio (adjust as needed for latency vs accuracy)
            audio_data = []
            chunk_duration = 3  # seconds
            chunks_to_collect = int(chunk_duration * self.sample_rate / 1024)  # 1024 is the chunk size
            
            for _ in range(chunks_to_collect):
                if not self.is_running:
                    break
                try:
                    audio_data.append(self.audio_queue.get(timeout=1))
                except queue.Empty:
                    continue
            
            if audio_data:
                # Concatenate audio chunks
                audio = np.concatenate(audio_data)
                
                # Convert to float32 in range [-1, 1]
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                
                # Normalize audio
                if np.abs(audio).max() > 0:
                    audio = audio / np.abs(audio).max()
                
                # Skip transcription if audio is mostly silence
                if np.abs(audio).mean() < 0.005:
                    continue
                
                # Transcribe
                try:
                    segments, _ = self.model.transcribe(
                        audio,
                        beam_size=5,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )
                    
                    # Convert generator to list to process all segments
                    segments = list(segments)
                    
                    # Print transcription
                    for segment in segments:
                        transcript = segment.text.strip()
                        if transcript:
                            print(f"Transcription: {transcript}")
                            self.transcription_buffer.append(transcript)
                except Exception as e:
                    print(f"Transcription error: {e}")
    
    def start(self):
        """Start the transcription process"""
        self.is_running = True
        
        # Start audio processing thread
        process_thread = threading.Thread(target=self.process_audio)
        process_thread.daemon = True
        process_thread.start()
        
        try:
            # Start audio capture
            with sd.InputStream(samplerate=self.sample_rate,
                              channels=self.channels,
                              dtype=self.dtype,
                              blocksize=1024,
                              callback=self.audio_callback):
                print("Recording started. Press Ctrl+C to stop and save the transcript.")
                
                # Keep main thread alive
                while self.is_running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            print(f"Error in audio stream: {e}")
            self.stop()
    
    def stop(self):
        """Stop the transcription process and save results if requested"""
        self.is_running = False
        print("\nStopping transcription...")
        
        # Save transcription
        if self.transcription_buffer:
            with open("transcript.txt", "w") as f:
                f.write("\n".join(self.transcription_buffer))
            print(f"Transcript saved to transcript.txt")
        
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
    
    args = parser.parse_args()
    
    # Create and start transcriber
    transcriber = RealtimeTranscriber(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type
    )
    
    transcriber.start()

if __name__ == "__main__":
    main() 