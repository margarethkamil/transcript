# Real-Time Audio Transcription with Whisper

This is a Python tool for real-time audio transcription using OpenAI's Whisper model (via faster-whisper).

## Features

- Real-time audio capture and transcription
- Support for multiple Whisper model sizes
- CPU and GPU (CUDA) support
- Automatic transcript saving
- Silence detection to reduce processing load
- Voice Activity Detection (VAD) to improve accuracy
- GUI interface for easier transcript viewing (NEW!)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/margarethkamil/transcript.git
   cd transcript
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Additional dependencies for audio capture:
   - On Windows: No additional steps required
   - On macOS: No additional steps required
   - On Linux: Install PortAudio
     ```bash
     sudo apt-get install libportaudio2
     ```

## Usage

### Command-line Version

Run the transcriber with default settings (base model, CPU):

```bash
python transcriber.py
```

Command-line arguments:

- `--model`: Whisper model size (tiny, base, small, medium, large)
- `--device`: Device to run inference on (cpu, cuda)
- `--compute_type`: Computation type (int8, float16, float32)

Example with a different model:

```bash
python transcriber.py --model tiny --device cpu
```

### GUI Version (Recommended)

For a graphical interface that displays the transcript in real-time:

```bash
python transcriber_gui.py
```

The GUI version offers:
- Real-time transcript display in a scrollable window
- Model selection via dropdown menus
- Start/stop recording buttons
- Save transcript functionality
- Clear transcript button
- Status bar for tracking application state

## Model Size Comparison

| Model | Parameters | English-only | Multilingual | Relative Speed |
|-------|------------|--------------|--------------|----------------|
| tiny  | 39 M       | tiny.en      | tiny         | ~32x           |
| base  | 74 M       | base.en      | base         | ~16x           |
| small | 244 M      | small.en     | small        | ~6x            |
| medium| 769 M      | medium.en    | medium       | ~2x            |
| large | 1550 M     | N/A          | large        | 1x             |

Smaller models are faster but less accurate, while larger models are more accurate but slower.

## Using the Output

The transcriptions are:
1. Displayed in the GUI window in real-time (GUI version)
2. Printed to the console in real-time (command-line version)
3. Saved to a file when you click "Save Transcript" (GUI version) or when you stop the program with Ctrl+C (command-line version)

## Troubleshooting

If you encounter audio device errors:
- Check your microphone is properly connected
- Try running `python -m sounddevice` to list available devices
- Specify a device if needed:
  ```python
  with sd.InputStream(device=DEVICE_ID, ...)
  ``` 