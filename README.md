
# parakeet-tdt-0.6b-v2-SRT-GUI - A NeMo-based Tool for Generating SRT Subtitles from Video/Audio
[简体中文](README.zh-CN.md)

This project utilizes the `nvidia/parakeet-tdt-0.6b-v2` ASR (Automatic Speech Recognition) model to automatically generate timestamped SRT subtitle files from video or audio files. The interface is built with Gradio, making it easy for users to upload files and retrieve results.

## Key Features

  * Extracts audio from various common video formats (e.g., MP4, MKV, AVI) to generate SRT subtitles.
  * Directly processes various common audio formats (e.g., MP3, WAV, M4A, FLAC) to generate SRT subtitles.
  * Supports long video and audio inputs.
  * Supports loading pre-trained Parakeet models from NVIDIA NGC (default: `nvidia/parakeet-tdt-0.6b-v2`).
  * Supports loading local user `.nemo` model files.
  * Adjustable audio chunk length to balance between processing speed and contextual coherence.
  * Automatically detects CUDA-enabled GPUs and prioritizes them for accelerated processing; falls back to CPU if no GPU is available (slower).
  * User-friendly interface for simple operation.
  * Automatically saves the user's selected model and chunk length configuration.
  * Supports batch transcription and mixed processing of video/audio files.

## System Requirements

  * Python 3.12.2 or higher, to ensure compatibility with the latest NeMo library.
  * **FFmpeg**: Used for audio/video decoding, encoding, and format conversion. **Must be installed separately and added to the system's PATH environment variable.**
  * NVIDIA GPU (Recommended for acceleration, requires CUDA drivers). Can run on CPU if no GPU is available, but it will be very slow.

## Simplified Installation Steps (Windows)

1.  **Clone this repository:**

    ```bash
    git clone https://github.com/NINIYOYYO/parakeet-tdt-0.6b-v2-SRT-GUI.git
    ```

2.  **Double-click `install_dependencies.bat`**
    This will create and activate a Python virtual environment while also checking for and installing dependencies.
    The installation of Torch depends on whether you need GPU acceleration.

3.  **Install FFmpeg:**
    This project relies on FFmpeg for audio extraction and preprocessing. You need to install it separately and ensure its executable path is added to your system's PATH environment variable.

      * **Windows:**
        1.  Download a pre-compiled version from the official FFmpeg download page ([https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)) (e.g., builds from "gyan.dev" or "BtbN").
        2.  Unzip the downloaded file.
        3.  Add the path to the `bin` directory inside the unzipped folder (e.g., `C:\ffmpeg\bin`) to your system's `Path` environment variable.
      * **Linux (Ubuntu/Debian, etc.):**
        ```bash
        sudo apt update && sudo apt install ffmpeg
        ```
      * **macOS (using Homebrew):**
        ```bash
        brew install ffmpeg
        ```

4.  **Double-click `launcher.bat`**
    If your environment and dependencies are all set up correctly, you can run the project directly by double-clicking `launcher.bat`.

## Detailed Installation Steps

1.  **Clone this repository:**

    ```bash
    git clone https://github.com/NINIYOYYO/parakeet-tdt-0.6b-v2-SRT-GUI.git
    ```

2.  **Create and activate a Python virtual environment (Highly Recommended):**

    ```bash
    python -m venv .venv
    ```

      * **Windows:**
        ```bash
        .\.venv\Scripts\activate
        ```
      * **macOS / Linux:**
        ```bash
        source .venv/bin/activate
        ```

3.  **Install PyTorch (Important: GPU users, pay close attention\!):**
    If you want to use an NVIDIA GPU for accelerated processing (highly recommended), **be sure to manually install a PyTorch version compatible with your CUDA environment *before* installing other dependencies.**

      * Press `Win+R` to open the Windows Run dialog, type `cmd`, and press Enter to open the terminal. Then, type:

    <!-- end list -->

    ```bash
    nvidia-smi
    ```

    and press Enter to check your `CUDA Version`.

      * Visit the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).
      * Select the correct installation command based on your operating system, package manager (recommend `pip`), compute platform (e.g., CUDA 11.8, CUDA 12.1), and Python version.
      * For example, if you are using `pip` and your system has a CUDA 12.1 environment, you can run:
        ```bash
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```

    If you skip this step, or if your system does not have an NVIDIA GPU, the `nemo_toolkit` installed later may default to a CPU-only version of PyTorch.

4.  **Install project dependencies:**
    After activating the virtual environment and (optionally) installing a specific version of PyTorch, run:

    ```bash
    pip install -r requirements.txt
    ```

5.  **Install FFmpeg:**
    This project relies on FFmpeg for audio extraction and preprocessing. You need to install it separately and ensure its executable path is added to your system's PATH environment variable.

      * **Windows:**
        1.  Download a pre-compiled version from the official FFmpeg download page ([https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)) (e.g., builds from "gyan.dev" or "BtbN").
        2.  Unzip the downloaded file.
        3.  Add the path to the `bin` directory inside the unzipped folder (e.g., `C:\ffmpeg\bin`) to your system's `Path` environment variable.
      * **Linux (Ubuntu/Debian, etc.):**
        ```bash
        sudo apt update && sudo apt install ffmpeg
        ```
      * **macOS (using Homebrew):**
        ```bash
        brew install ffmpeg
        ```

6.  **Double-click `launcher.bat`**
    If your environment and dependencies are all set up correctly, you can run the project directly by double-clicking `launcher.bat`.

## Loading a Local Model

**If you want to load the `nvidia/parakeet-tdt-0.6b-v2` model from a local path, on the first launch, enter the local path of the model in the "Local Model Path (.nemo file)" field.**
**For example: `C:\Users\models--nvidia--parakeet-tdt-0.6b-v2\snapshots\30c5e6f557f6ba26e5819a9ed2e86f670186b43f\parakeet-tdt-0.6b-v2.nemo`**

## Interface Preview

## How to Use

Ensure you have completed the environment setup and dependency installation as described above.

There are two ways to start the application:

1.  **Run `launcher.bat` to start.**

2.  **Start from the terminal (after activating the virtual environment):**

    ```bash
    python main.py
    ```

After the script starts, it will print a local URL in the terminal (usually `http://127.0.0.1:7860` or similar). Open this URL in your browser to access the Gradio user interface.

**Model Selection and Loading:**

  * **Local Model:** Enter the full path to your `.nemo` model file in the "Local Model Path" input box, then click the "Load Local Model" button.
  * **Cloud Model:** Simply click the "Load Cloud Model" button to download and load the default Parakeet model from NVIDIA NGC.

The model loading status will be displayed in the text box below.

**Adjusting Audio Chunk Length:**
Use the slider to adjust the "Audio Chunk Length (seconds)". A larger chunk size can preserve more context but may increase processing time and memory consumption. A range of 60-180 seconds is recommended. This setting will be saved along with your model choice the next time you click either "Load Model" button.

**Uploading Files and Generating Subtitles:**

  * **From Video:** Switch to the "Generate from Video" tab, upload your video file by clicking the video upload area, and then click the "Start Generating SRT from Video" button.
  * **From Audio:** Switch to the "Generate from Audio" tab, upload your audio file by clicking the audio upload area, and then click the "Start Generating SRT from Audio" button.

**Viewing and Downloading Results:**

  * The processing status will update in real-time.
  * Once processing is complete, you can preview the generated subtitle content in the "SRT Subtitle Result" area and click the "Download SRT File" link to download the `.srt` subtitle file.

## Notes

  * Processing large files or running on a CPU may take a significant amount of time. Please be patient.
  * The first time you load a cloud model, the model files will need to be downloaded, which may take time depending on your network speed.
  * If you encounter any `ffmpeg`-related errors, please ensure that FFmpeg is installed and configured correctly in your system's PATH.
  * If the script indicates it is running on the CPU, but you have an NVIDIA GPU and wish to use it, please double-check that you have correctly installed the CUDA-enabled version of PyTorch (refer to step 3 of the "Installation Steps").