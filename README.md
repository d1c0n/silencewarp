# silencewarp

[![PyPI version](https://badge.fury.io/py/silencewarp.svg)](https://badge.fury.io/py/silencewarp)

**silencewarp** is a Python package that utilizes FFmpeg to automatically speed up silent portions of video files. This can be useful for reducing video length and improving viewer engagement by removing unnecessary pauses but still allowing viewers to follow what's happening by avoiding abrupt cuts.

## Features

- **Silence Detection:** Uses FFmpeg's `silencedetect` filter to identify silent periods in video audio.
- **EBU R128 Noise Threshold Calculation:** Optionally calculates a noise threshold dynamically based on the EBU R128 loudness standard, making it adaptable to different audio levels.
- **Speed Up Silence:** Generates FFmpeg filter commands to speed up detected silent segments while maintaining normal speed for non-silent parts.
- **Chunked Processing:** Splits large videos into chunks for processing, which can be helpful for memory management and handling long videos.

## Installation

Before installing `silencewarp`, ensure you have FFmpeg installed and accessible in your system's PATH. FFmpeg is a powerful command-line tool for handling video and audio files, and `silencewarp` relies on it for its core functionality.

**Installing FFmpeg:**

Installation instructions vary depending on your operating system. Here are some common methods:

- **Linux (Debian/Ubuntu):**
  ```bash
  sudo apt update && sudo apt install ffmpeg
  ```
- **Linux (Fedora/CentOS):**
  ```bash
  sudo dnf install ffmpeg
  ```
- **macOS (Homebrew):**
  ```bash
  brew install ffmpeg
  ```
- **Windows:**
  Download the latest version from the [FFmpeg website](https://ffmpeg.org/download.html). After downloading, you'll need to add the FFmpeg `bin` directory to your system's PATH environment variable.

**Installing `silencewarp`:**

Once FFmpeg is installed and in your PATH, you can install `silencewarp` using pip:

```bash
pip install silencewarp
```

## Usage

Here are examples of how to use the functions in `silencewarp`.

### `calculate_noise_threshold_ebur128(input_file, percentile=30.0)`

Calculates a noise threshold based on the given percentile of EBU R128 momentary loudness (M) values. This is useful for automatically determining an appropriate noise level for silence detection based on the audio characteristics of your video.

```python
from silencewarp import calculate_noise_threshold_ebur128

input_video_file = "path/to/your/video.mp4"
noise_threshold = calculate_noise_threshold_ebur128(input_video_file, percentile=30.0)

if noise_threshold is not None:
    print(f"Calculated noise threshold: {noise_threshold:.2f} dB")
else:
    print("Could not calculate noise threshold.")
```

**Parameters:**

- `input_file` (str): Path to the input video or audio file.
- `percentile` (float, optional): The percentile to use for the noise threshold. For example, `30.0` (default) means the threshold will be set at the loudness level below which 30% of the momentary loudness values fall. Must be between 0 and 100.

**Returns:**

- `Optional[float]`: The calculated noise threshold in dB, or `None` if loudness information cannot be extracted from the input file (e.g., if there's no audio stream or if FFmpeg fails to process the file).

### `detect_silence(input_file, noise_threshold, silence_duration=0.35, frame_margin=2, fps=None)`

Detects silence periods in an audio or video file using FFmpeg's `silencedetect` filter.

```python
from silencewarp import detect_silence

input_video_file = "path/to/your/video.mp4"
noise_threshold = -40.0  # Example noise threshold in dB
silence_periods = detect_silence(input_video_file, noise_threshold, silence_duration=0.2)

if silence_periods:
    print("Silence periods detected:")
    for start, end in silence_periods:
        print(f"Start: {start:.2f}s, End: {end:.2f}s")
else:
    print("No silence detected.")
```

**Parameters:**

- `input_file` (str): Path to the input video or audio file.
- `noise_threshold` (float): The noise threshold in dB. Audio levels below this threshold are considered silence.
- `silence_duration` (float, optional): Minimum duration of silence in seconds to be detected. Default is `0.35` seconds.
- `frame_margin` (int, optional): Number of frames to add before the start and subtract from the end of each detected silence period. This is useful to avoid abrupt cuts at the edges of silence. Default is `2` frames.
- `fps` (Optional[float], optional): Frame rate of the video. If provided, `frame_margin` is interpreted in frames and converted to seconds. If `None`, `frame_margin` in seconds is not applied.

**Returns:**

- `List[Tuple[float, float]]`: A list of tuples, where each tuple contains the start and end times (in seconds) of a detected silence period. Returns an empty list if no silence is detected or if an error occurs.

### `get_frame_rate(input_video)`

Retrieves the frame rate of a video file using `ffprobe`.

```python
from silencewarp import get_frame_rate

input_video_file = "path/to/your/video.mp4"
frame_rate = get_frame_rate(input_video_file)

print(f"Frame rate of the video: {frame_rate:.2f} FPS")
```

**Parameters:**

- `input_video` (str): Path to the input video file.

**Returns:**

- `float`: The frame rate of the video in frames per second (FPS).

### `create_ffmpeg_speedup_filter(silences, speed_factor=5.0, fps=30.0)`

Generates the FFmpeg `filter_complex` command string required to speed up the silent segments of a video, while keeping the non-silent parts at normal speed.

```python
from silencewarp import create_ffmpeg_speedup_filter

silence_periods = [(10.5, 12.3), (25.0, 28.7)] # Example silence periods
speed_factor = 5.0
fps = 29.97

filter_complex_command = create_ffmpeg_speedup_filter(silence_periods, speed_factor=speed_factor, fps=fps)
print(f"FFmpeg filter_complex command:\n{filter_complex_command}")
```

**Parameters:**

- `silences` (List[Tuple[float, float]]): A list of silence periods, where each period is a tuple containing the start and end time (in seconds).
- `speed_factor` (float, optional): The factor by which silent segments should be sped up. Must be greater than 1. Default is `5.0`.
- `fps` (float, optional): The frame rate of the video being processed. Default is `30.0`.

**Returns:**

- `str`: The FFmpeg `filter_complex` string that can be used with FFmpeg to apply the speedup effect.

### `split_video(input_video, chunk_duration=60, temp_dir=None)`

Splits a video file into smaller chunks of a specified duration. This can be helpful for processing large videos in segments.

```python
from silencewarp import split_video

input_video_file = "path/to/your/video.mp4"
chunk_files = split_video(input_video_file, chunk_duration=30) # Split into 30-second chunks

print("Video chunks created:")
for chunk_file in chunk_files:
    print(chunk_file)
```

**Parameters:**

- `input_video` (str): Path to the input video file.
- `chunk_duration` (int, optional): Duration of each chunk in seconds. Default is `60` seconds.
- `temp_dir` (str, optional): Path to a temporary directory where chunks will be saved. If `None`, the system's default temporary directory is used.

**Returns:**

- `List[str]`: A list of paths to the created chunk files.

### `merge_chunks(chunk_files, output_video)`

Merges a list of video chunk files back into a single video file.

```python
from silencewarp import merge_chunks

chunk_files = [
    "path/to/temp/video_chunks/chunk001.mp4",
    "path/to/temp/video_chunks/chunk002.mp4",
    "path/to/temp/video_chunks/chunk003.mp4",
] # Example chunk files from split_video
output_video_file = "path/to/output/merged_video.mp4"

merge_chunks(chunk_files, output_video_file)
print(f"Merged video saved to: {output_video_file}")
```

**Parameters:**

- `chunk_files` (List[str]): A list of paths to the video chunk files that need to be merged.
- `output_video` (str): Path to save the merged output video file.

**Returns:**

- `None`

### `process_video_silence_speedup(input_video, output_video, percentile=30, silence_duration=0.35, speed_factor=10, temp_dir=None)`

This is the main function to process a video by automatically detecting and speeding up silent parts. It orchestrates the entire process, including splitting, silence detection, filter generation, and merging.

```python
from silencewarp import process_video_silence_speedup

input_video_file = "path/to/your/input_video.mp4"
output_video_file = "path/to/your/output_video_speedup.mp4"

process_video_silence_speedup(
    input_video_file,
    output_video_file,
    percentile=20,         # Adjust percentile for noise threshold calculation
    silence_duration=0.4,  # Adjust minimum silence duration
    speed_factor=8.0       # Adjust speed-up factor
)

print(f"Processed video with silence speedup saved to: {output_video_file}")
```

**Parameters:**

- `input_video` (str): Path to the input video file.
- `output_video` (str): Path to save the processed output video file with sped-up silences.
- `percentile` (int, optional): Percentile value (0-100) for calculating the noise threshold using EBU R128 loudness. Default is `30`. A lower percentile means a lower (more sensitive) noise threshold.
- `silence_duration` (float, optional): Minimum duration of silence in seconds to be sped up. Default is `0.35` seconds.
- `speed_factor` (float, optional): The factor by which silent segments should be sped up. Default is `10.0`.
- `temp_dir` (str, optional): Path to a temporary directory to use for intermediate files (chunks, etc.). If `None`, a system temporary directory is used.

**Returns:**

- `str`: The path to the processed output video file.

## Build Information

`silencewarp` is managed using [Poetry](https://python-poetry.org/).

**To set up a development environment:**

1. Ensure you have Poetry installed. Follow the instructions on the [Poetry website](https://python-poetry.org/docs/#installation).
2. Clone the repository.
3. Navigate to the project directory in your terminal.
4. Install dependencies using Poetry:

   ```bash
   poetry install
   ```

**To build the package:**

```bash
poetry build
```

This will create distribution files in the `dist` directory.
