# silencewarp

[![PyPI version](https://badge.fury.io/py/silencewarp.svg)](https://badge.fury.io/py/silencewarp)

**silencewarp** is a Python package that utilizes FFmpeg to automatically speed up silent portions of video files. This can be useful for reducing video length and improving viewer engagement by renmoving unnecessary pauses but still allowing viewers to follow what's happening by avoiding abrupt cuts.

## Features

- **Silence Detection:** Uses FFmpeg's `silencedetect` filter to identify silent periods in video audio.
- **EBU R128 Noise Threshold Calculation:** Optionally calculates a noise threshold dynamically based on the EBU R128 loudness standard, making it adaptable to different audio levels.
- **Speed Up Silence:** Generates FFmpeg filter commands to speed up detected silent segments while maintaining normal speed for non-silent parts.
- **Chunked Processing:** Splits large videos into chunks for processing, which can be helpful for memory management and handling long videos.
- **Easy Integration:** Designed to be easily integrated into Python backend applications or scripts.

## Installation

```bash
pip install silencewarp
```
