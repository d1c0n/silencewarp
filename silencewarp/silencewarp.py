import subprocess
import re
import numpy as np
import tempfile
import logging
from typing import List, Tuple, Optional
from pathlib import Path

# Initialize logging
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FFmpegError(Exception):
    """Custom exception for FFmpeg related errors."""

    pass


def _run_ffmpeg_command(command: List[str]) -> Tuple[str, str]:
    """
    Executes an FFmpeg command and returns the output and error.

    Args:
        command: The FFmpeg command as a list of strings.

    Returns:
        A tuple containing stdout and stderr as strings.

    Raises:
        FFmpegError: If the FFmpeg command fails (non-zero return code).
    """
    logging.debug(f"Executing FFmpeg command: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output_bytes, error_bytes = process.communicate()
    output_text = output_bytes.decode("utf-8").strip()
    error_text = error_bytes.decode("utf-8").strip()

    if process.returncode != 0:
        logging.error(f"FFmpeg command failed with return code {process.returncode}")
        logging.error(f"Command: {' '.join(command)}")
        logging.error(f"Error output:\n{error_text}")
        # raise FFmpegError(f"FFmpeg command failed: {error_text}")

    return output_text, error_text


def calculate_noise_threshold_ebur128(
    input_file: Path, percentile: float = 30.0
) -> Optional[float]:
    """
    Calculates a noise threshold based on the given percentile of EBU R128 momentary loudness (M) values.

    Args:
        input_file: Path to the input file.
        percentile: The percentile to use for the noise threshold (e.g., 1.0 for the 1st percentile). Must be between 0 and 100.

    Returns:
        The calculated noise threshold in dB, or None if loudness information cannot be extracted.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the percentile is not within the valid range [0, 100].
        FFmpegError: If there's an issue executing the FFmpeg command.
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not 0 <= percentile <= 100:
        raise ValueError(f"Percentile must be between 0 and 100, but got {percentile}")

    command = [
        "ffmpeg",
        "-i",
        str(input_file),  # Convert Path to string for ffmpeg command
        "-af",
        "ebur128",  # Use ebur128 filter and enable metadata output
        "-f",
        "null",
        "-",
    ]

    try:
        _, error_text = _run_ffmpeg_command(command)
    except FFmpegError as e:
        logging.error(f"Error during EBU R128 loudness analysis: {e}")
        return None  # Or re-raise if you want to propagate the exception

    # Extract momentary loudness (M) values from FFmpeg output
    momentary_loudness_values = []
    for line in error_text.splitlines():
        if "M:" in line:
            match = re.search(r"M: ([-+]?\d+\.?\d*)", line)
            if match:
                momentary_loudness_values.append(float(match.group(1)))

    if not momentary_loudness_values:
        logging.warning(
            "Could not extract loudness information from FFmpeg output. Possibly no audio stream."
        )
        return None

    # Calculate the specified percentile
    threshold = np.percentile(momentary_loudness_values, percentile)
    logging.info(
        f"Calculated noise threshold ({percentile}th percentile EBU R128 M): {threshold:.2f} dB"
    )
    return threshold


def detect_silence(
    input_file: Path,
    noise_threshold: float,
    silence_duration: float = 0.35,
    frame_margin: int = 2,
    fps: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """
    Detects silence in an audio/video file using FFmpeg with silencedetect filter.

    Args:
        input_file: Path to the input file.
        noise_threshold: The noise threshold in dB.
        silence_duration: Minimum silence duration in seconds.
        frame_margin: Number of frames to add before and after the detected silence.
        fps: Frame rate of the video. Required for frame margin calculation if not None.

    Returns:
        A list of tuples, where each tuple contains the start and end times of a silence period in seconds.
        Returns an empty list if no silence is detected or an error occurs.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If silence_duration or frame_margin is invalid.
        FFmpegError: If there's an issue executing the FFmpeg command.
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if silence_duration <= 0:
        raise ValueError(
            f"silence_duration must be positive, but got {silence_duration}"
        )
    if frame_margin < 0:
        raise ValueError(f"frame_margin cannot be negative, but got {frame_margin}")

    command = [
        "ffmpeg",
        "-i",
        str(input_file),  # Convert Path to string for ffmpeg command
        "-af",
        f"silencedetect=noise={noise_threshold}dB:d={silence_duration}",
        "-f",
        "null",
        "-",
    ]

    try:
        _, error_text = _run_ffmpeg_command(command)
    except FFmpegError as e:
        logging.error(f"Error during silence detection: {e}")
        return []  # Return empty list on error, or re-raise if you want to propagate the exception

    silence_start_times = []
    silence_end_times = []

    for line in error_text.splitlines():
        if "silence_start" in line:
            match = re.search(r"silence_start: (\d+\.?\d*)", line)
            if match:
                silence_start_times.append(float(match.group(1)))
        if "silence_end" in line:
            match = re.search(r"silence_end: (\d+\.?\d*)", line)
            if match:
                silence_end_times.append(float(match.group(1)))

    # Combine start and end times into a list of tuples
    silence_periods: List[Tuple[float, float]] = []
    for i in range(min(len(silence_start_times), len(silence_end_times))):
        start_time = silence_start_times[i]
        end_time = silence_end_times[i]
        if fps is not None:
            start_time += frame_margin / fps
            end_time -= frame_margin / fps
        silence_periods.append((start_time, end_time))

    logging.info(f"Detected {len(silence_periods)} silence periods.")
    return silence_periods


def get_frame_rate(input_video: Path) -> float:
    """
    Gets the frame rate of the video using ffprobe.

    Args:
        input_video: Path to the input video file.

    Returns:
        The frame rate of the video as a float.

    Raises:
        FileNotFoundError: If the input video file does not exist.
        FFmpegError: If there's an issue executing the ffprobe command or parsing the output.
    """
    if not input_video.exists():
        raise FileNotFoundError(f"Input video file not found: {input_video}")

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_video),  # Convert Path to string for ffprobe command
    ]

    try:
        output_text, _ = _run_ffmpeg_command(command)
    except FFmpegError as e:
        raise FFmpegError(f"Error getting frame rate with ffprobe: {e}") from e

    try:
        rate_str = output_text.strip()
        num_str, den_str = rate_str.split("/")
        frame_rate = float(num_str) / float(den_str)
        logging.debug(f"Frame rate detected: {frame_rate} fps")
        return frame_rate
    except ValueError as e:
        logging.error(f"Could not parse frame rate from ffprobe output: {output_text}")
        raise FFmpegError(f"Failed to parse frame rate: {e}") from e
    except Exception as e:
        logging.error(f"Unexpected error parsing frame rate: {e}")
        raise FFmpegError(f"Unexpected error parsing frame rate: {e}") from e


def create_ffmpeg_speedup_filter(
    silences: List[Tuple[float, float]], speed_factor: float = 5.0, fps: float = 30.0
) -> str:
    """
    Generates the FFmpeg filter_complex command to speed up silent parts while keeping the rest of the video at normal speed.

    Args:
        silences: A list of silence periods, each as a tuple of (start_time, end_time) in seconds.
        speed_factor: The speed-up factor for silent segments. Must be greater than 1.
        fps: The frame rate of the video.

    Returns:
        The FFmpeg filter_complex string.

    Raises:
        ValueError: If speed_factor is not greater than 1 or fps is not positive.
    """
    if speed_factor <= 1:
        raise ValueError(f"speed_factor must be greater than 1, but got {speed_factor}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, but got {fps}")

    filter_commands = []
    concat_inputs = []
    last_end = 0.0
    segment_count = 0

    # Break a speed_factor > 2 in multiple atempo filters of max 2x speed
    # This is done to avoid memory issues with FFMPEG at higher speed factors
    audio_speed_factor = speed_factor
    if speed_factor > 2:
        speed_factor_parts = []
        while audio_speed_factor > 2:
            speed_factor_parts.append(2)
            audio_speed_factor /= 2
        speed_factor_parts.append(audio_speed_factor)
        logging.debug(f"Speed factor parts: {speed_factor_parts}")

    atempo_filter = ",".join([f"atempo={part}" for part in speed_factor_parts])

    for i, (start, end) in enumerate(silences):
        # Normal-speed segment before the silence
        if start > last_end:
            filter_commands.append(
                f"[0:v]trim=start={last_end}:end={start},setpts=PTS-STARTPTS[v{segment_count}];"
                f"[0:a]atrim=start={last_end}:end={start},asetpts=PTS-STARTPTS[a{segment_count}]"
            )
            concat_inputs.append(f"[v{segment_count}][a{segment_count}]")
            segment_count += 1

        # Speed-up silent segment
        filter_commands.append(
            f"[0:v]trim=start={start}:end={end},setpts=(PTS-STARTPTS)/{speed_factor}[v{segment_count}];"
            f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS,{atempo_filter}[a{segment_count}]"
        )
        concat_inputs.append(f"[v{segment_count}][a{segment_count}]")
        segment_count += 1

        last_end = end

    # Handle remaining part of the video
    if (
        last_end < float("inf")
    ):  # To handle cases where the last silence period doesn't reach the end of the video.
        filter_commands.append(
            f"[0:v]trim=start={last_end},setpts=PTS-STARTPTS[v{segment_count}];"
            f"[0:a]atrim=start={last_end},asetpts=PTS-STARTPTS[a{segment_count}]"
        )
        concat_inputs.append(f"[v{segment_count}][a{segment_count}]")

    # Concatenate all segments
    concat_filter = f"{''.join(concat_inputs)}concat=n={segment_count + (1 if last_end < float('inf') else 0)}:v=1:a=1[outv][outa]"

    # Join all filter parts
    filter_complex = (
        ";".join(filter_commands) + ";" + concat_filter + f";[outv]fps={fps}[outv]"
    )
    logging.debug(f"Generated filter_complex: {filter_complex}")
    return filter_complex


def split_video(
    input_video: Path, chunk_duration: int = 60, temp_dir: Optional[str] = None
) -> List[Path]:
    """
    Splits the video into chunks within a temporary directory.

    Args:
        input_video: Path to the input video.
        chunk_duration: Duration of each chunk in seconds (default: 60 seconds).
        temp_dir:  Path to a temporary directory. If None, system temp dir is used.

    Returns:
        A list of paths to the chunk files.

    Raises:
        FileNotFoundError: If the input video file does not exist.
        ValueError: If chunk_duration is not positive.
        FFmpegError: If there's an issue executing the FFmpeg command.
    """
    if not input_video.exists():
        raise FileNotFoundError(f"Input video file not found: {input_video}")
    if chunk_duration <= 0:
        raise ValueError(f"chunk_duration must be positive, but got {chunk_duration}")

    if temp_dir is None:
        temp_dir_path = Path(tempfile.gettempdir())  # Use Path for temp dir
    else:
        temp_dir_path = Path(temp_dir)

    chunks_dir = (
        temp_dir_path / "video_chunks"
    )  # Create chunks directory inside temp dir
    chunks_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = str(chunks_dir / "chunk%03d.mp4")  # Path for output pattern

    command = [
        "ffmpeg",
        "-i",
        str(input_video),  # Convert Path to string for ffmpeg command
        "-c",
        "copy",  # Use stream copying for faster processing
        "-map",
        "0",
        "-segment_time",
        str(chunk_duration),
        "-f",
        "segment",
        "-reset_timestamps",
        "1",  # Reset timestamps for proper concatenation
        output_pattern,
    ]

    try:
        _run_ffmpeg_command(command)
    except FFmpegError as e:
        raise FFmpegError(f"Error splitting video: {e}") from e

    # Get list of chunk files
    chunk_files = [
        chunk_file
        for chunk_file in chunks_dir.iterdir()
        if chunk_file.name.startswith("chunk") and chunk_file.name.endswith(".mp4")
    ]
    chunk_files.sort()  # Ensure chunks are in the correct order
    logging.info(f"Video split into {len(chunk_files)} chunks in {chunks_dir}")
    return chunk_files


def merge_chunks(chunk_files: List[Path], output_video: Path) -> None:
    """
    Merges the processed chunk files back into a single video.

    Args:
        chunk_files: List of paths to the chunk files to merge.
        output_video: The path to save the merged video.

    Raises:
        FileNotFoundError: If no chunk files are provided.
        FFmpegError: If there's an issue executing the FFmpeg command for merging.
    """
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files provided for merging.")

    # Create a temporary directory for the list file (in system temp)
    temp_list_dir = tempfile.mkdtemp()
    list_file_path = Path(temp_list_dir) / "chunks_list.txt"

    try:
        with open(list_file_path, "w") as f:
            for chunk in chunk_files:
                f.write(f"file '{str(chunk)}'\n")  # Convert Path to string in list file

        # Use FFmpeg concat demuxer to merge the chunks
        command = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file_path),  # Convert Path to string for ffmpeg command
            "-c",
            "copy",  # Use stream copying (fast)
            str(output_video),  # Convert Path to string for ffmpeg command
        ]

        try:
            _run_ffmpeg_command(command)
        except FFmpegError as e:
            raise FFmpegError(f"Error merging chunks: {e}") from e

        logging.info(f"Merged video saved to: {output_video}")

    finally:  # Ensure cleanup even if errors occur
        # Clean up the temporary list file and directory
        list_file_path.unlink()
        Path(temp_list_dir).rmdir()


def process_video_silence_speedup(
    input_video: Path,
    output_video: Path,
    percentile=30,
    silence_duration=0.35,
    speed_factor=10,
    temp_dir=None,
):
    """
    Processes the video to speed up silences.
    """
    # Create a main temporary directory for all processing if not provided
    if temp_dir is None:
        with (
            tempfile.TemporaryDirectory() as main_temp_dir_str
        ):  # Context manager for auto cleanup
            main_temp_dir = Path(main_temp_dir_str)
            return _process_video_with_temp_dir(
                input_video,
                output_video,
                percentile,
                silence_duration,
                speed_factor,
                main_temp_dir,
            )
    else:
        main_temp_dir = Path(temp_dir)
        return _process_video_with_temp_dir(
            input_video,
            output_video,
            percentile,
            silence_duration,
            speed_factor,
            main_temp_dir,
        )


def _process_video_with_temp_dir(
    input_video: Path,
    output_video: Path,
    percentile,
    silence_duration,
    speed_factor,
    temp_dir: Path,
):
    """
    Internal function to process video with a given temporary directory.
    """
    try:
        chunk_files = split_video(
            input_video, temp_dir=str(temp_dir)
        )  # Pass temp_dir as string to split_video

        for chunk_file in chunk_files:
            try:
                # Calculate the noise threshold (percentile) using EBU R128 momentary loudness
                noise_threshold = calculate_noise_threshold_ebur128(
                    chunk_file, percentile=percentile
                )

                if noise_threshold is not None:
                    logging.info(f"Processing chunk: {chunk_file}")
                    logging.info(
                        f"Calculated noise threshold (EBU R128 M): {noise_threshold:.2f} dB"
                    )

                    fps = get_frame_rate(chunk_file)
                    # Detect silence using the calculated threshold
                    silence = detect_silence(
                        chunk_file,
                        noise_threshold,
                        silence_duration=silence_duration,
                        fps=fps,
                    )

                    if silence:
                        logging.info(f"Silence periods detected: {silence}")

                        filter_complex = create_ffmpeg_speedup_filter(
                            silence, speed_factor=speed_factor, fps=fps
                        )

                        # Use a temporary file for the filter_complex in the main temp dir
                        with tempfile.NamedTemporaryFile(
                            mode="w+t",
                            delete=False,
                            suffix=".txt",
                            dir=str(temp_dir),  # Pass temp_dir as string to tempfile
                        ) as filter_complex_file:
                            filter_complex_file.write(
                                filter_complex
                            )  # Write string directly
                            filter_complex_file_name = (
                                filter_complex_file.name
                            )  # Capture name before closing
                        filter_complex_file_path = Path(filter_complex_file_name)

                        # Use a temporary output file for the processed chunk in the same chunk directory
                        temp_output_file = chunk_file.parent / (
                            "temp_" + chunk_file.name
                        )

                        # Apply the filter_complex
                        command = [
                            "ffmpeg",
                            "-i",
                            str(
                                chunk_file
                            ),  # Convert Path to string for ffmpeg command
                            "-filter_complex_script",
                            filter_complex_file_name,  # Use -filter_complex_script
                            "-map",
                            "[outv]",
                            "-map",
                            "[outa]",
                            str(
                                temp_output_file
                            ),  # Convert Path to string for ffmpeg command
                        ]
                        _run_ffmpeg_command(command)

                        # Replace original chunk with the processed one
                        chunk_file.unlink()
                        temp_output_file.rename(chunk_file)

                        filter_complex_file_path.unlink()  # Clean up the temp filter file

                    else:
                        logging.info(f"No silence detected in {chunk_file}.")

                else:
                    logging.warning(
                        f"Could not calculate noise threshold for {chunk_file}. Skipping speedup for this chunk."
                    )

            except FFmpegError as e:
                logging.error(f"FFmpeg processing error for chunk {chunk_file}: {e}")
            except Exception as e:
                logging.error(
                    f"An unexpected error occurred while processing chunk {chunk_file}: {e}",
                    exc_info=True,
                )

        # Merge all processed chunk files:
        merge_chunks(chunk_files, output_video)

        logging.info(f"Processing complete. Output video: {output_video}")
        return output_video  # Return output path for API

    except FileNotFoundError as e:
        logging.error(f"File not found error: {e}")
        raise
    except ValueError as e:
        logging.error(f"Value error: {e}")
        raise
    except FFmpegError as e:
        logging.error(f"FFmpeg general error: {e}")
        raise
    except Exception as e:
        logging.error(
            f"An unexpected error occurred in main execution: {e}", exc_info=True
        )
        raise
