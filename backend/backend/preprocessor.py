import subprocess
import re
import numpy as np
import os
import tempfile

def calculate_noise_threshold_ebur128(input_file, percentile=1):
    """
    Calculates a noise threshold based on the given percentile of EBU R128 momentary loudness (M) values.

    Args:
        input_file (str): Path to the input file.
        percentile (float): The percentile to use for the noise threshold (e.g., 1 for the 1st percentile).

    Returns:
        float: The calculated noise threshold in dB, or None if an error occurred.
    """

    command = [
        "ffmpeg",
        "-i", input_file,
        "-af", "ebur128",  # Use ebur128 filter and enable metadata output
        "-f", "null", "-"
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    output_text = error.decode("utf-8")

    # Extract momentary loudness (M) values from FFmpeg output
    momentary_loudness_values = []
    for line in output_text.splitlines():
        if "M:" in line:
            match = re.search(r"M: ([-+]?\d+\.?\d*)", line)
            if match:
                momentary_loudness_values.append(float(match.group(1)))

    if not momentary_loudness_values:
        print("Error: Could not extract loudness information from FFmpeg output.")
        return None

    # Calculate the specified percentile
    threshold = np.percentile(momentary_loudness_values, percentile)
    return threshold


def detect_silence(input_file, noise_threshold, silence_duration=0.1):
    """
    Detects silence in an audio/video file using FFmpeg, with a given noise threshold.

    Args:
        input_file (str): Path to the input file.
        noise_threshold (float): The noise threshold in dB.
        silence_duration (float): Minimum silence duration in seconds.

    Returns:
        list: A list of tuples, where each tuple contains the start and end times of a silence period.
    """

    command = [
        "ffmpeg",
        "-i", input_file,
        "-af", f"silencedetect=noise={noise_threshold}dB:d={silence_duration}",
        "-f", "null", "-"
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    output_text = error.decode("utf-8")

    silence_start_times = []
    silence_end_times = []

    for line in output_text.splitlines():
        if "silence_start" in line:
            match = re.search(r"silence_start: (\d+\.?\d*)", line)
            if match:
                silence_start_times.append(float(match.group(1)))
        if "silence_end" in line:
            match = re.search(r"silence_end: (\d+\.?\d*)", line)
            if match:
                silence_end_times.append(float(match.group(1)))

    # Combine start and end times into a list of tuples
    silence_periods = []
    for i in range(min(len(silence_start_times), len(silence_end_times))):
        silence_periods.append((silence_start_times[i], silence_end_times[i]))

    return silence_periods


def get_frame_rate(input_video):
    """
    Gets the frame rate of the video.
    """
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", input_video]
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    return float(result.stdout.split("/")[0]) / float(result.stdout.split("/")[1])

def create_ffmpeg_speedup_filter(silences, speed_factor=2.0, fps=30):
    """
    Generates the FFmpeg filter_complex command to speed up silent parts while keeping the rest of the video at normal speed.
    """
    filter_commands = []
    concat_inputs = []
    
    last_end = 0.0
    segment_count = 0

    for i, (start, end) in enumerate(silences):
        # Normal-speed segment before the silence
        if start > last_end:
            filter_commands.append(
                f"[0:v]trim=start={last_end}:end={start},setpts=(PTS-STARTPTS),fps={fps}[v{segment_count}];"
                f"[0:a]atrim=start={last_end}:end={start},asetpts=PTS-STARTPTS[a{segment_count}]"
            )
            concat_inputs.append(f"[v{segment_count}][a{segment_count}]")
            segment_count += 1
        
        # Speed-up silent segment
        filter_commands.append(
            f"[0:v]trim=start={start}:end={end},setpts=(PTS-STARTPTS)/{speed_factor},fps={fps}[v{segment_count}];"
            f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS,atempo={speed_factor}[a{segment_count}]"
        )
        concat_inputs.append(f"[v{segment_count}][a{segment_count}]")
        segment_count += 1

        last_end = end

    # Handle remaining part of the video
    filter_commands.append(
        f"[0:v]trim=start={last_end},setpts=(PTS-STARTPTS),fps={fps}[v{segment_count}];"
        f"[0:a]atrim=start={last_end},asetpts=PTS-STARTPTS[a{segment_count}]"
    )
    concat_inputs.append(f"[v{segment_count}][a{segment_count}]")
    
    # Concatenate all segments
    concat_filter = f"{''.join(concat_inputs)}concat=n={segment_count+1}:v=1:a=1[outv][outa]"

    # Join all filter parts
    filter_complex = ";".join(filter_commands) + ";" + concat_filter

    return filter_complex

def split_video(input_video):
    """
    Splits the video into 5 minutes chunks.
    """
    command = ["ffmpeg", "-i", input_video, "-c", "copy", "-map", "0", "-segment_time", "300", "-f", "segment", "./assets/output%03d.mp4"]
    subprocess.run(command)



if __name__ == "__main__":
    input_video = "./assets/input.mp4"  # Replace with your video file



    # Calculate the noise threshold (1st percentile) using EBU R128 momentary loudness
    noise_threshold = calculate_noise_threshold_ebur128(input_video, percentile=30)

    if noise_threshold is not None:
        print(f"Calculated noise threshold (EBU R128 M): {noise_threshold:.2f} dB")

        # Detect silence using the calculated threshold
        silence = detect_silence(input_video, noise_threshold, silence_duration=0.35)

        if silence:
            print("Silence periods detected:")
            for start, end in silence:
                print(f"Start: {start:.3f}s, End: {end:.3f}s, Duration: {end - start:.3f}s")

            fps = get_frame_rate(input_video)
            filter_complex = create_ffmpeg_speedup_filter(silence, speed_factor=10, fps=fps)

            # Save filter_complex to a file
            filter_complex_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
            with open(filter_complex_file.name, "w") as f:
                f.write(filter_complex)

            # Apply the filter_complex to speed up silent parts
            if os.path.exists(filter_complex_file.name):

    
                command = [
                    "ffmpeg", "-i", input_video, "-filter_complex_script", os.path.join(filter_complex_file.name),
                    "-map", "[outv]", "-map", "[outa]", "./assets/output.mp4"
                ]
            
                subprocess.run(command)
            
            # os.remove("./assets/filter_complex.txt")
        else:
            print("No silence detected.")
    else:
        print("Could not calculate noise threshold. Check FFmpeg output.")
