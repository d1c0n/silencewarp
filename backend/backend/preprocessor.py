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


def detect_silence(input_file, noise_threshold, fps, silence_duration=0.1, frame_margin=2):
    """
    Detects silence in an audio/video file using FFmpeg, with a given noise threshold.

    Args:
        input_file (str): Path to the input file.
        noise_threshold (float): The noise threshold in dB.
        silence_duration (float): Minimum silence duration in seconds.
        frame_margin (int): Number of frames to add before and after the detected silence.

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
        silence_periods.append((silence_start_times[i] + frame_margin/fps, silence_end_times[i] - frame_margin/fps))

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

def split_video(input_video, chunk_duration=300):
    """
    Splits the video into chunks.

    Args:
        input_video (str): Path to the input video.
        chunk_duration (int): Duration of each chunk in seconds (default: 300 seconds = 5 minutes).

    Returns:
        list: A list of paths to the chunk files.
    """
    output_dir = os.path.join(os.path.abspath(os.path.dirname(input_video)), "chunks")
    if not os.path.exists(output_dir):
         os.makedirs(output_dir)

    output_pattern = os.path.join(output_dir, "chunk%03d.mp4")

    command = [
        "ffmpeg",
        "-i", input_video,
        "-c", "copy",  # Use stream copying for faster processing
        "-map", "0",
        "-segment_time", str(chunk_duration),
        "-f", "segment",
        "-reset_timestamps", "1", # Reset timestamps for proper concatenation
        output_pattern
    ]
    subprocess.run(command)

    # Get list of chunk files
    chunk_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith("chunk") and f.endswith(".mp4")]
    chunk_files.sort()  # Ensure chunks are in the correct order
    return chunk_files



def merge_chunks(output_video):
    """
    Merges the processed chunk files back into a single video.

    Args:
        output_video (str):  The path to save the merged video.
    """

    chunks_dir = os.path.join(os.path.abspath(os.path.dirname(output_video)), "chunks")
    chunk_files = [os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir) if f.startswith("chunk") and f.endswith(".mp4")]
    chunk_files.sort()  # Very important to sort!

    # Create a text file listing the chunk files for FFmpeg
    list_file_path = os.path.join(chunks_dir, "chunks_list.txt")
    with open(list_file_path, "w") as f:
        for chunk in chunk_files:
            f.write(f"file '{chunk}'\n")

    # Use FFmpeg concat demuxer to merge the chunks
    command = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file_path,
        "-c", "copy",  # Use stream copying (fast)
        output_video
    ]

    subprocess.run(command)
    print(f"Merged video saved to: {output_video}")
    # Clean up the temporary list file.  We leave the chunks for debugging.
    os.remove(list_file_path)



if __name__ == "__main__":
    input_video = "./assets/input.mp4"  # Replace with your video file
    output_video = "./assets/output.mp4"


    chunk_files = split_video(input_video)

    processed_chunks = []
    for chunk_file in chunk_files:
        # Calculate the noise threshold (1st percentile) using EBU R128 momentary loudness
        noise_threshold = calculate_noise_threshold_ebur128(chunk_file, percentile=30)

        if noise_threshold is not None:
            print(f"Calculated noise threshold (EBU R128 M): {noise_threshold:.2f} dB for {chunk_file}")
            
            fps = get_frame_rate(chunk_file)
            # Detect silence using the calculated threshold
            silence = detect_silence(chunk_file, noise_threshold, silence_duration=0.35, fps=fps)

            if silence:
                # print(f"Silence periods detected in {chunk_file}:")
                # for start, end in silence:
                #     print(f"Start: {start:.3f}s, End: {end:.3f}s, Duration: {end - start:.3f}s")

                
                filter_complex = create_ffmpeg_speedup_filter(silence, speed_factor=10, fps=fps)

                # Save filter_complex to a file
                filter_complex_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
                with open(filter_complex_file.name, "w") as f:
                    f.write(filter_complex)

                # Use a temporary output file for the processed chunk
                temp_output_file = os.path.join(os.path.abspath(os.path.dirname(chunk_file)), "temp_" + os.path.basename(chunk_file))

                # Apply the filter_complex
                command = [
                    "ffmpeg",
                    "-i", chunk_file,
                    "-filter_complex_script", filter_complex_file.name,
                    "-map", "[outv]",
                    "-map", "[outa]",
                    temp_output_file
                ]
                subprocess.run(command)

                # Replace original chunk with the processed one
                os.remove(chunk_file)
                os.rename(temp_output_file, chunk_file) # Rename temp file to overwrite the original

                os.remove(filter_complex_file.name) # Clean up the temp filter file


            else:
                print(f"No silence detected in {chunk_file}.")

        else:
            print(f"Could not calculate noise threshold for {chunk_file}. Check FFmpeg output.")


    # Merge all processed chunk files:
    merge_chunks(output_video)