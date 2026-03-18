import ffmpeg
import os
import glob

def create_video_from_frames(input_pattern, output_file, framerate=30, ffmpeg_path=None):
    """
    Create a video from a sequence of numbered images.

    :param input_pattern: Input image pattern (e.g., 'output_frames/frame_%04d.png')
    :param output_file: Output video file name
    :param framerate: Frame rate (default: 30)
    :param ffmpeg_path: Path to ffmpeg executable (optional)
    """
    try:
        # Check if input images exist
        input_files = glob.glob(input_pattern.replace('%04d', '*'))
        if not input_files:
            raise FileNotFoundError(f"Input images not found: {input_pattern}")

        # Create video using ffmpeg-python
        stream = (
            ffmpeg
            .input(input_pattern, framerate=framerate)
            .output(output_file, vcodec='libx264', pix_fmt='yuv420p')
            .overwrite_output()
        )

        # Use the specified ffmpeg path if provided
        if ffmpeg_path:
            stream = stream.global_args('-y')
            stream.run(cmd=ffmpeg_path, capture_stdout=True, capture_stderr=True)
        else:
            stream.run(capture_stdout=True, capture_stderr=True)

        print(f"Video creation completed: {output_file}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
