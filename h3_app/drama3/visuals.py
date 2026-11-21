"""Plan illustrated shots on the finished mix and render them on its clock."""

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from h3_app.errors import H3Error
from h3_app.processes import run_media_process


FPS = 24


@dataclass(frozen=True)
class Shot:
    first: int
    frames: int
    dissolve: bool
    prompt: str
    beat: str = ""
    references: tuple[str, ...] = ()
    # The scene text with a slot after each subject's name, ⟦key⟧, where an edit puts that subject's
    # reference tag (<image2>, <Picture 2>...) so the model knows which input is which person.
    tagged: str = ""
    id: str = ""  # Stable across timing changes and wording corrections.


def _encode(arguments, cwd=None):
    result = run_media_process(["ffmpeg", "-nostdin", "-y", "-v", "error", *arguments], timeout=7200, cwd=cwd)
    if result.returncode:
        raise H3Error(f"Illustrated video export failed: {result.stderr.strip()[-600:]}")


def render_video(audio, shots, images, output_dir, subtitles: str | None = None, size=(1280, 720),
                 *, frame_batches=None, soundtrack=None):
    """Encode stills or batches of composited JPEG frames, then mux the uninterrupted mix.

    ``frame_batches`` produces (JPEG stream, frame count) pairs in the temporary workspace.
    ``subtitles``, an ASS script (``h3_app.burned_subtitles``), is burned in over the joined
    shots, which then encode once more instead of being copied. ``size`` is the video's width and
    height: 720x1280 for a portrait video.
    """
    width, height = (int(side) for side in size)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audio = Path(audio).resolve()  # the final step runs in the work folder
    soundtrack = Path(soundtrack).resolve() if soundtrack else audio
    # Scale the still once, then repeat that frame until the shot ends.
    canvas = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
              f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p,tpad=stop_mode=clone:stop=-1")
    with TemporaryDirectory(dir=output_dir, prefix=".illustrated-") as temp, ExitStack() as stack:
        folder = Path(temp)
        entries = []
        if frame_batches is None:
            inputs = _still_segments(shots, images, canvas)
        else:
            batches = frame_batches(folder)
            stack.callback(batches.close)
            inputs = ((["-f", "image2pipe", "-framerate", str(FPS), "-c:v", "mjpeg", "-i", str(file)], frames)
                      for file, frames in batches)
        for i, (args, frames) in enumerate(inputs):
            segment = folder / f"segment-{i}.mp4"
            _encode([*args, "-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                     "-preset", "veryfast", str(segment)])
            entries.append(f"file '{segment.name}'")
        playlist = folder / "shots.txt"
        playlist.write_text("\n".join(entries), encoding="ascii")
        partial = folder / "video.mp4"
        video = ["-c:v", "copy"]
        if subtitles:
            # Named relative to the working folder: a Windows drive colon would break the filter's syntax.
            (folder / "subtitles.ass").write_text(subtitles, encoding="utf-8")
            video = ["-vf", "ass=subtitles.ass", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast"]
        _encode(["-f", "concat", "-i", str(playlist), "-i", str(soundtrack), "-map", "0:v:0", "-map", "1:a:0",
                 *video, "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", str(partial)], cwd=folder)
        output = output_dir / f"{audio.stem}_illustrated_{uuid4().hex[:8]}.mp4"
        partial.replace(output)
    return output


def _still_segments(shots, images, canvas):
    still_input = ["-framerate", str(FPS), "-i"]
    for i, (shot, image) in enumerate(zip(shots, images, strict=True)):
        args = [*still_input, str(image)]
        if shot.dissolve and shot.frames > 1:
            # The first frame belongs to the outgoing still. Leave a full incoming frame.
            seconds = min(12, shot.frames - 1) / FPS
            args += [*still_input, str(images[i - 1]), "-filter_complex_threads", "1", "-filter_complex",
                     f"[1:v]{canvas}[prev];[0:v]{canvas}[next];"
                     f"[prev][next]xfade=duration={seconds}:offset=0"]
        else:
            args += ["-vf", canvas]
        yield args, shot.frames
