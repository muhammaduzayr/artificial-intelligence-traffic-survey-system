"""Video opening and metadata helpers shared by the pipeline and zone tools.

OpenCV is kept as the fast/default reader. PyAV is used as a fallback for
containers/codecs that OpenCV can open but cannot decode reliably, especially
some CCTV HEVC exports.
"""

import math
import os
import shutil
import subprocess
import tempfile

import cv2

try:
    import av
except ImportError:  # Optional at import time; requirements installs it.
    av = None


DEFAULT_FPS = 25.0
MIN_REASONABLE_FPS = 0.1
MAX_REASONABLE_FPS = 240.0


def _valid_fps(value):
    return math.isfinite(value) and MIN_REASONABLE_FPS <= value <= MAX_REASONABLE_FPS


def _valid_frame_count(value):
    return math.isfinite(value) and 0 < value < 10**12


class PyAVCapture:
    """Small VideoCapture-compatible adapter around a PyAV video stream."""

    def __init__(self, video_path):
        if av is None:
            raise RuntimeError("PyAV is not installed")

        self._path = os.fspath(video_path)
        self._container = av.open(self._path)
        self._stream = next((s for s in self._container.streams if s.type == "video"), None)
        if self._stream is None:
            self.release()
            raise RuntimeError("No video stream was found")

        rate = float(self._stream.average_rate) if self._stream.average_rate else 0.0
        self._fps = rate if _valid_fps(rate) else 0.0
        self._frame_count = int(self._stream.frames) if self._stream.frames else 0
        self._start_time_sec = (
            float(self._stream.start_time * self._stream.time_base)
            if self._stream.start_time is not None and self._stream.time_base
            else 0.0
        )
        self._iterator = self._frame_generator()
        self._pending_frame = None
        self._pending_msec = None
        self._frame_index = 0
        self._position_msec = 0.0

    def _frame_generator(self):
        for packet in self._container.demux(self._stream):
            for frame in packet.decode():
                yield frame

    def _seek(self, frame_index=None, msec=None):
        if msec is not None and self._stream.time_base:
            target_time = self._start_time_sec + msec / 1000.0
            timestamp = int(target_time / float(self._stream.time_base))
            self._container.seek(max(0, timestamp), stream=self._stream,
                                 backward=True, any_frame=False)
            target_time = msec / 1000.0
            self._iterator = self._frame_generator()
            self._frame_index = 0
            self._pending_frame = None
            while True:
                frame = next(self._iterator, None)
                if frame is None:
                    return
                relative_time = (
                    None if frame.time is None else frame.time - self._start_time_sec
                )
                if relative_time is None or relative_time >= target_time:
                    self._pending_frame = frame
                    return
        else:
            self._container.seek(0, stream=self._stream, backward=True, any_frame=False)
            self._iterator = self._frame_generator()
            self._frame_index = 0
            self._pending_frame = None
            target = max(0, int(frame_index or 0))
            # Discard frames 0..target-1, then cache the frame AT target
            # itself so read() lands exactly on the requested index instead
            # of one frame early.
            while self._frame_index < target:
                frame = next(self._iterator, None)
                if frame is None:
                    return
                self._frame_index += 1
            frame = next(self._iterator, None)
            if frame is None:
                return
            self._pending_frame = frame

    def isOpened(self):
        return self._container is not None

    def getBackendName(self):
        return "PYAV"

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._stream.width or 0)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._stream.height or 0)
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._frame_count)
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self._frame_index)
        if prop == cv2.CAP_PROP_POS_MSEC:
            return float(self._position_msec)
        return 0.0

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_MSEC:
            self._pending_msec = max(0.0, float(value))
            self._pending_frame = None
            return True
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._pending_frame = None
            self._pending_msec = None
            self._seek(frame_index=max(0, int(value)))
            return True
        return False

    def read(self):
        if self._pending_msec is not None:
            msec = self._pending_msec
            self._pending_msec = None
            self._seek(msec=msec)

        frame = self._pending_frame
        self._pending_frame = None
        if frame is None:
            frame = next(self._iterator, None)
        if frame is None:
            return False, None

        image = frame.to_ndarray(format="bgr24")
        self._frame_index += 1
        if frame.time is not None:
            self._position_msec = max(
                0.0, float(frame.time - self._start_time_sec) * 1000.0
            )
        elif self._fps:
            self._position_msec = self._frame_index / self._fps * 1000.0
        return True, image

    def release(self):
        if self._container is not None:
            self._container.close()
            self._container = None


def _try_opencv(video_path, backend):
    cap = cv2.VideoCapture()
    try:
        if not cap.open(os.fspath(video_path), backend):
            cap.release()
            return None, "could not open"
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            cap.release()
            return None, "opened but could not decode a frame"
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return cap, "ok"
    except Exception as exc:
        cap.release()
        return None, str(exc)


def _find_system_ffmpeg():
    """Locate a system-installed ffmpeg binary, distinct from bundled builds."""
    return shutil.which("ffmpeg")


def _normalize_with_system_ffmpeg(video_path, ffmpeg_path):
    """Re-encode a problem video to a known-safe temporary H.264 MP4."""
    fd, out_path = tempfile.mkstemp(suffix="_normalized.mp4")
    os.close(fd)

    # Hikvision exports can carry a 40-byte IMKH header before the MPEG-PS
    # payload, despite being given an .mp4 extension.  Removing that header
    # lets FFmpeg parse the underlying program stream consistently.
    input_options = []
    if _is_imkh_file(video_path):
        input_options = [
            "-f", "mpeg",
            "-skip_initial_bytes", "40",
        ]

    cmd = [
        ffmpeg_path,
        "-analyzeduration", "100000000",
        "-probesize", "100000000",
        "-err_detect", "ignore_err",
        "-fflags", "+discardcorrupt",
        *input_options,
        "-y", "-i", os.fspath(video_path),
        "-map", "0:v:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-an", out_path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.SubprocessError):
        try:
            os.remove(out_path)
        except OSError:
            pass
        raise
    if result.returncode != 0 or not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        try:
            os.remove(out_path)
        except OSError:
            pass
        raise RuntimeError(
            f"ffmpeg normalization failed (code {result.returncode}):\n"
            f"{result.stderr[-1500:]}"
        )
    return out_path


def _is_imkh_file(video_path):
    """Return whether a file has Hikvision's proprietary IMKH prefix."""
    try:
        with open(os.fspath(video_path), "rb") as source:
            return source.read(4) == b"IMKH"
    except OSError:
        return False


def _find_video_pes_start(source, start_pos):
    """Find the next MPEG-PS video PES start code without loading the file."""
    source.seek(start_pos)
    carry = b""
    absolute_pos = start_pos
    marker = b"\x00\x00\x01"
    while True:
        chunk = source.read(1024 * 1024)
        if not chunk:
            return None
        data = carry + chunk
        data_start = absolute_pos - len(carry)
        search_pos = 0
        while True:
            marker_pos = data.find(marker, search_pos)
            if marker_pos < 0:
                break
            stream_id_pos = marker_pos + len(marker)
            if stream_id_pos < len(data) and 0xE0 <= data[stream_id_pos] <= 0xEF:
                return data_start + marker_pos
            search_pos = marker_pos + 1
        carry = data[-2:]
        absolute_pos += len(chunk)


def _extract_imkh_video_stream(video_path):
    """Extract video PES payloads from an IMKH file into a raw Annex-B stream."""
    fd, raw_path = tempfile.mkstemp(suffix="_extracted_video.bin")
    os.close(fd)
    packet_count = 0
    try:
        with open(os.fspath(video_path), "rb") as source, open(raw_path, "wb") as output:
            packet_pos = _find_video_pes_start(source, 0)
            while packet_pos is not None:
                source.seek(packet_pos)
                header = source.read(9)
                if len(header) < 9 or header[:3] != b"\x00\x00\x01":
                    break

                pes_length = int.from_bytes(header[4:6], "big")
                header_data_length = header[8]
                payload_pos = packet_pos + 9 + header_data_length
                if pes_length:
                    payload_length = pes_length - 3 - header_data_length
                    next_search_pos = packet_pos + 6 + pes_length
                else:
                    next_search_pos = _find_video_pes_start(source, payload_pos)
                    payload_length = (
                        None if next_search_pos is None
                        else next_search_pos - payload_pos
                    )

                if payload_length is not None and payload_length > 0:
                    source.seek(payload_pos)
                    remaining = payload_length
                    while remaining:
                        block = source.read(min(1024 * 1024, remaining))
                        if not block:
                            break
                        output.write(block)
                        remaining -= len(block)
                    if remaining:
                        break
                    packet_count += 1

                if pes_length:
                    packet_pos = _find_video_pes_start(source, next_search_pos)
                else:
                    packet_pos = next_search_pos

        if packet_count == 0 or os.path.getsize(raw_path) == 0:
            raise RuntimeError("no video PES payloads were found in the IMKH file")
        return raw_path
    except Exception:
        try:
            os.remove(raw_path)
        except OSError:
            pass
        raise


def _normalize_extracted_imkh_video(raw_path, ffmpeg_path):
    """Try decoding an extracted IMKH stream as HEVC and then H.264."""
    attempts = []
    for input_format in ("hevc", "h264"):
        fd, out_path = tempfile.mkstemp(suffix="_normalized.mp4")
        os.close(fd)
        cmd = [
            ffmpeg_path,
            "-err_detect", "ignore_err",
            "-fflags", "+discardcorrupt",
            "-f", input_format, "-i", raw_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-an", "-y", out_path,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=1800,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            result = None
            attempts.append(f"{input_format}: {exc}")

        if result is not None and result.returncode == 0 and os.path.isfile(out_path):
            if os.path.getsize(out_path) > 0:
                return out_path

        if result is not None:
            attempts.append(
                f"{input_format}: code {result.returncode}: {result.stderr[-700:]}"
            )
        try:
            os.remove(out_path)
        except OSError:
            pass

    raise RuntimeError("IMKH raw-stream recovery failed:\n" + "\n".join(attempts))


def release_video(cap):
    """Release a capture and remove any temporary normalized video it owns."""
    temp_path = getattr(cap, "_normalized_temp_path", None)
    underlying_cap = getattr(cap, "_cap", cap)
    try:
        underlying_cap.release()
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


def open_video(video_path):
    """Open a video and validate that at least one frame can be decoded."""
    path = os.fspath(video_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video file not found: {path}")

    attempts = []
    for name, backend in (("FFmpeg", cv2.CAP_FFMPEG),
                          ("Media Foundation", cv2.CAP_MSMF),
                          ("OpenCV default", cv2.CAP_ANY)):
        cap, result = _try_opencv(path, backend)
        if cap is not None:
            return cap
        attempts.append(f"{name}: {result}")

    if av is not None:
        try:
            cap = PyAVCapture(path)
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return cap
            cap.release()
            attempts.append("PyAV: opened but could not decode a frame")
        except Exception as exc:
            attempts.append(f"PyAV: {exc}")
    else:
        attempts.append("PyAV: not installed")

    # Final fallback: none of the bundled decoders could read this file,
    # but a system ffmpeg install often handles codec profiles (e.g. some
    # NVR/CCTV HEVC variants) that opencv-python's/av's bundled builds
    # don't. Re-encode to a normalized H.264 copy and open that instead.
    ffmpeg_path = _find_system_ffmpeg()
    if ffmpeg_path:
        try:
            normalized_path = _normalize_with_system_ffmpeg(path, ffmpeg_path)
            cap, result = _try_opencv(normalized_path, cv2.CAP_FFMPEG)
            if cap is not None:
                # cv2.VideoCapture instances do not support arbitrary
                # attributes, so retain the path through a small proxy.
                return _NormalizedCapture(cap, normalized_path)
            attempts.append(f"Normalized re-encode: {result}")
            try:
                os.remove(normalized_path)
            except OSError:
                pass
        except Exception as exc:
            attempts.append(f"System ffmpeg normalize: {exc}")

        if _is_imkh_file(path):
            raw_path = None
            try:
                raw_path = _extract_imkh_video_stream(path)
                normalized_path = _normalize_extracted_imkh_video(raw_path, ffmpeg_path)
                cap, result = _try_opencv(normalized_path, cv2.CAP_FFMPEG)
                if cap is not None:
                    return _NormalizedCapture(cap, normalized_path)
                attempts.append(f"IMKH raw-stream re-encode: {result}")
                try:
                    os.remove(normalized_path)
                except OSError:
                    pass
            except Exception as exc:
                attempts.append(f"IMKH raw-stream recovery: {exc}")
            finally:
                if raw_path:
                    try:
                        os.remove(raw_path)
                    except OSError:
                        pass
    else:
        attempts.append(
            "System ffmpeg normalize: ffmpeg not found on PATH "
            "(install it and restart the app to enable this fallback)"
        )

    raise RuntimeError(
        "Video was found but no decoder could read a frame. "
        "The file may be corrupt, incomplete, or encoded with an unavailable codec.\n"
        + "\n".join(f"  - {attempt}" for attempt in attempts)
        + "\nTry exporting the clip again as H.264 MP4 if all decoders fail."
    )


class _NormalizedCapture:
    """Delegate VideoCapture operations while retaining its temp-file path."""

    def __init__(self, cap, normalized_temp_path):
        self._cap = cap
        self._normalized_temp_path = normalized_temp_path

    def __getattr__(self, name):
        return getattr(self._cap, name)

    def release(self):
        release_video(self)


def get_video_info(cap, default_fps=DEFAULT_FPS):
    """Return safe (fps, frame_count) values even when container metadata is bad."""
    raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = raw_fps if _valid_fps(raw_fps) else default_fps

    raw_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    total_frames = int(raw_count) if _valid_frame_count(raw_count) else None
    return fps, total_frames
