import dataclasses
import os
import zipfile
import typing
import logging
import io
import json

import moderngl as mgl
import numpy as np
import av

__all__ = (
    "MilRendererConfig",
    "MilRenderer"
)

logging.basicConfig(
    level = logging.INFO if not os.environ.get("DEBUG") else logging.DEBUG,
    format = "[%(asctime)s] %(levelname)s %(funcName)s: %(message)s",
    datefmt = "%H:%M:%S"
)

logger = logging.getLogger(__name__)
logger.debug("log level: debug")

@dataclasses.dataclass
class MilRendererConfig:
    width: int
    height: int
    fps: float
    input_path: str
    video_path: str

def _normZipPath(path: str) -> str:
    path = path.replace("\\", "/")
    while "//" in path:
        path = path.replace("//", "/")
    
    if path and path[0] == "/":
        path = path[1:]

    return path

def _hasZipFile(zip: zipfile.ZipFile, path: str) -> bool:
    path = _normZipPath(path)
    for f in zip.infolist():
        if f.filename == path:
            return True
    return False

def _readZipFileAs(zip: zipfile.ZipFile, path: str, dtype: typing.Literal["bytes", "str", "json"]):
    path = _normZipPath(path)
    data = zip.read(path)

    match dtype:
        case "bytes":
            return data

        case "str":
            return data.decode("utf-8")
        
        case "json":
            return json.loads(data)
        
        case _:
            raise ValueError(f"Invalid dtype: {dtype}")

def _decodeAudioBytes(data: bytes) -> np.ndarray:
    wrapped = io.BytesIO(data)

    resampler = av.AudioResampler(format="s16", layout="stereo", rate=44100)

    pcm_chunks = []
    with av.open(wrapped) as cont:
        for frame in cont.decode(audio=0):
            frame.pts = None
            for rframe in resampler.resample(frame):
                if rframe.samples > 0:
                    pcm_chunks.append(rframe.to_ndarray())

    if not pcm_chunks: 
        return np.empty((2, 0), dtype=np.int16)

    pcm = np.concatenate(pcm_chunks, axis=1)
    return pcm.astype(np.int16)

class MilRenderer:
    def __init__(self):
        try:
            self.ctx = mgl.create_context(standalone=True)
        except Exception as e:
            if "XOpenDisplay" in repr(e):
                self.ctx = mgl.create_context(standalone=True, backend="egl")
                logger.info("created rendering context with egl backend")
            else:
                raise RuntimeError("Failed to create rendering context") from e

        logger.info("created rendering context")
        logger.debug(f"rendering context info: {self.ctx.info}")
        
        self._initialized = False
        
    def initialize(self, cfg: MilRendererConfig):
        if self._initialized:
            raise RuntimeError("Renderer already initialized")

        if not isinstance(cfg, MilRendererConfig):
            raise ValueError(f"Invalid config type: {type(cfg)}")
        
        try:
            cfg.width = float(cfg.width)
            cfg.height = float(cfg.height)

            if not cfg.width.is_integer() or not cfg.height.is_integer():
                raise Exception("size must be integer")
            
            cfg.width = int(cfg.width)
            cfg.height = int(cfg.height)

            if cfg.width <= 0 or cfg.height <= 0:
                raise Exception("size must be positive")
        except Exception as e:
            raise ValueError(f"Invalid resolution: {cfg.width}x{cfg.height}") from e
        
        try:
            cfg.fps = float(cfg.fps)
            
            if cfg.fps <= 0.0:
                raise Exception("fps must be positive")
        except Exception as e:
            raise ValueError(f"Invalid fps: {cfg.fps}") from e
        
        if not os.path.exists(cfg.input_path):
            raise ValueError(f"Invalid input path: {cfg.input_path} is not exists")
        
        if not os.path.isfile(cfg.input_path):
            raise ValueError(f"Invalid input path: {cfg.input_path} is not a file")
        
        if os.path.isdir(cfg.video_path):
            raise ValueError(f"Invalid video path: {cfg.video_path} is a directory")
        
        if os.path.exists(cfg.video_path):
            raise ValueError(f"Invalid video path: {cfg.video_path} is already exists")

        self.ctx.viewport = (0, 0, cfg.width, cfg.height)
        logger.info(f"set context viewport to {cfg.width}x{cfg.height}")
        self.cfg = cfg
        self._initialized = True
    
    def run(self):
        try:
            chartZip = zipfile.ZipFile(self.cfg.input_path)
            logger.debug(f"opened chart zip file: {self.cfg.input_path}")
            meta = _readZipFileAs(chartZip, "/meta.json", "json")
            logger.debug(f"read meta.json: {meta}")

            if not isinstance(meta, dict):
                raise Exception("meta.json is not a dict")
            
            chartJson = _readZipFileAs(chartZip, meta["chart_file"], "json")
            audioBytes = _readZipFileAs(chartZip, meta["audio_file"], "bytes")
            imageBytes = _readZipFileAs(chartZip, meta["image_file"], "bytes")

            if not isinstance(chartJson, dict):
                raise Exception("chart.json is not a dict")
            
            logger.info("decoding audio")
            decoededAudio = _decodeAudioBytes(audioBytes)
        except Exception as e:
            raise ValueError(f"Invalid input path: {self.cfg.input_path} is not a zip file") from e
    
if __name__ == "__main__":
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument("-i", "--input", type=str, required=True)
    aparser.add_argument("-o", "--output", type=str, required=True)
    aparser.add_argument("-s-w", "--width", type=int, default=1920)
    aparser.add_argument("-s-h", "--height", type=int, default=1080)
    aparser.add_argument("-f", "--fps", type=float, default=60.0)

    args = aparser.parse_args()
    renderer = MilRenderer()
    
    renderer.initialize(MilRendererConfig(
        width = args.width,
        height = args.height,
        fps = args.fps,
        input_path = args.input,
        video_path = args.output
    ))

    renderer.run()
