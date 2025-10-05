import dataclasses
import os
import warnings
import contextlib

import moderngl as mgl
import numpy as np

@dataclasses.dataclass
class MilRendererConfig:
    width: int
    height: int
    fps: float
    input_path: str
    video_path: str

class MilRendererException(Exception):
    pass

class FuckUserException:
    def __new__(cls, *args, **kwargs):
        if args and "fuck user" in args:
            args = args[1:]

        return MilRendererException(*args, **kwargs)

FuckUserWarnning = type("MilRendererWarnning", (Warning, ), {})

class MilRenderer:
    @contextlib.contextmanager
    def _internal_context(self):
        self._internal_call = True
        yield
        self._internal_call = False

    def __init__(self):
        try:
            self.ctx = mgl.create_context(standalone=True)
        except Exception as e:
            if "XOpenDisplay" in repr(e):
                self.ctx = mgl.create_context(standalone=True, backend="egl")
            else:
                raise RuntimeError("Failed to create rendering context") from e
        
        with self._internal_context():
            self._initialized = False
        
    def initialize(self, cfg: MilRendererConfig):
        if self._initialized:
            raise RuntimeError("Renderer already initialized")

        if not isinstance(cfg, MilRendererConfig):
            raise ValueError(f"Invalid config type: {type(cfg)}")
        
        try:
            ctx.width = float(cfg.width)
            ctx.height = float(cfg.height)

            if not ctx.width.is_integer() or not ctx.height.is_integer():
                raise FuckUserException("fuck user", "size must be integer")
            
            ctx.width = int(ctx.width)
            ctx.height = int(ctx.height)

            if ctx.width <= 0 or ctx.height <= 0:
                raise FuckUserException("fuck user", "size must be positive")
        except Exception as e:
            raise ValueError(f"Invalid resolution: {cfg.width}x{cfg.height}") from e
        
        try:
            ctx.fps = float(cfg.fps)
            
            if ctx.fps <= 0.0:
                raise FuckUserException("fuck user", "fps must be positive")
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

        with self._internal_context():
            self._initialized = True
    
    def run(self):
        pass
    
    def __setattr__(self, name, value):
        if name.startswith("_") and not self._internal_call:
            warnings.warn(f"Accessing private attribute {name} is not allowed", FuckUserWarnning)
    
if __name__ == "__main__":
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument("-i", "--input", type=str, required=True)
    aparser.add_argument("-o", "--output", type=str, required=True)
    aparser.add_argument("-s-w", "--width", type=int, default=1920, required=True)
    aparser.add_argument("-s-h", "--height", type=int, default=1080, required=True)
    aparser.add_argument("-f", "--fps", type=float, default=60.0, required=True)

    args = aparser.parse_args()
    renderer = MilRenderer(MilRendererConfig(
        width = args.width,
        height = args.height,
        fps = args.fps,
        input_path = args.input,
        video_path = args.output
    ))

    renderer.run()
