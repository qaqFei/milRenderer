from __future__ import annotations

import dataclasses
import os
import zipfile
import typing
import logging
import io
import json
import struct
import time
import collections
import ctypes
from ctypes.util import find_library

import moderngl as mgl
import numpy as np
import av
import tqdm

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

MIL_SCRW = 1920
MIL_SCRH = 1080
logger.debug(f"{MIL_SCRW=}, {MIL_SCRH=}")

LINE_CIRCLE_WIDTH = 0.003
LINE_HEAD_SIZE = 0.0223
LINE_HEAD_BORDER = LINE_HEAD_SIZE * (18 / 186)
NOTE_SIZE = LINE_HEAD_SIZE
NOTE_SCALE = 335 / 185
SPEED_UNIT = 120
logger.debug(f"{LINE_CIRCLE_WIDTH=}, {LINE_HEAD_SIZE=}")
logger.debug(f"{LINE_HEAD_BORDER=}, {NOTE_SIZE=}, {NOTE_SCALE=}")
logger.debug(f"{SPEED_UNIT=}")

HOLD_DISAPPEAR_TIME = 0.2
FLOW_SPEED = 1.66
HOLD_SPWAN_HIT_EFFECT_SEP = 0.1
HIT_EFFECT_DUR = 0.5
HITEFFECT_SIZE = 0.12
HITEFFECT_PREPARE_GROUP_NUM = 16
logger.debug(f"{HOLD_DISAPPEAR_TIME=}, {FLOW_SPEED=}")
logger.debug(f"{HOLD_SPWAN_HIT_EFFECT_SEP=}, {HIT_EFFECT_DUR=}")
logger.debug(f"{HITEFFECT_SIZE=}")
logger.debug(f"{HITEFFECT_PREPARE_GROUP_NUM=}")

AUDIO_SAMPLE_RATE = 16000
AUDIO_LAYOUT = "stereo"
logger.debug(f"{AUDIO_SAMPLE_RATE=}, {AUDIO_LAYOUT=}")

RES_PATH = "./res"
logger.debug(f"{RES_PATH=}")

MIL_EASINGS: list[list[typing.Callable[[float], float]]] = [
    [
        lambda t: t, # linear
        lambda t: 1 - math.cos((t * math.pi) / 2), # in sine
        lambda t: t ** 2, # in quad
        lambda t: t ** 3, # in cubic
        lambda t: t ** 4, # in quart
        lambda t: t ** 5, # in quint
        lambda t: 0 if t == 0 else 2 ** (10 * t - 10), # in expo
        lambda t: 1 - (1 - t ** 2) ** 0.5, # in circ
        lambda t: 2.70158 * (t ** 3) - 1.70158 * (t ** 2), # in back
        lambda t: 0 if t == 0 else (1 if t == 1 else - 2 ** (10 * t - 10) * math.sin((t * 10 - 10.75) * (2 * math.pi / 3))), # in elastic
        lambda t: 1 - (7.5625 * ((1 - t) ** 2) if ((1 - t) < 1 / 2.75) else (7.5625 * ((1 - t) - (1.5 / 2.75)) * ((1 - t) - (1.5 / 2.75)) + 0.75 if ((1 - t) < 2 / 2.75) else (7.5625 * ((1 - t) - (2.25 / 2.75)) * ((1 - t) - (2.25 / 2.75)) + 0.9375 if ((1 - t) < 2.5 / 2.75) else (7.5625 * ((1 - t) - (2.625 / 2.75)) * ((1 - t) - (2.625 / 2.75)) + 0.984375)))), # in bounce
    ],
    [
        lambda t: t, # linear
        lambda t: math.sin((t * math.pi) / 2), # out sine
        lambda t: 1 - (1 - t) * (1 - t), # out quad
        lambda t: 1 - (1 - t) ** 3, # out cubic
        lambda t: 1 - (1 - t) ** 4, # out quart
        lambda t: 1 - (1 - t) ** 5, # out quint
        lambda t: 1 if t == 1 else 1 - 2 ** (-10 * t), # out expo
        lambda t: (1 - (t - 1) ** 2) ** 0.5, # out circ
        lambda t: 1 + 2.70158 * ((t - 1) ** 3) + 1.70158 * ((t - 1) ** 2), # out back
        lambda t: 0 if t == 0 else (1 if t == 1 else 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * (2 * math.pi / 3)) + 1), # out elastic
        lambda t: 7.5625 * (t ** 2) if (t < 1 / 2.75) else (7.5625 * (t - (1.5 / 2.75)) * (t - (1.5 / 2.75)) + 0.75 if (t < 2 / 2.75) else (7.5625 * (t - (2.25 / 2.75)) * (t - (2.25 / 2.75)) + 0.9375 if (t < 2.5 / 2.75) else (7.5625 * (t - (2.625 / 2.75)) * (t - (2.625 / 2.75)) + 0.984375))), # out bounce
    ],
    [
        lambda t: t, # linear
        lambda t: -(math.cos(math.pi * t) - 1) / 2, # io sine
        lambda t: 2 * (t ** 2) if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2, # io quad
        lambda t: 4 * (t ** 3) if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2, # io cubic
        lambda t: 8 * (t ** 4) if t < 0.5 else 1 - (-2 * t + 2) ** 4 / 2, # io quart
        lambda t: 16 * (t ** 5) if t < 0.5 else 1 - ((-2 * t + 2) ** 5) / 2, # io quint
        lambda t: 0 if t == 0 else (1 if t == 1 else (2 ** (20 * t - 10) if t < 0.5 else (2 - 2 ** (-20 * t + 10))) / 2), # io expo
        lambda t: (1 - (1 - (2 * t) ** 2) ** 0.5) / 2 if t < 0.5 else (((1 - (-2 * t + 2) ** 2) ** 0.5) + 1) / 2, # io circ
        lambda t: ((2 * t) ** 2 * ((2.5949095 + 1) * 2 * t - 2.5949095)) / 2 if t < 0.5 else ((2 * t - 2) ** 2 * ((2.5949095 + 1) * (t * 2 - 2) + 2.5949095) + 2) / 2, # io back
        lambda t: 0 if t == 0 else (1 if t == 0 else (-2 ** (20 * t - 10) * math.sin((20 * t - 11.125) * ((2 * math.pi) / 4.5))) / 2 if t < 0.5 else (2 ** (-20 * t + 10) * math.sin((20 * t - 11.125) * ((2 * math.pi) / 4.5))) / 2 + 1), # io elastic
        lambda t: (1 - (7.5625 * ((1 - 2 * t) ** 2) if ((1 - 2 * t) < 1 / 2.75) else (7.5625 * ((1 - 2 * t) - (1.5 / 2.75)) * ((1 - 2 * t) - (1.5 / 2.75)) + 0.75 if ((1 - 2 * t) < 2 / 2.75) else (7.5625 * ((1 - 2 * t) - (2.25 / 2.75)) * ((1 - 2 * t) - (2.25 / 2.75)) + 0.9375 if ((1 - 2 * t) < 2.5 / 2.75) else (7.5625 * ((1 - 2 * t) - (2.625 / 2.75)) * ((1 - 2 * t) - (2.625 / 2.75)) + 0.984375))))) / 2 if t < 0.5 else (1 +(7.5625 * ((2 * t - 1) ** 2) if ((2 * t - 1) < 1 / 2.75) else (7.5625 * ((2 * t - 1) - (1.5 / 2.75)) * ((2 * t - 1) - (1.5 / 2.75)) + 0.75 if ((2 * t - 1) < 2 / 2.75) else (7.5625 * ((2 * t - 1) - (2.25 / 2.75)) * ((2 * t - 1) - (2.25 / 2.75)) + 0.9375 if ((2 * t - 1) < 2.5 / 2.75) else (7.5625 * ((2 * t - 1) - (2.625 / 2.75)) * ((2 * t - 1) - (2.625 / 2.75)) + 0.984375))))) / 2, # io bounce
    ]
]

class EnumAnimationKey:
    Unknown = -1
    
    PositionX = 0
    PositionY = 1
    Transparency = 2
    Size = 3
    Rotation = 4
    FlowSpeed = 5
    RelativeX = 6
    RelativeY = 7
    LineBodyTransparency = 8
    LineHeadTransparency = 9
    StoryBoardWidth = 10
    StoryBoardHeight = 11
    Speed = 12
    WholeTransparency = 13
    StoryBoardLeftBottomX = 14
    StoryBoardLeftBottomY = 15
    StoryBoardRightBottomX = 16
    StoryBoardRightBottomY = 17
    StoryBoardLeftTopX = 18
    StoryBoardLeftTopY = 19
    StoryBoardRightTopX = 20
    StoryBoardRightTopY = 21
    Color = 22
    VisibleArea = 23

class EnumAnimationBearerType:
    Unknown = -1
    
    Line = 0
    Note = 1
    StoryBoard = 2

class EnumNoteType:
    Hit = 0
    Drag = 1

MAX_ANIMKEY = EnumAnimationKey.VisibleArea

def beatval(beat: list[int]):
    return beat[0] + beat[1] / beat[2]

def tosec(t: list[int], chart: MilChart):
    t = beatval(t)
    sec = chart.meta.offset

    if len(chart.bpms) == 1:
        sec += 60 / chart.bpms[0].bpm * t
    else:
        for i, e in enumerate(chart.bpms):
            if i != len(chart.bpms) - 1:
                et_beat = chart.bpms[i + 1].time - e.time
                
                if t >= et_beat:
                    sec += et_beat * (60 / e.bpm)
                    t -= et_beat
                else:
                    sec += t * (60 / e.bpm)
                    break
            else:
                sec += t * (60 / e.bpm)

    return sec

class _VideoWriter:
    def __init__(self, width: int, height: int, fps: float):
        libSVW_path = find_library("libSimpleVideoWriter") or find_library("SimpleVideoWriter")
        if libSVW_path is None:
            raise RuntimeError("libSimpleVideoWriter not found")

        self._lib = ctypes.CDLL(libSVW_path)

        CreateVideoContext = self._lib.CreateVideoContext
        CreateVideoContext.argtypes = (ctypes.c_long, ctypes.c_long, ctypes.c_double)
        CreateVideoContext.restype = ctypes.c_void_p

        self._ptr = CreateVideoContext(width, height, fps)
    
    def initialize(self, path: str, v_codec: str, a_codec: str, audio: np.ndarray, a_bitrate: int = 192000):
        InitializeVideoContext = self._lib.InitializeVideoContext
        InitializeVideoContext.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool, ctypes.c_void_p, ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_long)
        InitializeVideoContext.restype = ctypes.c_bool
        
        audio = _audio_to_f32(audio)
        audio = np.ascontiguousarray(audio)
        channels = _get_channels_from_layout(AUDIO_LAYOUT)

        res = InitializeVideoContext(
            self._ptr, path.encode("utf-8"),
            v_codec.encode("utf-8"), a_codec.encode("utf-8"),
            True, audio.ctypes.data,
            AUDIO_SAMPLE_RATE, channels,
            len(audio) // channels, a_bitrate
        )

        if not res:
            raise RuntimeError("InitializeVideoContext failed")
    
    def write_frame(self, rgbFrame: bytearray, width: int, height: int):
        PutFrame = self._lib.PutFrame
        PutFrame.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_long)
        PutFrame.restype = None

        carr = (ctypes.c_ubyte * len(rgbFrame)).from_buffer(rgbFrame)
        PutFrame(self._ptr, carr, width, height)

    def release(self):
        ReleaseVideoContext = self._lib.ReleaseVideoContext
        ReleaseVideoContext.argtypes = (ctypes.c_void_p, )
        ReleaseVideoContext.restype = None

        ReleaseVideoContext(self._ptr)
    
    def get_encoders(self, is_video: bool):
        GetEncoders = self._lib.GetEncoders
        GetEncoders.argtypes = (ctypes.c_bool,)
        GetEncoders.restype = ctypes.POINTER(ctypes.c_ubyte)

        raw_ptr = GetEncoders(is_video)
        size = 0
        while raw_ptr[size]:
            while raw_ptr[size]:
                size += 1
            size += 1
        data = ctypes.string_at(raw_ptr, size)

        FreeString = self._lib.FreeString
        FreeString.argtypes = (ctypes.c_void_p,)
        FreeString.restype = None
        FreeString(raw_ptr)

        return [i.decode("utf-8") for i in data.split(b"\0")][:-1]
    
    def has_encoder(self, name: str):
        hasEncoder = self._lib.HasEncoder
        hasEncoder.argtypes = (ctypes.c_char_p, )
        hasEncoder.restype = ctypes.c_bool

        return hasEncoder(name.encode("utf-8"))
    
class ChartMeta:
    def __init__(self, data: dict):
        self.background_dim = data["background_dim"]
        self.name = data["name"]
        self.background_artist = data["background_artist"]
        self.music_artist = data["music_artist"]
        self.charter = data["charter"]
        self.difficulty_name = data["difficulty_name"]
        self.difficulty = data["difficulty"]
        self.offset = data["offset"]

class BPMEvent:
    def __init__(self, data: dict):
        self.time = beatval(data["time"])
        self.bpm = data["bpm"]

class MilNote:
    def __init__(self, data: dict, master_anims: list[MilAnimation], master_chart: MilChart):
        self.time = tosec(data["time"], master_chart)
        self.type = data["type"]
        self.isFake = data["isFake"]
        self.isAlwaysPerfect = data["isAlwaysPerfect"]
        self.endTime = tosec(data["endTime"], master_chart)
        self.index = data["index"]

        self.acollection = MilAnimationCollectionGroup.from_filter_anims(master_anims, EnumAnimationBearerType.Note, self.index)
        self.ishit = self.type == EnumNoteType.Hit
        self.ishold = self.ishit and self.endTime > self.time
        self.master: typing.Optional[MilLine] = None
        self.floorPosition = 0.0
        self.endFloorPosition = 0.0
        self.morebets = False
        self.clicked = False
        self.holdLastSpwanHitEffectTime = self.time
        self.transform = (0.0, ) * 6
    
    def init(self):
        assert isinstance(self.master, MilLine), "master is not set"
        
        self.master.acollection.update(self.time, only=EnumAnimationKey.Speed)
        self.floorPosition = self.master.acollection.get_value(EnumAnimationKey.Speed)
        self.master.acollection.update(self.endTime, only=EnumAnimationKey.Speed)
        self.endFloorPosition = self.master.acollection.get_value(EnumAnimationKey.Speed)
        self.texname = ("ex" if self.isAlwaysPerfect else "") + (("hold" if self.ishold else "tap") if self.ishit else "drag") + ("_double" if self.morebets else "")
    
    def update(self, t: float):
        self.acollection.update(t)

class MilEase:
    def __init__(self, data: dict):
        self.type = data["type"]
        self.press = data["press"]
        self.isValueExp = data["isValueExp"]
        self.cusValueExp = data["cusValueExp"]
        self.clipLeft = data["clipLeft"]
        self.clipRight = data["clipRight"]

        if not self.isValueExp:
            try:
                self.doease = MIL_EASINGS[self.type][self.press]
            except IndexError:
                self.doease = MIL_EASINGS[0][0]
        else:
            self.doease = lambda p: p
    
    def interplate(self, p: float, start: float, end: float, etype: int):
        is_color = etype == EnumAnimationKey.Color
        p = self.doease(p)

        if not is_color:
            return start + (end - start) * p
        else:
            s_color = num2rgba(start)
            e_color = num2rgba(end)
            r = s_color[0] + (e_color[0] - s_color[0]) * p
            g = s_color[1] + (e_color[1] - s_color[1]) * p
            b = s_color[2] + (e_color[2] - s_color[2]) * p
            a = s_color[3] + (e_color[3] - s_color[3]) * p
            return (r, g, b, a)

class MilAnimation:
    def __init__(self, data: dict, master_chart: MilChart):
        self.startTime = tosec(data["startTime"], master_chart)
        self.endTime = tosec(data["endTime"], master_chart)
        self.type = data["type"]
        self.start = data["start"]
        self.end = data["end"]
        self.index = data["index"]
        self.bearer_type = data["bearer_type"]
        self.bearer = data["bearer"]
        self.ease = MilEase(data["ease"])
        self.index = data["index"]

        self.floorPosition = 0
    
    def interplate(self, t: float):
        p = 1 if self.startTime == self.endTime else (t - self.startTime) / (self.endTime - self.startTime)
        p = max(0, min(1, p))
        res = self.ease.interplate(p, self.start, self.end, self.type)
        return res

class MilAnimationCollectionGroup:
    def __init__(self, anims: list[MilAnimation], defaults: list[float]):
        self.values = defaults.copy()
        self.defaults = defaults.copy()
        self.indexs = [0 for _ in range(MAX_ANIMKEY + 1)]
        self.anim_groups = [[] for _ in range(MAX_ANIMKEY + 1)]
        self._t = 0

        for e in anims:
            self.anim_groups[e.type].append(e)
        
        for es in self.anim_groups:
            es.sort(key=lambda e: e.startTime)
        
        speed_es = self.anim_groups[EnumAnimationKey.Speed]
        fp = 0.0

        for e in speed_es:
            e.floorPosition = fp
            fp += (e.endTime - e.startTime) * (e.start + e.end) / 2
        
        self.is_effect_opt = any(map(lambda k: self.anim_groups[k], (
            EnumAnimationKey.PositionX,
            EnumAnimationKey.PositionY,
            EnumAnimationKey.Size,
            EnumAnimationKey.Rotation,
            EnumAnimationKey.FlowSpeed,
            EnumAnimationKey.RelativeX,
            EnumAnimationKey.RelativeY,
            EnumAnimationKey.Speed
        )))
    
    def update(self, t: float, *, only: typing.Optional[int] = None):
        if t < self._t:
            self.indexs = [0 for _ in range(MAX_ANIMKEY + 1)]
        
        self._t = t

        for i, es in enumerate(self.anim_groups):
            if len(es) == 0 or (only is not None and i != only):
                if i == EnumAnimationKey.Speed and (only is None or only == EnumAnimationKey.Speed):
                    self.values[i] = t * self.defaults[i]
                continue
            
            while self.indexs[i] < len(es) - 1 and es[self.indexs[i] + 1].startTime <= t:
                self.indexs[i] += 1
            
            e = es[self.indexs[i]]
            self.values[i] = e.interplate(t)

            if i == EnumAnimationKey.Speed:
                if t < e.startTime: self.values[i] = t * e.start
                elif e.startTime < t <  e.endTime: self.values[i] = e.floorPosition + (t - e.startTime) * (self.values[i] + e.start) / 2
                else: self.values[i] = e.floorPosition + (e.endTime - e.startTime) * (e.start + e.end) / 2 + (t - e.endTime) * e.end
    
    def get_value(self, key: int):
        return self.values[key]

    @staticmethod
    def from_filter_anims(anims: list[MilAnimation], bearer_type: int, bearer: typing.Optional[int] = None):
        anims = list(filter(lambda e: e.bearer_type == bearer_type and (bearer is None or e.bearer == bearer), anims))

        return MilAnimationCollectionGroup(anims, {
            0: [
                0.0,
                -350.0,
                1.0,
                1.0,
                90.0,
                1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                (255, 255, 255, 255),
                float("inf"),
            ],
            1: [
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                (255, 255, 255, 255),
                0.0,
            ],
            2: [
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -0.5,
                0.5,
                0.5,
                0.5,
                -0.5,
                -0.5,
                0.5,
                -0.5,
                (255, 255, 255, 255),
                float("inf"),
            ]
        }[bearer_type])

class MilLine:
    def __init__(self, data: dict, master_chart: MilChart):
        self.animations = list(map(lambda x: MilAnimation(x, master_chart), data["animations"]))
        self.notes = list(map(lambda x: MilNote(x, self.animations, master_chart), data["notes"]))
        self.index = data["index"]

        self.notes.sort(key=lambda e: e.time)
        self.acollection = MilAnimationCollectionGroup.from_filter_anims(self.animations, EnumAnimationBearerType.Line)
        self.note_groups = [IterRemovableList([], can_break=False), IterRemovableList([], can_break=True)]

        for note in self.notes:
            if note.acollection.is_effect_opt:
                self.note_groups[0].append(note)
            else:
                self.note_groups[1].append(note)
    
    def init(self):
        for n in self.notes:
            n.master = self
            n.init()
    
    def update(self, t: float):
        self.acollection.update(t)

        for n in self.notes:
            n.update(t)

class MilChart:
    def __init__(self, data: dict):
        if data["fmt"] != 2:
            raise ValueError(f"Unsupported chart format: {data['fmt']}")

        self.meta = ChartMeta(data["meta"])
        self.bpms = list(map(BPMEvent, data["bpms"]))
        self.bpms.sort(key=lambda e: e.time)

        self.lines = list(map(lambda x: MilLine(x, self), data["lines"]))
        self.lines.sort(key=lambda e: e.index)

        self.init()
    
    def init(self):
        morebets_map = {}

        for l in self.lines:
            for note in l.notes:
                if note.isFake:
                    continue

                if note.time not in morebets_map:
                    morebets_map[note.time] = 0

                morebets_map[note.time] += 1

        for l in self.lines:
            for note in l.notes:
                if note.isFake:
                    continue

                if morebets_map[note.time] > 1:
                    note.morebets = True

            l.init()
    
    def update(self, t: float):
        for l in self.lines:
            l.update(t)

class MilHitEffect:
    def __init__(self, note: MilNote, t: float):
        self.note = note
        self.t = t
        self.group = random.randint(0, HITEFFECT_PREPARE_GROUP_NUM - 1)

T = typing.TypeVar("")
class Node(typing.Generic[T]):
    __slots__ = ("value", "prev", "next")
    def __init__(self, value: T):
        self.value = value
        self.prev: typing.Optional[Node[T]] = None
        self.next: typing.Optional[Node[T]] = None

T = typing.TypeVar("")
class IterRemovableList(typing.Generic[T]):
    def __init__(self, lst: list[T], *, can_break: bool = True):
        self.head: typing.Optional[Node[T]] = None
        self.tail: typing.Optional[Node[T]] = None
        self._build_linked_list(lst)
        self.current: typing.Optional[Node[T]] = None
        self.can_break = can_break

    def _build_linked_list(self, lst: list[T]) -> None:
        prev_node = None
        for item in lst:
            new_node = Node(item)
            if not self.head:
                self.head = new_node
            if prev_node:
                prev_node.next = new_node
                new_node.prev = prev_node
            prev_node = new_node
        self.tail = prev_node

    def __iter__(self) -> typing.Iterator[tuple[T, typing.Callable[[], None]]]:
        self.current = self.head
        return self

    def __next__(self) -> tuple[T, typing.Callable[[], None]]:
        if self.current is None:
            raise StopIteration
        
        current_node = self.current
        self.current = current_node.next
        
        def remove_callback() -> None:
            prev_node = current_node.prev
            next_node = current_node.next
            
            if prev_node:
                prev_node.next = next_node
            else:
                self.head = next_node
            
            if next_node:
                next_node.prev = prev_node
            else:
                self.tail = prev_node
        
        return current_node.value, remove_callback
    
    def append(self, i: T):
        new = Node(i)
        new.prev = self.tail
        new.next = None
        
        if self.tail is None:
            self.head = new
            self.tail = new
        else:
            self.tail.next = new
            self.tail = new

@dataclasses.dataclass
class MilRendererConfig:
    width: int
    height: int
    fps: float
    input_path: str
    video_path: str
    v_codec: str
    a_codec: str

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
    resampler = av.AudioResampler(format="s16", layout=AUDIO_LAYOUT, rate=AUDIO_SAMPLE_RATE)

    pcm_chunks = []
    with av.open(wrapped) as cont:
        for frame in cont.decode(audio=0):
            frame.pts = None
            for rframe in resampler.resample(frame):
                if rframe.samples > 0:
                    pcm_chunks.append(rframe.to_ndarray()[0])

    if not pcm_chunks: 
        return np.empty((2, ), dtype=np.int16)

    return np.concatenate(pcm_chunks).astype(np.int32)

def _decodeAudioFromFile(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        return _decodeAudioBytes(f.read())

def _decodeImageBytes(data: bytes) -> WarppedImage:
    warpped = io.BytesIO(data)
    
    with av.open(warpped) as cont:
        for frame in cont.decode(video=0):
            frame.pts = None
            narr = frame.to_ndarray()
            c = frame.format.bits_per_pixel // 8
            h, w = narr.shape[:2]

            if c == 4:
                narr = narr.ravel()
                r, b = np.arange(len(narr)) % 4 == 0, np.arange(len(narr)) % 4 == 2
                narr[r], narr[b] = narr[b], narr[r]
            elif c == 3:
                narr = narr[:, :, ::-1].ravel()
            else:
                raise ValueError(f"Unsupported color depth: {c}")

            return WarppedImage(narr, w, h, c)
        else:
            raise ValueError("No frame found")

def _decodeImageFromFile(path: str) -> WarppedImage:
    with open(path, "rb") as f:
        return _decodeImageBytes(f.read())

def _loadGlTexture(ctx: mgl.Context, raw: WarppedImage):
    data = raw.data.tobytes()

    return ctx.texture(
        size = (raw.width, raw.height),
        components = raw.c,
        data = data,
        dtype = "u1"
    )

def _loadGlTextureFromFile(ctx: mgl.Context, path: str):
    return _loadGlTexture(ctx, _decodeImageFromFile(path))

def _get_audio_dur(audio: np.ndarray) -> float:
    return len(audio) / AUDIO_SAMPLE_RATE / _get_channels_from_layout(AUDIO_LAYOUT)

def _get_channels_from_layout(layout: str):
    match layout:
        case "mono":
            return 1
            
        case "stereo":
            return 2
        
        case _:
            raise ValueError(f"Invalid layout: {layout}")
            
def _overlay_audio(target: np.ndarray, source: np.ndarray, t: float) -> None:
    channels = _get_channels_from_layout(AUDIO_LAYOUT)
    t = int(t * AUDIO_SAMPLE_RATE) * channels

    if len(target) <= t or t <= -len(source):
        return
    
    if t + len(source) > len(target):
        source = source[:len(target) - t]
    
    target[t:t + len(source)] += source

def _audio_to_s16(audio: np.ndarray) -> np.ndarray:
    return audio.clip(-32768, 32767).astype(np.int16)

def _audio_to_f32(audio: np.ndarray) -> np.ndarray:
    return audio.astype(np.float32) / 32768

def _export_audio(audio: np.ndarray, path: str) -> None:
    channels = _get_channels_from_layout(AUDIO_LAYOUT)
    audio = _audio_to_s16(audio)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(audio) * 2))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 0x10))
        f.write(struct.pack("<H", 1)) # PCM
        f.write(struct.pack("<H", channels))
        f.write(struct.pack("<I", AUDIO_SAMPLE_RATE))
        f.write(struct.pack("<I", AUDIO_SAMPLE_RATE * channels * 2))
        f.write(struct.pack("<H", channels * 2))
        f.write(struct.pack("<H", 2 * 8))
        f.write(b"data")
        f.write(audio.tobytes())

class _2DTransform:
    def __init__(self, matrix: typing.Optional[tuple[float, float, float, float, float, float]] = None):
        self.matrix = matrix if matrix is not None else (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        self._states = collections.deque()
    
    def resetTransform(self):
        self.matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        return self

    def setTransform(self, a: float, b: float, c: float, d: float, e: float, f: float):
        self.matrix = (a, b, c, d, e, f)
        return self

    def transform(self, a: float, b: float, c: float, d: float, e: float, f: float):
        self.matrix = (
            self.matrix[0] * a + self.matrix[2] * b,
            self.matrix[1] * a + self.matrix[3] * b,
            self.matrix[0] * c + self.matrix[2] * d,
            self.matrix[1] * c + self.matrix[3] * d,
            self.matrix[0] * e + self.matrix[2] * f + self.matrix[4],
            self.matrix[1] * e + self.matrix[3] * f + self.matrix[5]
        )
        return self
    
    def scale(self, x: float, y: float):
        self.transform(x, 0.0, 0.0, y, 0.0, 0.0)
        return self
    
    def translate(self, x: float, y: float):
        self.transform(1.0, 0.0, 0.0, 1.0, x, y)
        return self
    
    def rotate(self, angle: float):
        self.transform(math.cos(angle), math.sin(angle), -math.sin(angle), math.cos(angle), 0.0, 0.0)
        return self
    
    def rotateDegree(self, angle: float):
        self.rotate(angle * math.pi / 180.0)
        return self
    
    def getPoint(self, x: float, y: float):
        return (
            self.matrix[0] * x + self.matrix[2] * y + self.matrix[4],
            self.matrix[1] * x + self.matrix[3] * y + self.matrix[5]
        )
    
    def getRectPoints(self, x: float, y: float, width: float, height: float):
        return (
            self.getPoint(x, y),
            self.getPoint(x + width, y),
            self.getPoint(x + width, y + height),
            self.getPoint(x, y + height)
        )
    
    def getCRectPoints(self, x: float, y: float, width: float, height: float):
        x -= width / 2
        y -= height / 2
        return self.getRectPoints(x, y, width, height)
    
    def getInverse(self):
        a, b, c, d, e, f = self.matrix
        det = a * d - b * c
        inv_det = 1.0 / det if det != 0.0 else 1e9
        inv = (
            d * inv_det,
            -b * inv_det,
            -c * inv_det,
            a * inv_det,
            (c * f - d * e) * inv_det,
            (b * e - a * f) * inv_det
        )
        return _2DTransform(inv)
    
    def save():
        self._states.append(self.matrix)
    
    def restore():
        if self._states.empty():
            return

        self.matrix = self._states.pop()

    def warp(self):
        return type("_2DTransformWarp", (object, ), {
            "__enter__": lambda *_: self.save(),
            "__exit__": lambda *_: self.restore()
        })

class _GlProgManager:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.progs: dict[str, mgl.Program] = {}
        self.trans = _2DTransform()
    
    def add_from_shader(self, name: str, vertex_shader: str, fragment_shader: str):
        prog = self.ctx.program(vertex_shader, fragment_shader)
        self.progs[name] = prog
        logger.debug(f"added gl program {name}")
        return prog
    
    def add_from_file(self, name: str, vertex_shader_path: str, fragment_shader_path: str):
        with open(vertex_shader_path, "r", encoding="utf-8") as f:
            vertex_shader = f.read()
            
        with open(fragment_shader_path, "r", encoding="utf-8") as f:
            fragment_shader = f.read()
            
        return self.add_from_shader(name, vertex_shader, fragment_shader)
    
    def add_from_folder(self, name: str):
        self.add_from_file(name, f"./res/glprogs/{name}/v.glsl", f"./res/glprogs/{name}/f.glsl")
    
    def _set_v_uniforms(self, prog: mgl.Program):
        prog["trans_abcd"].value = self.trans.matrix[:4]
        prog["trans_ef"].value = self.trans.matrix[4:6]
        prog["viewport"].value = self.ctx.viewport[2:4]

    def _get_vs_typ(self, typ: list[tuple[str, type, int]]):
        tns = []

        for name, t, n in typ:
            tns.append(str(n) + {
                np.float32: "f",
            }[t])
        
        return (" ".join(tns), *map(lambda x: x[0], typ))
    
    def _norm_xy(self, x: float, y: float, w: float, h: float):
        sw, sh = self.ctx.viewport[2:4]
        x1 = (x / sw) * 2 - 1
        y1 = 1 - (y / sh) * 2
        x2 = ((x + w) / sw) * 2 - 1
        y2 = 1 - ((y + h) / sh) * 2
        return x1, y1, x2, y2

    def p_simple_texture(self, tex: moderngl.Texture, x: float, y: float, w: float, h: float, u: flaot = 0.0, v: float = 0.0, uw: float = 1.0, vh: float = 1.0):
        progn = "simple_texture"
        prog = self.progs[progn]

        dtyp = [
            ("in_pos", np.float32, 2),
            ("in_uv", np.float32, 2),
        ]

        x1, y1, x2, y2 = self._norm_xy(x, y, w, h)

        vsbytes = np.array([
            ((x1, y2), (u, v + vh),),
            ((x2, y2), (u + uw, v + vh),),
            ((x1, y1), (u, v),),
            ((x2, y1), (u + uw, v),),
        ], dtype=dtyp).tobytes()

        vertices = self.ctx.buffer(vsbytes)
        ibo = self.ctx.buffer(np.array([0, 1, 2, 1, 3, 2], dtype=np.uint8).tobytes())

        vao = self.ctx.vertex_array(
            prog,
            [(vertices, *self._get_vs_typ(dtyp))],
            index_buffer=ibo,
            index_element_size=1
        )

        tex.use(location=0)
        prog["u_tex"].value = 0
        self._set_v_uniforms(prog)

        vao.render(mgl.TRIANGLES)
    
    def p_simple_rect(self, x: float, y: float, w: float, h: float, fill_color: typing.Iterable[float, float, float, float]):
        progn = "simple_rect"
        prog = self.progs[progn]

        dtyp = [
            ("in_pos", np.float32, 2),
        ]

        x1, y1, x2, y2 = self._norm_xy(x, y, w, h)

        u, v, uw, vh = 0, 0, 1, 1
        vsbytes = np.array([
            ((x1, y2), ),
            ((x2, y2), ),
            ((x1, y1), ),
            ((x2, y1), ),
        ], dtype=dtyp).tobytes()

        vertices = self.ctx.buffer(vsbytes)
        ibo = self.ctx.buffer(np.array([0, 1, 2, 1, 3, 2], dtype=np.uint8).tobytes())

        vao = self.ctx.vertex_array(
            prog,
            [(vertices, *self._get_vs_typ(dtyp))],
            index_buffer=ibo,
            index_element_size=1
        )

        prog["u_fill_color"].value = fill_color
        self._set_v_uniforms(prog)

        vao.render(mgl.TRIANGLES)

    def __getitem__(self, name: str):
        return self.progs[name]
        
    def __setitem__(self, name: str, prog: moderngl.Program):
        self.progs[name] = prog
        
    def __delitem__(self, name: str):
        del self.progs[name]

@dataclasses.dataclass
class WarppedImage:
    data: np.ndarray
    width: int
    height: int
    c: int

class MilResourceLoader:
    def __init__(self, ctx: mgl.Context):
        self.res = {}

        for name in ("tap", "drag", "hold"):
            self.res[f"{name}"] = _loadGlTextureFromFile(ctx, self.get_res_path(f"{name}.png"))
            self.res[f"{name}_double"] = _loadGlTextureFromFile(ctx, self.get_res_path(f"{name}_double.png"))
            
            if name != "drag":
                self.res[f"ex{name}"] = _loadGlTextureFromFile(ctx, self.get_res_path(f"ex{name}.png"))
                self.res[f"ex{name}_double"] = _loadGlTextureFromFile(ctx, self.get_res_path(f"ex{name}_double.png"))

            logger.debug(f"loaded note texture {name}")
        
        self.res["clicksound"] = {
            "hit": _decodeAudioFromFile(self.get_res_path("hit.ogg")),
            "drag": _decodeAudioFromFile(self.get_res_path("drag.ogg")),
        }
    
    @staticmethod
    def get_res_path(name: str):
        return os.path.join(RES_PATH, name)
    
    def clicksound_from_type(self, typ: int):
        match typ:
            case EnumNoteType.Hit:
                return self.res["clicksound"]["hit"]
            
            case EnumNoteType.Drag:
                return self.res["clicksound"]["drag"]

            case _:
                raise ValueError(f"Invalid note type: {typ}")

@dataclasses.dataclass
class _MilChartRes:
    imageTex: mgl.Texture

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
        
        self.ctx.enable(mgl.BLEND)
        self.ctx.blend_func = (mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA, mgl.ONE, mgl.ONE)

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

        logger.info("loading resouces")
        self.resloader = MilResourceLoader(self.ctx)

        logger.info("loading gl programs")
        self.glpman = _GlProgManager(self.ctx)
        self.glpman.add_from_folder("simple_texture")
        self.glpman.add_from_folder("simple_rect")

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

            logger.info("loading chart")
            chart = MilChart(chartJson)

            logger.info("loading image")
            imageTex = _loadGlTexture(self.ctx, _decodeImageBytes(imageBytes))

            chartRes = _MilChartRes(
                imageTex = imageTex,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to read files of chart: {e}") from e
        
        try:
            logger.info("mixing audio")
            collected_notetimes = []

            for line in chart.lines:
                for note in line.notes:
                    if note.isFake:
                        continue

                    collected_notetimes.append((note.type, note.time))
            
            logger.debug(f"collected {len(collected_notetimes)} notes to mix audio")

            mixing_st = time.perf_counter()
            for note_type, note_time in collected_notetimes:
                _overlay_audio(decoededAudio, self.resloader.clicksound_from_type(note_type), note_time)
            mixing_et = time.perf_counter()
            logger.info(f"audio mixing took {mixing_et - mixing_st:.5f} s, {len(collected_notetimes) / (mixing_et - mixing_st)}note/s")
            
            if os.environ.get("DEBUG_EXPORT_MIXED_AUDIO"):
                _export_audio(decoededAudio, "debug_created_mixed_audio.wav")
        except Exception as e:
            raise RuntimeError(f"Failed to mix audio: {e}") from e
        
        try:
            v_writer = _VideoWriter(self.cfg.width, self.cfg.height, self.cfg.fps)
        except Exception as e:
            raise RuntimeError(f"Failed to open output stream: {e}") from e
        
        try:
            v_codecs = v_writer.get_encoders(True)
            a_codecs = v_writer.get_encoders(False)
            logger.debug(f"supported video encoders: {v_codecs}")
            logger.debug(f"supported audio encoders: {a_codecs}")

            if self.cfg.v_codec not in v_codecs:
                raise Exception(f"unsupported video codec: {self.cfg.v_codec}")
            
            if self.cfg.a_codec not in a_codecs:
                raise Exception(f"unsupported audio codec: {self.cfg.a_codec}")
            
            logger.info(f"using video codec: {self.cfg.v_codec}")
            logger.info(f"using audio codec: {self.cfg.a_codec}")
            
            v_writer.initialize(self.cfg.video_path, self.cfg.v_codec, self.cfg.a_codec, decoededAudio)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize output stream: {e}") from e
        
        t = 0.0
        duration = _get_audio_dur(decoededAudio)
        frame_duration = 1.0 / self.cfg.fps
        num_frames = int(duration // frame_duration) + 1

        if os.environ.get("DEBUG_MAX_NUM_FRAMES"):
            num_frames = min(int(os.environ.get("DEBUG_MAX_NUM_FRAMES")), num_frames)
            logger.debug(f"limited num_frames to {num_frames}")

        fbo = self.ctx.simple_framebuffer((self.cfg.width, self.cfg.height), dtype="u1")
        fbo.use()
        pixels = bytearray(self.cfg.width * self.cfg.height * 3)
        for i in tqdm.trange(num_frames, desc="rendering frames"):
            fbo.clear(0, 0, 0, 0)
            t = i * frame_duration
            self._render_chart(chart, t, chartRes)

            fbo.read_into(pixels, (0, 0, self.cfg.width, self.cfg.height), 3, dtype="u1")
            v_writer.write_frame(pixels, self.cfg.width, self.cfg.height)

        try:
            v_writer.release()
        except Exception as e:
            raise RuntimeError(f"Failed to close output stream: {e}") from e
    
    def _render_chart(self, chart: MilChart, t: float, cres: _MilChartRes):
        w, h = self.cfg.width, self.cfg.height
        scr_ratio = w / h
        bg_ratio = cres.imageTex.width / cres.imageTex.height

        if scr_ratio > bg_ratio:
            bg_size = (self.cfg.width, self.cfg.height * bg_ratio / scr_ratio)
        else:
            bg_size = (self.cfg.width * scr_ratio / bg_ratio, self.cfg.height)
        
        self.glpman.p_simple_texture(cres.imageTex, bg_size[0] / 2 - w / 2, bg_size[1] / 2 - h / 2, *bg_size)
        self.glpman.p_simple_rect(0, 0, w, h / 3, (0, 0, 0, chart.meta.background_dim))
        
if __name__ == "__main__":
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument("-i", "--input", type=str, required=True)
    aparser.add_argument("-o", "--output", type=str, required=True)
    aparser.add_argument("-s-w", "--width", type=int, default=1920)
    aparser.add_argument("-s-h", "--height", type=int, default=1080)
    aparser.add_argument("-f", "--fps", type=float, default=60.0)
    aparser.add_argument("-y", "--yes", action="store_true", default=False)
    aparser.add_argument("-vc", "--video-codec", type=str, default="libx264")
    aparser.add_argument("-ac", "--audio-codec", type=str, default="aac")
    aparser.add_argument("-as", "--audio-sample-rate", type=int, default=44100)

    args = aparser.parse_args()

    AUDIO_SAMPLE_RATE = args.audio_sample_rate

    if os.path.isfile(args.output):
        if not args.yes and input(f"{args.output} is exists, are you sure to continue? [y/N] ").lower() != "y":
            argparse.exit(1, "aborting by user")
        
        os.remove(args.output)

    renderer = MilRenderer()
    renderer.initialize(MilRendererConfig(
        width = args.width,
        height = args.height,
        fps = args.fps,
        input_path = args.input,
        video_path = args.output,
        v_codec = args.video_codec,
        a_codec = args.audio_codec
    ))

    renderer.run()
