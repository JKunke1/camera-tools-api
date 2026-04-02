from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Tuple
import json
from math import atan, degrees
from pathlib import Path

app = FastAPI(title="Camera Tools API", version="1.0.0")

PRESETS_PATH = Path(__file__).with_name("camera_presets.json")
with open(PRESETS_PATH, "r") as f:
    CAMERA_PRESETS = json.load(f)

FORMAT_DEFAULTS: Dict[str, Tuple[int, int]] = {
    "4K": (4096, 2160),
    "6K": (6144, 3160),
    "8K": (7680, 4320),
    "12K": (12288, 6480),
    "18K": (18024, 17592),
}

FULL_FRAME_DIAGONAL_MM = 43.266615305567875  # 36x24mm


def resolve_resolution(width: Optional[int], height: Optional[int], format_label: Optional[str]) -> Tuple[int, int, list]:
    assumptions = []
    if width and height:
        return width, height, assumptions
    if format_label:
        if format_label not in FORMAT_DEFAULTS:
            raise HTTPException(status_code=400, detail=f"Unsupported format label: {format_label}")
        w, h = FORMAT_DEFAULTS[format_label]
        assumptions.append(f"Used default resolution for {format_label}: {w} x {h}")
        return w, h, assumptions
    raise HTTPException(status_code=400, detail="Provide width and height, or format_label")


def sensor_diagonal(w_mm: float, h_mm: float) -> float:
    return (w_mm ** 2 + h_mm ** 2) ** 0.5


def coc_from_sensor(sensor_width_mm: float, sensor_height_mm: float) -> float:
    return sensor_diagonal(sensor_width_mm, sensor_height_mm) / 1500.0


class RecordTimeRequest(BaseModel):
    width: Optional[int] = Field(default=None)
    height: Optional[int] = Field(default=None)
    format_label: Optional[str] = Field(default=None)
    frame_rate: float = Field(gt=0)
    bit_depth: int = Field(gt=0)
    media_size_gb: float = Field(gt=0)


class FileSizeRequest(BaseModel):
    width: Optional[int] = None
    height: Optional[int] = None
    format_label: Optional[str] = None
    duration_sec: float = Field(gt=0)
    frame_rate: float = Field(gt=0)
    bit_depth: int = Field(default=12, gt=0)
    codec: str = Field(default="uncompressed")


class RenderTimeRequest(BaseModel):
    width: Optional[int] = None
    height: Optional[int] = None
    format_label: Optional[str] = None
    duration_sec: float = Field(gt=0)
    frame_rate: float = Field(gt=0)
    codec: str = Field(default="uncompressed")
    system_speed_factor: float = Field(default=1.0, gt=0)


class CameraPresetRequest(BaseModel):
    camera_name: str


class CropFactorRequest(BaseModel):
    camera_name: Optional[str] = None
    sensor_width_mm: Optional[float] = None
    sensor_height_mm: Optional[float] = None
    reference_width_mm: float = 36.0
    reference_height_mm: float = 24.0


class FieldOfViewRequest(BaseModel):
    camera_name: Optional[str] = None
    sensor_width_mm: Optional[float] = None
    sensor_height_mm: Optional[float] = None
    focal_length_mm: float = Field(gt=0)


class DOFRequest(BaseModel):
    camera_name: Optional[str] = None
    sensor_width_mm: Optional[float] = None
    sensor_height_mm: Optional[float] = None
    focal_length_mm: float = Field(gt=0)
    f_stop: float = Field(gt=0)
    focus_distance_m: float = Field(gt=0)


class LensEquivalencyRequest(BaseModel):
    source_camera_name: Optional[str] = None
    source_sensor_width_mm: Optional[float] = None
    source_sensor_height_mm: Optional[float] = None
    target_camera_name: Optional[str] = None
    target_sensor_width_mm: Optional[float] = None
    target_sensor_height_mm: Optional[float] = None
    focal_length_mm: float = Field(gt=0)


@app.get("/health")
def health():
    return {
        "status": "ok",
    }


@app.post("/estimate-record-time")
def estimate_record_time(req: RecordTimeRequest):
    width, height, assumptions = resolve_resolution(req.width, req.height, req.format_label)
    pixels_per_frame = width * height
    bits_per_frame = pixels_per_frame * req.bit_depth
    bytes_per_frame = bits_per_frame / 8
    bytes_per_second = bytes_per_frame * req.frame_rate
    total_media_bytes = req.media_size_gb * 1024**3
    total_seconds = total_media_bytes / bytes_per_second
    total_minutes = total_seconds / 60
    assumptions.append("Theoretical estimate based on simplified uncompressed storage assumption.")
    return {
        "estimated_record_time_minutes": round(total_minutes, 2),
        "estimated_record_time_seconds": round(total_seconds, 2),
        "resolution": {"width": width, "height": height},
        "pixels_per_frame": pixels_per_frame,
        "bits_per_frame": bits_per_frame,
        "bytes_per_frame": int(bytes_per_frame),
        "bytes_per_second": int(bytes_per_second),
        "total_media_bytes": int(total_media_bytes),
        "assumptions": assumptions,
        "limitation_note": "Actual record time can vary significantly depending on codec, compression ratio, and camera format."
    }


@app.post("/estimate-file-size")
def estimate_file_size(req: FileSizeRequest):
    width, height, assumptions = resolve_resolution(req.width, req.height, req.format_label)
    bits_per_pixel = {
        "uncompressed": req.bit_depth,
        "prores": 8,
        "raw": 12,
    }.get(req.codec.lower(), req.bit_depth)
    total_frames = req.duration_sec * req.frame_rate
    bytes_per_frame = width * height * bits_per_pixel / 8
    total_gb = total_frames * bytes_per_frame / 1024**3
    assumptions.append(f"Codec assumption used: {req.codec}")
    return {
        "estimated_size_gb": round(total_gb, 2),
        "total_frames": round(total_frames, 2),
        "bytes_per_frame": round(bytes_per_frame, 2),
        "assumptions": assumptions,
        "limitation_note": "File size is an estimate and depends on codec behavior and implementation."
    }


@app.post("/estimate-render-time")
def estimate_render_time(req: RenderTimeRequest):
    width, height, assumptions = resolve_resolution(req.width, req.height, req.format_label)
    bits_per_pixel = {
        "uncompressed": 16,
        "prores": 8,
        "raw": 12,
    }.get(req.codec.lower(), 12)
    total_frames = req.duration_sec * req.frame_rate
    estimated_size_gb = total_frames * width * height * bits_per_pixel / 8 / 1024**3
    estimated_render_time_minutes = (total_frames / req.system_speed_factor) / 60
    assumptions.append(f"Codec assumption used: {req.codec}")
    assumptions.append(f"System speed factor used: {req.system_speed_factor}")
    return {
        "estimated_render_time_minutes": round(estimated_render_time_minutes, 2),
        "estimated_size_gb": round(estimated_size_gb, 2),
        "total_frames": round(total_frames, 2),
        "assumptions": assumptions,
        "limitation_note": "Render time is an estimate and depends on hardware, software, and workflow."
    }


@app.post("/camera-preset-lookup")
def camera_preset_lookup(req: CameraPresetRequest):
    for name, preset in CAMERA_PRESETS.items():
        if name.lower() == req.camera_name.lower():
            return {
                "camera_name": name,
                **preset,
            }
    raise HTTPException(status_code=404, detail="That camera is not currently in the uploaded preset list.")


@app.post("/crop-factor")
def crop_factor(req: CropFactorRequest):
    assumptions = []
    if req.camera_name:
        preset = CAMERA_PRESETS.get(req.camera_name)
        if not preset:
            raise HTTPException(status_code=404, detail="That camera is not currently in the uploaded preset list.")
        sw = preset["sensor_width_mm"]
        sh = preset["sensor_height_mm"]
    else:
        if req.sensor_width_mm is None or req.sensor_height_mm is None:
            raise HTTPException(status_code=400, detail="Provide a camera_name or sensor_width_mm and sensor_height_mm")
        sw, sh = req.sensor_width_mm, req.sensor_height_mm
    crop = sensor_diagonal(req.reference_width_mm, req.reference_height_mm) / sensor_diagonal(sw, sh)
    return {
        "crop_factor": round(crop, 4),
        "sensor_width_mm": sw,
        "sensor_height_mm": sh,
        "reference_width_mm": req.reference_width_mm,
        "reference_height_mm": req.reference_height_mm,
        "assumptions": assumptions,
    }


@app.post("/field-of-view")
def field_of_view(req: FieldOfViewRequest):
    if req.camera_name:
        preset = CAMERA_PRESETS.get(req.camera_name)
        if not preset:
            raise HTTPException(status_code=404, detail="That camera is not currently in the uploaded preset list.")
        sw = preset["sensor_width_mm"]
        sh = preset["sensor_height_mm"]
    else:
        if req.sensor_width_mm is None or req.sensor_height_mm is None:
            raise HTTPException(status_code=400, detail="Provide a camera_name or sensor_width_mm and sensor_height_mm")
        sw, sh = req.sensor_width_mm, req.sensor_height_mm
    h_fov = degrees(2 * atan(sw / (2 * req.focal_length_mm)))
    v_fov = degrees(2 * atan(sh / (2 * req.focal_length_mm)))
    d_fov = degrees(2 * atan(sensor_diagonal(sw, sh) / (2 * req.focal_length_mm)))
    return {
        "horizontal_fov_degrees": round(h_fov, 2),
        "vertical_fov_degrees": round(v_fov, 2),
        "diagonal_fov_degrees": round(d_fov, 2),
        "sensor_width_mm": sw,
        "sensor_height_mm": sh,
        "focal_length_mm": req.focal_length_mm,
    }


@app.post("/depth-of-field")
def depth_of_field(req: DOFRequest):
    if req.camera_name:
        preset = CAMERA_PRESETS.get(req.camera_name)
        if not preset:
            raise HTTPException(status_code=404, detail="That camera is not currently in the uploaded preset list.")
        sw = preset["sensor_width_mm"]
        sh = preset["sensor_height_mm"]
    else:
        if req.sensor_width_mm is None or req.sensor_height_mm is None:
            raise HTTPException(status_code=400, detail="Provide a camera_name or sensor_width_mm and sensor_height_mm")
        sw, sh = req.sensor_width_mm, req.sensor_height_mm

    coc = coc_from_sensor(sw, sh)
    f = req.focal_length_mm
    N = req.f_stop
    s = req.focus_distance_m * 1000.0
    H = (f * f) / (N * coc) + f
    near_mm = (H * s) / (H + (s - f))
    if H > s:
        far_mm = (H * s) / (H - (s - f))
        far_m = far_mm / 1000.0
        total = far_m - near_mm / 1000.0
    else:
        far_m = None
        total = None
    return {
        "circle_of_confusion_mm": round(coc, 4),
        "hyperfocal_distance_m": round(H / 1000.0, 3),
        "near_focus_distance_m": round(near_mm / 1000.0, 3),
        "far_focus_distance_m": None if far_m is None else round(far_m, 3),
        "total_depth_of_field_m": None if total is None else round(total, 3),
        "sensor_width_mm": sw,
        "sensor_height_mm": sh,
        "focal_length_mm": f,
        "f_stop": N,
        "focus_distance_m": req.focus_distance_m,
    }


@app.post("/lens-equivalency")
def lens_equivalency(req: LensEquivalencyRequest):
    if req.source_camera_name:
        source = CAMERA_PRESETS.get(req.source_camera_name)
        if not source:
            raise HTTPException(status_code=404, detail="That camera is not currently in the uploaded preset list.")
        ssw, ssh = source["sensor_width_mm"], source["sensor_height_mm"]
    else:
        if req.source_sensor_width_mm is None or req.source_sensor_height_mm is None:
            raise HTTPException(status_code=400, detail="Provide source camera or source sensor dimensions")
        ssw, ssh = req.source_sensor_width_mm, req.source_sensor_height_mm

    if req.target_camera_name:
        target = CAMERA_PRESETS.get(req.target_camera_name)
        if not target:
            raise HTTPException(status_code=404, detail="That camera is not currently in the uploaded preset list.")
        tsw, tsh = target["sensor_width_mm"], target["sensor_height_mm"]
    else:
        if req.target_sensor_width_mm is None or req.target_sensor_height_mm is None:
            raise HTTPException(status_code=400, detail="Provide target camera or target sensor dimensions")
        tsw, tsh = req.target_sensor_width_mm, req.target_sensor_height_mm

    ratio = sensor_diagonal(tsw, tsh) / sensor_diagonal(ssw, ssh)
    target_focal = req.focal_length_mm * ratio
    return {
        "source_focal_length_mm": req.focal_length_mm,
        "equivalent_target_focal_length_mm": round(target_focal, 2),
        "source_sensor": {"width_mm": ssw, "height_mm": ssh},
        "target_sensor": {"width_mm": tsw, "height_mm": tsh},
        "note": "Equivalent match is based on approximate field of view from sensor size comparison."
    }
