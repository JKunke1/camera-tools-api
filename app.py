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

CAMERA_ALIASES = {
    "sony venice": "Sony Venice",
    "venice": "Sony Venice",

    "sony venice 2": "Sony Venice 2 8.6K",
    "venice 2": "Sony Venice 2 8.6K",
    "venice2": "Sony Venice 2 8.6K",
    "sony venice 2 8.6k": "Sony Venice 2 8.6K",

    "sony burano": "Sony BURANO",
    "burano": "Sony BURANO",

    "alexa 35": "Alexa 35",
    "alexa mini": "Alexa Mini",
    "alexa lf": "Alexa LF",
    "alexa mini lf": "Alexa Mini LF",
    "alexa 65": "Alexa 65",

    "red monstro 8k vv": "RED Monstro 8K VV",
    "red v raptor 8k vv": "RED V-Raptor 8K VV",
    "red v-raptor 8k vv": "RED V-Raptor 8K VV",
    "red v raptor x 8k vv": "RED V-Raptor [X] 8K VV",
    "red v-raptor [x] 8k vv": "RED V-Raptor [X] 8K VV",
    "red v raptor xl x 8k vv": "RED V-Raptor XL [X] 8K VV",
    "red v-raptor xl [x] 8k vv": "RED V-Raptor XL [X] 8K VV",

    "red komodo": "RED Komodo 6K S35",
    "red komodo 6k s35": "RED Komodo 6K S35",
    "komodo": "RED Komodo 6K S35",

    "red komodo x": "RED Komodo-X 6K S35",
    "red komodo-x": "RED Komodo-X 6K S35",
    "red komodo x 6k s35": "RED Komodo-X 6K S35",
    "red komodo-x 6k s35": "RED Komodo-X 6K S35",
    "komodo x": "RED Komodo-X 6K S35",
    "komodo-x": "RED Komodo-X 6K S35",
}

MODE_ALIASES_BY_CAMERA = {
    "Sony Venice 2 8.6K": {
        "ff 8.6k 3:2": "8.6K 3:2",
        "full frame 8.6k 3:2": "8.6K 3:2",
        "full-frame 8.6k 3:2": "8.6K 3:2",

        "ff 8.6k 17:9": "8.6K 17:9",
        "full frame 8.6k 17:9": "8.6K 17:9",
        "full-frame 8.6k 17:9": "8.6K 17:9",

        "8.6k 3:2": "8.6K 3:2",
        "8.6k 17:9": "8.6K 17:9",
        "8.2k 17:9": "8.2K 17:9",
        "8.1k 16:9": "8.1K 16:9",
        "7.6k 16:9": "7.6K 16:9",
        "8.2k 2.39:1": "8.2K 2.39:1",
        "5.8k 17:9 s35": "5.8K 17:9 S35",
        "5.8k 6:5 s35": "5.8K 6:5 S35",
        "5.8k 4:3 s35": "5.8K 4:3 S35",
        "5.5k 2.39:1 s35": "5.5K 2.39:1 S35",
        "5.4k 16:9 s35": "5.4K 16:9 S35",
    },
    "Sony BURANO": {
        "ff 8.6k 16:9": "FF 8.6K 16:9",
        "full frame 8.6k 16:9": "FF 8.6K 16:9",
        "full-frame 8.6k 16:9": "FF 8.6K 16:9",
        "8.6k 16:9": "FF 8.6K 16:9",

        "ff 8.6k 17:9": "FF 8.6K 17:9",
        "full frame 8.6k 17:9": "FF 8.6K 17:9",
        "full-frame 8.6k 17:9": "FF 8.6K 17:9",
        "8.6k 17:9": "FF 8.6K 17:9",

        "ffc 6k 16:9": "FFc 6K 16:9",
        "ffc 6k 17:9": "FFc 6K 17:9",
        "s35 5.8k 16:9": "S35 5.8K 16:9",
        "s35 5.8k 17:9": "S35 5.8K 17:9",
        "s35 4.3k 4:3": "S35 4.3K 4:3",
        "s35c 4k 17:9": "S35c 4K 17:9",
        "s35 1.9k 16:9": "S35 1.9K 16:9",
    },
}

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


def media_to_bytes(media_size: float, media_unit: str = "GB", storage_base: str = "decimal") -> float:
    media_unit = media_unit.upper()
    storage_base = storage_base.lower()

    if media_unit not in ("GB", "TB"):
        raise HTTPException(status_code=400, detail="media_unit must be 'GB' or 'TB'")

    if storage_base not in ("decimal", "binary"):
        raise HTTPException(status_code=400, detail="storage_base must be 'decimal' or 'binary'")

    if storage_base == "decimal":
        if media_unit == "GB":
            return media_size * 1_000_000_000
        return media_size * 1_000_000_000_000

    if media_unit == "GB":
        return media_size * 1024**3
    return media_size * 1024**4


def format_duration(total_seconds: float) -> str:
    total_seconds = max(0, int(total_seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    if hours > 0:
        return f"{hours} hr {minutes} min {seconds} sec"
    if minutes > 0:
        return f"{minutes} min {seconds} sec"
    return f"{seconds} sec"


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().replace("-", " ").split())


def normalize_camera_name(camera_name: str) -> str:
    key = normalize_text(camera_name)
    return CAMERA_ALIASES.get(key, camera_name)


def normalize_mode_name(camera_name: str, mode_name: str) -> str:
    key = normalize_text(mode_name)
    aliases = MODE_ALIASES_BY_CAMERA.get(camera_name, {})
    return aliases.get(key, mode_name)


def find_camera_preset(camera_name: str) -> Tuple[str, dict]:
    normalized_camera = normalize_camera_name(camera_name)
    normalized_camera_key = normalize_text(normalized_camera)

    for name, preset in CAMERA_PRESETS.items():
        if normalize_text(name) == normalized_camera_key:
            return name, preset

    raise HTTPException(
        status_code=404,
        detail="That camera is not currently available in the live preset data."
    )


def resolve_camera_mode(camera_name: str, mode_name: Optional[str] = None) -> Tuple[str, Optional[str], float, float, list, dict]:
    assumptions = []
    canonical_name, preset = find_camera_preset(camera_name)

    # Backward compatibility with old flat preset structure
    if "sensor_width_mm" in preset and "sensor_height_mm" in preset:
        assumptions.append("Used flat preset sensor dimensions.")
        if mode_name:
            assumptions.append(f"Ignored requested mode '{mode_name}' because this preset does not define modes.")
        return (
            canonical_name,
            None,
            preset["sensor_width_mm"],
            preset["sensor_height_mm"],
            assumptions,
            preset,
        )

    modes = preset.get("modes")
    if not modes or not isinstance(modes, dict):
        raise HTTPException(status_code=500, detail=f"Preset for {canonical_name} is missing a valid modes block.")

    resolved_mode_name = mode_name or preset.get("default_mode")
    if not resolved_mode_name:
        raise HTTPException(status_code=500, detail=f"Preset for {canonical_name} has no default_mode defined.")

    normalized_mode_name = normalize_mode_name(canonical_name, resolved_mode_name)

    matched_mode_name = None
    mode = None
    for candidate_name, candidate in modes.items():
        if candidate_name.lower() == normalized_mode_name.lower():
            matched_mode_name = candidate_name
            mode = candidate
            break

    if mode is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Mode '{resolved_mode_name}' is not defined for camera '{canonical_name}'.",
                "supported_modes": list(modes.keys())
            }
        )

    if "sensor_width_mm" not in mode or "sensor_height_mm" not in mode:
        raise HTTPException(
            status_code=500,
            detail=f"Mode '{matched_mode_name}' for camera '{canonical_name}' is missing sensor dimensions."
        )

    assumptions.append(f"Used preset camera '{canonical_name}' in mode '{matched_mode_name}'.")
    if resolved_mode_name != matched_mode_name:
        assumptions.append(f"Normalized requested mode '{resolved_mode_name}' to '{matched_mode_name}'.")
    if matched_mode_name == preset.get("default_mode"):
        assumptions.append("Used default mode for this camera preset.")

    return (
        canonical_name,
        matched_mode_name,
        mode["sensor_width_mm"],
        mode["sensor_height_mm"],
        assumptions,
        mode,
    )


class RecordTimeRequest(BaseModel):
    width: Optional[int] = Field(default=None)
    height: Optional[int] = Field(default=None)
    format_label: Optional[str] = Field(default=None)

    frame_rate: float = Field(gt=0)
    bit_depth: int = Field(gt=0)

    media_size: float = Field(gt=0)
    media_unit: str = Field(default="GB")
    compression_ratio: float = Field(default=1, ge=1)
    usable_media_percent: float = Field(default=95, gt=0, le=100)
    card_count: int = Field(default=1, ge=1)
    storage_base: str = Field(default="decimal")
    playback_fps: Optional[float] = Field(default=None, gt=0)


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
    mode_name: Optional[str] = None


class CropFactorRequest(BaseModel):
    camera_name: Optional[str] = None
    mode_name: Optional[str] = None
    sensor_width_mm: Optional[float] = None
    sensor_height_mm: Optional[float] = None
    reference_width_mm: float = 36.0
    reference_height_mm: float = 24.0


class FieldOfViewRequest(BaseModel):
    camera_name: Optional[str] = None
    mode_name: Optional[str] = None
    sensor_width_mm: Optional[float] = None
    sensor_height_mm: Optional[float] = None
    focal_length_mm: float = Field(gt=0)


class DOFRequest(BaseModel):
    camera_name: Optional[str] = None
    mode_name: Optional[str] = None
    sensor_width_mm: Optional[float] = None
    sensor_height_mm: Optional[float] = None
    focal_length_mm: float = Field(gt=0)
    f_stop: float = Field(gt=0)
    focus_distance_m: float = Field(gt=0)


class LensEquivalencyRequest(BaseModel):
    source_camera_name: Optional[str] = None
    source_mode_name: Optional[str] = None
    source_sensor_width_mm: Optional[float] = None
    source_sensor_height_mm: Optional[float] = None
    target_camera_name: Optional[str] = None
    target_mode_name: Optional[str] = None
    target_sensor_width_mm: Optional[float] = None
    target_sensor_height_mm: Optional[float] = None
    focal_length_mm: float = Field(gt=0)


@app.get("/")
def root():
    return {
        "name": "Camera Tools API",
        "status": "live"
    }


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
    raw_bytes_per_second = bytes_per_frame * req.frame_rate
    effective_bytes_per_second = raw_bytes_per_second / req.compression_ratio

    media_bytes_per_card = media_to_bytes(
        media_size=req.media_size,
        media_unit=req.media_unit,
        storage_base=req.storage_base
    )

    usable_bytes_per_card = media_bytes_per_card * (req.usable_media_percent / 100)
    total_usable_bytes = usable_bytes_per_card * req.card_count

    runtime_per_card_seconds = usable_bytes_per_card / effective_bytes_per_second
    runtime_total_seconds = total_usable_bytes / effective_bytes_per_second

    data_rate_mb_per_second = effective_bytes_per_second / 1_000_000
    storage_gb_per_minute = (effective_bytes_per_second * 60) / 1_000_000_000
    storage_gb_per_hour = (effective_bytes_per_second * 3600) / 1_000_000_000

    assumptions.append("Estimated using a simplified per-pixel storage model with optional compression ratio.")
    assumptions.append(f"Compression ratio used: {req.compression_ratio}:1")
    assumptions.append(f"Usable media percentage used: {req.usable_media_percent}%")
    assumptions.append(f"Storage base used: {req.storage_base}")
    assumptions.append(f"Card count used: {req.card_count}")

    if req.playback_fps is not None:
        assumptions.append(f"Playback fps provided for display/reference: {req.playback_fps}")

    return {
        "estimated_runtime_per_card_seconds": round(runtime_per_card_seconds, 2),
        "estimated_runtime_total_seconds": round(runtime_total_seconds, 2),
        "estimated_runtime_per_card_readable": format_duration(runtime_per_card_seconds),
        "estimated_runtime_total_readable": format_duration(runtime_total_seconds),

        "estimated_data_rate_bytes_per_second": round(effective_bytes_per_second, 2),
        "estimated_data_rate_mb_per_second": round(data_rate_mb_per_second, 2),
        "estimated_storage_gb_per_minute": round(storage_gb_per_minute, 2),
        "estimated_storage_gb_per_hour": round(storage_gb_per_hour, 2),

        "resolution": {"width": width, "height": height},
        "capture_fps": req.frame_rate,
        "playback_fps": req.playback_fps,
        "bit_depth": req.bit_depth,
        "compression_ratio": req.compression_ratio,
        "media_per_card": {
            "value": req.media_size,
            "unit": req.media_unit
        },
        "usable_media_percent": req.usable_media_percent,
        "card_count": req.card_count,
        "storage_base": req.storage_base,

        "debug": {
            "pixels_per_frame": pixels_per_frame,
            "bits_per_frame": bits_per_frame,
            "bytes_per_frame": round(bytes_per_frame, 2),
            "raw_bytes_per_second": round(raw_bytes_per_second, 2),
            "effective_bytes_per_second": round(effective_bytes_per_second, 2),
            "media_bytes_per_card": round(media_bytes_per_card, 2),
            "usable_bytes_per_card": round(usable_bytes_per_card, 2),
            "total_usable_bytes": round(total_usable_bytes, 2),
        },

        "assumptions": assumptions,
        "limitation_note": "Actual record time can vary depending on codec behavior, manufacturer implementation, overhead, and workflow."
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
    canonical_name, preset = find_camera_preset(req.camera_name)

    if req.mode_name:
        _, matched_mode_name, sw, sh, assumptions, mode = resolve_camera_mode(req.camera_name, req.mode_name)
        response = {
            "camera_name": canonical_name,
            "mode_name": matched_mode_name,
            "sensor_width_mm": sw,
            "sensor_height_mm": sh,
            "assumptions": assumptions,
        }
        if isinstance(mode, dict):
            for key, value in mode.items():
                if key not in response:
                    response[key] = value
        return response

    # flat preset support
    if "sensor_width_mm" in preset and "sensor_height_mm" in preset:
        return {
            "camera_name": canonical_name,
            "sensor_width_mm": preset["sensor_width_mm"],
            "sensor_height_mm": preset["sensor_height_mm"],
        }

    default_mode = preset.get("default_mode")
    modes = preset.get("modes", {})
    if not default_mode or default_mode not in modes:
        raise HTTPException(status_code=500, detail=f"Preset for {canonical_name} has an invalid default mode.")

    default_mode_data = modes[default_mode]
    response = {
        "camera_name": canonical_name,
        "default_mode": default_mode,
        "available_modes": list(modes.keys()),
        "sensor_width_mm": default_mode_data["sensor_width_mm"],
        "sensor_height_mm": default_mode_data["sensor_height_mm"],
    }
    for key, value in default_mode_data.items():
        if key not in response:
            response[key] = value
    return response


@app.post("/crop-factor")
def crop_factor(req: CropFactorRequest):
    assumptions = []
    if req.camera_name:
        _, matched_mode_name, sw, sh, mode_assumptions, _ = resolve_camera_mode(req.camera_name, req.mode_name)
        assumptions.extend(mode_assumptions)
    else:
        if req.sensor_width_mm is None or req.sensor_height_mm is None:
            raise HTTPException(status_code=400, detail="Provide a camera_name or sensor_width_mm and sensor_height_mm")
        sw, sh = req.sensor_width_mm, req.sensor_height_mm
        matched_mode_name = None

    crop = sensor_diagonal(req.reference_width_mm, req.reference_height_mm) / sensor_diagonal(sw, sh)
    return {
        "crop_factor": round(crop, 4),
        "sensor_width_mm": sw,
        "sensor_height_mm": sh,
        "mode_name": matched_mode_name,
        "reference_width_mm": req.reference_width_mm,
        "reference_height_mm": req.reference_height_mm,
        "assumptions": assumptions,
    }


@app.post("/field-of-view")
def field_of_view(req: FieldOfViewRequest):
    assumptions = []
    if req.camera_name:
        _, matched_mode_name, sw, sh, mode_assumptions, _ = resolve_camera_mode(req.camera_name, req.mode_name)
        assumptions.extend(mode_assumptions)
    else:
        if req.sensor_width_mm is None or req.sensor_height_mm is None:
            raise HTTPException(status_code=400, detail="Provide a camera_name or sensor_width_mm and sensor_height_mm")
        sw, sh = req.sensor_width_mm, req.sensor_height_mm
        matched_mode_name = None

    h_fov = degrees(2 * atan(sw / (2 * req.focal_length_mm)))
    v_fov = degrees(2 * atan(sh / (2 * req.focal_length_mm)))
    d_fov = degrees(2 * atan(sensor_diagonal(sw, sh) / (2 * req.focal_length_mm)))
    return {
        "horizontal_fov_degrees": round(h_fov, 2),
        "vertical_fov_degrees": round(v_fov, 2),
        "diagonal_fov_degrees": round(d_fov, 2),
        "sensor_width_mm": sw,
        "sensor_height_mm": sh,
        "mode_name": matched_mode_name,
        "focal_length_mm": req.focal_length_mm,
        "assumptions": assumptions,
    }


@app.post("/depth-of-field")
def depth_of_field(req: DOFRequest):
    assumptions = []
    if req.camera_name:
        _, matched_mode_name, sw, sh, mode_assumptions, _ = resolve_camera_mode(req.camera_name, req.mode_name)
        assumptions.extend(mode_assumptions)
    else:
        if req.sensor_width_mm is None or req.sensor_height_mm is None:
            raise HTTPException(status_code=400, detail="Provide a camera_name or sensor_width_mm and sensor_height_mm")
        sw, sh = req.sensor_width_mm, req.sensor_height_mm
        matched_mode_name = None

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
        "mode_name": matched_mode_name,
        "focal_length_mm": f,
        "f_stop": N,
        "focus_distance_m": req.focus_distance_m,
        "assumptions": assumptions,
    }


@app.post("/lens-equivalency")
def lens_equivalency(req: LensEquivalencyRequest):
    source_assumptions = []
    target_assumptions = []

    if req.source_camera_name:
        _, source_mode_name, ssw, ssh, source_assumptions, _ = resolve_camera_mode(
            req.source_camera_name, req.source_mode_name
        )
    else:
        if req.source_sensor_width_mm is None or req.source_sensor_height_mm is None:
            raise HTTPException(status_code=400, detail="Provide source camera or source sensor dimensions")
        ssw, ssh = req.source_sensor_width_mm, req.source_sensor_height_mm
        source_mode_name = None

    if req.target_camera_name:
        _, target_mode_name, tsw, tsh, target_assumptions, _ = resolve_camera_mode(
            req.target_camera_name, req.target_mode_name
        )
    else:
        if req.target_sensor_width_mm is None or req.target_sensor_height_mm is None:
            raise HTTPException(status_code=400, detail="Provide target camera or target sensor dimensions")
        tsw, tsh = req.target_sensor_width_mm, req.target_sensor_height_mm
        target_mode_name = None

    ratio = sensor_diagonal(tsw, tsh) / sensor_diagonal(ssw, ssh)
    target_focal = req.focal_length_mm * ratio
    return {
        "source_focal_length_mm": req.focal_length_mm,
        "equivalent_target_focal_length_mm": round(target_focal, 2),
        "source_sensor": {"width_mm": ssw, "height_mm": ssh, "mode_name": source_mode_name},
        "target_sensor": {"width_mm": tsw, "height_mm": tsh, "mode_name": target_mode_name},
        "assumptions": source_assumptions + target_assumptions,
        "note": "Equivalent match is based on approximate field of view from sensor size comparison."
    }
