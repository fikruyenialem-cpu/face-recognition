"""Configuration loading and saving."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .constants import (
    DEFAULT_DET_SIZE,
    DEFAULT_LICENSE_STATUS,
    DEFAULT_MODEL_NAME,
    DEFAULT_PROVIDER,
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
)
from .paths import default_config_path, default_workspace, ensure_workspace, expand_path


@dataclass
class AppConfig:
    workspace_path: str = str(default_workspace())
    database_path: str = ""
    crop_dir: str = ""
    export_dir: str = ""
    report_dir: str = ""
    log_dir: str = ""
    cache_dir: str = ""
    mode: str = "Personal / Research"
    model_name: str = DEFAULT_MODEL_NAME
    model_root: str = "~/.insightface"
    custom_model_dir: str = ""
    provider: str = DEFAULT_PROVIDER
    det_size: list[int] = None  # type: ignore[assignment]
    recognition_threshold: float = DEFAULT_THRESHOLD
    default_top_k: int = DEFAULT_TOP_K
    min_detection_score: float = 0.45
    min_face_size: int = 32
    save_crops: bool = True
    save_recognition_logs: bool = True
    anonymize_report_paths: bool = False
    batch_worker_count: int = 2
    video_frame_interval: int = 10
    camera_frame_skip: int = 3
    ui_theme: str = "azure_lab"
    ui_language: str = "system"
    ui_default_mode: str = "face_verification"
    ui_last_mode: str = "face_verification"
    ui_last_page_face_verification: str = "verification"
    ui_last_page_album_management: str = "album"
    ui_last_page_face_swap: str = "image_face_swap"
    ui_last_page_enterprise_evaluation: str = "enterprise_dashboard"
    ui_window_width: int = 1320
    ui_window_height: int = 860
    ui_sidebar_width: int = 240
    ui_settings_dialog_last_tab: int = 0
    ui_sidebar_compact: bool = False
    ui_show_status_chips: bool = True
    license_status: str = DEFAULT_LICENSE_STATUS
    safe_mode: bool = False
    auto_load_model: bool = True
    swap_model_path: str = ""
    enable_gfpgan: bool = False
    gfpgan_model_path: str = ""

    def __post_init__(self) -> None:
        if self.det_size is None:
            self.det_size = [DEFAULT_DET_SIZE[0], DEFAULT_DET_SIZE[1]]
        self.apply_workspace_defaults()

    def apply_workspace_defaults(self) -> None:
        paths = ensure_workspace(self.workspace_path)
        self.workspace_path = str(paths["workspace"])
        self.database_path = self.database_path or str(paths["database"])
        self.crop_dir = self.crop_dir or str(paths["crops"])
        self.export_dir = self.export_dir or str(paths["exports"])
        self.report_dir = self.report_dir or str(paths["reports"])
        self.log_dir = self.log_dir or str(paths["logs"])
        self.cache_dir = self.cache_dir or str(paths["cache"])

    @property
    def det_size_tuple(self) -> tuple[int, int]:
        return int(self.det_size[0]), int(self.det_size[1])

    @property
    def det_size_label(self) -> str:
        width, height = self.det_size_tuple
        if width <= 0 or height <= 0:
            return "Auto"
        return f"{width}x{height}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


def load_config(path: Optional[str | Path] = None) -> tuple[AppConfig, bool]:
    config_path = expand_path(path) if path else default_config_path()
    if not config_path.exists():
        if path is not None:
            return AppConfig(workspace_path=str(config_path.parent)), False
        return AppConfig(), False
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return AppConfig.from_dict(data), True
    except Exception:
        return AppConfig(), False


def save_config(config: AppConfig, path: Optional[str | Path] = None) -> Path:
    config.apply_workspace_defaults()
    config_path = expand_path(path) if path else Path(config.workspace_path) / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return config_path
