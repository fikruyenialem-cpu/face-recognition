"""Application bootstrap for InsightFace Evaluation Studio."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Iterable, Optional

from pathlib import Path

from .core.config import AppConfig, load_config, save_config
from .core.face_engine import FaceEngine, is_cuda_provider_available, providers_from_choice
from .core.logging import setup_logging
from .core.storage import Storage


@dataclass
class StudioContext:
    config: AppConfig
    config_exists: bool
    storage: Storage
    engine: FaceEngine
    log_file: str
    runtime_safe_mode: bool = False


def _unique_existing_paths(paths: Iterable[Path]) -> list[Path]:
    seen = set()
    unique_paths = []
    for path in paths:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists():
            unique_paths.append(path)
    return unique_paths


def qt_plugin_root_candidates() -> list[Path]:
    """Return plausible Qt plugin roots for PySide6 wheel and conda layouts."""

    try:
        import PySide6
    except ImportError:
        return []

    candidates = []
    for package_dir_raw in getattr(PySide6, "__path__", []):
        package_dir = Path(package_dir_raw)
        candidates.extend(
            [
                package_dir / "Qt" / "plugins",
                package_dir / "plugins",
            ]
        )

    try:
        from PySide6.QtCore import QLibraryInfo

        candidates.append(Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)))
    except Exception:
        pass

    candidates.extend(
        [
            Path(sys.prefix) / "plugins",
            Path(sys.prefix) / "Library" / "plugins",
            Path(sys.prefix) / "lib" / "qt6" / "plugins",
        ]
    )
    return _unique_existing_paths(candidates)


def configure_qt_plugin_paths() -> None:
    """Point Qt at PySide6-Essentials plugins when the PySide6 meta package is absent."""

    for plugins in qt_plugin_root_candidates():
        platforms = plugins / "platforms"
        if platforms.exists():
            if not os.environ.get("QT_PLUGIN_PATH"):
                os.environ["QT_PLUGIN_PATH"] = str(plugins)
            if not os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
                os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms)
            return


def create_context(args=None) -> StudioContext:
    config_path = None
    runtime_safe_mode = bool(args is not None and getattr(args, "safe_mode", False))
    if args is not None and getattr(args, "workspace", None):
        config_path = Path(args.workspace).expanduser() / "config.json"
    config, exists = load_config(config_path)
    if args is not None:
        if getattr(args, "workspace", None):
            config.workspace_path = args.workspace
            config.database_path = ""
            config.crop_dir = ""
            config.export_dir = ""
            config.report_dir = ""
            config.log_dir = ""
            config.cache_dir = ""
            config.apply_workspace_defaults()
        if getattr(args, "model", None):
            config.model_name = args.model
        if getattr(args, "provider", None):
            value = str(args.provider).upper()
            if value == "CPU":
                config.provider = "CPU"
            elif value == "CUDA":
                config.provider = "CUDA" if is_cuda_provider_available() else "Auto"
            else:
                config.provider = "Auto"
    if config.safe_mode:
        # safe-mode is a startup troubleshooting flag. Older builds persisted it
        # into config.json, which made normal launches silently skip model load.
        config.safe_mode = False
        if not config.auto_load_model:
            config.auto_load_model = True
    if str(config.provider).strip().lower() == "cuda" and not is_cuda_provider_available():
        config.provider = "Auto"
    config.apply_workspace_defaults()
    save_config(config)
    log_file = setup_logging(config.log_dir)
    storage = Storage(config.database_path)
    engine = FaceEngine(
        model_name=config.model_name,
        providers=providers_from_choice(config.provider),
        det_size=config.det_size_tuple,
        root=config.model_root,
        custom_model_dir=config.custom_model_dir,
    )
    return StudioContext(
        config=config,
        config_exists=exists,
        storage=storage,
        engine=engine,
        log_file=str(log_file),
        runtime_safe_mode=runtime_safe_mode,
    )


def run_app(args=None) -> int:
    configure_qt_plugin_paths()
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("InsightFace GUI requires PySide6.")
        print("Please install with: pip install insightface[gui]")
        return 1

    from .main_window import MainWindow
    from .resources import configure_application_metadata

    app = QApplication.instance() or QApplication(sys.argv[:1])
    configure_application_metadata(app)
    context = create_context(args)
    window = MainWindow(context)
    window.resize(1320, 860)
    window.show()
    return int(app.exec())
