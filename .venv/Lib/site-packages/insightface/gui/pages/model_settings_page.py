"""Model settings page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QLabel, QLineEdit, QTextEdit

from ..core.config import save_config
from ..core.face_engine import FaceEngine, is_cuda_provider_available, providers_from_choice
from ..core.model_downloads import is_model_package_installed, list_installed_gfpgan_models, list_installed_swap_models
from .base import BasePage


class ModelSettingsPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(context, "Model Settings", "Configure model packs, execution provider, face swap models, and runtime checks.", parent)
        form = QFormLayout()
        self.model_combo = QComboBox()
        self.model_packages = ["buffalo_l", "buffalo_m", "buffalo_s", "buffalo_sc", "antelopev2"]
        self._rebuild_model_combo()
        self.custom_dir = QLineEdit(context.config.custom_model_dir)
        self.custom_dir_label = None
        self._update_model_availability()
        self.model_combo.currentIndexChanged.connect(self._update_custom_dir_visibility)
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["Auto", "CPU", "CUDA"])
        self.provider_combo.setCurrentText(context.config.provider)
        self._update_provider_availability()
        self.det_combo = QComboBox()
        self.det_combo.addItems(["Auto", "128x128", "320x320", "640x640", "1024x1024"])
        self.det_combo.setCurrentText(context.config.det_size_label)
        self.swap_model_combo = QComboBox()
        self._update_swap_model_choices()
        self.gfpgan_enabled = QCheckBox("Enable GFPGAN restore after face swap")
        self.gfpgan_enabled.setChecked(bool(getattr(context.config, "enable_gfpgan", False)))
        self.gfpgan_model_combo = QComboBox()
        self._update_gfpgan_model_choices()
        form.addRow("Model package", self.model_combo)
        form.addRow("Custom model directory", self.custom_dir)
        self.custom_dir_label = form.labelForField(self.custom_dir)
        form.addRow("Provider", self.provider_combo)
        form.addRow("Detection size", self.det_combo)
        form.addRow("Face swap model", self.swap_model_combo)
        form.addRow("GFPGAN post-processing", self.gfpgan_enabled)
        form.addRow("GFPGAN model", self.gfpgan_model_combo)
        self.content.addLayout(form)
        self.runtime = QTextEdit()
        self.runtime.setReadOnly(True)
        self.content.addWidget(QLabel("Runtime information"))
        self.content.addWidget(self.runtime)
        self.content.addWidget(
            self.row(
                self.button("Save Settings", self.save),
                self.button("Open Model Downloads", lambda: self.window().open_page("Model Downloads")),
                self.button("Test Model Load", self.test_load),
                self.button("Warmup", self.warmup),
            )
        )
        self.refresh()
        self._update_custom_dir_visibility()

    def _apply_to_config(self) -> None:
        cfg = self.context.config
        chosen = self.model_combo.currentData()
        cfg.model_name = chosen if chosen != "custom model directory" else self.custom_dir.text().strip()
        cfg.custom_model_dir = self.custom_dir.text().strip()
        provider = self.provider_combo.currentText()
        cfg.provider = "Auto" if provider == "CUDA" and not is_cuda_provider_available() else provider
        if self.det_combo.currentText() == "Auto":
            cfg.det_size = [0, 0]
        else:
            size = self.det_combo.currentText().split("x")
            cfg.det_size = [int(size[0]), int(size[1])]
        cfg.swap_model_path = str(self.swap_model_combo.currentData() or "")
        cfg.gfpgan_model_path = str(self.gfpgan_model_combo.currentData() or "")
        cfg.enable_gfpgan = bool(self.gfpgan_enabled.isChecked() and cfg.gfpgan_model_path)

    def save(self) -> None:
        self._apply_to_config()
        save_config(self.context.config)
        self.set_status("Model settings saved.")
        self.refresh()

    def test_load(self) -> None:
        self._apply_to_config()

        def task():
            engine = FaceEngine(
                model_name=self.context.config.model_name,
                providers=providers_from_choice(self.context.config.provider),
                det_size=self.context.config.det_size_tuple,
                root=self.context.config.model_root,
                custom_model_dir=self.context.config.custom_model_dir,
            )
            engine.load()
            return engine

        def done(engine):
            self.context.engine = engine
            self.window().context.engine = engine
            self.refresh()
            if engine.is_loaded():
                self.set_status("Model loaded successfully.")
            else:
                self.show_error(engine.last_error or "Model load failed.")

        self.run_task("Loading model", task, done)

    def warmup(self) -> None:
        if not self.context.engine.is_loaded():
            self.show_error("Model is not loaded. Please open Models.")
            return
        self.run_task("Model warmup", self.context.engine.warmup, lambda info: self.set_status(f"Warmup complete: {info['warmup_ms']:.1f} ms"))

    def refresh(self) -> None:
        self._update_model_availability()
        self._update_provider_availability()
        self._update_swap_model_choices()
        self._update_gfpgan_model_choices()
        info = self.context.engine.get_runtime_info()
        self.runtime.setPlainText("\n".join(f"{key}: {value}" for key, value in info.items()))

    def _update_provider_availability(self) -> None:
        cuda_available = is_cuda_provider_available()
        cuda_index = self.provider_combo.findText("CUDA")
        if cuda_index >= 0:
            item = self.provider_combo.model().item(cuda_index)
            if item is not None:
                item.setEnabled(cuda_available)
                item.setToolTip(
                    "CUDAExecutionProvider is available."
                    if cuda_available
                    else "CUDAExecutionProvider is not available. Install a matching onnxruntime-gpu, CUDA runtime, and GPU driver first."
                )
        if self.provider_combo.currentText() == "CUDA" and not cuda_available:
            self.provider_combo.setCurrentText("Auto")
            self.provider_combo.setToolTip("CUDA is unavailable on this machine, so Auto will use CPU.")
        else:
            self.provider_combo.setToolTip("Auto uses CUDA when CUDAExecutionProvider is available, otherwise CPU.")

    def _update_model_availability(self) -> None:
        model = self.model_combo.model()
        for index, package in enumerate(self.model_packages):
            item = model.item(index)
            if item is None:
                continue
            installed = is_model_package_installed(package, self.context.config.model_root)
            item.setEnabled(installed)
            item.setText(package if installed else f"{package} (not downloaded)")
            item.setData(package, Qt.UserRole)
            if installed:
                item.setData(None, Qt.ForegroundRole)
            else:
                item.setForeground(QBrush(QColor("#9ca3af")))
            item.setToolTip(
                f"{package} is installed under {self.context.config.model_root}/models."
                if installed
                else f"{package} is not downloaded. Open Models > Downloads to install it."
            )
        custom_item = model.item(len(self.model_packages))
        if custom_item is not None:
            custom_item.setEnabled(True)
            custom_item.setText("custom model directory")
            custom_item.setData("custom model directory", Qt.UserRole)
        current = self.context.config.model_name if self.context.config.model_name in self.model_packages else "custom model directory"
        index = self.model_combo.findData(current)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        self._update_custom_dir_visibility()

    def _rebuild_model_combo(self) -> None:
        self.model_combo.clear()
        for package in self.model_packages:
            self.model_combo.addItem(package, package)
        self.model_combo.addItem("custom model directory", "custom model directory")

    def _update_custom_dir_visibility(self) -> None:
        is_custom = self.model_combo.currentData() == "custom model directory"
        self.custom_dir.setVisible(is_custom)
        if hasattr(self, "custom_dir_label") and self.custom_dir_label is not None:
            self.custom_dir_label.setVisible(is_custom)

    def _update_swap_model_choices(self) -> None:
        current = getattr(self.context.config, "swap_model_path", "")
        paths = list_installed_swap_models(self.context.config.model_root)
        self.swap_model_combo.blockSignals(True)
        self.swap_model_combo.clear()
        if not paths:
            self.swap_model_combo.addItem("No downloaded swap models", "")
            self.swap_model_combo.setEnabled(False)
            self.swap_model_combo.setToolTip("Download inswapper_128.onnx from Models > Downloads first.")
        else:
            self.swap_model_combo.setEnabled(True)
            self.swap_model_combo.setToolTip("Only downloaded swap models are shown.")
            for path in paths:
                label = f"{Path(path).parent.name}/{Path(path).name}"
                self.swap_model_combo.addItem(label, str(path))
            index = self.swap_model_combo.findData(current)
            if index >= 0:
                self.swap_model_combo.setCurrentIndex(index)
            elif current and Path(current).exists():
                self.swap_model_combo.addItem(Path(current).name, current)
                self.swap_model_combo.setCurrentIndex(self.swap_model_combo.count() - 1)
        self.swap_model_combo.blockSignals(False)

    def _update_gfpgan_model_choices(self) -> None:
        current = getattr(self.context.config, "gfpgan_model_path", "")
        paths = list_installed_gfpgan_models(self.context.config.model_root)
        self.gfpgan_model_combo.blockSignals(True)
        self.gfpgan_model_combo.clear()
        if not paths:
            self.gfpgan_model_combo.addItem("No downloaded GFPGAN models", "")
            self.gfpgan_model_combo.setEnabled(False)
            self.gfpgan_enabled.setEnabled(False)
            self.gfpgan_model_combo.setToolTip("Download GFPGANv1.4.onnx from Models > Downloads first.")
            self.gfpgan_enabled.setToolTip("GFPGAN is unavailable until a GFPGAN model is downloaded.")
        else:
            self.gfpgan_model_combo.setEnabled(True)
            self.gfpgan_enabled.setEnabled(True)
            self.gfpgan_model_combo.setToolTip("Only downloaded GFPGAN restore models are shown.")
            self.gfpgan_enabled.setToolTip("Run GFPGAN face restoration after face swap.")
            for path in paths:
                label = f"{Path(path).parent.name}/{Path(path).name}"
                self.gfpgan_model_combo.addItem(label, str(path))
            index = self.gfpgan_model_combo.findData(current)
            if index >= 0:
                self.gfpgan_model_combo.setCurrentIndex(index)
            elif current and Path(current).exists():
                self.gfpgan_model_combo.addItem(Path(current).name, current)
                self.gfpgan_model_combo.setCurrentIndex(self.gfpgan_model_combo.count() - 1)
        self.gfpgan_model_combo.blockSignals(False)
