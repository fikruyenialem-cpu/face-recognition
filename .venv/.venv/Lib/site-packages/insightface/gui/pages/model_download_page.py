"""Manual GitHub release model downloads."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QAbstractItemView, QFrame, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout

from ..core.config import save_config
from ..core.constants import LICENSE_NOTICE
from ..core.i18n import tr
from ..core.model_downloads import (
    GITHUB_RELEASES_URL,
    ModelAsset,
    download_model_asset,
    load_cached_assets,
    local_model_status,
    refresh_model_assets,
)
from ..widgets.table_utils import configure_table_columns, refresh_table_columns
from .base import BasePage


class ModelDownloadPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(
            context,
            "Model Downloads",
            "Manually refresh GitHub release model URLs and download selected model packages locally.",
            parent,
        )
        self.assets: list[ModelAsset] = []
        self.content.addWidget(
            self.notice(
                "Downloads are manual only. The GUI does not auto-download models. "
                "Model files may have different licenses from code; review usage before deployment."
            )
        )
        self.content.addWidget(self.notice(LICENSE_NOTICE))
        self.content.addWidget(
            self.row(
                self.button("Refresh Download URLs", self.refresh_urls),
                self.button("Download Selected", self.download_selected),
                self.button("Use Selected Model", self.use_selected_model),
                self.button("Open Model Folder", self.open_model_folder),
                self.button("Open GitHub Releases", self.open_releases),
            )
        )
        self.table = QTableWidget(0, 8)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setHorizontalHeaderLabels(
            ["asset", "source", "kind", "release", "size", "updated_at", "local status", "download url"]
        )
        configure_table_columns(self.table, [210, 100, 150, 130, 90, 170, 150, 360])
        self.table.setMinimumHeight(400)
        self.content.addWidget(self.table, 1)
        self.content.addSpacing(8)
        self.source_footer = QFrame()
        self.source_footer.setObjectName("downloadSourceFooter")
        footer_layout = QVBoxLayout(self.source_footer)
        footer_layout.setContentsMargins(10, 8, 10, 8)
        self.url_label = QLabel()
        self.url_label.setWordWrap(True)
        self.url_label.setProperty("role", "muted")
        footer_layout.addWidget(self.url_label)
        self.content.addWidget(self.source_footer)
        self.refresh()

    def refresh(self) -> None:
        self.assets = load_cached_assets(self.context.config.cache_dir)
        self.populate()

    def populate(self) -> None:
        self.table.setRowCount(len(self.assets))
        for row, asset in enumerate(self.assets):
            values = [
                asset.name,
                asset.source,
                asset.kind,
                asset.tag_name,
                self._format_size(asset.size),
                asset.updated_at,
                local_model_status(asset, self.context.config.model_root),
                asset.browser_download_url,
            ]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(str(value)))
        refresh_table_columns(self.table)
        language = self.context.config.ui_language
        self.url_label.setText(
            f"{tr('Refresh source', language)}: {GITHUB_RELEASES_URL}\n"
            f"{tr('Local model root', language)}: {Path(self.context.config.model_root).expanduser() / 'models'}"
        )

    def selected_asset(self) -> ModelAsset | None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self.assets):
            self.show_error("Select a model asset first.")
            return None
        return self.assets[row]

    def refresh_urls(self) -> None:
        def task():
            return refresh_model_assets(self.context.config.cache_dir)

        def done(payload):
            self.assets, message = payload
            self.populate()
            self.set_status(message)

        self.run_task("Refreshing model download URLs", task, done)

    def download_selected(self) -> None:
        asset = self.selected_asset()
        if asset is None:
            return

        def task(progress=None, is_cancelled=None):
            del is_cancelled
            if progress:
                progress(0, asset.size or 1, f"Connecting to download {asset.name}")
            return download_model_asset(
                asset,
                model_root=self.context.config.model_root,
                gui_cache_dir=self.context.config.cache_dir,
                progress=progress,
            )

        def done(path):
            path = Path(path)
            lower_name = asset.name.lower()
            if "gfpgan" in lower_name:
                self.context.config.gfpgan_model_path = str(path)
                save_config(self.context.config)
            elif "swap" in lower_name or "inswapper" in lower_name:
                self.context.config.swap_model_path = str(path)
                save_config(self.context.config)
            self.populate()
            manager = self.window()
            if hasattr(manager, "refresh_model_pages"):
                manager.refresh_model_pages()
            self.set_status(f"Downloaded {asset.name} to {path}")

        self.run_task(f"Downloading {asset.name}", task, done)

    def use_selected_model(self) -> None:
        asset = self.selected_asset()
        if asset is None:
            return
        if asset.name.endswith(".zip"):
            self.context.config.model_name = asset.stem
            self.context.config.custom_model_dir = ""
            save_config(self.context.config)
            self.set_status(f"Model set to {asset.stem}. Open Models and test model load.")
        elif asset.name.endswith(".onnx"):
            path = Path(self.context.config.model_root).expanduser() / "models" / asset.stem / asset.name
            if "gfpgan" in asset.name.lower():
                self.context.config.gfpgan_model_path = str(path)
                save_config(self.context.config)
                self.set_status(f"GFPGAN model set to {path}. Enable GFPGAN in Models > Runtime to use it after face swap.")
            elif "swap" in asset.name.lower() or "inswapper" in asset.name.lower():
                self.context.config.swap_model_path = str(path)
                save_config(self.context.config)
                self.set_status(f"Face swap model set to {path}.")
            else:
                self.set_status(f"ONNX model path: {path}")
        else:
            self.set_status("Selected asset cannot be used as a model package.")

    def open_model_folder(self) -> None:
        folder = Path(self.context.config.model_root).expanduser() / "models"
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def open_releases(self) -> None:
        QDesktopServices.openUrl(QUrl(GITHUB_RELEASES_URL))

    @staticmethod
    def _format_size(size: int) -> str:
        if not size:
            return ""
        return f"{size / (1024 * 1024):.1f} MB"
