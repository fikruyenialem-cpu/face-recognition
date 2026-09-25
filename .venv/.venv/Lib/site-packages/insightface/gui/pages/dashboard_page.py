"""Dashboard page."""

from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QPushButton, QWidget

from ..core.constants import LOCAL_PROCESSING_NOTICE, SUBTITLE
from ..core.tooltips import set_button_tooltip
from ..widgets.metric_card import MetricCard
from .base import BasePage


class DashboardPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(context, "Dashboard", "Welcome to InsightFace Evaluation Studio.\n" + SUBTITLE, parent)
        self.content.addWidget(self.notice(LOCAL_PROCESSING_NOTICE))
        grid_widget = QWidget()
        self.grid = QGridLayout(grid_widget)
        self.cards = {
            "workspace": MetricCard("Workspace", context.config.workspace_path),
            "model": MetricCard("Model", context.config.model_name),
            "provider": MetricCard("Provider", context.config.provider),
            "license": MetricCard("License", context.config.license_status),
            "people": MetricCard("People", "0"),
            "samples": MetricCard("Face samples", "0"),
            "media": MetricCard("Indexed photos", "0"),
            "faces": MetricCard("Detected faces", "0"),
        }
        for index, card in enumerate(self.cards.values()):
            self.grid.addWidget(card, index // 2, index % 2)
        self.content.addWidget(grid_widget)
        shortcuts = QWidget()
        shortcut_layout = QGridLayout(shortcuts)
        for index, (label, page) in enumerate(
            [
                ("Start 1:1 Compare", "1:1 Compare"),
                ("Add People", "People Library"),
                ("Scan Folder", "Batch Folder Processing"),
                ("Run Enterprise Evaluation", "Enterprise Evaluation"),
                ("Open License Center", "License Center"),
            ]
        ):
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, target=page: self.window().open_page(target))
            set_button_tooltip(button)
            shortcut_layout.addWidget(button, index // 3, index % 3)
        self.content.addWidget(shortcuts)
        self.content.addStretch(1)

    def refresh(self) -> None:
        counts = self.context.storage.counts()
        self.cards["workspace"].set_value(self.context.config.workspace_path)
        self.cards["model"].set_value(self.context.config.model_name)
        self.cards["provider"].set_value(self.context.config.provider)
        self.cards["license"].set_value(self.context.config.license_status)
        self.cards["people"].set_value(counts["people"])
        self.cards["samples"].set_value(counts["face_samples"])
        self.cards["media"].set_value(counts["media_items"])
        self.cards["faces"].set_value(counts["media_faces"])
