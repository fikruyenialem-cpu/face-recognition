"""License Center page."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QTableWidget, QTableWidgetItem

from ..core.i18n import tr
from ..core.links import open_insightface_url
from ..core.licensing import allowed_usage_summary
from ..core.constants import APP_VERSION
from ..widgets.table_utils import configure_table_columns, refresh_table_columns
from .base import BasePage

HOMEPAGE_URL = "https://www.insightface.ai"
ENTERPRISE_HELP_URL = "https://www.insightface.ai/contact"


class LicenseCenterPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(context, "License Center", "A concise local-use and commercial-deployment summary.", parent)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.content.addWidget(self.summary)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Usage", "Status"])
        self.table.setMinimumHeight(260)
        configure_table_columns(self.table, [260, 420])
        self.content.addWidget(self.table)
        self.content.addWidget(
            self.row(
                self.button("Visit Homepage", self.open_homepage),
                self.button("Contact Enterprise Support", self.open_enterprise_help),
            )
        )
        self.refresh()

    def refresh(self) -> None:
        cfg = self.context.config
        self.summary.setText(
            " ".join(
                [
                    tr(
                        "InsightFace Evaluation Studio runs locally and does not upload images, embeddings, videos, or reports automatically.",
                        cfg.ui_language,
                    ),
                    tr(
                        "Code and model files may have different licenses, and commercial deployment requires appropriate model authorization.",
                        cfg.ui_language,
                    ),
                    f"{tr('Current package', cfg.ui_language)}: insightface {APP_VERSION}, GUI {APP_VERSION}, "
                    f"{tr('Model', cfg.ui_language).lower()} {cfg.model_name}, "
                    f"{tr('Provider', cfg.ui_language).lower()} {cfg.provider}, "
                    f"{tr('License', cfg.ui_language).lower()} {tr(cfg.license_status, cfg.ui_language)}.",
                ]
            )
        )
        summary = allowed_usage_summary()
        self.table.setRowCount(len(summary))
        for row, (usage, status) in enumerate(summary.items()):
            self.table.setItem(row, 0, QTableWidgetItem(tr(usage, cfg.ui_language)))
            self.table.setItem(row, 1, QTableWidgetItem(tr(status, cfg.ui_language)))
        refresh_table_columns(self.table)

    def open_homepage(self) -> None:
        open_insightface_url(HOMEPAGE_URL, content="license_homepage")
        self.set_status("Opened InsightFace homepage.")

    def open_enterprise_help(self) -> None:
        open_insightface_url(ENTERPRISE_HELP_URL, content="license_enterprise_support")
        self.set_status("Opened InsightFace enterprise contact page.")
