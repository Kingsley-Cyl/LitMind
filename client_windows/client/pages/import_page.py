from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QStyle, QToolButton, QVBoxLayout, QWidget
from qfluentwidgets import BodyLabel, CaptionLabel, LineEdit, PlainTextEdit, PrimaryPushButton, ProgressBar


class ImportPage(QWidget):
    import_requested = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("import-page")
        self._build_ui()

    def _build_ui(self) -> None:
        self.title = BodyLabel("导入 PDF 目录（支持服务器路径或本地 Windows 目录）")
        self.directory_edit = LineEdit()
        self.directory_edit.setPlaceholderText("例如: /data/papers 或 C:\\Users\\xxx\\Documents\\papers")
        self.browse_button = PrimaryPushButton("选择目录")
        self.start_button = PrimaryPushButton("开始导入")
        self.online_toggle = QToolButton()
        self.online_toggle.setCheckable(True)
        self.online_toggle.setToolTip("开启后，服务器会尝试联网校正文献标题、作者和年份")
        network_icon = getattr(QStyle, "SP_DriveNetIcon", QStyle.SP_FileDialogContentsView)
        self.online_toggle.setIcon(self.style().standardIcon(network_icon))
        self.online_toggle.setAutoRaise(True)
        self.online_hint = CaptionLabel("联网校正")
        self.progress = ProgressBar()
        self.progress.setRange(0, 100)
        self.status_label = QLabel("等待导入")
        self.log_edit = PlainTextEdit()
        self.log_edit.setReadOnly(True)

        top_bar = QHBoxLayout()
        top_bar.addWidget(self.directory_edit, 1)
        top_bar.addWidget(self.browse_button)
        top_bar.addWidget(self.online_toggle)
        top_bar.addWidget(self.online_hint)
        top_bar.addWidget(self.start_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.title)
        layout.addLayout(top_bar)
        layout.addWidget(self.progress)
        layout.addWidget(self.status_label)
        layout.addWidget(self.log_edit, 1)

        self.browse_button.clicked.connect(self._choose_directory)
        self.start_button.clicked.connect(self._emit_import)

    def _choose_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择 PDF 目录")
        if directory:
            self.directory_edit.setText(directory)

    def _emit_import(self) -> None:
        directory = self.directory_edit.text().strip()
        if directory:
            self.import_requested.emit(
                {
                    "directory": directory,
                    "enable_online_metadata": self.online_toggle.isChecked(),
                }
            )

    def set_job_payload(self, job_payload: dict) -> None:
        job = job_payload.get("data", {})
        total = max(job.get("total", 0), 1)
        completed = job.get("completed", 0)
        failed = job.get("failed", 0)
        progress = int(((completed + failed) / total) * 100)
        self.progress.setValue(progress)
        self.status_label.setText(
            f"状态: {job.get('status', 'unknown')} | 完成: {completed}/{job.get('total', 0)} | 失败: {failed} | 步骤: {job.get('current_step', '')}"
        )
        self.log_edit.setPlainText("\n".join(job.get("logs", [])))
