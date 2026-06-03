from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QTableWidgetItem, QVBoxLayout, QWidget
from qfluentwidgets import PrimaryPushButton, SearchLineEdit, TableWidget


class SearchPage(QWidget):
    search_requested = pyqtSignal(str)
    paper_selected = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("search-page")
        self._build_ui()

    def _build_ui(self) -> None:
        self.query_edit = SearchLineEdit()
        self.query_edit.setPlaceholderText("输入查询内容，系统将结合论文全文进行检索")
        self.search_button = PrimaryPushButton("开始检索")
        self.table = TableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["标题", "分数", "关键词", "摘要片段", "匹配片段"])

        top = QHBoxLayout()
        top.addWidget(self.query_edit, 1)
        top.addWidget(self.search_button)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.table, 1)

        self.search_button.clicked.connect(self._emit_search)
        self.query_edit.searchSignal.connect(lambda *_: self._emit_search())
        self.table.cellDoubleClicked.connect(self._open_selected)

    def _emit_search(self) -> None:
        query = self.query_edit.text().strip()
        if query:
            self.search_requested.emit(query)

    def set_results(self, results: list[dict]) -> None:
        self.table.setRowCount(len(results))
        self._paper_ids = []
        for row, result in enumerate(results):
            self._paper_ids.append(result["paper_id"])
            self.table.setItem(row, 0, QTableWidgetItem(result["title"]))
            self.table.setItem(row, 1, QTableWidgetItem(str(result["score"])))
            self.table.setItem(row, 2, QTableWidgetItem(", ".join(result.get("keywords", []))[:60]))
            self.table.setItem(row, 3, QTableWidgetItem(result.get("abstract_snippet", "")[:120]))
            self.table.setItem(row, 4, QTableWidgetItem(result.get("matched_passage", "")[:180]))
        self.table.resizeColumnsToContents()

    def _open_selected(self, row: int, _: int) -> None:
        if 0 <= row < len(getattr(self, "_paper_ids", [])):
            self.paper_selected.emit(self._paper_ids[row])
