from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QTableWidgetItem, QVBoxLayout, QWidget
from qfluentwidgets import LineEdit, PrimaryPushButton, TableWidget


class LibraryPage(QWidget):
    refresh_requested = pyqtSignal(dict)
    paper_selected = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("library-page")
        self._build_ui()

    def _build_ui(self) -> None:
        self.keyword_edit = LineEdit()
        self.keyword_edit.setPlaceholderText("关键词过滤")
        self.year_edit = LineEdit()
        self.year_edit.setPlaceholderText("年份")
        self.topic_edit = LineEdit()
        self.topic_edit.setPlaceholderText("主题")
        self.refresh_button = PrimaryPushButton("刷新文献库")
        self.table = TableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["标题", "作者", "年份", "关键词", "摘要"])

        filter_bar = QHBoxLayout()
        filter_bar.addWidget(self.keyword_edit, 2)
        filter_bar.addWidget(self.year_edit, 1)
        filter_bar.addWidget(self.topic_edit, 1)
        filter_bar.addWidget(self.refresh_button)

        layout = QVBoxLayout(self)
        layout.addLayout(filter_bar)
        layout.addWidget(self.table, 1)

        self.refresh_button.clicked.connect(self._emit_refresh)
        self.table.cellDoubleClicked.connect(self._open_selected)

    def _emit_refresh(self) -> None:
        self.refresh_requested.emit(
            {
                "keyword": self.keyword_edit.text().strip(),
                "year": self.year_edit.text().strip(),
                "topic": self.topic_edit.text().strip(),
            }
        )

    def set_papers(self, papers: list[dict]) -> None:
        self.table.setRowCount(len(papers))
        self._paper_ids = []
        for row, paper in enumerate(papers):
            authors = ", ".join(paper.get("authors", []))
            keywords = ", ".join(paper.get("keywords", []))
            abstract = paper.get("abstract", "")
            self._paper_ids.append(paper["paper_id"])
            self.table.setItem(row, 0, self._make_item(paper["title"]))
            self.table.setItem(row, 1, self._make_item(authors))
            self.table.setItem(row, 2, self._make_item(str(paper.get("year") or "")))
            self.table.setItem(row, 3, self._make_item(keywords, keywords[:60]))
            self.table.setItem(row, 4, self._make_item(abstract, abstract[:120]))
        self.table.resizeColumnsToContents()

    def _open_selected(self, row: int, _: int) -> None:
        if 0 <= row < len(getattr(self, "_paper_ids", [])):
            self.paper_selected.emit(self._paper_ids[row])

    def _make_item(self, full_text: str, display_text: str | None = None) -> QTableWidgetItem:
        item = QTableWidgetItem(display_text if display_text is not None else full_text)
        item.setToolTip(full_text)
        return item
