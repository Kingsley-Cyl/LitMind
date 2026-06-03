from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import BodyLabel, CardWidget, PrimaryPushButton, TextBrowser

try:
    import fitz
except ImportError:  # pragma: no cover - optional dependency at runtime
    fitz = None


class PdfPreviewWidget(QScrollArea):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setAlignment(Qt.AlignTop)
        self.setWidget(self._container)
        self.set_placeholder("正在等待 PDF 预览...")

    def clear_pages(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def set_placeholder(self, message: str) -> None:
        self.clear_pages()
        label = QLabel(message)
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        self._layout.addWidget(label)

    def set_pdf_bytes(self, pdf_bytes: bytes) -> None:
        if fitz is None:
            self.set_placeholder("客户端未安装 PyMuPDF，无法渲染 PDF 预览。")
            return
        if not pdf_bytes:
            self.set_placeholder("未获取到 PDF 内容。")
            return

        self.clear_pages()
        try:
            document = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page_index in range(len(document)):
                page = document.load_page(page_index)
                pix = page.get_pixmap(matrix=fitz.Matrix(1.35, 1.35), alpha=False)
                image_bytes = pix.tobytes("png")
                label = QLabel()
                label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
                pixmap = QPixmap()
                pixmap.loadFromData(image_bytes)
                label.setPixmap(pixmap)
                label.setToolTip(f"第 {page_index + 1} 页")
                self._layout.addWidget(label)
            document.close()
        except Exception as exc:
            self.set_placeholder(f"PDF 预览加载失败：{exc}")


class DetailPage(QWidget):
    recommendation_selected = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("detail-page")
        self._build_ui()

    def _build_ui(self) -> None:
        self._recommend_paper_ids = []
        self._current_paper_id = None
        self.title_label = BodyLabel("请选择一篇论文")
        self.pdf_preview = PdfPreviewWidget()
        self.abstract_text = TextBrowser()
        self.keyword_label = QLabel("")
        self.meta_label = QLabel("")
        self.sections_text = TextBrowser()
        self.recommend_list = QListWidget()
        self.open_recommend_button = PrimaryPushButton("查看所选文献")
        self.open_recommend_button.setEnabled(False)
        self.keyword_label.setWordWrap(True)
        self.meta_label.setWordWrap(True)

        right_card = CardWidget()
        right_layout = QVBoxLayout(right_card)
        right_layout.addWidget(QLabel("摘要"))
        right_layout.addWidget(self.abstract_text, 1)
        right_layout.addWidget(QLabel("关键词"))
        right_layout.addWidget(self.keyword_label)
        right_layout.addWidget(QLabel("元数据"))
        right_layout.addWidget(self.meta_label)
        right_layout.addWidget(QLabel("章节摘要"))
        right_layout.addWidget(self.sections_text, 1)
        right_layout.addWidget(QLabel("相似文献推荐"))
        right_layout.addWidget(self.recommend_list, 1)
        right_layout.addWidget(self.open_recommend_button)

        splitter = QSplitter()
        splitter.addWidget(self.pdf_preview)
        splitter.addWidget(right_card)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout = QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addWidget(splitter, 1)

        self.recommend_list.itemDoubleClicked.connect(self._emit_recommendation)
        self.recommend_list.currentRowChanged.connect(self._update_recommend_button_state)
        self.open_recommend_button.clicked.connect(self._emit_current_recommendation)

    def prepare_for_paper(self, paper_id: str) -> None:
        self._current_paper_id = paper_id
        self.title_label.setText("正在加载论文详情...")
        self.pdf_preview.set_placeholder("正在从服务器加载 PDF 预览...")
        self.abstract_text.clear()
        self.keyword_label.clear()
        self.meta_label.clear()
        self.sections_text.clear()
        self._recommend_paper_ids = []
        self.recommend_list.clear()
        self.open_recommend_button.setEnabled(False)

    def set_paper(self, paper: dict) -> None:
        self._current_paper_id = paper.get("paper_id")
        self.title_label.setText(paper.get("title", "未命名论文"))
        self.abstract_text.setPlainText(paper.get("abstract", ""))
        self.keyword_label.setText(", ".join(paper.get("keywords", [])))
        self.meta_label.setText(
            f"作者: {', '.join(paper.get('authors', []))}\n年份: {paper.get('year') or ''}\n语言: {paper.get('language', '')}"
        )
        sections = paper.get("sections", {})
        compact_sections = []
        for name in ("abstract", "introduction", "method", "experiments", "conclusion"):
            if sections.get(name):
                compact_sections.append(f"[{name}]\n{sections[name][:500]}")
        self.sections_text.setPlainText("\n\n".join(compact_sections))

    def set_pdf_bytes(self, pdf_bytes: bytes) -> None:
        self.pdf_preview.set_pdf_bytes(pdf_bytes)

    def set_pdf_error(self, message: str) -> None:
        self.pdf_preview.set_placeholder(f"无法加载 PDF 预览：{message}")

    def set_recommendations(self, items: list[dict]) -> None:
        self._recommend_paper_ids = []
        self.recommend_list.clear()
        for item in items:
            self._recommend_paper_ids.append(item["paper_id"])
            self.recommend_list.addItem(f"{item['title']} | {item['score']} | {item['reason']}")
        self._update_recommend_button_state(self.recommend_list.currentRow())

    def _emit_recommendation(self, item) -> None:
        row = self.recommend_list.row(item)
        if 0 <= row < len(getattr(self, "_recommend_paper_ids", [])):
            self.recommendation_selected.emit(self._recommend_paper_ids[row])

    def _emit_current_recommendation(self) -> None:
        row = self.recommend_list.currentRow()
        if 0 <= row < len(getattr(self, "_recommend_paper_ids", [])):
            self.recommendation_selected.emit(self._recommend_paper_ids[row])

    def _update_recommend_button_state(self, row: int) -> None:
        self.open_recommend_button.setEnabled(0 <= row < len(getattr(self, "_recommend_paper_ids", [])))
