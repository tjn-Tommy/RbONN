"""Dark theme stylesheet for the Santec SLM control app."""

DARK_STYLESHEET = """
QMainWindow, QWidget {
    background: #0f1419;
    color: #d8dee9;
    font-family: Segoe UI, Arial, sans-serif;
    font-size: 10.5pt;
}
#Navigation {
    background: #121920;
    border: none;
    border-right: 1px solid #1d2a33;
    padding: 12px 10px;
    outline: none;
}
#Navigation::item {
    border-radius: 8px;
    padding: 14px 14px;
    margin: 3px 0;
    border-left: 3px solid transparent;
}
#Navigation::item:hover {
    background: #1a242d;
}
#Navigation::item:selected {
    background: #1f6f78;
    border-left: 3px solid #f5c542;
    color: white;
}
#AppBrand {
    font-size: 13pt;
    font-weight: 700;
    color: #f4f7f8;
    padding: 14px 8px 10px 8px;
}
#AppBrandSub {
    font-size: 8.5pt;
    color: #6d8a93;
    padding: 0 8px 12px 8px;
}
#PageTitle {
    font-size: 22pt;
    font-weight: 650;
    color: #f4f7f8;
}
#PageSubtitle {
    color: #7d97a0;
    font-size: 10pt;
}
QGroupBox#Panel {
    border: 1px solid #243440;
    border-radius: 10px;
    margin-top: 16px;
    padding: 18px 12px 12px 12px;
    background: #121b22;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #9fc9cf;
    font-weight: 600;
}
QLabel {
    background: transparent;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTableWidget {
    background: #0b1116;
    border: 1px solid #2e4350;
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: #1f6f78;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus {
    border-color: #2f8995;
}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled {
    color: #5b6a73;
    border-color: #233039;
}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background: #1a2730;
    border: none;
    width: 18px;
    border-radius: 3px;
    margin: 1px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: #25525a;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    width: 7px;
    height: 7px;
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid #9fc9cf;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 7px;
    height: 7px;
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #9fc9cf;
}
QComboBox::drop-down {
    border: none;
    width: 26px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #9fc9cf;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background: #121b22;
    border: 1px solid #2e4350;
    selection-background-color: #1f6f78;
    outline: none;
}
QCheckBox {
    spacing: 8px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #3a505c;
    border-radius: 4px;
    background: #0b1116;
}
QCheckBox::indicator:hover {
    border-color: #2f8995;
}
QCheckBox::indicator:checked {
    background: #f5c542;
    border-color: #f5c542;
}
QPushButton {
    background: #236d77;
    border: 1px solid #2f8995;
    border-radius: 6px;
    color: white;
    padding: 8px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background: #2b7f8a;
}
QPushButton:pressed {
    background: #1c5a63;
}
QPushButton:disabled {
    background: #26313a;
    border-color: #2d3942;
    color: #6d7b84;
}
QPushButton[variant="danger"] {
    background: #8c3a3a;
    border-color: #b05050;
}
QPushButton[variant="danger"]:hover {
    background: #a14545;
}
QPushButton[variant="danger"]:disabled {
    background: #26313a;
    border-color: #2d3942;
    color: #6d7b84;
}
QPushButton[variant="ghost"] {
    background: transparent;
    border: 1px solid #3a505c;
    color: #b8c8cf;
}
QPushButton[variant="ghost"]:hover {
    background: #1a242d;
    border-color: #2f8995;
}
QLabel[status="ok"] {
    color: #8fd6a0;
    background: #14281c;
    border: 1px solid #2c6e44;
    border-radius: 10px;
    padding: 3px 10px;
}
QLabel[status="error"] {
    color: #f0a3a3;
    background: #2c1518;
    border: 1px solid #8c3a3a;
    border-radius: 10px;
    padding: 3px 10px;
}
QLabel[status="off"] {
    color: #8b9aa2;
    background: #18222a;
    border: 1px solid #2c3a43;
    border-radius: 10px;
    padding: 3px 10px;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #273640;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #1f6f78;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #f5c542;
    width: 18px;
    margin: -7px 0;
    border-radius: 9px;
}
QSlider::handle:horizontal:hover {
    background: #ffd765;
}
QProgressBar {
    background: #0b1116;
    border: 1px solid #2e4350;
    border-radius: 6px;
    text-align: center;
    height: 20px;
    color: #d8dee9;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #d9a92f, stop:1 #f5c542);
    border-radius: 5px;
}
QTableWidget {
    gridline-color: #243440;
    alternate-background-color: #101921;
}
QHeaderView::section {
    background: #18242d;
    color: #9fc9cf;
    border: none;
    border-bottom: 1px solid #2e4350;
    padding: 6px 8px;
    font-weight: 600;
}
QTableCornerButton::section {
    background: #18242d;
    border: none;
}
QScrollBar:vertical {
    background: #0f1419;
    width: 10px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #2c3a43;
    border-radius: 5px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover {
    background: #3a505c;
}
QScrollBar:horizontal {
    background: #0f1419;
    height: 10px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #2c3a43;
    border-radius: 5px;
    min-width: 24px;
}
QScrollBar::handle:horizontal:hover {
    background: #3a505c;
}
QScrollBar::add-line, QScrollBar::sub-line {
    height: 0;
    width: 0;
}
QScrollBar::add-page, QScrollBar::sub-page {
    background: none;
}
QSplitter::handle {
    background: #1d2a33;
    width: 2px;
}
QToolTip {
    background: #18242d;
    color: #d8dee9;
    border: 1px solid #2f8995;
    padding: 5px 8px;
    border-radius: 4px;
}
QStatusBar {
    background: #121920;
    color: #8fa6af;
    border-top: 1px solid #1d2a33;
}
#Preview {
    border: 1px solid #243440;
    border-radius: 10px;
    background: #0b1116;
}
#LogBox {
    font-family: Consolas, monospace;
    font-size: 9.5pt;
}
"""
