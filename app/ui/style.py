"""Global Qt stylesheet for a polished dark UI.

Single source of truth for colours, paddings and typography. The window
title, buttons, dropdowns, status bar and result list all pull from here so
the app feels consistent across platforms (PySide6 on Linux/macOS/Windows).
"""

from __future__ import annotations

# Palette
COLOR_BG = "#1c1f24"
COLOR_BG_PANEL = "#21252c"
COLOR_BG_ELEV = "#272c34"
COLOR_BORDER = "#2f343d"
COLOR_BORDER_STRONG = "#3a4150"
COLOR_TEXT = "#e6e8ee"
COLOR_TEXT_MUTED = "#8a93a4"
COLOR_ACCENT = "#3aa0ff"
COLOR_ACCENT_HOVER = "#54b0ff"
COLOR_ACCENT_PRESSED = "#2d8de0"
COLOR_OK = "#3ec27b"
COLOR_WARN = "#f0a050"
COLOR_DANGER = "#e85a5a"

FONT_FAMILY = (
    '"Inter", "SF Pro Text", "Segoe UI", "Cantarell", '
    '"Helvetica Neue", "Arial", sans-serif'
)

APP_STYLESHEET = f"""
* {{
    font-family: {FONT_FAMILY};
    font-size: 13px;
    color: {COLOR_TEXT};
}}

QMainWindow,
QWidget {{
    background-color: {COLOR_BG};
}}

QToolBar {{
    background-color: {COLOR_BG_PANEL};
    border: 0;
    border-bottom: 1px solid {COLOR_BORDER};
    spacing: 6px;
    padding: 8px 12px;
}}

QToolBar QLabel {{
    color: {COLOR_TEXT_MUTED};
    padding: 0 4px;
}}

QToolBar::separator {{
    background-color: {COLOR_BORDER_STRONG};
    width: 1px;
    margin: 4px 8px;
}}

QPushButton {{
    background-color: {COLOR_BG_ELEV};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_BORDER_STRONG};
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: #2e333d;
    border-color: #46506a;
}}
QPushButton:pressed {{
    background-color: #232830;
}}
QPushButton:disabled {{
    color: {COLOR_TEXT_MUTED};
    background-color: #20242b;
    border-color: {COLOR_BORDER};
}}
QPushButton#primary {{
    background-color: {COLOR_ACCENT};
    color: #ffffff;
    border-color: {COLOR_ACCENT};
}}
QPushButton#primary:hover {{
    background-color: {COLOR_ACCENT_HOVER};
    border-color: {COLOR_ACCENT_HOVER};
}}
QPushButton#primary:pressed {{
    background-color: {COLOR_ACCENT_PRESSED};
    border-color: {COLOR_ACCENT_PRESSED};
}}
QPushButton#primary:disabled {{
    background-color: #2a4660;
    border-color: #2a4660;
    color: #9fb3c8;
}}

/* Segmented mode switch — three joined buttons that behave like a tab bar.
   Outer corners are rounded; inner edges share a single 1px divider so the
   whole control reads as one widget. */
QFrame#modeSwitch {{
    background-color: {COLOR_BG_ELEV};
    border: 1px solid {COLOR_BORDER_STRONG};
    border-radius: 8px;
}}

QPushButton#modeSegmentLeft,
QPushButton#modeSegmentMiddle,
QPushButton#modeSegmentRight {{
    background-color: transparent;
    color: {COLOR_TEXT_MUTED};
    border: 0;
    border-radius: 0;
    padding: 6px 18px;
    font-weight: 500;
    min-width: 72px;
    text-align: center;
}}
QPushButton#modeSegmentLeft {{
    border-top-left-radius: 7px;
    border-bottom-left-radius: 7px;
    border-right: 1px solid {COLOR_BORDER_STRONG};
}}
QPushButton#modeSegmentMiddle {{
    border-right: 1px solid {COLOR_BORDER_STRONG};
}}
QPushButton#modeSegmentRight {{
    border-top-right-radius: 7px;
    border-bottom-right-radius: 7px;
}}
QPushButton#modeSegmentLeft:hover,
QPushButton#modeSegmentMiddle:hover,
QPushButton#modeSegmentRight:hover {{
    background-color: #2e333d;
    color: {COLOR_TEXT};
}}
QPushButton#modeSegmentLeft:checked,
QPushButton#modeSegmentMiddle:checked,
QPushButton#modeSegmentRight:checked {{
    background-color: {COLOR_ACCENT};
    color: #ffffff;
    font-weight: 600;
}}
QPushButton#modeSegmentLeft:disabled,
QPushButton#modeSegmentMiddle:disabled,
QPushButton#modeSegmentRight:disabled {{
    color: #555a66;
    background-color: transparent;
}}

QProgressBar {{
    background-color: {COLOR_BG_ELEV};
    border: 1px solid {COLOR_BORDER_STRONG};
    border-radius: 6px;
    text-align: center;
    color: {COLOR_TEXT};
    min-width: 180px;
    max-height: 18px;
}}
QProgressBar::chunk {{
    background-color: {COLOR_ACCENT};
    border-radius: 5px;
}}

QStatusBar {{
    background-color: {COLOR_BG_PANEL};
    color: {COLOR_TEXT_MUTED};
    border-top: 1px solid {COLOR_BORDER};
    padding: 4px 8px;
}}
QStatusBar QLabel {{
    color: {COLOR_TEXT_MUTED};
}}

QLabel#sectionTitle {{
    color: {COLOR_TEXT};
    font-size: 14px;
    font-weight: 600;
    padding: 2px 4px 8px 4px;
    border-bottom: 1px solid {COLOR_BORDER};
}}

QLabel#libraryLabel {{
    color: {COLOR_TEXT_MUTED};
    padding: 8px 4px 0 4px;
}}

QLabel#statusDot {{
    color: {COLOR_TEXT_MUTED};
    padding: 0 6px;
}}

QListWidget {{
    background-color: {COLOR_BG_PANEL};
    border: 1px solid {COLOR_BORDER};
    border-radius: 6px;
    padding: 4px;
    outline: 0;
    font-family: "JetBrains Mono", "Cascadia Mono", "Menlo", "Consolas",
                 monospace;
    font-size: 12px;
}}
QListWidget::item {{
    padding: 6px 8px;
    border-radius: 4px;
    margin: 1px 0;
}}
QListWidget::item:hover {{
    background-color: {COLOR_BG_ELEV};
}}
QListWidget::item:selected {{
    background-color: #2a4660;
    color: #ffffff;
}}

QToolTip {{
    background-color: {COLOR_BG_ELEV};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_BORDER_STRONG};
    padding: 4px 8px;
}}

QMessageBox {{
    background-color: {COLOR_BG_PANEL};
}}
QMessageBox QLabel {{
    color: {COLOR_TEXT};
}}
"""
