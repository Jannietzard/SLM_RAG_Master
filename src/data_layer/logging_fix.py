"""
Windows UTF-8 Encoding Fix für Logging.

Problem: Windows PowerShell nutzt cp1252, nicht UTF-8.
Checkmarks (✓) und Pfeile (→) werden nicht korrekt encoded.

Lösung: Forciere UTF-8 für File- und Console-Handler.
"""

import logging
import sys
from pathlib import Path


def setup_logging_windows_safe(log_file: Path = Path("./logs/edge_rag.log")) -> logging.Logger:
    """
    Windows-kompatibles Logging Setup mit UTF-8 support.
    
    Diese Funktion ersetzt setup_logging() in main.py
    und handles UTF-8 auf Windows korrekt.

    Args:
        log_file: Pfad zur Log-Datei

    Returns:
        Logger Instance
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler mit UTF-8 (für Windows)
    # WICHTIG: encoding='utf-8' macht Unicode-Chars sicher
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Windows Fix: Force UTF-8 für Console
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8')
    else:
        # Fallback für ältere Python-Versionen
        import io
        console_handler.stream = io.TextIOWrapper(
            sys.stdout.buffer, encoding='utf-8', line_buffering=True
        )

    # File Handler mit UTF-8
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Root Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return root_logger