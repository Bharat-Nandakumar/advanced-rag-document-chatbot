# run.py
from __future__ import annotations

from rag.config import get_settings
from app.ui import build_app


def main() -> None:
    settings = get_settings()
    app = build_app(settings=settings)
    app.launch(debug=True)


if __name__ == "__main__":
    main()
