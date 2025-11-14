from __future__ import annotations

from .app import MywApp
from .config import ConfigError


def main() -> None:
    try:
        app = MywApp()
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return
    app.run()


if __name__ == "__main__":
    main()

