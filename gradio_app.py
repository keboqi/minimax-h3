#!/usr/bin/env python3
"""Launch H3, or expose its temporary compatibility API to existing callers."""

from __future__ import annotations

if __name__ == "__main__":
    from h3_ui.application import main

    main()
else:
    # Preserve module identity so legacy integrations patch the same callbacks
    # that composition injects into services. New code imports h3_app directly.
    import sys
    from h3_ui import application

    sys.modules[__name__] = application
