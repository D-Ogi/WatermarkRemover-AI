"""Windowed entry point for source installs and the portable executable."""
from desktop_runtime import configure_runtime

configure_runtime()

if __name__ == "__main__":
    from remwmgui import main
    main()
