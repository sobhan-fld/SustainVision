"""SustainVision entrypoint.

Launches the TUI to review/edit the configuration and saves it to disk.
Subsequent steps (training loop + CodeCarbon integration) can import and use
the resulting configuration object returned here.
"""

from __future__ import annotations

from sustainvision.tui import run_config_tui


def main() -> None:
    """Run the configuration TUI and print the final configuration."""
    cfg = run_config_tui()

    # Placeholder: kick off training here with CodeCarbon around the loop
    print("\nFinal configuration:")
    print(cfg)


if __name__ == "__main__":
    main()


