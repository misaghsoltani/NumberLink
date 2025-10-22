from __future__ import annotations

from .registration import register_numberlink_v0


def main() -> None:
    """Register the NumberLink environment with Gymnasium.

    This function is the package entry point invoked by ``python -m numberlink``. It registers the environment id
    ``NumberLinkRGB-v0`` so callers can create environments with ``gymnasium.make("NumberLinkRGB-v0")``.

    :raises RuntimeError: If registration fails for an unexpected reason.
    """
    try:
        register_numberlink_v0()

    except Exception as exc:
        raise RuntimeError("Failed to register NumberLink environment") from exc

    print('The environment "NumberLinkRGB-v0" registered. Use gymnasium.make("NumberLinkRGB-v0")')


if __name__ == "__main__":
    main()
