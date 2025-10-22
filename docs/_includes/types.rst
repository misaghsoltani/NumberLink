.. py:module:: numberlink.types

:py:mod:`numberlink.types`
==========================

Type Aliases
------------

.. py:data:: RenderMode
   :canonical: numberlink.types.RenderMode
   :type: Literal["rgb_array", "ansi", "human"]

   Rendering mode for the environment. Can be ``"rgb_array"``, ``"ansi"``, or ``"human"``.

.. py:data:: Coord
   :canonical: numberlink.types.Coord
   :type: tuple[int, int]

   A coordinate pair ``(row, column)`` representing a grid position.

.. py:data:: Lane
   :canonical: numberlink.types.Lane
   :type: Literal["n", "v", "h"]

   Lane indicator for bridge cells. ``"n"`` marks a normal cell, ``"v"`` marks the vertical lane, and ``"h"`` marks the horizontal lane.

.. py:data:: CellLane
   :canonical: numberlink.types.CellLane
   :type: tuple[int, int, Lane]

   A cell coordinate and the active lane encoded as ``(row, column, lane)``.

.. py:data:: RGBInt
   :canonical: numberlink.types.RGBInt
   :type: tuple[int, int, int]

   An RGB color stored as three integers ``(red, green, blue)``.

.. py:data:: ActType
   :canonical: numberlink.types.ActType
   :type: numpy.integer | int

   Action type for the environment. Accepts numpy integer scalars or built-in ``int`` values.

.. py:data:: ObsType
   :canonical: numberlink.types.ObsType
   :type: numpy.typing.NDArray[numpy.uint8]

   Observation type for the environment as an RGB image array.

Utility Functions
-----------------

.. autofunction:: select_signed_dtype

.. autofunction:: select_unsigned_dtype

Typed dicts
-----------

.. py:class:: Snapshot
   :canonical: numberlink.types.Snapshot

   Runtime snapshot captured by :py:meth:`numberlink.viewer.NumberLinkViewer._snapshot_state`.

   The snapshot contains internal environment arrays and viewer selection state used to restore a previous
   runtime configuration via :py:meth:`numberlink.viewer.NumberLinkViewer._restore_state`.
