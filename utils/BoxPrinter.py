"""Fixed-layout ANSI array display. Python 3.9+, standard library only.

Use one printer per terminal, preferably as a context manager. The first print
draws the screen; later prints write only changed, fixed-width value fields.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass
from decimal import Decimal
from numbers import Number
import operator
import os
import shutil
import sys
from typing import Any, Optional, TextIO
import unicodedata

__all__ = ["BoxPrinter", "TerminalSizeError", "ValueOverflowError"]

_CSI = "\x1b["


class TerminalSizeError(ValueError):
    """The display cannot fit, or the terminal changed size after drawing."""


class ValueOverflowError(ValueError):
    """A formatted value no longer fits its fixed field."""


def _positive_int(value: Any, label: str, minimum: int = 1) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be an integer >= {minimum}") from exc
    if isinstance(value, bool) or result < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return result


def _title_width(title: str) -> int:
    """Width of ordinary Unicode text, including combining marks and CJK.

    Reject terminal controls and emoji sequences whose widths are terminal
    dependent. East Asian ambiguous characters are assumed to occupy one cell.
    """
    width = 0
    for char in title:
        code = ord(char)
        category = unicodedata.category(char)
        if (category.startswith("C") or char in "\u2028\u2029"
                or 0xFE00 <= code <= 0xFE0F or 0x1F000 <= code <= 0x1FFFF):
            raise ValueError("box names must be plain text without controls or emoji")
        if category not in ("Mn", "Me"):
            width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    if width == 0 or not title.strip():
        raise ValueError("box name must not be empty or blank")
    return width


def _cursor(row: int, col: int) -> str:
    return f"{_CSI}{row};{col}H"


@dataclass(frozen=True)
class _Box:
    name: str
    values: Any  # Keep the original container, never a permanent values copy.
    rows: int
    cols: int
    number_format: str
    column_width: Optional[int]
    title_width: int


@dataclass(frozen=True)
class _Cell:
    row: int
    col: int
    width: int
    move: str


@dataclass
class _PlacedBox:
    box: _Box
    top: int
    left: int
    width: int
    height: int
    column_widths: tuple[int, ...]
    cells: tuple[_Cell, ...]
    previous: list[str]


class BoxPrinter:
    """Display mutable 1D numeric containers in fixed, bordered boxes.

    Args:
        stream: Terminal output stream; defaults to sys.stdout.
        width_padding: Extra characters per auto-sized column (default 2).
        overflow: "mark" fills an overflowing field with #; "raise" fails
            before emitting any part of the new frame.
        ascii_borders: Use +, -, | instead of Unicode box-drawing characters.
        terminal_size: Optional (columns, lines) override, useful for tests.
        force_ansi: Permit a non-TTY stream, e.g. an ANSI-aware capture tool.

    Boxes use row-major order. Configuration changes trigger one complete
    redraw on the next print. Ordinary prints never redraw static content.
    Names must be unique. This class and its input containers are not locked;
    use an external lock if another thread must update a whole frame atomically.
    """

    def __init__(
        self, *, stream: Optional[TextIO] = None, width_padding: int = 2,
        overflow: str = "mark", ascii_borders: bool = False,
        terminal_size: Optional[tuple[int, int]] = None,
        force_ansi: bool = False,
    ) -> None:
        self._stream = sys.stdout if stream is None else stream
        self._padding = _positive_int(width_padding, "width_padding", 0)
        if overflow not in ("mark", "raise"):
            raise ValueError('overflow must be "mark" or "raise"')
        self._overflow = overflow
        self._ascii = ascii_borders
        if terminal_size is not None:
            if len(terminal_size) != 2:
                raise ValueError("terminal_size must be (columns, lines)")
            terminal_size = (
                _positive_int(terminal_size[0], "terminal columns"),
                _positive_int(terminal_size[1], "terminal lines"),
            )
        self._size_override = terminal_size
        self._force_ansi = force_ansi
        self._boxes: dict[str, _Box] = {}
        self._layout: list[_PlacedBox] = []
        self._dirty = True
        self._drawn_size: Optional[tuple[int, int]] = None
        self._height = 0
        self._active = False
        self._stopped = False
        self._restore_windows = None

    def _check_open(self) -> None:
        if self._stopped:
            raise RuntimeError("this BoxPrinter has been stopped; create a new one")

    def add_box(
        self, name: str, values: Any, rows: int, cols: int,
        number_format: str = ".3f", *, column_width: Optional[int] = None,
    ) -> None:
        """Register a box, retaining values by reference.

        column_width optionally fixes every column to an exact character width.
        Otherwise each column is sized from its formatted initial contents plus
        width_padding. Initial values must fit an explicitly requested width.
        Adding after initialization redraws the layout on the next print().
        """
        self._check_open()
        if not isinstance(name, str):
            raise ValueError("box name must be a nonempty string")
        title_width = _title_width(name)
        if name in self._boxes:
            raise ValueError(f"duplicate box name: {name!r}")
        rows = _positive_int(rows, "rows")
        cols = _positive_int(cols, "cols")
        if not isinstance(number_format, str):
            raise ValueError("number_format must be a Python format specification string")
        if column_width is not None:
            column_width = _positive_int(column_width, "column_width")
        box = _Box(name, values, rows, cols, number_format, column_width, title_width)
        formatted = self._format_values(box)
        if column_width is not None and any(len(s) > column_width for s in formatted):
            raise ValueOverflowError(f"{name!r}: initial values exceed column_width")
        self._boxes[name] = box
        self._dirty = True

    def remove_box(self, name: str) -> None:
        """Remove a box; the next print redraws. Unknown names raise KeyError."""
        self._check_open()
        del self._boxes[name]
        self._dirty = True

    def clear_boxes(self) -> None:
        """Remove all boxes and invalidate the layout; next print clears it."""
        self._check_open()
        self._boxes.clear()
        self._dirty = True

    def refresh_layout(self) -> None:
        """Immediately recompute widths/positions and clear and redraw once.

        Call explicitly after resizing the terminal or to fit wider values.
        """
        self._check_open()
        self._dirty = True
        self.print()

    @staticmethod
    def _format_values(box: _Box) -> list[str]:
        values = box.values
        if getattr(values, "ndim", 1) != 1:
            raise ValueError(f"{box.name!r}: values must be one-dimensional")
        if isinstance(values, (str, bytes, bytearray, dict, set, frozenset)):
            raise ValueError(f"{box.name!r}: values must be a 1D numeric sequence")
        try:
            count = len(values)
        except TypeError as exc:
            raise ValueError(f"{box.name!r}: values must support len() and indexing") from exc
        if count > box.rows * box.cols:
            raise ValueError(
                f"{box.name!r}: {count} values exceed {box.rows * box.cols} cells"
            )
        # A per-frame snapshot keeps size changes from altering the frame halfway
        # through formatting. External locking is still needed for atomic writers.
        try:
            snapshot = [values[i] for i in range(count)]
        except (TypeError, IndexError, KeyError) as exc:
            raise ValueError(f"{box.name!r}: values cannot be indexed consistently") from exc
        result = []
        for index, value in enumerate(snapshot):
            if not isinstance(value, (Number, Decimal)):
                raise ValueError(f"{box.name!r}: value {index} is not numeric")
            try:
                rendered = format(value, box.number_format)
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError(
                    f"{box.name!r}: cannot format value {index} with {box.number_format!r}"
                ) from exc
            BoxPrinter._check_number_text(rendered, box.name)
            result.append(rendered)
        if not result:
            # Validate empty arrays too; integer-only formats such as '04d' work.
            for probe in (0, 0.0, 0j):
                try:
                    rendered = format(probe, box.number_format)
                except (ValueError, TypeError, OverflowError):
                    continue
                BoxPrinter._check_number_text(rendered, box.name)
                break
            else:
                raise ValueError(f"{box.name!r}: invalid number_format {box.number_format!r}")
        return result

    @staticmethod
    def _check_number_text(rendered: str, name: str) -> None:
        # ANSI controls, tabs, newlines, or wide Unicode fills could corrupt the
        # terminal or invalidate our stored coordinates. Numeric fields are ASCII.
        if not rendered or not rendered.isascii() or not rendered.isprintable():
            raise ValueError(f"{name!r}: formatted numbers must be printable ASCII")

    def _get_terminal_size(self) -> tuple[int, int]:
        if self._size_override is not None:
            return self._size_override
        try:
            size = os.get_terminal_size(self._stream.fileno())
        except (AttributeError, OSError, ValueError):
            size = shutil.get_terminal_size(fallback=(80, 24))
        return size.columns, size.lines

    def _field(self, text: str, width: int, name: str, index: int) -> str:
        if len(text) > width:
            if self._overflow == "raise":
                raise ValueOverflowError(
                    f"{name!r}: value {index} ({text}) exceeds field width {width}; "
                    "call refresh_layout() or reserve a larger column_width"
                )
            return "#" * width
        return text.rjust(width)

    def _build_layout(
        self, frames: list[list[str]], size: tuple[int, int],
    ) -> tuple[list[_PlacedBox], list[str], int]:
        # Reserve the last column to avoid autowrap and a bottom row for stop().
        available_cols, available_rows = size[0] - 1, size[1] - 1
        top, left, shelf_height = 1, 1, 0
        layout: list[_PlacedBox] = []
        drawing: list[str] = []
        height = 0
        tl, tr, bl, br, sep_l, sep_r, horizontal, vertical = (
            ("+", "+", "+", "+", "+", "+", "-", "|") if self._ascii else
            ("┌", "┐", "└", "┘", "├", "┤", "─", "│")
        )
        for box, strings in zip(self._boxes.values(), frames):
            # Empty columns get the widest present value, or a formatted zero.
            fallback = max((len(s) for s in strings), default=0)
            if fallback == 0:
                for probe in (0, 0.0, 0j):
                    try:
                        fallback = len(format(probe, box.number_format))
                        break
                    except (ValueError, TypeError, OverflowError):
                        continue
            widths = tuple(
                box.column_width if box.column_width is not None else
                max((len(s) for s in strings[col::box.cols]), default=fallback) + self._padding
                for col in range(box.cols)
            )
            content_width = sum(widths) + 2 * (box.cols - 1)
            inside_width = max(content_width + 2, box.title_width + 2)
            width, box_height = inside_width + 2, box.rows + 4
            if width > available_cols:
                raise TerminalSizeError(
                    f"{box.name!r} needs {width + 1} terminal columns; only {size[0]} available"
                )
            if left + width - 1 > available_cols:
                top += shelf_height + 1
                left, shelf_height = 1, 0
            bottom = top + box_height - 1
            if bottom > available_rows:
                raise TerminalSizeError(
                    f"layout needs at least {bottom + 1} terminal lines at width {size[0]}; "
                    f"only {size[1]} available; enlarge the terminal or reduce the boxes"
                )
            title_space = inside_width - box.title_width
            title_line = " " * (title_space // 2) + box.name + " " * (title_space - title_space // 2)
            lines = [tl + horizontal * inside_width + tr,
                     vertical + title_line + vertical,
                     sep_l + horizontal * inside_width + sep_r]
            cells: list[_Cell] = []
            fields: list[str] = []
            for row in range(box.rows):
                row_fields = []
                col_position = left + 2
                for col, field_width in enumerate(widths):
                    index = row * box.cols + col
                    raw = strings[index] if index < len(strings) else ""
                    if len(raw) > field_width:
                        raise ValueOverflowError(f"{box.name!r}: initial values exceed column_width")
                    field = raw.rjust(field_width)
                    row_fields.append(field)
                    fields.append(field)
                    cells.append(_Cell(top + 3 + row, col_position, field_width,
                                       _cursor(top + 3 + row, col_position)))
                    col_position += field_width + 2
                lines.append(vertical + " " + "  ".join(row_fields)
                             + " " * (inside_width - 1 - content_width) + vertical)
            lines.append(bl + horizontal * inside_width + br)
            drawing.extend(_cursor(top + offset, left) + line for offset, line in enumerate(lines))
            layout.append(_PlacedBox(box, top, left, width, box_height, widths, tuple(cells), fields))
            left += width + 2
            shelf_height = max(shelf_height, box_height)
            height = max(height, bottom)
        return layout, drawing, height

    def _enable_windows_ansi(self) -> None:
        if os.name != "nt" or not getattr(self._stream, "isatty", lambda: False)():
            return
        import ctypes
        from ctypes import wintypes
        import msvcrt

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetConsoleMode.restype = wintypes.BOOL
        kernel.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.SetConsoleMode.restype = wintypes.BOOL
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(self._stream.fileno()))
        original = wintypes.DWORD()
        if not kernel.GetConsoleMode(handle, ctypes.byref(original)):
            raise OSError(ctypes.get_last_error(), "could not read Windows console mode")
        # ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING
        if not kernel.SetConsoleMode(handle, original.value | 0x0001 | 0x0004):
            raise OSError(ctypes.get_last_error(), "could not enable Windows ANSI processing")
        self._restore_windows = lambda: kernel.SetConsoleMode(handle, original.value)

    def print(self) -> None:
        """Draw once, then overwrite only changed value fields in one write.

        Validation and formatting complete before terminal output begins. A
        resize raises TerminalSizeError; call refresh_layout() to redraw.
        No call emits a newline, including the initial draw.
        """
        self._check_open()
        if not self._force_ansi and not getattr(self._stream, "isatty", lambda: False)():
            raise RuntimeError("BoxPrinter requires an ANSI terminal (or force_ansi=True)")
        size = self._get_terminal_size()
        if not self._dirty and size != self._drawn_size:
            raise TerminalSizeError("terminal size changed; call refresh_layout() to redraw")
        frames = [self._format_values(box) for box in self._boxes.values()]
        if self._dirty:
            layout, drawing, height = self._build_layout(frames, size)
            payload = _CSI + "?25l" + _CSI + "2J" + _CSI + "H" + "".join(drawing) + _cursor(height + 1, 1)
            # Fail on encoding limitations before clearing or hiding the cursor.
            encoding = getattr(self._stream, "encoding", None)
            if encoding:
                payload.encode(encoding)
            if not self._active:
                self._enable_windows_ansi()
                self._active = True
                atexit.register(self.stop)
            self._write(payload)
            self._layout, self._height = layout, height
            self._drawn_size, self._dirty = size, False
            return

        updates: list[str] = []
        next_fields: list[list[str]] = []
        for placed, strings in zip(self._layout, frames):
            fields = []
            for index, cell in enumerate(placed.cells):
                raw = strings[index] if index < len(strings) else ""
                field = self._field(raw, cell.width, placed.box.name, index)
                fields.append(field)
                if field != placed.previous[index]:
                    updates.append(cell.move + field)
            next_fields.append(fields)
        if updates:
            self._write("".join(updates))
            for placed, fields in zip(self._layout, next_fields):
                placed.previous = fields

    def _write(self, payload: str) -> None:
        try:
            self._stream.write(payload)
            self._stream.flush()
        except Exception:
            # A failed write may leave a partial screen; never resume using it.
            try:
                self.stop()
            except Exception:
                pass
            raise

    def stop(self) -> None:
        """Restore cursor visibility and Windows console mode; safe to repeat.

        Leaves the final display on screen. No newline or clear is emitted.
        Use a context manager or try/finally for reliable cleanup on exceptions.
        """
        if self._stopped:
            return
        self._stopped = True
        atexit.unregister(self.stop)
        try:
            if self._active:
                lines = self._get_terminal_size()[1]
                self._stream.write(_cursor(max(1, min(self._height + 1, lines)), 1) + _CSI + "?25h")
                self._stream.flush()
        finally:
            self._active = False
            if self._restore_windows is not None:
                self._restore_windows()
                self._restore_windows = None

    def __enter__(self) -> BoxPrinter:
        self._check_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()
