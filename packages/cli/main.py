#!/usr/bin/env python3
"""
A fuzzy finder CLI tool built with Textual
Usage: python fuzzy_finder.py [file_or_directory]
"""

import sys
import os
from pathlib import Path
from typing import List, Optional
import argparse

from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Input, Static, ListView, ListItem, Label
from textual.binding import Binding


class FuzzyListView(ListView):
    """Custom ListView that handles fuzzy filtering"""

    def __init__(self, items: List[str], **kwargs):
        super().__init__(**kwargs)
        self.all_items = items
        self.filtered_items = items.copy()

    def filter_items(self, query: str) -> None:
        """Filter items based on fuzzy matching"""
        if not query:
            self.filtered_items = self.all_items.copy()
        else:
            self.filtered_items = []
            query_lower = query.lower()

            for item in self.all_items:
                # Simple fuzzy matching: check if all query chars appear in order
                if self._fuzzy_match(item.lower(), query_lower):
                    self.filtered_items.append(item)

        self.refresh_list()

    def _fuzzy_match(self, text: str, query: str) -> bool:
        """Check if query characters appear in text in order"""
        text_idx = 0
        for char in query:
            while text_idx < len(text) and text[text_idx] != char:
                text_idx += 1
            if text_idx >= len(text):
                return False
            text_idx += 1
        return True

    def refresh_list(self) -> None:
        """Refresh the ListView with filtered items"""
        self.clear()
        for item in self.filtered_items:
            list_item = ListItem(Label(item))
            list_item.item_data = item  # type: ignore
            self.append(list_item)

    def get_selected_item(self) -> Optional[str]:
        """Get the currently selected item"""
        if self.index is not None and 0 <= self.index < len(self.filtered_items):
            return self.filtered_items[self.index]
        return None


class FuzzyFinder(App):
    """A fuzzy finder application"""

    CSS = """
    Screen {
        background: $surface;
    }

    .container {
        height: 100%;
    }

    Input {
        border: solid $primary;
    }

    .list-container {
        height: 1fr;
    }

    FuzzyListView {
        border: solid $primary;
    }

    .status {
        height: 1;
        background: $primary-darken-2;
        color: $text;
        text-align: center;
    }

    ListItem {
        padding: 0 1;
    }

    ListItem:hover {
        background: $primary-lighten-1;
    }

    ListView:focus ListItem.--highlight {
        background: $primary;
    }
    """

    BINDINGS = [
        Binding("escape,ctrl+c,q", "quit", "Quit"),
        Binding("enter", "select", "Select"),
        Binding("up", "cursor_up", "Up"),
        Binding("down", "cursor_down", "Down"),
    ]

    def __init__(self, items: List[str]):
        super().__init__()
        self.items = items
        self.selected_item: Optional[str] = None

    def compose(self) -> ComposeResult:
        """Create the UI layout"""
        with Vertical(classes="container"):
            with Vertical(classes="list-container"):
                yield FuzzyListView(self.items, id="list", initial_index=0)

            yield Static(
                f"[bold]Items: {len(self.items)}[/] | ↑↓ Navigate | Enter Select | Esc Quit",
                classes="status"
            )

            # with Horizontal(classes="input-container"):
            yield Input(placeholder="Type to filter...", id="search")

    def on_mount(self) -> None:
        """Initialize the app when mounted"""
        list_view = self.query_one("#list", FuzzyListView)
        list_view.refresh_list()

        # Focus the input initially
        self.query_one("#search", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes"""
        if event.input.id == "search":
            list_view = self.query_one("#list", FuzzyListView)
            list_view.filter_items(event.value)

            # Update status bar
            status = self.query_one(".status", Static)
            filtered_count = len(list_view.filtered_items)
            total_count = len(list_view.all_items)
            status.update(
                f"[bold]Items: {filtered_count}/{total_count}[/] | ↑↓ Navigate | Enter Select | Esc Quit"
            )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle enter key press in search input"""
        self.action_select()

    def action_select(self) -> None:
        """Handle item selection"""
        list_view = self.query_one("#list", FuzzyListView)
        selected = list_view.get_selected_item()
        if selected:
            self.selected_item = selected
            self.exit()

    def action_cursor_up(self) -> None:
        """Move cursor up in the list"""
        list_view = self.query_one("#list", FuzzyListView)
        list_view.action_cursor_up()

    def action_cursor_down(self) -> None:
        """Move cursor down in the list"""
        list_view = self.query_one("#list", FuzzyListView)
        list_view.action_cursor_down()

    def action_quit(self) -> None:
        """Quit the application"""
        self.exit()


def get_file_list(path: str) -> List[str]:
    """Get list of files from a directory or read from a file"""
    path_obj = Path(path)

    if path_obj.is_file():
        # Read lines from file
        try:
            with open(path_obj, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading file {path}: {e}", file=sys.stderr)
            return []

    elif path_obj.is_dir():
        # List files in directory
        try:
            items = []
            for item in path_obj.rglob('*'):
                if item.is_file():
                    items.append(str(item.relative_to(path_obj)))
            return sorted(items)
        except Exception as e:
            print(f"Error reading directory {path}: {e}", file=sys.stderr)
            return []

    else:
        print(f"Path not found: {path}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(description="Fuzzy finder CLI tool")
    parser.add_argument(
        "path",
        nargs='?',
        default=".",
        help="File or directory to search (default: current directory)"
    )

    args = parser.parse_args()

    # Get items to search through
    if not sys.stdin.isatty():
        # Read from stdin
        items = [line.strip() for line in sys.stdin if line.strip()]
    else:
        # Read from file or directory
        items = get_file_list(args.path)

    if not items:
        print("No items to search", file=sys.stderr)
        sys.exit(1)

    # Run the fuzzy finder
    app = FuzzyFinder(items)
    app.run()

    # Output the selected item
    if app.selected_item:
        print(app.selected_item)
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
