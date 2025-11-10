#!/usr/bin/env python3
"""
EZF TUI - A fuzzy finder interface using Textual
"""

import json
import sys
import asyncio
from typing import List, Optional
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Input, Static, ListView, ListItem, Label
from textual.binding import Binding
from textual.worker import Worker, WorkerState

from .client import Client
from .models import Update
from .config import get_config


class SearchListView(ListView):
    """Custom ListView for search results"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.displayed_items = []  # Store items in display order

    def update_results(self, results: List[dict]) -> None:
        """Update the ListView with new search results (displayed bottom to top)"""
        self.displayed_items = []  # Reset displayed items
        self.clear()
        
        # Add items in reverse order so first result appears at bottom
        for item in reversed(results):
            # Extract file info from nested structure
            file_info = item.get('file', {})
            title = file_info.get('title')
            filename = file_info.get('filename', 'N/A')
            score = item.get('relevance_score', 0)
            
            # Build the full filename with extension
            # full_filename = f"{filename}{extension}" if extension else filename
            
            # Build label text: title (if exists), filename (muted), score (more muted)
            if title:
                label_text = f"{title} [dim]{filename}[/] [dim italic]({score:.2f})[/]"
            else:
                label_text = f"[dim]{filename}[/] [dim italic]({score:.2f})[/]"
            
            list_item = ListItem(Label(label_text))
            list_item.item_data = item  # type: ignore
            self.append(list_item)
            self.displayed_items.append(item)  # Store in display order
        
        # Reset to last item if results exist (which is actually the first result)
        if self.displayed_items:
            self.index = len(self.displayed_items) - 1

    def get_selected_item(self) -> Optional[dict]:
        """Get the currently selected item"""
        try:
            if self.index is not None and 0 <= self.index < len(self.displayed_items):
                item = self.displayed_items[self.index]
                print(f"DEBUG: index={self.index}, item={item}", flush=True)
                return item
            else:
                print(f"DEBUG: index={self.index}, len_displayed={len(self.displayed_items)}, out of range", flush=True)
            return None
        except Exception as e:
            print(f"DEBUG: Error in get_selected_item: {e}", flush=True)
            return None


class EZFApp(App):
    """EZF Fuzzy Finder Application"""

    CSS = """
    Screen {
        background: $surface;
    }

    .container {
        height: 100%;
    }

    .list-wrapper {
        height: 1fr;
        align: center bottom;
    }

    Input {
        border: solid $primary;
    }

    SearchListView {
        border: solid $primary;
        height: 100%;
    }

    ListView {
        align: center bottom;
    }

    .status {
        height: 1;
        background: $primary-darken-2;
        color: $text;
        padding: 0 1;
    }

    ListItem {
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("escape,ctrl+c,q", "quit", "Quit"),
        Binding("enter", "select", "Select"),
        Binding("up", "cursor_up", "Up"),
        Binding("down", "cursor_down", "Down"),
    ]

    def __init__(self, directory: str = "./test2", base_url: str = "http://127.0.0.1:8080"):
        super().__init__()
        self.directory = str(Path(directory).resolve())
        self.base_url = base_url
        self.client = Client(base_url=base_url)
        self.selected_item: Optional[str] = None
        self.status_message: str = "Initializing..."
        self.running = True
        self.search_throttle_task: Optional[asyncio.Task] = None
        self.is_searching: bool = False
        self.pending_query: Optional[str] = None
        self.last_search_time: float = 0.0
        
        # Load and apply theme from config
        config = get_config()
        theme = config.get_theme()
        if theme:
            self.theme = theme

    def compose(self) -> ComposeResult:
        """Create the UI layout"""
        with Vertical(classes="container"):
            with Vertical(classes="list-wrapper"):
                yield SearchListView(id="list", initial_index=0)

            yield Static(self.status_message, classes="status", id="status")
            yield Input(placeholder="Type to search...", id="search")

    def on_mount(self) -> None:
        """Initialize the app when mounted"""
        # Focus the input initially
        self.query_one("#search", Input).focus()
        
        # Start indexing in the background
        self.update_status(f"Indexing {self.directory}...")
        self.run_worker(self.index_directory(), exclusive=True, group="indexing")
        
        # Start SSE listener as a worker
        self.run_worker(self.listen_to_updates(), exclusive=False, group="sse")

    async def listen_to_updates(self) -> None:
        """Listen to server-sent events and update status bar"""
        try:
            self.log("Starting SSE listener...")
            stream = self.client.stream_updates()
            self.log(f"Got stream object: {stream}")
            self.log("About to enter async for loop...")
            async for update in stream:
                self.log(f"Inside async for loop, got update: {update}")
                if not self.running:
                    break
                # Update is now an Update instance, use its display message
                if isinstance(update, Update):
                    self.update_status(update.get_display_message())
                else:
                    # Fallback for unexpected data
                    self.update_status(str(update))
        except Exception as e:
            self.log(f"SSE Exception: {e}")
            self.update_status(f"SSE Error: {str(e)}")

    def update_status(self, message: str) -> None:
        """Update the status bar from any thread"""
        status = self.query_one("#status", Static)
        # Add loading indicator if searching
        if self.is_searching:
            message = f"⏳ {message}"
        status.update(message)

    async def index_directory(self) -> dict:
        """Index the directory (runs in worker)"""
        return await self.client.index_directory(self.directory)
    
    def handle_index_result(self, result) -> None:
        """Handle indexing result"""
        try:
            if result.get('success'):
                # files_indexed = result.get('files_indexed', 0)
                self.update_status("Indexing files. Type to search...")
            else:
                self.update_status("Indexing failed")
        except Exception as e:
            self.update_status(f"Error: {str(e)}")

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes with throttling"""
        if event.input.id == "search":
            query = event.value.strip()
            if query:
                self.pending_query = query
                # Start throttle task if not already running
                if self.search_throttle_task is None or self.search_throttle_task.done():
                    self.search_throttle_task = asyncio.create_task(
                        self.throttled_search()
                    )
            else:
                # Clear results if query is empty
                self.pending_query = None
                list_view = self.query_one("#list", SearchListView)
                list_view.update_results([])
                self.is_searching = False
                self.update_status("Type to search...")
    
    async def throttled_search(self) -> None:
        """Throttle searches to max one per 0.5 seconds"""
        import time
        
        while self.pending_query is not None:
            current_query = self.pending_query
            current_time = time.time()
            time_since_last = current_time - self.last_search_time
            
            # Wait if we searched too recently
            if time_since_last < 0.5:
                await asyncio.sleep(0.5 - time_since_last)
            
            # Check if query changed during wait
            if self.pending_query != current_query:
                continue
            
            # Perform the search
            self.last_search_time = time.time()
            self.is_searching = True
            self.update_status("Searching...")
            self.run_worker(
                self.perform_search(current_query), 
                exclusive=True, 
                group="search"
            )
            
            # Wait a bit to see if more input comes
            await asyncio.sleep(0.1)
            
            # If query hasn't changed, we're done
            if self.pending_query == current_query:
                self.pending_query = None
                break

    async def perform_search(self, query: str) -> dict:
        """Perform search (runs in worker)"""
        return await self.client.search(query, None, 50)
    
    def handle_search_result(self, result) -> None:
        """Handle search result"""
        try:
            self.is_searching = False
            if 'results' in result:
                results = result['results']
                list_view = self.query_one("#list", SearchListView)
                list_view.update_results(results)
                self.update_status(f"Found {len(results)} results")
        except Exception as e:
            self.is_searching = False
            self.update_status(f"Search error: {str(e)}")
    
    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle worker state changes"""
        if event.state == WorkerState.SUCCESS:
            if event.worker.group == "indexing":
                self.handle_index_result(event.worker.result)
            elif event.worker.group == "search":
                self.handle_search_result(event.worker.result)
        elif event.state == WorkerState.ERROR:
            self.update_status(f"Worker error: {event.worker.error}")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle enter key press in search input"""
        try:
            self.log(f"Input.Submitted event: {event}")
            self.log(f"Input value: {event.input.value}")
            self.log("Triggering action_select from input submitted")
            self.action_select()
        except Exception as e:
            self.log(f"Error in on_input_submitted: {e}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")
    
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle list item selection (mouse click)"""
        try:
            self.log(f"ListView.Selected event: {event}")
            self.log(f"Event attributes: {dir(event)}")
            # Get the list view that triggered the event
            list_view = self.query_one("#list", SearchListView)
            self.action_select()
        except Exception as e:
            self.log(f"Error in on_list_view_selected: {e}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")

    def action_select(self) -> None:
        """Handle item selection"""
        try:
            self.log("=== action_select called ===")
            list_view = self.query_one("#list", SearchListView)
            self.log(f"list_view.index: {list_view.index}")
            self.log(f"list_view.displayed_items length: {len(list_view.displayed_items)}")
            
            selected = list_view.get_selected_item()
            self.log(f"selected item: {selected}")
            
            if selected:
                # Get the file path to print
                file_info = selected.get('file', {})
                self.log(f"file_info: {file_info}")
                file_path = file_info.get('file_path', selected.get('filename', ''))
                self.log(f"file_path: {file_path}")
                
                # Print the selected file path (like fzf)
                print(file_path, flush=True)
                self.log(f"Printed file path: {file_path}")
                
                # Store the selected item and exit
                self.selected_item = file_path
                self.running = False
                self.log("Calling self.exit()")
                self.exit(self.selected_item)
            else:
                self.log("No selected item found")
        except Exception as e:
            self.log(f"Error in action_select: {e}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")

    def action_cursor_up(self) -> None:
        """Move cursor up in the list"""
        list_view = self.query_one("#list", SearchListView)
        list_view.action_cursor_up()

    def action_cursor_down(self) -> None:
        """Move cursor down in the list"""
        list_view = self.query_one("#list", SearchListView)
        list_view.action_cursor_down()

    async def action_quit(self) -> None:
        """Quit the application"""
        self.running = False
        await self.client.close()
        self.exit()

    async def on_unmount(self) -> None:
        """Cleanup when app is unmounting"""
        self.running = False
        await self.client.close()


def run_tui(directory: str = ".", base_url: str = "http://127.0.0.1:8080") -> Optional[str]:
    """Run the TUI and return the selected item"""
    app = EZFApp(directory=directory, base_url=base_url)
    
    # Apply theme from config
    config = get_config()
    theme = config.get_theme()
    if theme:
        app.theme = theme
    
    selected_item = app.run()
    return selected_item
