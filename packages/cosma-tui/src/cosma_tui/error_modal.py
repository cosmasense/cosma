import sys
from textual.app import ComposeResult
from textual.containers import Grid
from textual.screen import ModalScreen
from textual.widgets import Button, Label
from textual.binding import Binding


class ConnectionErrorModal(ModalScreen[bool]):  
    """Screen with a dialog to start backend."""

    BINDINGS = [
        Binding("escape", "quit", "Quit"),
    ]

    CSS = """
    ConnectionErrorModal {
        align: center middle;
        background: rgba(0,0,0,0.5);
    }

    #dialog {
        grid-size: 2;
        grid-gutter: 1 2;
        grid-rows: 1fr 1fr 3;
        padding: 0 1;
        width: 60;
        height: 16;
        border: thick $background 80%;
        background: $surface;
    }

    #message {
        column-span: 2;
        height: 1fr;
        width: 1fr;
        content-align: center middle;
    }

    #instruction {
        column-span: 2;
        height: 1fr;
        width: 1fr;
        content-align: center middle;
    }

    #command {
        column-span: 2;
        height: 1fr;
        width: 1fr;
        content-align: center middle;
        text-style: bold;
        background: $panel;
        border: solid $secondary;
        padding: 0 1;
    }

    Button {
        width: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        yield Grid(
            Label("Could not connect to backend", id="message"),
            Label("Please open a new terminal and run:", id="instruction"),
            Label("cosma serve", id="command"),
            Button("Retry", variant="primary", id="retry"),
            Button("Quit", variant="error", id="quit"),
            id="dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit":
            self.dismiss(True)
        else:
            self.dismiss(False)

    def action_quit(self) -> None:
        """Handle escape key press - quit the application."""
        self.dismiss(True)
