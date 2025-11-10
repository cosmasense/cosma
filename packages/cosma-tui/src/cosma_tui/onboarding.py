"""
Onboarding screen for EZF first-time setup
"""

from typing import Optional, List
from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal, Center
from textual.widgets import Label, Button, ListView, ListItem, Static
from textual.binding import Binding
from textual.screen import Screen
from textual.reactive import reactive

from .config import get_config


def get_available_themes() -> List[str]:
    """Get all available themes from Textual App with textual themes at top"""
    temp_app = App()
    all_themes = list(temp_app.available_themes)
    
    # Priority themes (textual themes first)
    priority_themes = ['textual-dark', 'textual-light']
    
    # Get the remaining themes, sorted alphabetically
    remaining_themes = [theme for theme in all_themes if theme not in priority_themes]
    remaining_themes.sort()
    
    # Combine: priority themes first, then the rest
    return priority_themes + remaining_themes


class ThemeSelectionScreen(Screen):
    """Screen for selecting a theme during onboarding"""
    
    BINDINGS = [
        Binding("escape,q", "quit", "Quit"),
        Binding("enter", "select_theme", "Select Theme"),
        Binding("up", "cursor_up", "Up"),
        Binding("down", "cursor_down", "Down"),
    ]

    selected_theme = reactive("textual-dark")

    def __init__(self, theme_options: List[str]):
        super().__init__()
        self.theme_options = theme_options
        self._is_previewing = False
    
    def watch_selected_theme(self, theme: str) -> None:
        """Apply theme when selection changes"""
        if theme and hasattr(self.app, 'theme'):
            self.app.theme = theme

    def compose(self) -> ComposeResult:
        """Create the onboarding UI"""
        with Center():
            with Vertical(id="onboarding-container"):
                yield Label("Welcome to EZF!", id="welcome-title")
                yield Label(
                    "Let's set up your preferences to get started.\n"
                    "Choose a theme for the interface:",
                    id="welcome-subtitle"
                )
                
                # Theme list that takes remaining space
                yield ListView(id="theme-list")
                
                with Horizontal(id="button-container"):
                    yield Button("Continue", id="continue-btn", variant="primary")
                    yield Button("Quit", id="quit-btn", variant="error")

    def on_mount(self) -> None:
        """Initialize the theme list"""
        list_view = self.query_one("#theme-list", ListView)
        
        # Add themes to the list
        for theme in self.theme_options:
            list_item = ListItem(Label(theme))
            list_item.item_data = theme
            list_view.append(list_item)
        
        # Focus the theme list
        list_view.focus()
        
        # Set initial selection to first theme
        if self.theme_options:
            list_view.index = 0
            self.selected_theme = self.theme_options[0]

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle theme selection from list - select immediately on click"""
        list_view = self.query_one("#theme-list", ListView)
        if list_view.index is not None and 0 <= list_view.index < len(self.theme_options):
            self.selected_theme = self.theme_options[list_view.index]
            # Auto-select when clicked
            self.action_select_theme()

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Handle theme preview on hover/cursor change"""
        list_view = self.query_one("#theme-list", ListView)
        if list_view.index is not None and 0 <= list_view.index < len(self.theme_options):
            theme = self.theme_options[list_view.index]
            self.selected_theme = theme

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "continue-btn":
            self.action_select_theme()
        elif event.button.id == "quit-btn":
            self.app.exit()

    def action_select_theme(self) -> None:
        """Save the selected theme and exit onboarding"""
        list_view = self.query_one("#theme-list", ListView)
        if list_view.index is not None and 0 <= list_view.index < len(self.theme_options):
            self.selected_theme = self.theme_options[list_view.index]
            
            # Save the theme to config
            config = get_config()
            config.set_theme(self.selected_theme)
            
            # Return the selected theme
            self.app.exit(self.selected_theme)

    def action_cursor_up(self) -> None:
        """Move cursor up in the theme list"""
        list_view = self.query_one("#theme-list", ListView)
        current_index = list_view.index
        list_view.action_cursor_up()
        # Apply theme preview if index changed
        if (list_view.index is not None and 
            list_view.index != current_index and 
            0 <= list_view.index < len(self.theme_options)):
            self.selected_theme = self.theme_options[list_view.index]

    def action_cursor_down(self) -> None:
        """Move cursor down in the theme list"""
        list_view = self.query_one("#theme-list", ListView)
        current_index = list_view.index
        list_view.action_cursor_down()
        # Apply theme preview if index changed
        if (list_view.index is not None and 
            list_view.index != current_index and 
            0 <= list_view.index < len(self.theme_options)):
            self.selected_theme = self.theme_options[list_view.index]

    def action_quit(self) -> None:
        """Quit the application"""
        self.app.exit()


class OnboardingApp(App):
    """Simple app for onboarding flow"""
    
    CSS = """
    Screen {
        background: $surface;
        overflow: hidden;
    }

    #onboarding-container {
        width: 60;
        height: 22;
        padding: 2;
        border: solid $primary;
        background: $surface;
    }

    #welcome-title {
        text-align: center;
        text-style: bold;
        color: $primary;
        margin: 0 0 1 0;
        height: 1;
    }

    #welcome-subtitle {
        text-align: center;
        margin: 0 0 1 0;
        height: 2;
    }

    #theme-list {
        height: 1fr;
        border: solid $primary;
        overflow-y: scroll;
        margin-bottom: 1;
    }

    #button-container {
        align: center middle;
        height: 3;
    }

    Button {
        margin: 0 1;
    }

    ListItem {
        padding: 0 1;
        height: 1;
    }

    ListItem:hover {
        background: $primary-lighten-1;
    }

    ListView:focus ListItem.--highlight {
        background: $primary;
    }

    /* Ensure proper scrolling */
    ListView {
        scrollbar-background: $surface;
        scrollbar-color: $primary;
    }
    """

    def on_mount(self) -> None:
        """Show the theme selection screen"""
        # Get all available themes dynamically
        themes = get_available_themes()
        self.push_screen(ThemeSelectionScreen(themes))


def run_onboarding() -> Optional[str]:
    """Run the onboarding flow and return the selected theme"""
    app = OnboardingApp()
    return app.run()