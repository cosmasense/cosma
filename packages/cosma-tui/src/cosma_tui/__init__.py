from .config import get_config
from .onboarding import run_onboarding
from .tui import run_tui

def start_tui(directory: str = '.', base_url: str = 'http://localhost:8080'):
    # Check if this is first run and show onboarding
    config = get_config()
    if config.is_first_run():
        # Run onboarding screen with all available themes
        selected_theme = run_onboarding()
        if selected_theme is None:
            return
    
    return run_tui(directory=directory, base_url=base_url)