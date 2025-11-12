from .config import get_config
from .tui import run_tui

def start_tui(directory: str = '.', base_url: str = 'http://localhost:60534'):
    # Check if this is first run and pass flag to show onboarding
    config = get_config()
    show_onboarding = config.is_first_run()
    
    return run_tui(directory=directory, base_url=base_url, show_onboarding=show_onboarding)