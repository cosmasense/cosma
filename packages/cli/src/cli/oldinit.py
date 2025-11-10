#!/usr/bin/env python3
"""
EZ Find CLI tool - A command-line interface for file indexing and searching
"""

import argparse
import pathlib
import sys

from rich import print
from rich.console import Console
from rich.table import Table

from tui import start_tui

console = Console()


def cmd_index(args):
    """Index a directory"""
    client = Client(base_url=args.base_url)
    
    try:
        path = pathlib.Path(args.directory).resolve()
        console.print(f"[yellow]Indexing directory:[/yellow] {str(path)}")
        result = client.index_directory(str(path))
        
        if result.get('success'):
            console.print(f"[green]✓[/green] {result.get('message', 'Success')}")
            console.print(f"Files indexed: {result.get('files_indexed', 0)}")
        else:
            console.print(f"[red]✗[/red] Indexing failed")
            print(result)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    finally:
        client.close()


def cmd_search(args):
    """Search for files"""
    client = Client(base_url=args.base_url)
    
    try:
        console.print(f"[yellow]Searching for:[/yellow] {args.query}")
        
        filters = {}
        if args.extension:
            filters['extension'] = args.extension
        if args.date_from:
            filters['date_from'] = args.date_from
        
        result = client.search(
            query=args.query,
            filters=filters if filters else None,
            limit=args.limit
        )
        
        if 'results' in result:
            results = result['results']
            total = result.get('total', len(results))
            
            console.print(f"\n[green]Found {total} result(s)[/green]\n")
            
            if results:
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Filename", style="dim")
                table.add_column("Match type", style="cyan")
                table.add_column("Combined Score", style="green", width=10)
                table.add_column("Semantic Score", justify="right", width=10)
                
                for item in results:
                    table.add_row(
                        str(item.get('filename', 'N/A')),
                        item.get('match_type', 'N/A'),
                        f"{item.get('combined_score', 0):.2f}",
                        f"{item.get('semantic_score', 0):.2f}"
                    )
                
                console.print(table)
                
                # Show summaries if available
                if args.show_summary:
                    console.print("\n[bold]Summaries:[/bold]\n")
                    for item in results:
                        if item.get('summary'):
                            console.print(f"[cyan]{item.get('filename')}:[/cyan]")
                            console.print(f"  {item.get('summary')}\n")
            else:
                console.print("[yellow]No results found[/yellow]")
        else:
            console.print("[red]Unexpected response format[/red]")
            print(result)
            
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    finally:
        client.close()


def cmd_status(args):
    """Check indexing status"""
    client = Client(base_url=args.base_url)
    
    try:
        result = client.index_status()
        
        console.print("\n[bold]Indexing Status[/bold]\n")
        console.print(f"Status: {'[yellow]Indexing[/yellow]' if result.get('is_indexing') else '[green]Idle[/green]'}")
        
        if result.get('is_indexing'):
            console.print(f"Current file: {result.get('current_file', 'N/A')}")
            console.print(f"Progress: {result.get('files_processed', 0)}/{result.get('total_files', 0)}")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    finally:
        client.close()


def main():
    # Check if first arg is a subcommand
    subcommands = ['index', 'search', 'status']
    is_subcommand = len(sys.argv) > 1 and sys.argv[1] in subcommands
    
    parser = argparse.ArgumentParser(
        description="EZ Find - File indexing and search CLI tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--base-url',
        default='http://127.0.0.1:8080',
        help='API base URL (default: http://127.0.0.1:8080)'
    )
    
    # Only add directory argument if not using a subcommand
    if not is_subcommand:
        parser.add_argument(
            'directory',
            nargs='?',
            default='.',
            help='Directory to index and search (default: current directory)'
        )
    
    # subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # # Index command
    # index_parser = subparsers.add_parser('index', help='Index a directory')
    # index_parser.add_argument(
    #     'directory',
    #     help='Directory path to index'
    # )
    # index_parser.set_defaults(func=cmd_index)
    
    # # Search command
    # search_parser = subparsers.add_parser('search', help='Search for files')
    # search_parser.add_argument(
    #     'query',
    #     help='Search query'
    # )
    # search_parser.add_argument(
    #     '--limit',
    #     type=int,
    #     default=50,
    #     help='Maximum number of results (default: 50)'
    # )
    # search_parser.add_argument(
    #     '--extension',
    #     help='Filter by file extension (e.g., py, txt)'
    # )
    # search_parser.add_argument(
    #     '--date-from',
    #     help='Filter by date from (YYYY-MM-DD)'
    # )
    # search_parser.add_argument(
    #     '--show-summary',
    #     action='store_true',
    #     help='Show file summaries in results'
    # )
    # search_parser.set_defaults(func=cmd_search)
    
    # # Status command
    # status_parser = subparsers.add_parser('status', help='Check indexing status')
    # status_parser.set_defaults(func=cmd_status)
    
    args = parser.parse_args()
    
    start_tui()
