from rich.console import Console


class Printer:
    def __init__(self):
        self.console = Console()

    def thinking_start(self):
        self.console.print(f"[dim cyan]◌ Thinking[/dim cyan]")

    def thinking(self, text):
        self.console.print(f"[dim cyan]{text}[/dim cyan]", end="")

    def tool_name(self, name):
        self.console.print(f"[yellow]⚙ {name}[/yellow]")

    def tool_result(self, text):
        self.console.print(f"[dim]{text}[/dim]")

    def response_start(self):
        self.console.print(f"[bold green]● Response[/bold green]")

    def response(self, text):
        self.console.print(text, end="")

    def newline(self):
        self.console.print()