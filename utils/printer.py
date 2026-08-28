from rich.console import Console

_printer = None


def get_printer() -> Printer:
    global _printer
    if _printer is None:
        _printer = Printer()
    return _printer


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