import json
import sys
import tty
import termios
from rich.console import Console


def print_generation_result(rprint, row):
    index = row["index"]
    input = row["input"]
    target = row["target"]
    token_count = row["token_count"]
    generation = row["generation"]
    metrics = row["metrics"]

    rprint(f"[red]{'-' * 20} Sample {index + 1} [/red]")
    rprint("[cyan]------- Input text:[/cyan]")
    print(input)
    rprint("[cyan]------- Target text:[/cyan]")
    print(target)

    rprint(f"[green]------- Generated text ({token_count} tokens):[/green]")
    print(generation)

    rprint("[green]------- Metrics:[/green]")
    for k, v in metrics.items():
        try:
            rprint(f"{k}: {float(v):.5f}")
        except Exception:
            rprint(f"{k}: {v}")


def print_unknown_row(rprint, row):
    for k, v in row.items():
        rprint(f"[cyan]{k}[/cyan]")
        print(v)


def read_jsonl_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def get_single_keypress():
    """Capture a single keypress from stdin without Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def prompt_user_input(prompt):
    """Restore terminal mode and get full input from the user."""
    fd = sys.stdin.fileno()
    termios.tcsetattr(fd, termios.TCSADRAIN, termios.tcgetattr(fd))
    return input(prompt)


def main(filepath):
    try:
        entries = read_jsonl_file(filepath)
    except Exception as e:
        print(f"Failed to read JSONL file: {e}")
        sys.exit(1)

    if not entries:
        print("No data found in the file.")
        return

    console = Console()
    index = 0
    num_entries = len(entries)

    def render_entry():
        console.clear()
        if (
            "index" in entries[index]
            and "input" in entries[index]
            and "target" in entries[index]
            and "generation" in entries[index]
        ):
            print_generation_result(console.print, entries[index])
        else:
            print_unknown_row(console.print, entries[index])
        console.print(
            "\n[dim]Press 'h' (back), 'l' (forward), 'j' (jump), 'q' (quit)[/dim]"
        )

    render_entry()

    while True:
        key = get_single_keypress()
        if key == "q":
            break
        elif key == "l":
            if index < num_entries - 1:
                index += 1
                render_entry()
        elif key == "h":
            if index > 0:
                index -= 1
                render_entry()
        elif key == "j":
            # Restore terminal state to accept full input
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            try:
                jump_str = input(
                    "Enter index or percentage (e.g. 10 or 75%): "
                ).strip()
            except KeyboardInterrupt:
                continue  # allow Ctrl+C to cancel
            finally:
                # Re-enter raw mode for keypress
                tty.setraw(fd)

            if jump_str.endswith("%"):
                try:
                    percent = float(jump_str.rstrip("%"))
                    new_index = int((percent / 100) * num_entries)
                    if 0 <= new_index < num_entries:
                        index = new_index
                        render_entry()
                except ValueError:
                    pass
            else:
                try:
                    new_index = int(jump_str)
                    if 0 <= new_index < num_entries:
                        index = new_index
                        render_entry()
                except ValueError:
                    pass


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_generations.py <path_to_file.jsonl>")
        sys.exit(1)

    filepath = sys.argv[1]
    main(filepath)
