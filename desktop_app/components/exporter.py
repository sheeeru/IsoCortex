"""
IsoCortex Desktop App — Export Utility
=======================================
Export chat conversations, AI answers, and search results
as Markdown files or to the clipboard.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from desktop_app.engine import IsoCortexEngine


def export_conversation_markdown(
    engine: IsoCortexEngine,
    conversation_id: str,
    output_path: Optional[str] = None,
) -> str:
    """Export an entire conversation as a Markdown file.

    Args:
        engine: The IsoCortex engine instance.
        conversation_id: UUID of the conversation.
        output_path: Where to save. If None, saves to ~/Downloads/.

    Returns:
        Path to the saved file.
    """
    messages = engine.get_conversation_messages(conversation_id)
    if not messages:
        raise ValueError("No messages to export")

    # Get conversation title
    convs = [c for c in engine.list_conversations() if c["conversation_id"] == conversation_id]
    title = convs[0]["title"] if convs else "Chat Export"

    md_lines = [
        f"# {title}",
        f"",
        f"*Exported from IsoCortex on {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        f"",
        f"---",
        f"",
    ]

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])

        if role == "user":
            md_lines.append(f"## You")
            md_lines.append(f"")
            md_lines.append(f"{content}")
            md_lines.append(f"")
        elif role == "assistant":
            md_lines.append(f"## IsoCortex AI")
            md_lines.append(f"")
            md_lines.append(f"{content}")
            md_lines.append(f"")

            if sources:
                md_lines.append(f"**Sources:**")
                for src in sources:
                    file_name = src.get("file", "unknown")
                    page = src.get("page", 0)
                    score = src.get("score", 0)
                    ref = f"- [{file_name}]"
                    if page:
                        ref += f" (page {page})"
                    ref += f" — relevance: {score:.0%}"
                    md_lines.append(ref)
                md_lines.append(f"")

        md_lines.append(f"---")
        md_lines.append(f"")

    md_content = "\n".join(md_lines)

    if output_path is None:
        downloads = Path.home() / "Downloads"
        downloads.mkdir(exist_ok=True)
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
        output_path = str(downloads / f"isocortex_{safe_title}.md")

    Path(output_path).write_text(md_content, encoding="utf-8")
    return output_path


def export_single_answer_markdown(
    query: str,
    answer: str,
    sources: list | None = None,
    output_path: Optional[str] = None,
) -> str:
    """Export a single Q&A pair as Markdown.

    Args:
        query: The user's question.
        answer: The AI's response.
        sources: List of source dicts.
        output_path: Where to save. If None, saves to ~/Downloads/.

    Returns:
        Path to the saved file.
    """
    md_lines = [
        f"# IsoCortex AI Answer",
        f"",
        f"*Exported on {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        f"",
        f"---",
        f"",
        f"## Question",
        f"",
        f"{query}",
        f"",
        f"## Answer",
        f"",
        f"{answer}",
        f"",
    ]

    if sources:
        md_lines.append(f"## Sources")
        md_lines.append(f"")
        for i, src in enumerate(sources):
            file_name = src.get("file", "unknown")
            page = src.get("page", 0)
            score = src.get("score", 0)
            md_lines.append(f"{i+1}. **{file_name}**" + (f" (page {page})" if page else "") + f" — {score:.0%}")
        md_lines.append(f"")

    md_content = "\n".join(md_lines)

    if output_path is None:
        downloads = Path.home() / "Downloads"
        downloads.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(downloads / f"isocortex_answer_{timestamp}.md")

    Path(output_path).write_text(md_content, encoding="utf-8")
    return output_path


def copy_answer_to_clipboard(answer: str, root_widget=None) -> bool:
    """Copy an AI answer to the system clipboard.

    Args:
        answer: The text to copy.
        root_widget: An existing Tk/CTk root widget. If None, attempts
                     to use the clipboard without creating a new root.

    Returns:
        True if successful.
    """
    try:
        # Reuse the app's existing root window — creating a second
        # tk.Tk() in the same process raises TclError.
        if root_widget is not None:
            root_widget.clipboard_clear()
            root_widget.clipboard_append(answer)
            root_widget.update()
            return True

        # Fallback: try to grab the active root via CTk internals
        try:
            import customtkinter as ctk
            existing = ctk._get_current_root()
            if existing is not None:
                existing.clipboard_clear()
                existing.clipboard_append(answer)
                existing.update()
                return True
        except Exception:
            pass

        # Last resort: create a temporary root (only works if no root exists)
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(answer)
        root.update()
        root.destroy()
        return True
    except Exception:
        return False