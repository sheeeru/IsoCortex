"""
IsoCortex — CLI Entry Point
============================

Command-line interface for IsoCortex semantic search engine.

SRS References:
  - FR-CLI-001: Command structure
  - NFR-11: CLI usability (--help, --json, progress bars, color output)

Usage:
  isocortex index create --name my-docs --path ./documents
  isocortex index list
  isocortex index info my-docs
  isocortex index delete my-docs
  isocortex search my-docs --query "How does auth work?" --top-k 5
  isocortex serve --host 0.0.0.0 --port 8000
  isocortex user create --username admin --role admin

Author : Shaheer Qureshi
Project: IsoCortex
"""

from __future__ import annotations

import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="isocortex",
    help="IsoCortex — Production-grade local semantic search engine",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

# Register sub-commands
index_app = typer.Typer(name="index", help="Manage HNSW indices", no_args_is_help=True)
search_app = typer.Typer(name="search", help="Search across indices", no_args_is_help=True)
user_app = typer.Typer(name="user", help="User management", no_args_is_help=True)
job_app = typer.Typer(name="jobs", help="Manage background jobs", no_args_is_help=True)

app.add_typer(index_app, name="index")
app.add_typer(search_app, name="search")
app.add_typer(user_app, name="user")
app.add_typer(job_app, name="jobs")


# =============================================================================
# Index Commands (SRS FR-CLI-001)
# =============================================================================

@index_app.command("create")
def index_create(
    name: str = typer.Option(..., "--name", "-n", help="Index name (unique)"),
    path: str = typer.Option(..., "--path", "-p", help="File/directory path to index"),
    description: str = typer.Option("", "--description", "-d", help="Index description"),
    chunk_size: int = typer.Option(512, help="Tokens per chunk"),
    chunk_overlap: int = typer.Option(50, help="Token overlap between chunks"),
    m: int = typer.Option(16, help="HNSW M parameter"),
    ef_construction: int = typer.Option(200, help="HNSW ef_construction"),
    ef_search: int = typer.Option(50, help="HNSW ef_search"),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
) -> None:
    """Create a new HNSW index from files or a directory."""
    from isocortex.config import load_config
    from isocortex.engine.indexing.manager import IndexManager

    config = load_config()
    mgr = IndexManager(config.storage.indices_dir)

    try:
        mgr.create_index(
            name=name,
            description=description,
            hnsw_params={
                "M": m,
                "ef_construction": ef_construction,
                "ef_search": ef_search,
                "metric": "cosine",
            },
            chunk_config={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        )
        if json_output:
            import json
            console.print_json(json.dumps({"status": "created", "name": name}))
        else:
            console.print(f"[green]✓[/green] Index '{name}' created successfully")
    except FileExistsError:
        console.print(f"[red]✗[/red] Index '{name}' already exists")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]✗[/red] Error: {exc}")
        raise typer.Exit(1)


@index_app.command("list")
def index_list(
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
) -> None:
    """List all indexes."""
    from isocortex.config import load_config
    from isocortex.engine.indexing.manager import IndexManager

    config = load_config()
    mgr = IndexManager(config.storage.indices_dir)
    indexes = mgr.list_indexes()

    if json_output:
        import json
        console.print_json(json.dumps([i.to_dict() for i in indexes]))
    else:
        if not indexes:
            console.print("[yellow]No indexes found[/yellow]")
            return

        table = Table(title="IsoCortex Indexes")
        table.add_column("Name", style="cyan")
        table.add_column("Vectors", justify="right")
        table.add_column("Deleted", justify="right")
        table.add_column("Created", style="dim")
        table.add_column("Status")

        for idx in indexes:
            status = "[green]healthy[/green]" if idx.healthy else "[red]unhealthy[/red]"
            table.add_row(
                idx.name,
                str(idx.vector_count),
                str(idx.deleted_count),
                idx.created_at[:19] if idx.created_at else "",
                status,
            )
        console.print(table)


@index_app.command("info")
def index_info(
    name: str = typer.Argument(..., help="Index name"),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
) -> None:
    """Show detailed index information."""
    from isocortex.config import load_config
    from isocortex.engine.indexing.manager import IndexManager

    config = load_config()
    mgr = IndexManager(config.storage.indices_dir)
    stats = mgr.get_index(name)

    if stats is None:
        console.print(f"[red]✗[/red] Index '{name}' not found")
        raise typer.Exit(1)

    if json_output:
        import json
        console.print_json(json.dumps(stats.to_dict()))
    else:
        console.print(f"[bold]Index:[/bold] {stats.name}")
        console.print(f"  Vectors:     {stats.vector_count}")
        console.print(f"  Active:      {stats.active_count}")
        console.print(f"  Deleted:     {stats.deleted_count}")
        console.print(f"  Size:        {stats.index_size_mb:.2f} MB")
        console.print(f"  Model:       {stats.embedding_model}")
        console.print(f"  Dimension:   {stats.dimension}")
        console.print(f"  Format:      v{stats.format_version}")
        console.print(f"  HNSW:        M={stats.hnsw_params.get('M')}, "
                      f"ef_c={stats.hnsw_params.get('ef_construction')}, "
                      f"ef_s={stats.hnsw_params.get('ef_search')}")
        console.print(f"  Status:      {'[green]healthy[/green]' if stats.healthy else '[red]unhealthy[/red]'}")
        console.print(f"  Created:     {stats.created_at[:19] if stats.created_at else 'N/A'}")
        console.print(f"  Updated:     {stats.updated_at[:19] if stats.updated_at else 'N/A'}")


@index_app.command("delete")
def index_delete(
    name: str = typer.Argument(..., help="Index name"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete an index permanently."""
    if not force:
        confirm = typer.confirm(f"Are you sure you want to delete index '{name}'?")
        if not confirm:
            raise typer.Abort()

    from isocortex.config import load_config
    from isocortex.engine.indexing.manager import IndexManager

    config = load_config()
    mgr = IndexManager(config.storage.indices_dir)

    try:
        mgr.delete_index(name)
        console.print(f"[green]✓[/green] Index '{name}' deleted")
    except FileNotFoundError:
        console.print(f"[red]✗[/red] Index '{name}' not found")
        raise typer.Exit(1)


@index_app.command("export")
def index_export(
    name: str = typer.Argument(..., help="Index name"),
    output: str = typer.Option("./export.isocortex", "--output", "-o", help="Output path"),
) -> None:
    """Export an index as a .isocortex archive."""
    from isocortex.config import load_config
    from isocortex.engine.indexing.manager import IndexManager
    from pathlib import Path

    config = load_config()
    mgr = IndexManager(config.storage.indices_dir)

    try:
        result = mgr.export_index(name, Path(output))
        console.print(f"[green]✓[/green] Exported to {result['archive_path']}")
        console.print(f"  Size:   {result['archive_size_mb']} MB")
        console.print(f"  SHA256: {result['sha256_checksum'][:16]}...")
    except FileNotFoundError:
        console.print(f"[red]✗[/red] Index '{name}' not found")
        raise typer.Exit(1)


@index_app.command("import")
def index_import(
    archive: str = typer.Argument(..., help="Path to .isocortex archive"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Custom index name"),
) -> None:
    """Import an index from a .isocortex archive."""
    from isocortex.config import load_config
    from isocortex.engine.indexing.manager import IndexManager
    from pathlib import Path

    config = load_config()
    mgr = IndexManager(config.storage.indices_dir)

    try:
        index_name = mgr.import_index(Path(archive), name)
        console.print(f"[green]✓[/green] Imported index '{index_name}'")
    except Exception as exc:
        console.print(f"[red]✗[/red] Import failed: {exc}")
        raise typer.Exit(1)


# =============================================================================
# Search Commands
# =============================================================================

@search_app.command("query")
def search_query(
    index_name: str = typer.Argument(..., help="Index name"),
    query: str = typer.Option(..., "--query", "-q", help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
) -> None:
    """Search an index with a natural language query."""
    from isocortex.config import load_config
    from isocortex.engine.indexing.manager import IndexManager
    from isocortex.core.search.engine import SearchEngine

    config = load_config()
    mgr = IndexManager(config.storage.indices_dir)

    try:
        mgr.load_index(index_name)
    except FileNotFoundError:
        console.print(f"[red]✗[/red] Index '{index_name}' not found")
        raise typer.Exit(1)

    search_fn, meta_getter, count_fn = mgr.get_search_components(index_name)

    if count_fn() == 0:
        console.print(f"[yellow]Index '{index_name}' is empty[/yellow]")
        raise typer.Exit(0)

    # Load embedding model and embed the query
    with console.status("[bold green]Embedding query..."):
        from isocortex.core.embedding.embedder import EmbeddingEngine
        embed_engine = EmbeddingEngine(config)
        query_vector = embed_engine.embed(query)

    # Perform the search
    with console.status("[bold green]Searching..."):
        results = search_fn(query_vector, top_k)

    if not results:
        console.print("[yellow]No results found[/yellow]")
        raise typer.Exit(0)

    if json_output:
        import json
        output = []
        for r in results:
            meta = meta_getter(r.id)
            output.append({
                "rank": r.rank,
                "score": round(r.score, 4),
                "text_preview": meta.get("text_preview", "")[:200] if meta else "",
                "source_file": meta.get("source_file", "") if meta else "",
                "chunk_index": meta.get("chunk_index", "") if meta else "",
            })
        console.print_json(json.dumps(output))
    else:
        table = Table(title=f'Search Results for: "{query}"')
        table.add_column("#", justify="right", style="dim")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Source", style="cyan")
        table.add_column("Preview", max_width=60)

        for r in results:
            meta = meta_getter(r.id)
            source = meta.get("source_file", "unknown") if meta else "unknown"
            preview = meta.get("text_preview", "")[:100] if meta else ""
            # Show relative path
            source_short = source.replace(config.storage.data_dir, "").lstrip("/")
            table.add_row(
                str(r.rank),
                f"{r.score:.4f}",
                source_short,
                preview,
            )
        console.print(table)
        console.print(f"\n[dim]Found {len(results)} results in '{index_name}' "
                      f"({count_fn()} total vectors)[/dim]")


# =============================================================================
# User Commands
# =============================================================================

@user_app.command("create")
def user_create(
    username: str = typer.Option(..., "--username", "-u", help="Username"),
    password: str = typer.Option(..., "--password", "-p", help="Password (min 12 chars)"),
    role: str = typer.Option("user", "--role", "-r", help="Role (admin/user)"),
    email: Optional[str] = typer.Option(None, "--email", "-e", help="Email"),
) -> None:
    """Create a new user."""
    from isocortex.auth import get_user_manager

    mgr = get_user_manager()
    try:
        user = mgr.create_user(
            username=username,
            password=password,
            role=role,
            email=email,
        )
        console.print(f"[green]✓[/green] User '{username}' created (id: {user.user_id}, role: {user.role})")
    except ValueError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(1)


@user_app.command("list")
def user_list(
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
) -> None:
    """List all users."""
    from isocortex.auth import get_user_manager

    mgr = get_user_manager()
    users = mgr.list_users()

    if json_output:
        import json
        console.print_json(json.dumps([{
            "user_id": u.user_id, "username": u.username,
            "role": u.role, "is_active": u.is_active,
        } for u in users]))
    else:
        if not users:
            console.print("[yellow]No users found[/yellow]")
            return

        table = Table(title="Users")
        table.add_column("Username", style="cyan")
        table.add_column("Role")
        table.add_column("Active")
        table.add_column("Created", style="dim")

        for u in users:
            active = "[green]yes[/green]" if u.is_active else "[red]no[/red]"
            table.add_row(u.username, u.role, active, u.created_at[:19] if u.created_at else "")
        console.print(table)


@user_app.command("delete")
def user_delete(
    user_id: str = typer.Argument(..., help="User ID to delete"),
) -> None:
    """Delete a user by ID."""
    from isocortex.auth import get_user_manager

    mgr = get_user_manager()
    try:
        mgr.delete_user(user_id)
        console.print(f"[green]✓[/green] User '{user_id}' deleted")
    except Exception as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(1)


# =============================================================================
# Job Commands
# =============================================================================

@job_app.command("list")
def job_list(
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
    limit: int = typer.Option(20, "--limit", "-l", help="Max results"),
) -> None:
    """List background jobs."""
    from isocortex.config import load_config
    from isocortex.engine.jobs.scheduler import JobScheduler, JobStatus

    config = load_config()
    scheduler = JobScheduler(config.storage.db_path)

    job_status = None
    if status:
        try:
            job_status = JobStatus(status)
        except ValueError:
            console.print(f"[red]✗[/red] Invalid status: {status}")
            raise typer.Exit(1)

    jobs = scheduler.list_jobs(status=job_status, limit=limit)

    if not jobs:
        console.print("[yellow]No jobs found[/yellow]")
        return

    table = Table(title="Jobs")
    table.add_column("Job ID", style="cyan")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Progress")
    table.add_column("Created", style="dim")

    for j in jobs:
        progress = f"{j.progress.percentage:.0f}%" if j.progress else "—"
        status_style = {
            "completed": "[green]",
            "running": "[blue]",
            "failed": "[red]",
            "pending": "[yellow]",
            "queued": "[dim]",
        }.get(j.status.value, "")

        table.add_row(
            j.job_id[:12],
            j.job_type,
            f"{status_style}{j.status.value}",
            progress,
            j.created_at[:19] if j.created_at else "",
        )
    console.print(table)


# =============================================================================
# Serve Command
# =============================================================================

@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address"),
    port: int = typer.Option(8900, "--port", "-p", help="Port number"),
    workers: int = typer.Option(1, "--workers", "-w", help="Uvicorn worker count"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
) -> None:
    """Start the IsoCortex API server."""
    console.print(f"[bold cyan]IsoCortex[/bold cyan] API Server")
    console.print(f"  Host:    {host}:{port}")
    console.print(f"  Workers: {workers}")
    console.print(f"  Docs:    http://{host}:{port}/docs")
    console.print()

    import uvicorn
    uvicorn.run(
        "isocortex.api:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )


# =============================================================================
# Version Command
# =============================================================================

@app.command()
def version() -> None:
    """Show IsoCortex version."""
    from isocortex import __version__
    console.print(f"IsoCortex v{__version__}")
    console.print(f"Author: Shaheer Qureshi")


if __name__ == "__main__":
    app()
