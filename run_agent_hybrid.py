#!/usr/bin/env python3
"""
Main entry point for the Retail Analytics Copilot.

Usage:
    python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
    python run_agent_hybrid.py --question "What is the return policy for Beverages?"
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from agent.graph_hybrid import RetailAnalyticsCopilot


console = Console()


def load_questions(batch_file: str) -> list[dict]:
    """Load questions from JSONL file."""
    questions = []
    with open(batch_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def save_outputs(outputs: list[dict], output_file: str) -> None:
    """Save outputs to JSONL file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for output in outputs:
            # Remove internal trace before saving
            output_clean = {k: v for k, v in output.items() if not k.startswith('_')}
            f.write(json.dumps(output_clean) + '\n')


def display_result(result: dict) -> None:
    """Display a single result nicely."""
    table = Table(title=f"Result: {result.get('id', 'N/A')}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Final Answer", str(result.get('final_answer', '')))
    table.add_row("Confidence", f"{result.get('confidence', 0):.2f}")
    table.add_row("Explanation", result.get('explanation', '')[:100])
    table.add_row("SQL", result.get('sql', '')[:100] + ('...' if len(result.get('sql', '')) > 100 else ''))
    table.add_row("Citations", ', '.join(result.get('citations', [])))
    
    console.print(table)


@click.command()
@click.option('--batch', type=click.Path(exists=True), help='Path to JSONL file with questions')
@click.option('--out', type=click.Path(), help='Output JSONL file path')
@click.option('--question', type=str, help='Single question to answer')
@click.option('--format-hint', type=str, default='', help='Format hint for single question')
@click.option('--db-path', type=click.Path(), default='data/northwind.sqlite', help='Path to SQLite database')
@click.option('--docs-dir', type=click.Path(), default='docs', help='Path to documents directory')
@click.option('--model', type=str, default='phi3.5:3.8b-mini-instruct-q4_K_M', help='Ollama model name')
@click.option('--optimize', is_flag=True, help='Run DSPy optimization before inference')
@click.option('--verbose', is_flag=True, help='Show detailed output')
def main(
    batch: str,
    out: str,
    question: str,
    format_hint: str,
    db_path: str,
    docs_dir: str,
    model: str,
    optimize: bool,
    verbose: bool
):
    """Retail Analytics Copilot - Answer retail questions using RAG + SQL."""
    
    console.print("[bold blue]Retail Analytics Copilot[/bold blue]")
    console.print(f"Database: {db_path}")
    console.print(f"Documents: {docs_dir}")
    console.print(f"Model: {model}")
    console.print()
    
    # Check if database exists
    if not Path(db_path).exists():
        console.print(f"[red]Error: Database not found at {db_path}[/red]")
        console.print("Run ./setup.sh to download the database")
        sys.exit(1)
    
    # Check if docs directory exists
    if not Path(docs_dir).exists():
        console.print(f"[red]Error: Documents directory not found at {docs_dir}[/red]")
        sys.exit(1)
    
    # Initialize the copilot
    console.print("[yellow]Initializing copilot...[/yellow]")
    try:
        copilot = RetailAnalyticsCopilot(
            db_path=db_path,
            docs_dir=docs_dir,
            model_name=model,
            optimize=optimize
        )
        console.print("[green]Copilot initialized successfully![/green]")
    except Exception as e:
        console.print(f"[red]Error initializing copilot: {e}[/red]")
        # Try to continue in mock mode for testing
        console.print("[yellow]Attempting to continue without LLM...[/yellow]")
        try:
            # Create copilot without DSPy LM
            copilot = RetailAnalyticsCopilot(
                db_path=db_path,
                docs_dir=docs_dir,
                model_name=model,
                optimize=False
            )
        except Exception as e2:
            console.print(f"[red]Failed to initialize: {e2}[/red]")
            sys.exit(1)
    
    console.print()
    
    # Handle batch mode
    if batch:
        console.print(f"[yellow]Loading questions from {batch}...[/yellow]")
        questions = load_questions(batch)
        console.print(f"Loaded {len(questions)} questions")
        console.print()
        
        results = []
        for i, q in enumerate(questions, 1):
            console.print(f"[cyan]Processing {i}/{len(questions)}: {q.get('id', 'N/A')}[/cyan]")
            if verbose:
                console.print(f"  Question: {q.get('question', '')[:80]}...")
            
            try:
                result = copilot.run(
                    question=q.get('question', ''),
                    format_hint=q.get('format_hint', ''),
                    question_id=q.get('id', '')
                )
                results.append(result)
                
                if verbose:
                    display_result(result)
                else:
                    console.print(f"  Answer: {result.get('final_answer')}")
                    console.print(f"  Confidence: {result.get('confidence', 0):.2f}")
            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")
                results.append({
                    "id": q.get('id', ''),
                    "final_answer": None,
                    "sql": "",
                    "confidence": 0.0,
                    "explanation": f"Error: {str(e)}",
                    "citations": []
                })
            
            console.print()
        
        # Save outputs
        if out:
            console.print(f"[yellow]Saving outputs to {out}...[/yellow]")
            save_outputs(results, out)
            console.print(f"[green]Saved {len(results)} results to {out}[/green]")
        
        # Summary
        console.print()
        console.print("[bold]Summary:[/bold]")
        successful = sum(1 for r in results if r.get('confidence', 0) > 0)
        console.print(f"  Total questions: {len(questions)}")
        console.print(f"  Successful: {successful}")
        console.print(f"  Average confidence: {sum(r.get('confidence', 0) for r in results) / len(results):.2f}")
    
    # Handle single question mode
    elif question:
        console.print(f"[yellow]Processing question...[/yellow]")
        console.print(f"Question: {question}")
        if format_hint:
            console.print(f"Format hint: {format_hint}")
        console.print()
        
        try:
            result = copilot.run(
                question=question,
                format_hint=format_hint,
                question_id="interactive"
            )
            display_result(result)
            
            if verbose and result.get('_trace'):
                console.print()
                console.print("[bold]Execution Trace:[/bold]")
                for entry in result['_trace']:
                    console.print(f"  {entry.get('node', 'N/A')}: {entry.get('output', {})}")
        
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
    
    else:
        console.print("[yellow]No input provided. Use --batch or --question.[/yellow]")
        console.print()
        console.print("Examples:")
        console.print("  python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl")
        console.print("  python run_agent_hybrid.py --question 'Top 3 products by revenue' --format-hint 'list[{product:str, revenue:float}]'")
        sys.exit(1)


if __name__ == "__main__":
    main()
