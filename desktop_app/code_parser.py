"""
IsoCortex Desktop App — Code Parser
=====================================
Parses source code files into structured representations
(functions, classes, methods) for enhanced code search.

Supports: Python, JavaScript, TypeScript, Java, C, C++, Go, Rust, Ruby.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("IsoCortex.code_parser")

# Code file extensions
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h",
    ".hpp", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".cs",
}

# Language → comment patterns for stripping comments
COMMENT_PATTERNS = {
    ".py":    [(r'#.*$', '')],
    ".rb":    [(r'#.*$', '')],
    ".js":    [(r'//.*$', ''), (r'/\*.*?\*/', '', re.DOTALL)],
    ".ts":    [(r'//.*$', ''), (r'/\*.*?\*/', '', re.DOTALL)],
    ".tsx":   [(r'//.*$', ''), (r'/\*.*?\*/', '', re.DOTALL)],
    ".jsx":   [(r'//.*$', ''), (r'/\*.*?\*/', '', re.DOTALL)],
    ".java":  [(r'//.*$', ''), (r'/\*.*?\*/', '', re.DOTALL)],
    ".c":     [(r'//.*$', ''), (r'/\*.*?\*/', '', re.DOTALL)],
    ".cpp":   [(r'//.*$', ''), (r'/\*.*?\*/', '', re.DOTALL)],
    ".h":     [(r'//.*$', ''), (r'/\*.*?\*/', '', re.DOTALL)],
    ".hpp":   [(r'//.*$', ''), (r'/\*.*?\*/', '', re.DOTALL)],
    ".go":    [(r'//.*$', ''), (r'/\*.*?\*/', '', re.DOTALL)],
    ".rs":    [(r'//.*$', ''), (r'/\*.*?\*/', '', re.DOTALL)],
    ".php":   [(r'//.*$', ''), (r'/\*.*?\*/', '', re.DOTALL), (r'#.*$', '')],
    ".swift": [(r'//.*$', '')],
    ".kt":    [(r'//.*$', ''), (r'/\*.*?\*/', '', re.DOTALL)],
    ".cs":    [(r'//.*$', ''), (r'/\*.*?\*/', '', re.DOTALL)],
}


@dataclass
class CodeSymbol:
    """A parsed code symbol (function, class, method, etc.)."""
    name: str
    kind: str          # "function", "class", "method", "variable"
    line_number: int
    signature: str     # Full signature line
    docstring: str = ""  # Associated docstring/comment
    body_preview: str = ""  # First few lines of the body


@dataclass
class ParsedCodeFile:
    """Result of parsing a code file."""
    file_path: str
    language: str
    symbols: list[CodeSymbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    total_lines: int = 0


def parse_code_file(file_path: Path) -> Optional[ParsedCodeFile]:
    """Parse a code file and extract symbols.
    
    Returns None if the file is not a recognized code file.
    """
    ext = file_path.suffix.lower()
    if ext not in CODE_EXTENSIONS:
        return None
    
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    
    lines = content.split("\n")
    language = _ext_to_language(ext)
    
    if ext == ".py":
        symbols = _parse_python(lines, content)
    elif ext in {".js", ".ts", ".tsx", ".jsx"}:
        symbols = _parse_javascript(lines, content)
    elif ext in {".java", ".kt", ".cs", ".c", ".cpp", ".h", ".hpp"}:
        symbols = _parse_c_family(lines, content)
    elif ext == ".go":
        symbols = _parse_go(lines, content)
    elif ext == ".rs":
        symbols = _parse_rust(lines, content)
    elif ext == ".rb":
        symbols = _parse_ruby(lines, content)
    else:
        symbols = _parse_generic(lines, content)
    
    imports = _extract_imports(lines, ext)
    
    return ParsedCodeFile(
        file_path=str(file_path),
        language=language,
        symbols=symbols,
        imports=imports,
        total_lines=len(lines),
    )


def strip_comments(content: str, ext: str) -> str:
    """Remove comments from source code."""
    patterns = COMMENT_PATTERNS.get(ext, [])
    result = content
    for pattern, repl, *flags in patterns:
        re_flags = flags[0] if flags else 0
        result = re.sub(pattern, repl, result, flags=re_flags)
    return result


def generate_code_summary(parsed: ParsedCodeFile) -> str:
    """Generate a text summary of a parsed code file for embedding.
    
    This summary is what gets embedded and searched, so it should
    contain symbol names, signatures, and docstrings.
    """
    parts = [f"File: {Path(parsed.file_path).name}"]
    parts.append(f"Language: {parsed.language}")
    
    if parsed.imports:
        parts.append(f"Imports: {', '.join(parsed.imports[:10])}")
    
    for sym in parsed.symbols:
        part = f"[{sym.kind}] {sym.name}"
        if sym.signature:
            part += f" — {sym.signature}"
        if sym.docstring:
            doc_preview = sym.docstring[:200]
            part += f"\n  {doc_preview}"
        if sym.body_preview:
            body_preview = sym.body_preview[:150]
            part += f"\n  {body_preview}"
        parts.append(part)
    
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# Language-specific parsers
# ═══════════════════════════════════════════════════════════════

def _parse_python(lines: list[str], content: str) -> list[CodeSymbol]:
    """Parse Python file for classes and functions."""
    symbols = []
    
    # Pattern: class ClassName(BaseClass):
    class_pat = re.compile(r'^(\s*)class\s+(\w+)\s*(\(.*?\))?\s*:')
    # Pattern: def function_name(args):
    func_pat = re.compile(r'^(\s*)def\s+(\w+)\s*\((.*?)\)\s*.*:')
    
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        
        # Classes
        m = class_pat.match(line)
        if m:
            bases = m.group(3) or ""
            sig = f"class {m.group(2)}({bases})"
            docstring = _get_docstring(lines, i + 1)
            symbols.append(CodeSymbol(
                name=m.group(2),
                kind="class",
                line_number=i + 1,
                signature=sig,
                docstring=docstring,
            ))
            continue
        
        # Functions (top-level or methods)
        m = func_pat.match(line)
        if m:
            name = m.group(2)
            args = m.group(3)
            kind = "method" if indent > 0 else "function"
            sig = f"def {name}({args})"
            docstring = _get_docstring(lines, i + 1)
            body = _get_body_preview(lines, i + 1, max_lines=3)
            symbols.append(CodeSymbol(
                name=name,
                kind=kind,
                line_number=i + 1,
                signature=sig,
                docstring=docstring,
                body_preview=body,
            ))
    
    return symbols


def _parse_javascript(lines: list[str], content: str) -> list[CodeSymbol]:
    """Parse JavaScript/TypeScript for functions, classes, and exports."""
    symbols = []
    
    # function name(args) {
    func_pat = re.compile(r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\((.*?)\)')
    # const/let/var name = (args) => {
    arrow_pat = re.compile(r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\((.*?)\)\s*=>')
    # class Name {
    class_pat = re.compile(r'(?:export\s+)?(?:default\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?')
    # method(args) {
    method_pat = re.compile(r'(?:async\s+)?(\w+)\s*\((.*?)\)\s*\{')
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        m = class_pat.search(stripped)
        if m:
            bases = f" extends {m.group(2)}" if m.group(2) else ""
            sig = f"class {m.group(1)}{bases}"
            symbols.append(CodeSymbol(
                name=m.group(1), kind="class", line_number=i + 1, signature=sig,
            ))
            continue
        
        m = func_pat.search(stripped)
        if m:
            sig = f"function {m.group(1)}({m.group(2)})"
            body = _get_body_preview(lines, i + 1, max_lines=3)
            symbols.append(CodeSymbol(
                name=m.group(1), kind="function", line_number=i + 1,
                signature=sig, body_preview=body,
            ))
            continue
        
        m = arrow_pat.search(stripped)
        if m:
            sig = f"{m.group(1)}({m.group(2)}) =>"
            symbols.append(CodeSymbol(
                name=m.group(1), kind="function", line_number=i + 1, signature=sig,
            ))
    
    return symbols


def _parse_c_family(lines: list[str], content: str) -> list[CodeSymbol]:
    """Parse C/C++/Java/C#/Kotlin for classes, methods, and functions."""
    symbols = []
    
    class_pat = re.compile(r'(?:class|struct)\s+(\w+)(?:\s*:\s*public\s+(\w+))?')
    func_pat = re.compile(r'(?:\w[\w\s*&:]+?)\s+(\w+)\s*\(([^)]*)\)\s*(?:const\s*)?(?:\{|$|;)')
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        m = class_pat.search(stripped)
        if m:
            bases = f" : {m.group(2)}" if m.group(2) else ""
            sig = f"class {m.group(1)}{bases}"
            symbols.append(CodeSymbol(
                name=m.group(1), kind="class", line_number=i + 1, signature=sig,
            ))
            continue
        
        m = func_pat.search(stripped)
        if m and not stripped.startswith("//") and not stripped.startswith("/*"):
            name = m.group(2)
            # Skip common keywords that look like function calls
            if name in ("if", "for", "while", "switch", "return", "else", "do"):
                continue
            sig = f"{name}({m.group(3)})"
            body = _get_body_preview(lines, i + 1, max_lines=3)
            symbols.append(CodeSymbol(
                name=name, kind="function", line_number=i + 1,
                signature=sig, body_preview=body,
            ))
    
    return symbols


def _parse_go(lines: list[str], content: str) -> list[CodeSymbol]:
    """Parse Go files for functions and types."""
    symbols = []
    
    func_pat = re.compile(r'func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(([^)]*)\)')
    type_pat = re.compile(r'type\s+(\w+)\s+(struct|interface)\s*\{')
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        m = type_pat.search(stripped)
        if m:
            symbols.append(CodeSymbol(
                name=m.group(1), kind=m.group(2), line_number=i + 1,
                signature=f"type {m.group(1)} {m.group(2)}",
            ))
            continue
        
        m = func_pat.search(stripped)
        if m:
            sig = f"func {m.group(1)}({m.group(2)})"
            body = _get_body_preview(lines, i + 1, max_lines=3)
            symbols.append(CodeSymbol(
                name=m.group(1), kind="function", line_number=i + 1,
                signature=sig, body_preview=body,
            ))
    
    return symbols


def _parse_rust(lines: list[str], content: str) -> list[CodeSymbol]:
    """Parse Rust files for functions, structs, and impls."""
    symbols = []
    
    func_pat = re.compile(r'(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)')
    struct_pat = re.compile(r'(?:pub\s+)?struct\s+(\w+)')
    impl_pat = re.compile(r'impl(?:<[^>]*>)?\s+(\w+)')
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        m = impl_pat.search(stripped)
        if m:
            symbols.append(CodeSymbol(
                name=m.group(1), kind="impl", line_number=i + 1,
                signature=f"impl {m.group(1)}",
            ))
            continue
        
        m = struct_pat.search(stripped)
        if m:
            symbols.append(CodeSymbol(
                name=m.group(1), kind="struct", line_number=i + 1,
                signature=f"struct {m.group(1)}",
            ))
            continue
        
        m = func_pat.search(stripped)
        if m:
            sig = f"fn {m.group(1)}({m.group(2)})"
            body = _get_body_preview(lines, i + 1, max_lines=3)
            symbols.append(CodeSymbol(
                name=m.group(1), kind="function", line_number=i + 1,
                signature=sig, body_preview=body,
            ))
    
    return symbols


def _parse_ruby(lines: list[str], content: str) -> list[CodeSymbol]:
    """Parse Ruby files for classes, modules, and methods."""
    symbols = []
    
    class_pat = re.compile(r'(?:class|module)\s+(\w+)')
    method_pat = re.compile(r'def\s+(?:self\.)?(\w+)\s*(?:\(([^)]*)\))?')
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        m = class_pat.search(stripped)
        if m:
            symbols.append(CodeSymbol(
                name=m.group(1), kind=m.group(0).split()[0],
                line_number=i + 1,
                signature=f"{m.group(0)}",
            ))
            continue
        
        m = method_pat.search(stripped)
        if m:
            args = m.group(2) or ""
            sig = f"def {m.group(1)}({args})"
            body = _get_body_preview(lines, i + 1, max_lines=3)
            symbols.append(CodeSymbol(
                name=m.group(1), kind="method", line_number=i + 1,
                signature=sig, body_preview=body,
            ))
    
    return symbols


def _parse_generic(lines: list[str], content: str) -> list[CodeSymbol]:
    """Generic parser — extracts indented blocks as potential functions."""
    symbols = []
    func_pat = re.compile(r'^\s*(?:public|private|protected|static|internal|\w+\s+)*\s*(\w+)\s*\(')
    
    for i, line in enumerate(lines):
        m = func_pat.match(line)
        if m and m.group(1) not in ("if", "for", "while", "switch", "return", "else", "do", "class", "struct"):
            symbols.append(CodeSymbol(
                name=m.group(1), kind="function", line_number=i + 1,
                signature=line.strip().rstrip("{;:"),
            ))
    
    return symbols


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _get_docstring(lines: list[str], start_idx: int) -> str:
    """Extract a docstring or comment block following a definition."""
    if start_idx >= len(lines):
        return ""
    
    line = lines[start_idx].strip()
    
    # Python triple-quote docstring
    if line.startswith('"""') or line.startswith("'''"):
        quote = line[:3]
        if line.endswith(quote) and len(line) > 6:
            return line[3:-3].strip()
        
        doc_lines = [line[3:].strip()]
        for j in range(start_idx + 1, min(start_idx + 20, len(lines))):
            if lines[j].strip().endswith(quote):
                doc_lines.append(lines[j].strip()[:-3])
                break
            doc_lines.append(lines[j].strip())
        
        return "\n".join(doc_lines).strip()
    
    # JSDoc / JavaDoc / Go doc
    if line.startswith("/**") or line.startswith("/*"):
        doc_lines = []
        for j in range(start_idx, min(start_idx + 20, len(lines))):
            doc_lines.append(lines[j].strip().lstrip("/* ").rstrip("*/ "))
            if "*/" in lines[j]:
                break
        return "\n".join(doc_lines).strip()
    
    # Single-line comment doc
    if line.startswith("#") or line.startswith("//"):
        return line.lstrip("#/ ").strip()
    
    return ""


def _get_body_preview(lines: list[str], start_idx: int, max_lines: int = 3) -> str:
    """Get a preview of the function/method body."""
    preview_lines = []
    for j in range(start_idx, min(start_idx + max_lines + 10, len(lines))):
        line = lines[j].strip()
        if not line:
            continue
        if line in ("{", "}", ""):
            if line == "{":
                continue
            if line == "}" and preview_lines:
                break
        preview_lines.append(line)
        if len(preview_lines) >= max_lines:
            break
    
    return " ".join(preview_lines)


def _extract_imports(lines: list[str], ext: str) -> list[str]:
    """Extract import statements from a code file."""
    imports = []
    
    if ext == ".py":
        pat = re.compile(r'^(?:from\s+([\w.]+)\s+)?import\s+(.+)')
    elif ext in {".js", ".ts", ".tsx", ".jsx"}:
        pat = re.compile(r'^import\s+.*?from\s+["\'](.+?)["\']|require\(["\'](.+?)["\']\)')
    elif ext == ".go":
        pat = re.compile(r'^import\s+["\'](.+?)["\']|^\s+["\'](.+?)["\']')
    elif ext == ".rs":
        pat = re.compile(r'^use\s+(.+?);')
    elif ext == ".java":
        pat = re.compile(r'^import\s+(.+?);')
    else:
        pat = re.compile(r'^#include\s+[<"](.+?)[>"]|import\s+(.+)')
    
    for line in lines:
        line = line.strip()
        m = pat.match(line)
        if m:
            # Get first non-None group
            imp = next((g for g in m.groups() if g is not None), "")
            if imp:
                imports.append(imp)
    
    return imports[:20]  # Limit to prevent bloating


def _ext_to_language(ext: str) -> str:
    """Map file extension to language name."""
    mapping = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
        ".tsx": "TypeScript", ".jsx": "JavaScript", ".java": "Java",
        ".c": "C", ".cpp": "C++", ".h": "C/C++ Header", ".hpp": "C++ Header",
        ".go": "Go", ".rs": "Rust", ".rb": "Ruby",
        ".php": "PHP", ".swift": "Swift", ".kt": "Kotlin", ".cs": "C#",
    }
    return mapping.get(ext, ext.lstrip("."))