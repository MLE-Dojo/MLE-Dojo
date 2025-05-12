from pathlib import Path
from typing import Dict, Any
from mledojo.gym.sandbox import Sandbox
from mledojo.gym.error import SandboxError, ArchiveError


# Utility Functions
def run_in_sandbox(code_path: Path, sandbox: Sandbox) -> Dict[str, Any]:
    """Execute code in a sandbox environment and return the result.
    
    Args:
        code_path: Path to the code file to execute
        sandbox: Sandbox instance to use for execution
        
    Returns:
        Dictionary containing execution results with status, output, error, and execution time
        
    Raises:
        SandboxError: If execution in sandbox fails
    """
    try:
        result = sandbox.run_code(str(code_path))
        return {
            "status": "SUCCESS" if result["status"] == 0 else "FAILED",
            "output": result["stdout"],
            "error": result["stderr"],
            "execution_time": f"{result['execution_time']:.2f}s"
        }
    except Exception as e:
        raise SandboxError(
            message="Failed to execute code in sandbox",
            details=str(e)
        )


def archive_file(file_path: Path, output_dir: Path, prefix: str, extension: str) -> Path:
    """Archive a file by renaming it with a numbered suffix.
    
    Args:
        file_path: Path to the file to archive
        output_dir: Directory to store the archived file
        prefix: Prefix for the archived filename
        extension: File extension for the archived file
        
    Returns:
        Path to the newly archived file
        
    Raises:
        ArchiveError: If file archiving fails
    """
    try:
        count = len(list(output_dir.glob(f"{prefix}_*.{extension}"))) + 1
        new_name = output_dir / f"{prefix}_{count}.{extension}"
        file_path.rename(new_name)
        return new_name
    except Exception as e:
        raise ArchiveError(
            message=f"Failed to archive file {file_path}",
            details=str(e)
        )


def save_code_file(code: str, output_dir: Path, prefix: str) -> Path:
    """Save code to a numbered file in the output directory.
    
    Args:
        code: Code content to save
        output_dir: Directory to save the code file
        prefix: Prefix for the code filename
        
    Returns:
        Path to the saved code file
        
    Raises:
        ArchiveError: If file saving fails
    """
    try:
        count = len(list(output_dir.glob(f"{prefix}_*.py"))) + 1
        code_file = output_dir / f"{prefix}_{count}.py"
        code_file.write_text(code)
        return code_file
    except Exception as e:
        raise ArchiveError(
            message=f"Failed to save code file to {output_dir}",
            details=str(e)
        )


def build_tree(path: Path, prefix: str = "", is_last: bool = True, max_items: int = 10) -> str:
    """Generate a text representation of a directory tree structure.
    
    Args:
        path: Path to the directory to represent
        prefix: Prefix string for the current line (used for recursion)
        is_last: Whether this is the last item in its parent directory
        max_items: Maximum number of items to show per directory
        
    Returns:
        String representation of the directory tree
    """
    if not path.exists():
        return f"{prefix}[Error accessing {path}]\n"
        
    result = f"{prefix}{'└── ' if is_last else '├── '}{path.name}/\n"
    
    try:
        items = sorted(path.iterdir())
    except PermissionError:
        return f"{result}{prefix}{'    ' if is_last else '│   '}[Permission denied]\n"
    
    dirs = [p for p in items if p.is_dir()]
    files = [p for p in items if p.is_file()]
    
    new_prefix = prefix + ('    ' if is_last else '│   ')
    
    # Process directories
    for i, d in enumerate(dirs[:max_items]):
        is_last_item = (i == len(dirs[:max_items]) - 1) and not files
        result += build_tree(d, new_prefix, is_last_item)
    
    if len(dirs) > max_items:
        result += f"{new_prefix}├── [and {len(dirs) - max_items} more directories]\n"
        
    # Process files  
    for i, f in enumerate(files[:max_items]):
        file_prefix = '└── ' if i == len(files[:max_items]) - 1 else '├── '
        result += f"{new_prefix}{file_prefix}{f.name}\n"
        
    if len(files) > max_items:
        result += f"{new_prefix}└── [and {len(files) - max_items} more files]\n"
        
    return result