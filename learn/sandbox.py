'''

🗂️ L7: Sandboxed File Tools

The Problem: Giving an LLM direct access to open() is a massive security vulnerability.
The agent could run os.system("rm -rf /") or read your private SSH keys if injected.
The Fix: Build a Directory-Chrooted Sandbox File Tool that restricts all read/write 
paths strictly inside a designated ./sandbox directory.'''

from pathlib import Path

class SandboxedFileTools:
    """Safe file system tools that cannot escape a specific directory."""
    def __init__(self, base_sandbox_dir: str = "./sandbox"):
        self.sandbox_path = Path(base_sandbox_dir).resolve()
        self.sandbox_path.mkdir(exist_ok=True)

    def _safe_path(self, target_path: str) -> Path:
        """Enforces jail limits, preventing path traversal attacks (e.g., ../../)"""
        resolved = (self.sandbox_path / target_path).resolve()
        if not resolved.is_relative_to(self.sandbox_path):
            raise PermissionError("Access Denied: Path traversal detected!")
        return resolved

    def write_file(self, filename: str, content: str) -> str:
        try:
            safe_p = self._safe_path(filename)
            safe_p.write_text(content, encoding="utf-8")
            return f"Successfully wrote to {filename}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

    def read_file(self, filename: str) -> str:
        try:
            safe_p = self._safe_path(filename)
            if not safe_p.exists():
                return f"Error: File {filename} does not exist."
            return safe_p.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file: {str(e)}"
