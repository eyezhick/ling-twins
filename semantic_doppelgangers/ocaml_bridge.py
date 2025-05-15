import subprocess
import json
from typing import List, Tuple, Dict, Any
import os

class OCamlBridge:
    def __init__(self):
        self.ocaml_binary = os.path.join(os.path.dirname(__file__), "..", "ocaml", "_build", "default", "src", "linguistic.exe")
        self._ensure_ocaml_built()

    def _ensure_ocaml_built(self):
        """Ensure the OCaml binary is built."""
        if not os.path.exists(self.ocaml_binary):
            subprocess.run(["dune", "build"], cwd=os.path.join(os.path.dirname(__file__), "..", "ocaml"))

    def process_text(self, text: str) -> Tuple[List[Dict[str, Any]], List[Tuple[str, List[str]]]]:
        """Process text using OCaml linguistic processing."""
        result = subprocess.run(
            [self.ocaml_binary, "process", text],
            capture_output=True,
            text=True
        )
        return json.loads(result.stdout)

    def compare_texts(self, text1: str, text2: str) -> float:
        """Compare two texts using OCaml similarity calculation."""
        result = subprocess.run(
            [self.ocaml_binary, "compare", text1, text2],
            capture_output=True,
            text=True
        )
        return float(result.stdout.strip())

    def extract_features(self, text: str) -> List[Tuple[str, List[str]]]:
        """Extract linguistic features using OCaml processing."""
        tokens, features = self.process_text(text)
        return features 