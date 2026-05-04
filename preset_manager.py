from pathlib import Path
import json
from datetime import datetime
from typing import List, Tuple, Dict, Optional

class PresetManager:
    def __init__(self, presets_dir: Optional[str] = None):
        if presets_dir is None:
            # Use a standard location relative to user's home or app directory
            self.presets_dir = Path.home() / ".sound_effects_generator" / "presets"
        else:
            self.presets_dir = Path(presets_dir)
        
        # Ensure directory exists
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        
        self._cache: List[Tuple[str, Path]] = []
        self._cache_dirty = True

    def save_preset(self, name: str, params: Dict) -> Path:
        # Sanitize name for filename
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in name)
        filepath = self.presets_dir / f"{safe_name}.json"
        
        data = {
            "name": name,
            "version": "2.1",
            "created": datetime.now().isoformat(),
            "parameters": params
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        self._cache_dirty = True
        return filepath

    def load_preset(self, filepath: str) -> Dict:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Preset file not found: {filepath}")
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("parameters", data)
        except (json.JSONDecodeError, OSError) as e:
            raise ValueError(f"Failed to load preset {filepath}: {e}") from e

    def list_presets(self) -> List[Tuple[str, Path]]:
        if not self._cache_dirty:
            return self._cache

        presets: List[Tuple[str, Path]] = []
        # Check if directory exists to avoid error on first run before creation logic
        if self.presets_dir.exists():
            for fp in sorted(self.presets_dir.glob("*.json")):
                try:
                    with open(fp, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    name = data.get("name", fp.stem)
                    presets.append((name, fp))
                except (json.JSONDecodeError, OSError):
                    # Fallback to filename if json is corrupt
                    presets.append((fp.stem, fp))

        presets.sort(key=lambda x: x[0].lower())
        self._cache = presets
        self._cache_dirty = False
        return presets

    def invalidate_cache(self):
        self._cache_dirty = True