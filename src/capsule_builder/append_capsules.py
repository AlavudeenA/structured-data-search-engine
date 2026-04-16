"""Module for programmatically appending new capsule definitions to the registry."""

import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def append_to_capsule_definitions_file(new_capsule: dict) -> bool:
    """Read capsule_definitions.py, inject the new dictionary before the final list bracket, and save."""
    target_file = Path(__file__).parent.parent / "business_schema" / "capsule_definitions.py"
    if not target_file.exists():
        logger.error(f"Cannot find {target_file}")
        return False
        
    try:
        content = target_file.read_text(encoding="utf-8")
        
        # Find the last closing bracket of the CAPSULE_DEFINITIONS list.
        # It's highly likely to be at the very end of the file.
        last_bracket_idx = content.rfind("]")
        
        if last_bracket_idx == -1:
            logger.error("Could not find the closing ] in capsule_definitions.py")
            return False
            
        # Format the new capsule as a pretty-printed JSON string (which is valid Python dict syntax)
        # However, we must ensure booleans/nulls are Python-style if they exist.
        # The easiest safe path is to string-replace JSON styles back to Python after dumping.
        # But since we control the schema, we can also just format it manually or rely on json dump replacing true->True
        dict_str = json.dumps(new_capsule, indent=4)
        # Convert JSONisms to Pythonisms
        dict_str = dict_str.replace("false", "False").replace("true", "True").replace("null", "None")
        
        # Indent the dictionary text to properly align within the python list
        indented_dict_str = "\n".join("    " + line for line in dict_str.splitlines())
        
        # Splice the content
        prefix = content[:last_bracket_idx].rstrip()
        
        # If the list isn't empty, check if we need a trailing comma on the previous element
        if not prefix.endswith(",") and not prefix.endswith("["):
            prefix += ","
            
        new_content = prefix + "\n\n" + indented_dict_str + "\n\n]\n"
        
        target_file.write_text(new_content, encoding="utf-8")
        return True
        
    except Exception as e:
        logger.error(f"Failed to append to capsule_definitions: {e}")
        return False
