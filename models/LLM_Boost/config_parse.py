import tomllib  # Python 3.11+ standard library for TOML
import toml  # Third-party TOML library for older Python versions


def load_toml_config(file_path):
    """
    Loads a TOML configuration file.

    Args:
        file_path (str): The path to the TOML file.

    Returns:
        dict: A dictionary containing the parsed TOML data, 
              or None if an error occurs.
    """
    try:
        with open(file_path, 'rb') as f: # Open in binary read mode
            if hasattr(tomllib, 'load'): # Check if tomllib (Python 3.11+) is available and has 'load'
                config_data = tomllib.load(f)
                print("Loaded TOML using 'tomllib' (Python 3.11+)")
            elif hasattr(toml, 'load'): # Check if toml (third-party) is available
                # The third-party 'toml' library expects a text file stream
                # So we need to reopen in text mode or decode if we stick to 'rb'
                # For simplicity, let's assume we reopen in text mode if using 'toml'
                with open(file_path, 'r', encoding='utf-8') as text_f:
                    config_data = toml.load(text_f)
                print("Loaded TOML using 'toml' (third-party library)")
            else:
                print("No suitable TOML library available.")
                return None
        return config_data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        # Catching tomllib.TOMLDecodeError or toml.TomlDecodeError
        print(f"Error parsing TOML file '{file_path}': {e}")
        return None