import os


def get_file_name(file_path: str) -> str:
    """Get the file name from a file path."""
    files = os.listdir(file_path)
    numbering = [int(file.split(".")[0]) for file in files]
    return str(max(numbering) + 1) + ".png"
