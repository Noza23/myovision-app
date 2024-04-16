from typing import Any
import uuid
import os


def get_fp(directory: str, suffix: str = ".png") -> str:
    """Get UUID based file path."""
    return os.path.join(directory, f"{uuid.uuid4().hex}{suffix}")


def clean_dir(directory: str) -> None:
    """Clean the directory."""
    for f in os.listdir(directory):
        os.remove(os.path.join(directory, f))


def preprocess_ws_resp(data: dict[str, Any], exclude: list[str] = []) -> dict:
    """preprocess data before sending through websocket."""
    data_post: dict[str, Any] = {}
    for k, v in data.items():
        if k not in exclude:
            if isinstance(v, (list, tuple)):
                v = list(v)
                for i, _v in enumerate(v):
                    if isinstance(_v, float):
                        v[i] = _v.__round__(2)
                data_post[k] = v
            elif isinstance(v, float):
                data_post.update({k: v.__round__(2)})
            else:
                data_post.update({k: v})
    return data_post
