from typing import Any
import uuid
import os


def get_fp(base: str, suffix: str = ".png") -> str:
    return os.path.join(base, f"{uuid.uuid4().hex}{suffix}")


def clean_dir(base: str) -> None:
    for f in os.listdir(base):
        os.remove(os.path.join(base, f))


def preprocess_ws_resp(data: dict[str, Any], exclude: list[str] = []) -> dict:
    """preprocess data before sending to front."""
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
