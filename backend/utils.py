import uuid
import os


def get_fp(base: str, suffix: str = ".png") -> str:
    return os.path.join(base, f"{uuid.uuid4().hex}{suffix}")


def preprocess_ws_resp(data: dict) -> dict:
    """preprocess data before sending to front."""
    exc = ["roi_coords", "nuclei_ids"]
    data_post = {}
    for k, v in data.items():
        if k not in exc:
            if isinstance(v, tuple):
                v = list(v)
            if isinstance(v, list):
                for i, _v in enumerate(v):
                    if isinstance(_v, float):
                        v[i] = round(_v, 2)
                data_post[k] = v
            elif isinstance(v, float):
                data_post[k] = round(v, 2)
            elif isinstance(v, int):
                data_post[k] = v
    return data_post
