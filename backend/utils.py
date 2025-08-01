from typing import Any


def preprocess_ws_resp(data: dict[str, Any], exclude: list[str] | None) -> dict:
    """Preprocess data before sending through websocket."""
    if not exclude:
        exclude = []
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
