import uuid


def get_fn(suffix: str = ".png") -> str:
    return uuid.uuid4().hex + suffix
