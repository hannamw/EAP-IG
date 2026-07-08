from .model_adapter import EAPModelAdapter, EAPModelProtocol, prepare_model_for_eap


def hello() -> str:
    return "Hello from eap!"


__all__ = ["EAPModelAdapter", "EAPModelProtocol", "prepare_model_for_eap", "hello"]
