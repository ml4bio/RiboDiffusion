import torch

_MODELS = {}

def register_model(cls=None, *, name=None):
    """"A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f"Already registerd model")
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

def create_model(config):
    model = _MODELS[config.model.name](config)
    model = model.to(config.device)
    model = torch.nn.DataParallel(model)
    return model