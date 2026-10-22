"""Find which application object owns a name tests patch: the module, its media controller or its model controller."""


def owner(app, name):
    """``app`` for its own functions, else the controller instance whose method is called."""
    for target in (app, app.MEDIA, app.MODELS):
        if name in vars(target) or name in dir(type(target)) and target is not app:
            return target
    raise AttributeError(f"Neither the application nor its controllers define {name!r}.")
