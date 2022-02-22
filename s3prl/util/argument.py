def bind_function(instance, func, as_name=None):
    """
    Bind the function *func* to *instance*, with either provided name *as_name*
    or the existing name of *func*. The provided *func* should accept the
    instance as the first argument, i.e. "self".
    """
    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    return bound_method


if __name__ == "__main__":

    class Thing:
        def __init__(self, val):
            self.val = val

    something = Thing(21)

    def double(self):
        return 2 * self.val

    bind_function(something, double)
    something.double()  # returns 42
