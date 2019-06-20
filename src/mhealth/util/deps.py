""" Optional dependencies. If not available, a fake is used to
allow for singledispatch in other modules
"""
try:
    import pandas as pd
except ImportError:
    class pd:
        class DataFrame(dict):
            pass
