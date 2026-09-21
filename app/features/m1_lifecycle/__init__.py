import importlib
import pkgutil
from flask import Blueprint

m1_bp = Blueprint('m1_lifecycle', __name__)

# Import all modules in this directory so they can register their routes with m1_bp
import os
for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    importlib.import_module(f'.{module[:-3]}', package=__name__)
