# Import config first to set up cafpyana in sys.path
from . import config

# Import all modules
from . import utils
from . import io
from . import plotting
from . import syst
from . import selection
from . import analysis
from . import classes
from . import funcs
from . import preprocess
from . import detvar
from . import exclusive
from . import new_variables
from . import bdt

from .utils import *
from .io import *
from .plotting import *
from .syst import *
from .selection import *
from .analysis import *
from .classes import *
from .funcs import *
from .preprocess import *
from .detvar import *
from .exclusive import *
from .new_variables import *
from .bdt import *

# This allows both:
# import nueana; nueana.cutPreselection(df)
# from nueana import cutPreselection; cutPreselection(df)