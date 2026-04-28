# Import config first to set up cafpyana in sys.path
from . import config

# Import all modules
from . import constants
from . import utils
from . import io
from . import geometry
from . import histogram
from . import plotting
from . import syst
from . import selection
from . import classes
from . import funcs
from . import new_variables
from . import bdt

from .constants import *
from .utils import *
from .io import *
from .geometry import *
from .histogram import *
from .plotting import *
from .syst import *
from .selection import *
from .classes import *
from .funcs import *
from .new_variables import *
from .bdt import *

# This allows both:
# import nueana; nueana.cutPreselection(df)
# from nueana import cutPreselection; cutPreselection(df)