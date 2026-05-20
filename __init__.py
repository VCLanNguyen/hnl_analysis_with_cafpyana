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
from . import analysis
from . import classes
from . import variables
from . import funcs
from . import preprocess
from . import detvar_store
from . import detvar_recomb

from .constants import *
from .utils import *
from .io import *
from .geometry import *
from .histogram import *
from .plotting import *
from .syst import *
from .selection import *
from .analysis import *
from .classes import *
from .variables import *
from .funcs import *
from .preprocess import *
from .detvar_store import *
from .detvar_recomb import *

# This allows both:
# import nueana; nueana.cutPreselection(df)
# from nueana import cutPreselection; cutPreselection(df)