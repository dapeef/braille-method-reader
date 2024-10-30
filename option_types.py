from enum import Enum, auto

class TitlePos(Enum):
    # Where to put the title
    TOP = auto()
    BOTTOM = auto()
    LEFT = auto()
    RIGHT = auto()
    NONE = auto()

class TitleLanguage(Enum):
    # Whether to write the title in braille or latin script
    BRAILLE = auto()
    LATIN = auto()
    BOTH = auto()

class TitleText(Enum):
    # What text to write for the title
    FULL = auto() # eg "Cambridge Surprise Major"
    FULL_LOWER = auto() # eg "cambridge surprise major"
    SHORT = auto() # eg "Cambridge 8"
    SHORT_LOWER = auto() # eg "cambridge 8"

class LengthTypes(Enum):
    PLAIN_COURSE = auto()
    SINGLE_LEAD = auto()

class BaseType(Enum):
    HOLE = auto()
    NO_HOLE = auto()

class TrebleType(Enum):
    DOTTED = auto()
    CROSS = auto()
    SOLID = auto()
    NONE = auto()


class PathCrossSection(Enum):
    CYLINDER = auto()
    RECTANGLE = auto()
