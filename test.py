from geometry_utils import Plate, PlateConfig
from method_utils import Method


config = PlateConfig()
config.UNIT_THICKNESS = .4 # mm
config.UNIT_WIDTH = 10 # mm
config.UNIT_HEIGHT = 5 # mm

config.THICK_LINE_WIDTH = 2 # mm
config.THICK_LINE_HEIGHT = 1 # mm
config.THIN_LINE_WIDTH = 1 # mm
config.THIN_LINE_HEIGHT = .5 # mm
config.DOT_DIAMETER = 1.5 # mm
config.DOT_HEIGHT = config.DOT_DIAMETER / 2 # mm
config.DOT_SEPARATION = 2.3 # mm

# Norwich 6 = 14317
# Cambridge 8 = 16694
method = Method.create_from_complib_id("16694")

plate = Plate(method, config)

plate.create_base()
plate.create_vertical_lines()
plate.create_thick_line(4)
plate.create_dotted_line()

plate.save_to_stl()
