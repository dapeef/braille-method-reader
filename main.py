from geometry_utils import Plate, PlateConfig
from option_types import *
from method_utils import Method


config = PlateConfig()
config.unit_thickness = .4 # mm
config.unit_width = 5 # mm
config.unit_height = 2.5 # mm

config.length_type = LengthTypes.SINGLE_LEAD

config.treble_type = TrebleType.CROSS
config.treble_line_config.cross_section = PathCrossSection.CYLINDER
config.treble_line_config.height = config.thick_line_config.height
config.treble_line_config.width = config.thick_line_config.width / 2


# config.reverse_method = True


# Plain bob 5 = 10550
# Norwich 6 = 14317
# Cambridge 8 = 16694
# Cambridge 10 = 21250
# Double Norwich 8 = 12470
method = Method.create_from_complib_id("16694")
# method.name = "C'bridge"

plate = Plate(method, config)

plate.create_base()
plate.create_vertical_lines()
bell = 2
plate.create_thick_line(bell)
plate.create_lead_end_dots(bell)
plate.create_treble_line(bell)
plate.create_half_lead_lines()

plate.save_to_stl()
