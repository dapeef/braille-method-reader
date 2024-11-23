from plate import Plate
from config import PlateConfig
from option_types import *
from method_utils import Method


config = PlateConfig()
config.plate_thickness = .4 # mm
config.title_text = TitleText.FULL
config.braille_config.dot_type = BrailleDotType.DOME
config.reverse_method = True


# Plain bob 5 = 10550
# Norwich 6 = 14317
# Cambridge 8 = 16694
# Cambridge 10 = 21250
# Double Norwich 8 = 12470
# Mareham 8 = 25093
method = Method.create_from_complib_id("25093")
# method.name = "C'bridge"

for i in range(2, 9):
    # Method plate
    plate = Plate(method, config, i)
    plate.create_base()
    plate.create_vertical_lines()
    plate.create_thick_line()
    plate.create_lead_end_markers()
    plate.create_treble_line()
    plate.create_half_lead_lines()
    plate.save_to_stl(f"./stl-files/mareham/{i}.stl")

# # Title plate
# plate = Plate(method, config)
# plate.create_base()
# plate.create_title()
# plate.save_to_stl("stl-files/mk-4/type-2.stl")
