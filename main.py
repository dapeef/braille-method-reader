from geometry_utils import Plate, PlateConfig, TitlePos, TitleLanguage, TitleText, LengthTypes
from method_utils import Method


config = PlateConfig()
config.unit_thickness = .4 # mm
config.unit_width = 10 # mm
config.unit_height = 5 # mm

config.title_position = TitlePos.LEFT 
config.title_language = TitleLanguage.BOTH
config.title_text = TitleText.SHORT

config.length_type = LengthTypes.SINGLE_LEAD


# Plain bob 5 = 10550
# Norwich 6 = 14317
# Cambridge 8 = 16694
# Cambridge 10 = 21250
# Double Norwich 8 = 12470
method = Method.create_from_complib_id("16694")
# method.name = "C'bridge"

plate = Plate(method, config)

plate.create_base()
plate.create_title()
plate.create_vertical_lines()
bell = 2
plate.create_thick_line(bell)
plate.create_lead_end_dots(bell)
plate.create_place_bell_label(bell)
plate.create_dotted_line()

plate.save_to_stl()
