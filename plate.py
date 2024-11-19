from geometry_utils import *
from method_utils import Method
from option_types import *


class Plate:
    def __init__(self, method: Method, config: PlateConfig) -> None:
        self.method = method
        self.config = config

        self.base_width = self.config.unit_width * (self.method.stage - 1)
        match self.config.length_type:
            case LengthTypes.PLAIN_COURSE:
                self.drawable_rows = self.method.rows
                self.base_height = self.config.unit_height * (len(self.method.rows) - 1)
            case LengthTypes.SINGLE_LEAD:
                self.drawable_rows = self.method.get_first_lead()
                self.base_height = self.config.unit_height * self.method.lead_length

        if self.config.reverse_method:
            self.drawable_rows = self.drawable_rows[::-1]

        self.shapes = []

    def create_base(self):
        # Base plate
        bottom_left = np.array([
            -self.config.margin,
            -self.base_height - self.config.margin,
            -self.config.plate_thickness])
        top_right = np.array([
            self.base_width + self.config.margin,
            self.config.margin,
            0])

        match self.config.title_position:
            case TitlePos.TOP:
                top_right[1] += self.config.braille_config.cell_spacing_y
            case TitlePos.BOTTOM:
                bottom_left[1] -= self.config.braille_config.cell_spacing_y
            case TitlePos.LEFT:
                bottom_left[0] -= self.config.braille_config.cell_spacing_y
            case TitlePos.RIGHT:
                top_right[0] += self.config.braille_config.cell_spacing_y

        match self.config.base_type:
            case BaseType.HOLE:
                file_name = "templates/base_template_hole.stl"
                has_hole = True
            case BaseType.NO_HOLE:
                file_name = "templates/base_template_no_hole.stl"
                has_hole = False
            case _:
                raise Exception(f"Bad base_type of {self.config.base_type}")

        self.shapes.append(create_base_from_template(
            file_name,
            bottom_left,
            top_right,
            self.config.margin,
            has_hole))

    def create_title(self):
        if self.config.title_position == TitlePos.NONE:
            print("Warning: Title creation called, but TITLE_POSITION is NONE")

        else:
            match self.config.title_text:
                case TitleText.FULL:
                    title_text = self.method.title
                case TitleText.FULL_LOWER:
                    title_text = self.method.title.lower()
                case TitleText.SHORT:
                    title_text = f"{self.method.name} {self.method.stage}"
                case TitleText.SHORT_LOWER:
                    title_text = f"{self.method.name} {self.method.stage}".lower()

            match self.config.title_position:
                case TitlePos.TOP:
                    start_position = np.array([0, self.config.braille_config.cell_spacing_y, 0])
                    angle = 0
                    alignment_x = AlignmentX.LEFT
                    alignment_y = AlignmentY.TOP
                    max_length = self.base_width
                case TitlePos.BOTTOM:
                    start_position = np.array([0, -self.base_height - self.config.braille_config.cell_gap_y, 0])
                    angle = 0
                    alignment_x = AlignmentX.LEFT
                    alignment_y = AlignmentY.TOP
                    max_length = self.base_width
                case TitlePos.LEFT:
                    start_position = np.array([-self.config.braille_config.cell_gap_y, 0, 0])
                    angle = -np.pi / 2
                    alignment_x = AlignmentX.LEFT
                    alignment_y = AlignmentY.TOP
                    max_length = self.base_height
                case TitlePos.RIGHT:
                    start_position = np.array([self.base_width + self.config.braille_config.cell_spacing_y, 0, 0])
                    angle = -np.pi / 2
                    alignment_x = AlignmentX.LEFT
                    alignment_y = AlignmentY.TOP
                    max_length = self.base_height
                case TitlePos.CENTER_HORIZONTAL:
                    start_position = np.array([self.base_width / 2, -self.base_height / 2, 0])
                    angle = -np.pi / 2
                    alignment_x = AlignmentX.CENTER
                    alignment_y = AlignmentY.CENTER
                    max_length = self.base_width
                case TitlePos.CENTER_VERTICAL:
                    start_position = np.array([self.base_width / 2, -self.base_height / 2, 0])
                    angle = -np.pi / 2
                    alignment_x = AlignmentX.CENTER
                    alignment_y = AlignmentY.CENTER
                    max_length = self.base_height

            points = str_to_dots(title_text, self.config.braille_config, max_length)
            points = align_points(points, alignment_x)
            points = align_points(points, alignment_y)
            points = rotate_points(points, angle)
            points += start_position

            for point in points:
                self.shapes.append(create_hemisphere(point,
                                                     self.config.braille_config.dot_diameter,
                                                     self.config.braille_config.dot_height))

    def create_place_bell_label(self, bell: int | str):
        if self.config.title_position == TitlePos.NONE:
            print("Warning: Place bell label creation called, but TITLE_POSITION is NONE")

        else:
            text = str(bell)

            match self.config.title_position:
                case TitlePos.TOP:
                    start_position = np.array([self.base_width,
                                               self.config.braille_config.cell_spacing_y,
                                               0])
                    angle = 0
                case TitlePos.BOTTOM:
                    start_position = np.array([self.base_width,
                                               -self.base_height - self.config.braille_config.cell_gap_y,
                                               0])
                    angle = 0
                case TitlePos.LEFT:
                    start_position = np.array([-self.config.braille_config.cell_gap_y,
                                               -self.base_height,
                                               0])
                    angle = -np.pi / 2
                case TitlePos.RIGHT:
                    start_position = np.array([self.base_width + self.config.braille_config.cell_spacing_y,
                                               -self.base_height,
                                               0])
                    angle = -np.pi / 2

            points = str_to_dots(text, self.config.braille_config)
            points = align_points(points, AlignmentX.RIGHT)
            points = rotate_points(points, angle)
            points += start_position

            for point in points:
                self.shapes.append(create_hemisphere(point,
                                                     self.config.braille_config.dot_diameter,
                                                     self.config.braille_config.dot_height))

    def create_vertical_lines(self):
        # Vertical lines
        for i in range(self.method.stage):
            self.shapes.append(create_path_object(
                np.array([np.array([i * self.config.unit_width,
                                    -x / (len(self.drawable_rows) - 1) * self.base_height,
                                    0]) for x in range(len(self.drawable_rows))]),
                line_config=self.config.thin_line_config))

    def create_thick_line(self, bell: int):
        path = Method.path_from_method(self.drawable_rows, bell, self.config)
        path[:, 1] *= self.base_height / -np.min(path[:, 1])  # Normalise to fit on plate
        self.shapes.append(create_path_object(path, line_config=self.config.thick_line_config))

    def create_lead_end_markers(self, bell: int):
        i = 0

        while i < len(self.drawable_rows):
            place = self.drawable_rows[i].index(str(bell))

            position = np.array([
                place * self.config.unit_width,
                -i * self.config.unit_height,
                0])

            match self.config.lead_end_marker_type:
                case LeadEndMarkerType.DOME:
                    self.shapes.append(create_hemisphere(position, self.config.lead_end_marker_width,
                                                         self.config.lead_end_marker_height))
                case LeadEndMarkerType.T:
                    left_position = np.array(
                        [max(position[0] - self.config.unit_width, -self.config.lead_end_t_overflow), position[1], 0])
                    right_position = np.array(
                        [min(position[0] + self.config.unit_width, self.base_width + self.config.lead_end_t_overflow),
                         position[1], 0])

                    self.shapes.append(
                        create_path_object(np.array([left_position, right_position]), self.config.thick_line_config))
                case LeadEndMarkerType.NONE:
                    pass
                case _:
                    raise Exception("Bad lead_end_marker_option")

            i += self.method.lead_length

    def create_half_lead_lines(self):
        if self.config.reverse_method:
            i = (self.method.lead_length + 1) / 2
        else:
            i = (self.method.lead_length - 1) / 2

        while i < len(self.drawable_rows):
            self.shapes.append(create_path_object(np.array(
                [np.array([x / (self.method.stage - 1) * self.base_width,
                           -i * self.config.unit_height,
                           0]) for x in range(self.method.stage)]),
                line_config=self.config.half_lead_line_config))

            i += self.method.lead_length

    def create_treble_line(self, treble_bell: int = 1, blue_line_bell: int | None = None):
        match self.config.treble_type:
            case TrebleType.SOLID:
                path = Method.path_from_method(self.drawable_rows, treble_bell, self.config.unit_width,
                                               self.config.unit_height)
                self.shapes.append(create_path_object(path, line_config=self.config.treble_line_config))

            case TrebleType.CROSS:
                paths = Method.passing_point_paths_from_method(self.drawable_rows,
                                                               blue_line_bell,
                                                               treble_bell,
                                                               self.config.unit_width,
                                                               self.config.unit_height)

                for path in paths:
                    self.shapes.append(create_path_object(path, line_config=self.config.treble_line_config))

    def save_to_stl(self, file_name="stl-files/output.stl"):
        # Write the mesh to file
        combined = mesh.Mesh(np.concatenate([shape.data for shape in self.shapes]))
        combined.save(file_name)
