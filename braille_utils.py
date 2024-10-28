import numpy as np

ascii_chars = " A1B'K2L@CIF/MSP\"E3H9O6R^DJG>NTQ,*5<-U8V.%[$+X!&;:4\\0Z7(_?W]#Y)=".lower()
braille_chars = "⠀⠁⠂⠃⠄⠅⠆⠇⠈⠉⠊⠋⠌⠍⠎⠏⠐⠑⠒⠓⠔⠕⠖⠗⠘⠙⠚⠛⠜⠝⠞⠟⠠⠡⠢⠣⠤⠥⠦⠧⠨⠩⠪⠫⠬⠭⠮⠯⠰⠱⠲⠳⠴⠵⠶⠷⠸⠹⠺⠻⠼⠽⠾⠿"
numbers = "0123456789"


class BrailleConfig:
    DOT_DIAMETER = 1.5 # mm
    DOT_HEIGHT = 0.6 # mm
    DOT_SPACING = 2.3 # mm
    CELL_SPACING_X = 6.1 # mm
    CELL_SPACING_Y = 10 # mm

    CELL_GAP_X = CELL_SPACING_X - DOT_SPACING
    CELL_GAP_Y = CELL_SPACING_Y - DOT_SPACING * 2

    INCLUDE_CAPS = True # Should the braille display capital letters, or should they all be lower


def str_to_braille(string:str, legible:bool=False) -> str:
    braille_string = ""
    was_previous_number = False

    for char in string:
        if char.lower() in ascii_chars:
            # Handle prefix characters
            if char.isupper():
                braille_string += braille_chars[ascii_chars.index(",")]

            if char in numbers:
                if not was_previous_number:
                    braille_string += braille_chars[ascii_chars.index("#")]
                was_previous_number = True
            else:
                was_previous_number = False

            # Handle letter
            braille_string += braille_chars[ascii_chars.index(char.lower())]

        else:
            raise Exception(f"Ascii character \"{char}\" can't be converted to braille.")
    
    if legible:
        legible_string = ""
        for char in braille_string:
            legible_string += char + " "
        
        braille_string = legible_string[:-1]

        
    return braille_string

def char_to_dots(braille_char:str, position:np.ndarray, config:BrailleConfig) -> np.ndarray:
    points = []

    binary_string = f"{braille_chars.index(braille_char):06b}" # eg 5 -> 000101
    binary_string = binary_string[::-1] # Get lsb first

    for i in range(6):
        if binary_string[i] == "1":
            points.append(np.array([
                np.floor_divide(i, 3),
                -(i % 3),
                0]) * config.DOT_SPACING + position)

    return points

def str_to_dots(string:str,
                config:BrailleConfig=BrailleConfig()) -> np.ndarray:
    points = []
    position = np.array([0, 0, 0])

    braille_string = str_to_braille(string)


    for char in braille_string:
        if char == "\n":
            position[0] = 0
            position[1] += config.CELL_SPACING_Y

        else:
            # Add letter
            points += char_to_dots(char, position, config)
            position[0] += config.CELL_SPACING_X
    
    return np.array(points)


if __name__ == "__main__":
    from geometry_utils import plot_3d_path

    string = "Cambridge 888"

    print(str_to_braille(string, legible=True))
    plot_3d_path(str_to_dots(string))