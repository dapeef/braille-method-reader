import requests
import numpy as np

class Method:
    def __init__(self, complib_data) -> None:
        self.title = complib_data["title"]
        self.stage = complib_data["stage"]
        self.raw_place_notation = complib_data["placeNotation"]
        self.place_notation = Method.parse_place_notation(self.raw_place_notation)
        self.lead_length = len(self.place_notation)
        self.rows = [row[0] for row in complib_data["rows"][1:]]

    @staticmethod
    def parse_place_notation(raw_place_notation: str) -> list[str]:
        def is_num(token):
            return token in "0123456789"
        
        place_notation = []

        was_last_token_number = False

        for token in raw_place_notation:
            if token == ",":
                for i in range(len(place_notation)-2, -1, -1):
                    place_notation.append(place_notation[i])

            elif token == ".":
                was_last_token_number = False
                
            elif was_last_token_number and is_num(token):
                place_notation[-1] += token
            
            else:
                place_notation.append(token)

            was_last_token_number = is_num(token)
        
        return place_notation

    @staticmethod
    def path_from_method(rows:list[str], bell:int, unit_width:float, unit_height:float):
        bell = str(bell)

        path = []

        y = 0
        
        for row in rows:
            path.append(np.array([row.index(bell) * unit_width, y, 0]))
            y -= unit_height
        
        return np.array(path)
    
    def get_first_lead(self) -> list[str]:
        return self.rows[0:self.lead_length+1]


def make_api_call(url:str):
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()  # Parse JSON response
    else:
        raise Exception(f"Network error: {response.status_code}")

def get_method_id(search_string:str):
    data = make_api_call(f"https://api.complib.org/method/search/{search_string}")

def get_method_from_id(method_id: str) -> list[str]:
    data = make_api_call(f"https://api.complib.org/method/{method_id}/rows")
    
    return data

if __name__ == "__main__":
    # print(get_method_id("cambridge"))
    data = get_method_from_id("16694")

    cambridge = Method(data)
    rows = cambridge.get_first_lead()
    path = Method.path_from_method(rows, 2, 10, 10)

    pass