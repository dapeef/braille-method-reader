import requests
import numpy as np

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

def path_from_method(method, bell:str, unit_width:float, unit_height:float):
    path = []

    y = 0
    
    for row in method["rows"]:
        path.append(np.array([row[0].index(bell) * unit_width, y, 0]))
        y += unit_height
    
    return np.array(path)

if __name__ == "__main__":
    # print(get_method_id("cambridge"))
    print(get_method_from_id("16694"))