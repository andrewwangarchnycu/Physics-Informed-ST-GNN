import os

def get_all_extensions(directory_path):
    extensions = set()
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # os.path.splitext returns a tuple: ('filename', '.ext')
            ext = os.path.splitext(file)[1].lower()
            if ext:
                extensions.add(ext)
                
    return extensions

# Usage
path = "C:/Users/user/Desktop/UTIC GNN/Physics-Informed ST-GNN/urban-thermal-gnn"
print(get_all_extensions(path))