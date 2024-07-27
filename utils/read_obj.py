import numpy as np


def read_obj_file(obj_path):
    with open(obj_path, 'r') as f:
        lines = f.readlines()
    vertices = np.zeros((1, 3))
    k = 0
    for i in range(len(lines)):
        if lines[i][0] == 'v' and lines[i][1] == ' ':
            vert = lines[i].replace('\n', '').split(' ')
            vert_x = float(vert[1])
            vert_y = float(vert[2])
            vert_z = float(vert[3])
            if k == 0:
                vertices = np.array([[vert_x, vert_y, vert_z]])
            else:
                vertices = np.append(vertices, np.array([[vert_x, vert_y, vert_z]]), axis=0)
            k = k+1

    return vertices

