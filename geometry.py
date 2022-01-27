#All math assumed RHR. Returned stuff is corrected for switching to LHR
import numpy as np
import math

#parameters (for TDW images)
image_dim_x = 256.
image_dim_y = 256.
horizontal_FoV = 55.
vertical_FoV = 55.


"""
    get_abc_plane(p1::Coordinate, p2::Coordinate, p3::Coordinate)

Given 3 points on a plane (p1, p2, p3), get coefficients for the plane
determined by the 3 points.
"""
def get_abc_plane(p1, p2, p3):
    #using Method 2 from wikipedia: https://en.wikipedia.org/wiki/Plane_(geometry)#:~:text=In%20mathematics%2C%20a%20plane%20is,)%20and%20three%2Ddimensional%20space.
    D = np.linalg.det(np.array([p1, p2, p3]))
    #print(f"D {D}")
    if D==0:
        print("crap! determinant D=0")
        print(p1, p2, p3)

    #implicitly going to say d=-1 to obtain solution set
    #print(np.column_stack(([1, 1, 1], p2, p3)))
    a = np.linalg.det(np.column_stack(([1, 1, 1], [p1[1], p2[1], p3[1]], [p1[2], p2[2], p3[2]])))/D
    b = np.linalg.det(np.column_stack(([p1[0], p2[0], p3[0]], [1, 1, 1], [p1[2], p2[2], p3[2]])))/D
    c = np.linalg.det(np.column_stack(([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], [1, 1, 1])))/D

    return np.array([a, b, c])

"""
    get_horizontal_plane(camera_params::Camera_Params, a_vertical, b_vertical, c_vertical)

Returns coefficients a, b, c for the horizontally oriented plane that includes
the vector between the camera focus and position

This implementation depends on already having determined the vertically-oriented plane.
Note that our implementation returns a placeholder value when the camera is
pointing upwards.
"""
def get_horizontal_plane(camera_pos, camera_focus, vertical_abc):
    p1 = camera_pos
    #print(f"p1 {p1}")
    p2 = camera_focus
    p3 = p1 + vertical_abc#adding normal vector (a, b, c) to the point to make a third point on the horizontal plane
    #print(f"p3 {p3}")
    h = get_abc_plane(p1, p2, p3)
    #make sure that this normal is upright, so b > 0
    if h[1] < 0:
        return (-1 * h)
    elif h[1] > 0:
        return h
    else: #when b==0, camera is looking straight up or straight down.
        print("uh oh. camera looking straight up or straight down")
        return h


"""
    get_vertical_plane(camera_params::Camera_Params)

Returns coefficients a,b,c for the vertically-oriented plane which includes
the vector between the camera focus and position.

Note that the y-axis is the upward one.
"""
def get_vertical_plane(camera_pos, camera_focus):
    p1 = camera_pos
    p2 = camera_focus
    p3 = np.array([p1[0], p1[1]+1, p1[2]])
    plane_abc = get_abc_plane(p1, p2, p3)

    #want normal to be be on the "righthand" side of camera
    #x and z component of vec for the direction camera is pointing
    camera_pointing = camera_focus - camera_pos

    result = np.cross(plane_abc, camera_pointing)
    #if y-component < 0, multiply by -1
    if result[1] < 0:
        plane_abc = -1 * plane_abc
    return plane_abc


def get_vector(camera_pos, camera_focus, detection):

    # camera_pos[2] = -camera_pos[2]
    # camera_focus[2] = -camera_focus[2]

    x = detection[0] - image_dim_x/2
    y = detection[1] - image_dim_y/2 #now (0,0) is the center of the image

    x = -x #for getting left-handed coordinates

    angle_from_vertical = x / (image_dim_x/2) * math.radians(horizontal_FoV/2)
    angle_from_horizontal = -1 * y / (image_dim_y/2) * math.radians(vertical_FoV/2)

    #print(f"angle_from_vertical{angle_from_vertical}")

    v = get_vertical_plane(camera_pos, camera_focus)
    h = get_horizontal_plane(camera_pos, camera_focus, v)

    l = math.dist(camera_pos, camera_focus)
    x = l * math.tan(angle_from_vertical)
    #print(f"x {x}")
    y = l * math.tan(angle_from_horizontal)

    normalized_v = v/np.linalg.norm(v)
    normalized_h = h/np.linalg.norm(h)

    #print(f"normalized_v {normalized_v}")

    point = camera_focus + x * normalized_v + y * normalized_h
    #print(f"point {point}")
    vector = point - camera_pos
    vec = vector/np.linalg.norm(vector) #return the normalized vector
    #Flip vector into the LHR coordinate system for TDW
    vec = np.array([vec[0], vec[1], vec[2]])
    return vec

"""
    proj_vec_to_plane(a, b, c, x, y , z)

Projects a vector onto a plane.
"""
def proj_vec_to_plane(plane, point):
    numerator = np.dot(plane, point)
    denominator = np.linalg.norm(plane) * np.linalg.norm(plane) #check this
    const = numerator/denominator
    return point - const*plane

"""
    get_angle(a, b, c, x, y, z)

Returns the angle between the vector ``(x, y, z)`` and the plane ``(a, b, c)``

Whether the sin is positive or negative depends on the normal.
"""
def get_angle(plane, point):
    numerator = np.dot(plane, point)
    denominator = np.linalg.norm(plane) * np.linalg.norm(point)
    return math.asin(numerator/denominator)

"""
    locate(camera_params::Camera_Params, params::Video_Params, object::Coordinate)

Given camera info and object's location, finds object's location on 2-D image
"""
def locate(camera_pos, camera_focus, object_pos):
    v = get_vertical_plane(camera_pos, camera_focus)
    h = get_horizontal_plane(camera_pos, camera_focus, v)
    s = object_pos - camera_pos
    s_v = proj_vec_to_plane(v, s)
    s_h = proj_vec_to_plane(h, s)

    angle_from_vertical = get_angle(v, s_h)
    angle_from_horizontal = get_angle(h, s_v)

    x = image_dim_x/2 * (angle_from_vertical / math.radians(horizontal_FoV/2))
    y = image_dim_y/2 * (-angle_from_horizontal / math.radians(vertical_FoV/2))

    x = x + image_dim_x/2
    y = y + image_dim_y/2
    return [x, y]

"""
    on_right_side(camera_params::Camera_Params, object::Coordinate)

Checks if the object is on the right side of camera.

The object is on the right side of the camera if
``\\vec{(\\mathrm{focus} - \\mathrm{camera})} \\cdot \\vec{(\\mathrm{object} - \\mathrm{camera})} > 0``
"""
def on_right_side(camera_pos, camera_focus, object_pos):
    return np.dot((camera_focus - camera_pos),  (object_pos - camera_pos))

"""
    get_image_xy(camera_params::Camera_Params, params::Video_Params, object::Coordinate)

Returns the object's position on the 2D image in pixel space.

(0,0) is the center of the image
"""
def get_image_xy(camera_pos, camera_focus, object_pos):
    #for left-handed coordinate system like TDW, flip the sign of the z-coordinate
    camera_pos = np.array([camera_pos[0], camera_pos[1], -camera_pos[2]])
    camera_focus = np.array([camera_focus[0], camera_focus[1], -camera_focus[2]])
    object_pos = np.array([object_pos[0], object_pos[1], -object_pos[2]])

    if on_right_side(camera_pos, camera_focus, object_pos) > 0:
        object_2d = np.array(locate(camera_pos, camera_focus, object_pos))
    else:
        object_2d = np.array([np.inf, np.inf])

    return object_2d

def within_frame(object_2d):
    a = (object_2d[0] <= image_dim_x) & (object_2d[0] >= 0)
    b = (object_2d[1] <= image_dim_y) & (object_2d[1] >= 0)
    return a & b
# camera_pos = np.array([0.1, 0.01, 0.001])
# camera_focus = np.array([1., 0.0001, 0.00001])
# obj_pos = np.array([0.99591707, -0.3424945,  0.28579714])
# obj_2D = get_image_xy(camera_pos, camera_focus, obj_pos)
# print(obj_2D)

#obj_2D = np.array([46., 225.])

#vec = get_vector(camera_pos, camera_focus, obj_2D)
#print(vec)
