import numpy as np
import matplotlib.pyplot as plt
import os

two_arm_pos=np.load("../data/arm_data.npz")
dir_id = "./images"
position_rod1 = two_arm_pos["position_rod"]
radius_rod1 = two_arm_pos["radii_rod"]
n_elems_rod1 = two_arm_pos["n_elems_rod"]

"""
position_sphere = two_arm_pos["position_sphere"]
radius_sphere = two_arm_pos["radii_sphere"]
"""

scale=1 #scales scene. Set at 3 to avoid resetting camera angles. 

#snap shot every (time_inc) frames
time_inc=0.1#int(pos.shape[0]/5)
time = position_rod1.shape[0]

for k in range(time):
    file1 = open("%s/moving_arm%02d.inc" % (dir_id, k),"w")

    # Arm
    file1.writelines("sphere_sweep\n{b_spline %d"% int(n_elems_rod1))
    radius_scale = 1.0 #np.flip(np.linspace(0.25,1.0,n_elems_rod1))
    for i in range(0,n_elems_rod1):
        file1.writelines(
            ",\n<%f,%f,%f>,%f" % (scale*position_rod1[k][0][i] , scale*(position_rod1[k][1][i]), scale*position_rod1[k][2][i], scale*radius_rod1[k][i]*radius_scale))
    file1.writelines("\ntexture{")
    file1.writelines("pigment{ color rgb<0.45,0.39,1>  transmit %f }" % (0.1))
    file1.writelines("finish{ phong 1 } }")
    file1.writelines("scale<1,1,1> rotate<0,0,0> translate<0,0,0>    }\n")

    file2 = open("%s/moving_arm%02d.pov" % (dir_id, k) ,"w")
    file2.writelines("#include \"../camera_position.inc\"\n")
    file2.writelines("#include \"moving_arm%02d.inc\"\n"%k)
    file2.close()