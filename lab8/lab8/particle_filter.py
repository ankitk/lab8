from grid import *
from particle import Particle
from utils import *
from setting import *
from time import sleep
import numpy as np


def motion_update(particles, odom):
    motion_particles = []
    for particle in particles:
        tempX = odom[0]
        tempY = odom[1]
        x, y = rotate_point(tempX, tempY, particle.h)
        particle.x += x
        particle.y += y
        particle.x = add_gaussian_noise(particle.x, ODOM_TRANS_SIGMA)
        particle.y = add_gaussian_noise(particle.y, ODOM_TRANS_SIGMA)
        particle.h += odom[2]
        particle.h = add_gaussian_noise(particle.h, ODOM_HEAD_SIGMA)
        motion_particles.append(particle)

    return motion_particles

# ------------------------------------------------------------------------
def measurement_update(particles, measured_marker_list, grid):
    weight = []
    count = 0
    if len(measured_marker_list) == 0:
        s = 1
        for particle in particles:
            weight.append((particle, 1/len(particles)))
    else:
        for particle in particles:
            visible_markers = particle.read_markers(grid)
            if (particle.x, particle.y) in grid.occupied or particle.x < 0 or particle.x >= grid.width or particle.y < 0 or particle.y >= grid.height:
                weight.append((particle, 0))
                continue

            match = []
            diff = int(math.fabs(len(measured_marker_list)-len(visible_markers)))

            for cm in measured_marker_list:
                if len(visible_markers) == 0:
                    break
                cmx, cmy, cmh = add_marker_measurement_noise(cm, MARKER_TRANS_SIGMA, MARKER_ROT_SIGMA)

                minp = visible_markers[0]
                mind = grid_distance(cmx, cmy, minp[0], minp[1])

                for mvp in visible_markers:
                    mvpx, mvpy, mvph = mvp[0], mvp[1], mvp[2]
                    dist = grid_distance(cmx, cmy, mvpx, mvpy)
                    if dist < mind:
                        mind = dist
                        minp = mvp

                match.append((minp, cm))
                visible_markers.remove(minp)

            prob = 1

            maxc1 = 0
            maxc2 = (45 ** 2) / (2*(MARKER_ROT_SIGMA ** 2))
            c1 = 2*(MARKER_TRANS_SIGMA ** 2)
            c2 = 2*(MARKER_ROT_SIGMA ** 2)

            for i, j in match:
                distBetweenMarkers = grid_distance(i[0], i[1], j[0], j[1])
                angleBetweenMarkers = diff_heading_deg(i[2], j[2])
                const1 = (distBetweenMarkers ** 2) / c1
                const2 = (angleBetweenMarkers ** 2) / c2
                maxc1 = max(maxc1, const1)
                prob *= np.exp(-const1-const2)

            for _ in range(diff):
                prob *= np.exp(-maxc1-maxc2)

            weight.append((particle, prob))

        s = 0
        weight.sort(key=lambda x: x[1])
        delete = int(PARTICLE_COUNT/100)
        weight = weight[delete:]
        for i, j in weight:
            if j == 0:
                count+=1
            else:
                s += j
        weight = weight[count:]
        count += delete

    plist = []
    wlist = []

    for i, j in weight:
        newi = Particle(i.x, i.y, i.h)
        wlist.append(j/s)
        plist.append(newi)

    newplist = []

    if plist != []:
        newplist = np.random.choice(plist, size=len(plist), replace = True, p=wlist)

    measured_particles = Particle.create_random(count, grid)[:]

    for p in newplist:
        ph = add_gaussian_noise(p.h, ODOM_HEAD_SIGMA)
        px = add_gaussian_noise(p.x, ODOM_TRANS_SIGMA)
        py = add_gaussian_noise(p.y, ODOM_TRANS_SIGMA)
        newp = Particle(px, py, ph)
        measured_particles.append(newp)

    return measured_particles