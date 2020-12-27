# TODO fish shoal that avoids predators
# TODO split rendering code
# Copyright (c) 2012 Tom Marble
# Licensed under the MIT license http://opensource.org/licenses/MIT
# https://github.com/tmarble/pyboids/blob/master/boids.py

"""Implements the Peter Keller boids program
which is an adaptation of
http://www.cs.toronto.edu/~dt/siggraph97-course/cwr87/
"""
from __future__ import annotations

import math
import random
import time

import pygame
import pygame.locals as pyg
from OpenGL import GL, GLU

COHESION_RANGE = 10
ALIGNMENT_RANGE = 10
SEPARATION_RANGE = 10
BOUND_RANGE_RATIO = 0.8  # Point at which boids start turning round

COHESION_WEIGHT = 0.001
ALIGNMENT_WEIGHT = 0.04
SEPARATION_WEIGHT = 0.05
BOUND_WEIGHT = 0.001

CONSTRAIN_TO_CUBE = True  # False for pac-man wrapping
EDGE = 15  # cube size
UPDATE_INTERVAL = 0.032
NUM_BOIDS = 150
BASE_SPEED = 0.03


class CustomVector3:
    """
    numpy arrays aren't optimised for small lengths.
    This tries to replicate the bare pygame Vector3 behaviour we need.
    It's way way slower, but it's still better than numpy.
    """

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def length(self):
        return self._length(*self)

    @classmethod
    def _length(cls, x, y, z):
        return math.sqrt(sum(map(lambda a: a ** 2, (x, y, z))))

    def normalize(self):
        length = self.length()
        if not length:
            raise ValueError("Can't normalize Vector of length Zero")
        return CustomVector3(self.x / length, self.y / length, self.z / length)

    def distance_to(self, other):
        return self._length(*map(lambda x: x[0] - x[1], zip(self, other)))

    def __add__(self, other):
        return self._naive_op(other, lambda v: v[0] + v[1])

    def __sub__(self, other):
        return self._naive_op(other, lambda v: v[0] - v[1])

    def __mul__(self, other):
        if isinstance(
            other,
            (
                int,
                float,
            ),
        ):
            return self._naive_op([other] * 3, lambda v: v[0] * v[1])
        raise NotImplementedError("too much maths")

    def __truediv__(self, other):
        if isinstance(
            other,
            (
                int,
                float,
            ),
        ):
            return self._naive_op([other] * 3, lambda v: v[0] / v[1])
        raise TypeError("Can't divide two vectors")

    def _naive_op(self, other, func):
        return CustomVector3(*map(func, zip(self, other)))

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __str__(self):
        return str(list(self))

    def __repr__(self):
        return "<Vector3%s>" % (tuple(self),)


try:
    from pygame.math import Vector3
except ImportError:
    print("Warning: using slow custom Vector3 class. Consider installing pygame")
    Vector3 = CustomVector3


def rand_vector3(lower=0.0, upper=1.0):
    return Vector3(
        random.uniform(lower, upper),
        random.uniform(lower, upper),
        random.uniform(lower, upper),
    )


def zero_vector3():
    return Vector3(0, 0, 0)


class Rule:
    """
    Template for rules of flocking
    """

    # max distance at which rule is applied
    RANGE = NotImplemented

    def __init__(self):
        self.change = zero_vector3()  # velocity correction
        self.num = 0  # number of participants

    def accumulate(self, boid, other, distance):
        """
        Save any corrections based on other boid to self.change
        """
        raise NotImplementedError

    def add_adjustment(self, boid):
        """
        Add the accumulated self.change to boid.adjustment
        """
        raise NotImplementedError


class Cohesion(Rule):
    """
    Rule 1: Boids try to fly towards the centre of mass of neighbouring boids.
    """

    RANGE = COHESION_RANGE

    def accumulate(self, boid, other, distance):
        self.change += other.location
        self.num += 1

    def add_adjustment(self, boid):
        if self.num > 0:
            centroid = self.change / self.num
            desired = centroid - boid.location
            self.change = (desired - boid.velocity) * COHESION_WEIGHT
        boid.adjustment += self.change


class Alignment(Rule):
    """
    Rule 2: Boids try to match velocity with near boids.
    """

    RANGE = ALIGNMENT_RANGE

    def accumulate(self, boid, other, distance):
        self.change += other.velocity
        self.num += 1

    def add_adjustment(self, boid):
        if self.num > 0:
            group_velocity = self.change / self.num
            self.change = (group_velocity - boid.velocity) * ALIGNMENT_WEIGHT
        boid.adjustment += self.change


class Separation(Rule):
    """
    Rule 3: Boids try to keep a small distance away from other objects (including other boids).
    """

    RANGE = SEPARATION_RANGE

    def accumulate(self, boid, other, distance):
        separation = boid.location - other.location
        if distance > 0:
            self.change += separation.normalize() / distance
        self.num += 1

    def add_adjustment(self, boid):
        if self.change.length() > 0:
            group_separation = self.change / self.num
            self.change = (group_separation - boid.velocity) * SEPARATION_WEIGHT
        boid.adjustment += self.change


class Boid:
    def __init__(self, b_id, behaviour: AbstractWallBehaviour):
        self.id = b_id
        self.behaviour = behaviour
        self.color = rand_vector3(0.3, 0.7)  # R G B
        self.location = zero_vector3()  # x y z
        self.velocity = rand_vector3(-1.0, 1.0)  # vx vy vz
        self.adjustment = zero_vector3()  # to accumulate corrections

        self.is_near_wall = False

    def __repr__(self):
        return "color %s, location %s, velocity %s" % (
            self.color,
            self.location,
            self.velocity,
        )

    def orient(self, boids):
        """
        Calculate new position
        """
        rules = [Cohesion(), Alignment(), Separation()]
        for boid in boids:  # accumulate corrections
            if self.id == boid.id:
                continue

            distance = self.location.distance_to(boid.location)
            for rule in rules:
                if distance < rule.RANGE:
                    rule.accumulate(self, boid, distance)
        self.adjustment = zero_vector3()  # reset adjustment vector

        for rule in rules:
            rule.add_adjustment(self)

        self.behaviour.add_adjustment(self)

    def update(self):
        """
        Move to new position
        """
        self.velocity += self.adjustment
        # Hack: Add a constant velocity in whatever direction
        # they are moving so they don't ever stop.
        if self.velocity.length() > 0:
            self.velocity += self.velocity.normalize() * random.uniform(0.0, BASE_SPEED)
        self.limit_speed(1.0)
        self.location += self.velocity

        self.is_near_wall = self.behaviour.is_near_wall(self)

    def limit_speed(self, max_speed):
        """
        Ensure the speed does not exceed max_speed
        """
        if self.velocity.length() > max_speed:
            self.velocity = self.velocity.normalize() * max_speed


class Flock:
    """
    A flock of boids
    """

    def __init__(self, num_boids, cube_min, cube_max, behaviour: AbstractWallBehaviour):
        self.cube_min = cube_min
        self.cube_max = cube_max
        self.behaviour = behaviour
        self.boids = []
        for i in range(num_boids):
            self.boids.append(Boid(i, behaviour))

    def update(self):
        """
        update flock positions
        """
        t_0 = time.time()

        for boid in self.boids:
            boid.orient(self.boids)  # calculate new velocity
        for boid in self.boids:
            boid.update()  # move to new position

        t_1 = time.time()
        computation_time = t_1 - t_0
        if computation_time > UPDATE_INTERVAL:
            print(
                "WARNING: computation took %s (> %s)"
                % (computation_time, UPDATE_INTERVAL)
            )
        else:
            time.sleep(UPDATE_INTERVAL - computation_time)

    def __repr__(self):
        rep = "Flock of %d boids bounded by %s, %s" % (
            len(self.boids),
            self.cube_min,
            self.cube_max,
        )
        return rep


class AbstractWallBehaviour:
    """
    What to do when a boid is near the edge of its container
    """

    def __init__(self, cube_min, cube_max):
        self.cube_min = cube_min
        self.cube_max = cube_max

    def is_near_wall(self, boid) -> bool:
        raise NotImplementedError

    def add_adjustment(self, boid):
        raise NotImplementedError


class WrapBehaviour(AbstractWallBehaviour):
    """
    Wrap round to the opposite side like pac-man
    """

    def is_near_wall(self, boid):
        return False

    def add_adjustment(self, boid):
        loc = boid.location
        if loc.x < self.cube_min.x:
            loc.x += self.cube_max.x - self.cube_min.x
        elif loc.x > self.cube_max.x:
            loc.x -= self.cube_max.x - self.cube_min.x
        if loc.y < self.cube_min.y:
            loc.y += self.cube_max.y - self.cube_min.y
        elif loc.y > self.cube_max.y:
            loc.y -= self.cube_max.y - self.cube_min.y
        if loc.z < self.cube_min.z:
            loc.z += self.cube_max.z - self.cube_min.z
        elif loc.z > self.cube_max.z:
            loc.z -= self.cube_max.z - self.cube_min.z
        boid.location = loc


class BoundBehaviour(AbstractWallBehaviour):
    """
    Start turning towards the centre
    """

    def is_near_wall(self, boid):
        # TODO assumes centre is 0,0,0
        return any([abs(coord) >= EDGE * BOUND_RANGE_RATIO for coord in boid.location])

    def add_adjustment(self, boid):
        change = zero_vector3()
        if boid.is_near_wall:
            # TODO assumes centre is 0,0,0
            direction = zero_vector3() - boid.location
            change = direction * BOUND_WEIGHT
        boid.adjustment += change


class Renderer:
    def __init__(self, flock, cube_min, cube_max):
        self.flock = flock

        self.top_square = self.make_square_abstract(cube_min, cube_max, cube_max.z)
        self.bottom_square = self.make_square_abstract(cube_min, cube_max, cube_min.z)
        self.lines = self.make_vertical_lines(self.top_square, self.bottom_square)

    @staticmethod
    def make_square_abstract(a, b, z):
        return [
            [a.x, a.y, z],
            [a.x, b.y, z],
            [b.x, b.y, z],
            [b.x, a.y, z],
        ]

    @staticmethod
    def make_vertical_lines(top, bottom):
        """
        point pairs defining vertical lines of the cube
        """
        agg = []
        for tb in zip(top, bottom):
            agg.extend(tb)
        return agg

    def render_boundary(self):
        """
        Draw the bounding cube
        """
        GL.glColor(0.5, 0.5, 0.5)
        for loop in (self.top_square, self.bottom_square):
            GL.glBegin(GL.GL_LINE_LOOP)
            for point in loop:
                GL.glVertex(point)
            GL.glEnd()

        # The connecting lines in the Z direction
        GL.glBegin(GL.GL_LINES)
        for point in self.lines:
            GL.glVertex(point)
        GL.glEnd()

    def render_boids(self):
        GL.glBegin(GL.GL_LINES)
        for boid in self.flock.boids:
            GL.glColor(*boid.color)
            GL.glVertex(*boid.location)
            if boid.velocity.length() > 0:
                head = boid.location + boid.velocity.normalize()
            else:
                head = boid.location
            GL.glVertex(head.x, head.y, head.z)
        GL.glEnd()

    @staticmethod
    def render_point():
        """ unused so far """
        GL.glEnable(GL.GL_POINT_SMOOTH)
        GL.glPointSize(5)

        GL.glBegin(GL.GL_POINTS)
        GL.glColor3d(1, 1, 1)
        GL.glVertex3d(0, 0, 0)
        GL.glEnd()

    def render(self):
        self.render_boundary()
        self.render_boids()
        # self.render_point()


def main():

    random.seed(1)

    # initialize pygame and setup an opengl display
    #  angle = 0.4  # camera rotation angle
    angle = 0.1  # camera rotation angle
    # side = 700
    side = 1000
    xyratio = 1.0  # == xside / yside
    title = "pyboids 0.1"
    pygame.init()
    pygame.display.set_caption(title, title)
    pygame.display.set_mode((side, side), pyg.OPENGL | pyg.DOUBLEBUF)
    GL.glEnable(GL.GL_DEPTH_TEST)  # use our zbuffer
    GL.glClearColor(0, 0, 0, 0)
    # setup the camera
    GL.glMatrixMode(GL.GL_PROJECTION)
    GL.glLoadIdentity()
    GLU.gluPerspective(60.0, xyratio, 1.0, (6 * EDGE) + 10)  # setup lens
    GL.glMatrixMode(GL.GL_MODELVIEW)
    GL.glLoadIdentity()
    GLU.gluLookAt(0.0, 0.0, 3 * EDGE, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    GL.glPointSize(3.0)
    cube_min_vertex = Vector3(-EDGE, -EDGE, -EDGE)
    cube_max_vertex = Vector3(+EDGE, +EDGE, +EDGE)
    behaviour_class = BoundBehaviour if CONSTRAIN_TO_CUBE else WrapBehaviour
    behaviour = behaviour_class(cube_min_vertex, cube_max_vertex)
    flock = Flock(NUM_BOIDS, cube_min_vertex, cube_max_vertex, behaviour)

    renderer = Renderer(flock, cube_min_vertex, cube_max_vertex)

    while True:
        event = pygame.event.poll()
        if event.type == pyg.QUIT or (
            event.type == pyg.KEYDOWN
            and (event.key == pyg.K_ESCAPE or event.key == pyg.K_q)
        ):
            break
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glRotatef(angle, 0, 1, 0)  # orbit camera around by angle

        renderer.render()
        pygame.display.flip()
        flock.update()


if __name__ == "__main__":
    main()
