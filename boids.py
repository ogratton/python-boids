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

import pygame
import pygame.locals as pyg
from OpenGL import GL, GLU

CONSTRAIN_TO_CUBE = False  # False for pac-man hypertaurus


class CustomVector3:
    """
    numpy arrays aren't optimised for small lengths.

    This tries to replicate the bare pygame Vector3 behaviour we need

    But it's slow as HELL

    TODO just do it with numpy here anyway. It's probably still faster than this.
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
        return self._length(*map(lambda x: x[0]-x[1], zip(self, other)))

    def __add__(self, other):
        return self._naive_op(other, lambda v: v[0]+v[1])

    def __sub__(self, other):
        return self._naive_op(other, lambda v: v[0]-v[1])

    def __mul__(self, other):
        if isinstance(other, (int, float,)):
            return self._naive_op([other]*3, lambda v: v[0]*v[1])
        raise NotImplementedError("too much maths")

    def __truediv__(self, other):
        if isinstance(other, (int, float,)):
            return self._naive_op([other]*3, lambda v: v[0]/v[1])
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

    NEIGHBOURHOOD = 5  # max distance at which rule is applied

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

    def accumulate(self, boid, other, distance):
        self.change += other.location
        self.num += 1

    def add_adjustment(self, boid):
        if self.num > 0:
            centroid = self.change / self.num
            desired = centroid - boid.location
            self.change = (desired - boid.velocity) * 0.0006
        # TODO just return the change
        boid.adjustment += self.change


class Alignment(Rule):
    """
    Rule 2: Boids try to match velocity with near boids.
    """

    NEIGHBOURHOOD = 10

    def accumulate(self, boid, other, distance):
        self.change += other.velocity
        self.num += 1

    def add_adjustment(self, boid):
        if self.num > 0:
            group_velocity = self.change / self.num
            self.change = (group_velocity - boid.velocity) * 0.03
        boid.adjustment += self.change


class Separation(Rule):
    """
    Rule 3: Boids try to keep a small distance away from other objects (including other boids).
    """

    def accumulate(self, boid, other, distance):
        separation = boid.location - other.location
        if distance > 0:
            self.change += (separation.normalize() / distance)
        self.num += 1

    def add_adjustment(self, boid):
        if self.change.length() > 0:
            group_separation = self.change / self.num
            self.change = (group_separation - boid.velocity) * 0.01
        boid.adjustment += self.change


class Boid:
    def __init__(self, b_id, behaviour: AbstractWallBehaviour):
        self.id = b_id
        self.behaviour = behaviour
        self.color = rand_vector3(0.5)  # R G B
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
                if distance < rule.NEIGHBOURHOOD:
                    rule.accumulate(self, boid, distance)
        self.adjustment = zero_vector3()  # reset adjustment vector

        for rule in rules:
            rule.add_adjustment(self)

        self.behaviour.add_adjustment(self)  # TODO is this the right place?

    def limit_speed(self, max_speed):
        """
        Ensure the speed does not exceed max_speed
        """
        if self.velocity.length() > max_speed:
            self.velocity = self.velocity.normalize() * max_speed

    def update(self):
        """
        Move to new position
        """
        self.velocity += self.adjustment
        # Hack: Add a constant velocity in whatever direction
        # they are moving so they don't ever stop.
        if self.velocity.length() > 0:
            self.velocity += self.velocity.normalize() * random.uniform(0.0, 0.007)
        self.limit_speed(1.0)
        self.location += self.velocity

        self.is_near_wall = self.behaviour.is_near_wall(self)


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
        for boid in self.boids:
            boid.orient(self.boids)  # calculate new velocity
        for boid in self.boids:
            boid.update()  # move to new position

    def __repr__(self):
        rep = "Flock of %d boids bounded by %s, %s:\n" % (
            len(self.boids),
            self.cube_min,
            self.cube_max,
        )
        for i, b in enumerate(self.boids):
            rep += "%3d: %s\n" % (i, b)
        return rep

    def top_square(self):
        """
        loop of points defining top square
        """
        return [
            [self.cube_min.x, self.cube_min.y, self.cube_max.z],
            [self.cube_min.x, self.cube_max.y, self.cube_max.z],
            [self.cube_max.x, self.cube_max.y, self.cube_max.z],
            [self.cube_max.x, self.cube_min.y, self.cube_max.z],
        ]

    def bottom_square(self):
        """
        loop of points defining bottom square
        """
        return [
            [self.cube_min.x, self.cube_min.y, self.cube_min.z],
            [self.cube_min.x, self.cube_max.y, self.cube_min.z],
            [self.cube_max.x, self.cube_max.y, self.cube_min.z],
            [self.cube_max.x, self.cube_min.y, self.cube_min.z],
        ]

    def vertical_lines(self):
        """
        point pairs defining vertical lines of the cube
        """
        return [
            [self.cube_min.x, self.cube_min.y, self.cube_min.z],
            [self.cube_min.x, self.cube_min.y, self.cube_max.z],
            [self.cube_min.x, self.cube_max.y, self.cube_min.z],
            [self.cube_min.x, self.cube_max.y, self.cube_max.z],
            [self.cube_max.x, self.cube_max.y, self.cube_min.z],
            [self.cube_max.x, self.cube_max.y, self.cube_max.z],
            [self.cube_max.x, self.cube_min.y, self.cube_min.z],
            [self.cube_max.x, self.cube_min.y, self.cube_max.z],
        ]

    def render_cube(self):
        """
        Draw the bounding cube
        """
        GL.glColor(0.5, 0.5, 0.5)
        # XY plane, positive Z
        GL.glBegin(GL.GL_LINE_LOOP)
        for point in self.top_square():
            GL.glVertex(point)
        GL.glEnd()
        # XY plane, negative Z
        GL.glBegin(GL.GL_LINE_LOOP)
        for point in self.bottom_square():
            GL.glVertex(point)
        GL.glEnd()
        # The connecting lines in the Z direction
        GL.glBegin(GL.GL_LINES)
        for point in self.vertical_lines():
            GL.glVertex(point)
        GL.glEnd()

    def render(self):
        """
        draw a flock of boids
        """
        self.render_cube()
        GL.glBegin(GL.GL_LINES)
        for boid in self.boids:
            GL.glColor(*boid.color)
            GL.glVertex(*boid.location)
            if boid.velocity.length() > 0:
                head = boid.location + boid.velocity.normalize() * 3
            else:
                head = boid.location
            GL.glVertex(head.x, head.y, head.z)
        GL.glEnd()


class AbstractWallBehaviour:

    def __init__(self, cube_min, cube_max):
        self.cube_min = cube_min
        self.cube_max = cube_max

    def is_near_wall(self, boid: Boid) -> bool:
        raise NotImplementedError

    def add_adjustment(self, boid: Boid):
        raise NotImplementedError


class WrapBehaviour(AbstractWallBehaviour):

    def is_near_wall(self, boid):
        return False

    def add_adjustment(self, boid: Boid):
        loc = boid.location
        if loc.x < self.cube_min.x:
            loc.x = loc.x + (self.cube_max.x - self.cube_min.x)
        elif loc.x > self.cube_max.x:
            loc.x = loc.x - (self.cube_max.x - self.cube_min.x)
        if loc.y < self.cube_min.y:
            loc.y = loc.y + (self.cube_max.y - self.cube_min.y)
        elif loc.y > self.cube_max.y:
            loc.y = loc.y - (self.cube_max.y - self.cube_min.y)
        if loc.z < self.cube_min.z:
            loc.z = loc.z + (self.cube_max.z - self.cube_min.z)
        elif loc.z > self.cube_max.z:
            loc.z = loc.z - (self.cube_max.z - self.cube_min.x)
        boid.location = loc  # TODO this is just changing the location, but it's probably fine


class BoundBehaviour(AbstractWallBehaviour):

    def is_near_wall(self, boid):
        return any([abs(coord) >= EDGE * 0.95 for coord in boid.location])

    def add_adjustment(self, boid: Boid):
        change = zero_vector3()
        if boid.is_near_wall:
            # TODO assumes centre is 0,0,0
            direction = zero_vector3() - boid.location
            # TODO make it a more gradual turn
            change = direction * 0.005
        boid.adjustment += change


def main():

    random.seed(1)

    # initialize pygame and setup an opengl display
    #  angle = 0.4  # camera rotation angle
    angle = 0.1  # camera rotation angle
    # side = 700
    side = 1000
    delay = 0  # time to delay each rotation, in ms
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
    # GLU.gluPerspective(60.0, xyratio, 1.0, 250.0)   # setup lens
    GLU.gluPerspective(60.0, xyratio, 1.0, (6 * EDGE) + 10)  # setup lens
    GL.glMatrixMode(GL.GL_MODELVIEW)
    GL.glLoadIdentity()
    # GLU.gluLookAt(0.0, 0.0, 150, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    GLU.gluLookAt(0.0, 0.0, 3 * EDGE, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    GL.glPointSize(3.0)
    cube_min_vertex = Vector3(-EDGE, -EDGE, -EDGE)
    cube_max_vertex = Vector3(+EDGE, +EDGE, +EDGE)
    behaviour_class = BoundBehaviour if CONSTRAIN_TO_CUBE else WrapBehaviour
    behaviour = behaviour_class(cube_min_vertex, cube_max_vertex)
    # TODO the speed of the program is dependant on the number of boids
    #   this is really dumb
    flock = Flock(200, cube_min_vertex, cube_max_vertex, behaviour)
    while True:
        event = pygame.event.poll()
        if event.type == pyg.QUIT or (
            event.type == pyg.KEYDOWN
            and (event.key == pyg.K_ESCAPE or event.key == pyg.K_q)
        ):
            break
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        flock.render()
        GL.glRotatef(angle, 0, 1, 0)  # orbit camera around by angle
        pygame.display.flip()
        if delay > 0:
            pygame.time.wait(delay)
        flock.update()


if __name__ == "__main__":
    main()
