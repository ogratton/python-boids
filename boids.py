# Copyright (c) 2012 Tom Marble
# Licensed under the MIT license http://opensource.org/licenses/MIT
# https://github.com/tmarble/pyboids/blob/master/boids.py

"""Implements the Peter Keller boids program
which is an adaptation of
http://www.cs.toronto.edu/~dt/siggraph97-course/cwr87/
"""

import random
import pygame
import pygame.locals as pyg
import math

from OpenGL import GL, GLU


class CustomVector3:
    """
    numpy arrays aren't optimised for small lengths.

    This tries to replicate the bare pygame Vector3 behaviour we need
    """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def length(self):
        return math.sqrt(sum(map(lambda x: x**2, self)))

    def normalize(self):
        length = self.length()
        return CustomVector3(self.x/length, self.y/length, self.z/length)

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __repr__(self):
        return "<Vector3(%s, %s, %s)>" % (self.x, self.y, self.z)


try:
    from pygame.math import Vector3
except ImportError:
    Vector3 = CustomVector3


def random_vector3(lower=0.0, upper=1.0):
    return Vector3(
        random.uniform(lower, upper),
        random.uniform(lower, upper),
        random.uniform(lower, upper),
    )


class Rule:
    """
    Template for rules of flocking
    """

    NEIGHBOURHOOD = 5  # max distance at which rule is applied

    def __init__(self):
        self.change = Vector3(0, 0, 0)  # velocity correction
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
        if other != boid:
            self.change = self.change + other.location
            self.num += 1

    def add_adjustment(self, boid):
        if self.num > 0:
            centroid = self.change / self.num
            desired = centroid - boid.location
            self.change = (desired - boid.velocity) * 0.0006
        boid.adjustment = boid.adjustment + self.change


class Alignment(Rule):
    """
    Rule 2: Boids try to match velocity with near boids.
    """

    NEIGHBOURHOOD = 10

    def accumulate(self, boid, other, distance):
        if other != boid:
            self.change = self.change + other.velocity
            self.num += 1

    def add_adjustment(self, boid):
        if self.num > 0:
            group_velocity = self.change / self.num
            self.change = (group_velocity - boid.velocity) * 0.03
        boid.adjustment = boid.adjustment + self.change


class Separation(Rule):
    """
    Rule 3: Boids try to keep a small distance away from other objects (including other boids).
    """

    def accumulate(self, boid, other, distance):
        if other != boid:
            separation = boid.location - other.location
            if separation.length() > 0:
                self.change = self.change + (separation.normalize() / distance)
            self.num += 1

    def add_adjustment(self, boid):
        if self.change.length() > 0:
            group_separation = self.change / self.num
            self.change = (group_separation - boid.velocity) * 0.01
        boid.adjustment = boid.adjustment + self.change


class Boid:
    def __init__(self):
        self.color = random_vector3(0.5)  # R G B
        self.location = Vector3(0, 0, 0)  # x y z
        self.velocity = random_vector3(-1.0, 1.0)  # vx vy vz
        self.adjustment = Vector3(0, 0, 0)  # to accumulate corrections

    def __repr__(self):
        return "color %s, location %s, velocity %s" % (
            self.color,
            self.location,
            self.velocity,
        )

    def wrap(self, cube0, cube1):
        """
        implement hypertaurus
        """
        loc = self.location
        if loc.x < cube0.x:
            loc.x = loc.x + (cube1.x - cube0.x)
        elif loc.x > cube1.x:
            loc.x = loc.x - (cube1.x - cube0.x)
        if loc.y < cube0.y:
            loc.y = loc.y + (cube1.y - cube0.y)
        elif loc.y > cube1.y:
            loc.y = loc.y - (cube1.y - cube0.y)
        if loc.z < cube0.z:
            loc.z = loc.z + (cube1.z - cube0.z)
        elif loc.z > cube1.z:
            loc.z = loc.z - (cube1.z - cube0.x)
        self.location = loc

    def orient(self, others):
        """
        Calculate new position
        """
        rules = [Cohesion(), Alignment(), Separation()]
        for other in others:  # accumulate corrections
            distance = self.location.distance_to(other.location)
            for rule in rules:
                if distance < rule.NEIGHBOURHOOD:
                    rule.accumulate(self, other, distance)
        self.adjustment = [0.0, 0.0, 0.0]  # reset adjustment vector
        for rule in rules:  # save corrections to the adjustment
            rule.add_adjustment(self)

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
            self.velocity += (self.velocity.normalize() * random.uniform(0.0, 0.007))
        self.limit_speed(1.0)
        self.location += self.velocity


class Flock:
    """
    A flock of boids
    """

    def __init__(self, num_boids, cube0, cube1):
        self.cube0 = cube0  # cube min vertex
        self.cube1 = cube1  # cube max vertex
        self.boids = []
        for _ in range(num_boids):
            self.boids.append(Boid())

    def update(self):
        """
        update flock positions
        """
        for boid in self.boids:
            boid.orient(self.boids)  # calculate new velocity
        for boid in self.boids:
            boid.update()  # move to new position
            boid.wrap(self.cube0, self.cube1)

    def __repr__(self):
        rep = "Flock of %d boids bounded by %s, %s:\n" % (
            len(self.boids),
            self.cube0,
            self.cube1,
        )
        for i, b in enumerate(self.boids):
            rep += "%3d: %s\n" % (i, b)
        return rep

    def top_square(self):
        """
        loop of points defining top square
        """
        return [
            [self.cube0.x, self.cube0.y, self.cube1.z],
            [self.cube0.x, self.cube1.y, self.cube1.z],
            [self.cube1.x, self.cube1.y, self.cube1.z],
            [self.cube1.x, self.cube0.y, self.cube1.z],
        ]

    def bottom_square(self):
        """
        loop of points defining bottom square
        """
        return [
            [self.cube0.x, self.cube0.y, self.cube0.z],
            [self.cube0.x, self.cube1.y, self.cube0.z],
            [self.cube1.x, self.cube1.y, self.cube0.z],
            [self.cube1.x, self.cube0.y, self.cube0.z],
        ]

    def vertical_lines(self):
        """
        point pairs defining vertical lines of the cube
        """
        return [
            [self.cube0.x, self.cube0.y, self.cube0.z],
            [self.cube0.x, self.cube0.y, self.cube1.z],
            [self.cube0.x, self.cube1.y, self.cube0.z],
            [self.cube0.x, self.cube1.y, self.cube1.z],
            [self.cube1.x, self.cube1.y, self.cube0.z],
            [self.cube1.x, self.cube1.y, self.cube1.z],
            [self.cube1.x, self.cube0.y, self.cube0.z],
            [self.cube1.x, self.cube0.y, self.cube1.z],
        ]

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
                # head = boid.location + boid.velocity.normalize()
                head = boid.location + boid.velocity.normalize() * 3
            else:
                head = boid.location
            GL.glVertex(head.x, head.y, head.z)
        GL.glEnd()

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


def main():
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
    edge = 50
    GLU.gluPerspective(60.0, xyratio, 1.0, (6 * edge) + 10)  # setup lens
    GL.glMatrixMode(GL.GL_MODELVIEW)
    GL.glLoadIdentity()
    # GLU.gluLookAt(0.0, 0.0, 150, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    GLU.gluLookAt(0.0, 0.0, 3 * edge, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    GL.glPointSize(3.0)
    cube0 = Vector3(-edge, -edge, -edge)  # cube min vertex
    cube1 = Vector3(+edge, +edge, +edge)  # cube max vertex
    flock = Flock(200, cube0, cube1)
    # print(flock)
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
