# TODO fish shoal that avoids predators
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

COHESION_RANGE = 12
ALIGNMENT_RANGE = 6
SEPARATION_RANGE = 4

BOUND_RANGE_RATIO = 0.8  # Point at which boids start turning round
FREE_WILL_CHANCE = 0.0002  # chance to become a trend-setter

COHESION_WEIGHT = 0.001
ALIGNMENT_WEIGHT = 0.04
SEPARATION_WEIGHT = 0.05

BOUND_WEIGHT = 0.001
FREE_WILL_WEIGHT = 1
ATTRACTION_WEIGHT = 0.0008

RADIUS = 25  # bad name, but half of cube edge length
UPDATE_INTERVAL = 0.033
NUM_BOIDS = 150
BOID_BASE_SPEED = 0.03
BOID_MAX_SPEED = 0.6
BOID_PREDATOR_ALARM_RADIUS = RADIUS * 0.4
BOID_PREDATOR_AVOIDANCE = 0.009

NUM_PREDATORS = 4
PREDATOR_BASE_SPEED = 0.02
PREDATOR_MAX_SPEED = 0.5
PREDATOR_PATH_WEIGHT = 1

BOID_RENDER_LENGTH = 1
PREDATOR_RENDER_SIZE = 10


class CustomVector3:
    """
    numpy arrays aren't optimised for small lengths.
    This tries to replicate the minimal pygame Vector3 behaviour we need.
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
        if isinstance(other, (int, float)):
            return self._naive_op([other] * 3, lambda v: v[0] * v[1])
        raise NotImplementedError("too much maths")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
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


def rand_point_in_cube(radius, min_vertex):
    rand_unit_point = rand_vector3()
    scaled_point = rand_unit_point * radius * 2
    return scaled_point + min_vertex


def zero_vector3():
    return Vector3(0, 0, 0)


class SocialRule:
    """
    Template for rules of flocking
    """

    # Apply rule only to neighbours within this distance
    RANGE = NotImplemented

    def __init__(self):
        self.change = zero_vector3()  # velocity correction
        self.num = 0  # number of participants

    def accumulate(self, boid, other, distance):
        """
        Save any corrections based on other boid to self.change
        """
        raise NotImplementedError

    def apply(self, boid):
        """
        Add the accumulated self.change to boid.adjustment
        """
        raise NotImplementedError


class Cohesion(SocialRule):
    """
    Rule 1: Boids try to fly towards the centre of mass of neighbouring boids.
    """

    RANGE = COHESION_RANGE

    def accumulate(self, boid, other, distance):
        self.change += other.location
        self.num += 1

    def apply(self, boid):
        if self.num > 0:
            centroid = self.change / self.num
            desired = centroid - boid.location
            self.change = (desired - boid.velocity) * COHESION_WEIGHT
        boid.adjustment += self.change


class Alignment(SocialRule):
    """
    Rule 2: Boids try to match velocity with near boids.
    """

    RANGE = ALIGNMENT_RANGE

    def accumulate(self, boid, other, distance):
        self.change += other.velocity
        self.num += 1

    def apply(self, boid):
        if self.num > 0:
            group_velocity = self.change / self.num
            self.change = (group_velocity - boid.velocity) * ALIGNMENT_WEIGHT
        boid.adjustment += self.change


class Separation(SocialRule):
    """
    Rule 3: Boids try to keep a small distance away from other objects (including other boids).
    """

    RANGE = SEPARATION_RANGE

    def accumulate(self, boid, other, distance):
        separation = boid.location - other.location
        if distance > 0:
            self.change += separation.normalize() / distance
        self.num += 1

    def apply(self, boid):
        if self.change.length() > 0:
            group_separation = self.change / self.num
            self.change = (group_separation - boid.velocity) * SEPARATION_WEIGHT
        boid.adjustment += self.change


class SocialBehaviour:
    """
    Apply rules that relate to other entities
    """

    def __init__(self, rule_classes=None, avoid_predators=False):
        self.rule_classes = rule_classes or []
        self.avoid_predators = avoid_predators

    def apply(self, boid, all_boids, predators):
        rules = [r() for r in self.rule_classes]

        for other in all_boids:
            if boid.id == other.id:
                continue

            distance = boid.location.distance_to(other.location)

            for rule in rules:
                if distance < rule.RANGE:
                    rule.accumulate(boid, other, distance)

        boid.adjustment = zero_vector3()  # reset adjustment vector

        if self.avoid_predators:
            for pred in predators:
                distance = boid.location.distance_to(pred.location)
                if distance < BOID_PREDATOR_ALARM_RADIUS:
                    separation = boid.location - pred.location
                    change = separation * BOID_PREDATOR_AVOIDANCE
                    boid.adjustment += change

        for rule in rules:
            rule.apply(boid)


class IndividualRule:
    """
    Template for individual boid rules
    """

    def __init__(self, cube_min, cube_max):
        self.cube_min = cube_min
        self.cube_max = cube_max

    def apply(self, boid):
        raise NotImplementedError


class Wrap(IndividualRule):
    """
    If a boid manages to escape, teleport it to the opposite side like pac-man
    """

    def apply(self, boid):
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


class Bound(IndividualRule):
    """
    Start turning towards the centre when a boid approaches the edge
    """

    @staticmethod
    def _is_near_wall(boid):
        # TODO assumes centre is 0,0,0
        return any(
            [abs(coord) >= RADIUS * BOUND_RANGE_RATIO for coord in boid.location]
        )

    def apply(self, boid):
        change = zero_vector3()
        if self._is_near_wall(boid):
            # TODO assumes centre is 0,0,0
            direction = zero_vector3() - boid.location
            change = direction * BOUND_WEIGHT
        boid.adjustment += change


class FreeWill(IndividualRule):
    """
    Small chance to break the mold
    """

    def apply(self, boid):
        if random.random() < FREE_WILL_CHANCE:
            rand = rand_vector3(lower=-1)
            boid.adjustment += rand * FREE_WILL_WEIGHT


class PathFollow(IndividualRule):
    """
    Follow a path
    """

    def apply(self, boid):  # TODO rename to entity
        # TODO speed should be the same relative to all radii
        t = (boid.tick / 100) % 10  # randomise for random speed

        x = math.sin(math.pi * t * 0.4)
        y = math.cos(math.pi * t * 0.2)
        z = math.cos(math.pi * t)

        new_pos = (Vector3(x, y, z) * RADIUS)
        boid.location = new_pos
        # boid.adjustment += (new_pos - boid.location) * PREDATOR_PATH_WEIGHT


class Attraction(IndividualRule):
    """
    A constant deep-seated longing for home

    (Mild attraction to the centre)
    TODO make it some customisable point
    """

    def apply(self, boid):
        # TODO assumes centre is 0,0,0
        direction = zero_vector3() - boid.location
        change = direction * ATTRACTION_WEIGHT
        boid.adjustment += change


class IndividualBehaviour:
    """
    Apply rules that do not depend on other entities
    """

    def __init__(self, cube_min, cube_max, rule_classes=None):
        self.cube_min = cube_min
        self.cube_max = cube_max
        self.rule_classes = rule_classes or []

    def apply(self, boid):
        rules = [r(self.cube_min, self.cube_max) for r in self.rule_classes]

        for rule in rules:
            rule.apply(boid)


class Boid:
    BASE_SPEED = BOID_BASE_SPEED
    MAX_SPEED = BOID_MAX_SPEED

    def __init__(
        self,
        b_id,
        social_behaviour: SocialBehaviour,
        individual_behaviour: IndividualBehaviour,
        starting_pos=None,
    ):
        self.id = b_id
        self.social_behaviour = social_behaviour
        self.individual_behaviour = individual_behaviour
        self.color = self.decide_colour()
        self.location = starting_pos or zero_vector3()
        self.velocity = rand_vector3(-1.0, 1.0)
        self.adjustment = zero_vector3()  # to accumulate corrections

        self.tick = random.randint(0, 1000)  # TODO

    def __repr__(self):
        return "%s %s: color %s, location %s, velocity %s" % (
            self.__class__.__name__,
            self.id,
            self.color,
            self.location,
            self.velocity,
        )

    @staticmethod
    def decide_colour():
        return rand_vector3(0.3, 0.7)

    def apply_behaviours(self, boids, predators):
        """
        Calculate new position
        """
        self.social_behaviour.apply(self, boids, predators)
        self.individual_behaviour.apply(self)

    def update(self):
        """
        Move to new position
        """
        self.velocity += self.adjustment
        # Hack: Add a constant velocity in whatever direction
        # they are moving so they don't ever stop.
        if self.velocity.length() > 0:
            self.velocity += self.velocity.normalize() * random.uniform(
                0.0, self.BASE_SPEED
            )
        self.limit_speed()
        self.location += self.velocity

        self.tick += 1

    def limit_speed(self):
        """
        Ensure the speed does not exceed max_speed
        """
        if self.velocity.length() > self.MAX_SPEED:
            self.velocity = self.velocity.normalize() * self.MAX_SPEED


class Predator(Boid):
    BASE_SPEED = PREDATOR_BASE_SPEED
    MAX_SPEED = PREDATOR_MAX_SPEED

    @staticmethod
    def decide_colour():
        return Vector3(0.4, 0, 0)


class Flock:
    """
    A flock of boids
    """

    def __init__(
        self,
        cube_min,
        cube_max,
        num_boids,
        boid_social_behaviour: SocialBehaviour,
        boid_individual_behaviour: IndividualBehaviour,
        num_predators,
        predator_social_behaviour: SocialBehaviour,
        predator_individual_behaviour: IndividualBehaviour,
    ):
        self.cube_min = cube_min
        self.cube_max = cube_max
        self.boids = []
        for i in range(num_boids):
            starting_pos = rand_point_in_cube(RADIUS, cube_min)
            # starting_pos = None
            self.boids.append(
                Boid(
                    i,
                    boid_social_behaviour,
                    boid_individual_behaviour,
                    starting_pos=starting_pos,
                )
            )

        self.predators = []
        for i in range(num_predators):
            # starting_pos = rand_point_in_cube(RADIUS, cube_min)
            starting_pos = None
            self.predators.append(
                Predator(
                    i,
                    predator_social_behaviour,
                    predator_individual_behaviour,
                    starting_pos=starting_pos,
                )
            )

    def update(self):
        """
        update flock positions
        """
        t_0 = time.time()

        for entities in (self.boids, self.predators):
            for entity in entities:
                entity.apply_behaviours(
                    self.boids, self.predators
                )  # calculate new velocity
            for entity in entities:
                entity.update()  # move to new position

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
                head = boid.location + boid.velocity.normalize() * BOID_RENDER_LENGTH
            else:
                head = boid.location
            GL.glVertex(head.x, head.y, head.z)
        GL.glEnd()

    def render_predators(self):
        for pred in self.flock.predators:
            self.render_point(pred.color, pred.location)

    @staticmethod
    def render_point(colour, location):
        """ unused so far """
        GL.glEnable(GL.GL_POINT_SMOOTH)
        GL.glPointSize(PREDATOR_RENDER_SIZE)

        GL.glBegin(GL.GL_POINTS)
        GL.glColor3d(*colour)
        GL.glVertex3d(*location)
        GL.glEnd()

    def render(self):
        self.render_boundary()
        self.render_boids()
        self.render_predators()


def main():

    # random.seed(1)

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
    GLU.gluPerspective(60.0, xyratio, 1.0, (6 * RADIUS) + 10)  # setup lens
    GL.glMatrixMode(GL.GL_MODELVIEW)
    GL.glLoadIdentity()
    GLU.gluLookAt(0.0, 0.0, 3 * RADIUS, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    GL.glPointSize(3.0)
    cube_min_vertex = Vector3(-RADIUS, -RADIUS, -RADIUS)
    cube_max_vertex = Vector3(+RADIUS, +RADIUS, +RADIUS)
    boid_social_behaviour = SocialBehaviour(
        [
            Cohesion,
            Alignment,
            Separation,
        ],
        avoid_predators=True,
    )
    boid_individual_behaviour = IndividualBehaviour(
        cube_min_vertex,
        cube_max_vertex,
        [
            Bound,
            Wrap,
            FreeWill,
            Attraction,
        ],
    )
    predator_social_behaviour = SocialBehaviour()
    predator_individual_behaviour = IndividualBehaviour(
        cube_min_vertex,
        cube_max_vertex,
        [
            # Bound,
            # Wrap,
            PathFollow,
        ],
    )
    flock = Flock(
        cube_min_vertex,
        cube_max_vertex,
        NUM_BOIDS,
        boid_social_behaviour,
        boid_individual_behaviour,
        NUM_PREDATORS,
        predator_social_behaviour,
        predator_individual_behaviour,
    )
    # TODO make template boid/predator instead of passing all behaviours

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
