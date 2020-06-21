"""
TODO:
- check the Quaternion video source code to see how to add a short video to the background with transparency:
    https://www.youtube.com/watch?v=d4EgbgTm0Bg, at 27:44

Ideas:
When going from 2d to 3d, add the 3rd axis as 3Blue1Brown did in the Fourier DE2 video.
Then, also add the tangent plane.
- draw connection between gravity and direction of steepest descent?
"""

from manimlib.imports import *


def my_wiggle(t):
    return wiggle(t, wiggles=1)


def my_smooth(t):
    return smooth(t, inflection=10)


# deleted stuff
class MyParametricSurface:
    def __init__(self, func, **kwargs):
        self.func = func
        self.config = kwargs

    def surface(self):
        return ParametricSurface(
            lambda u, v: np.array([u, v, self.func(u, v)]),
            **self.config,
        )

    def tangent_plane_point(self, x0, y0, x, y):
        z0 = self.func(x0, y0)
        partial_x = (self.func(x0 + EPSILON, y0) - z0) / EPSILON
        partial_y = (self.func(x0, y0 + EPSILON) - z0) / EPSILON
        return z0 + partial_x * (x - x0) + partial_y * (y - y0)

    def tangent_plane(self, x0, y0):
        return ParametricSurface(
            lambda u, v: np.array([u, v, self.tangent_plane_point(x0, y0, u, v)]),
            u_min=x0 - 2, u_max=x0 + 2,
            v_min=y0 - 2, v_max=y0 + 2,
            resolution=10,
        ).set_style(
            fill_color=GREEN,
            fill_opacity=1.0,
            stroke_color=GREEN,
            stroke_width=0.5,
        )


class ThreeDExample(SpecialThreeDScene):
    CONFIG = {
        # 'axes_config': {
        #     'z_axis_config': {
        #         'tick_frequency': 10,
        #         'include_tip': False,
        #     },
        # },
        # 'default_surface_config': {
        #     'fill_opacity': 0.2,
        #     'stroke_color': WHITE,
        # },
        'three_d_axes_config': {
            'axis_config': {
                'unit_size': 1,
            },
            'x_min': -5, 'x_max': 5,
            'y_min': -5, 'y_max': 5,
            'z_min': -3, 'z_max': 4,
        },
        'my_surface': MyParametricSurface(
            lambda u, v: 1 + 0.1 * (u ** 2) + 0.1 * (v ** 2),
            u_min=-4, u_max=4,
            v_min=-4, v_max=4,
            # checkerboard_colors=[BLUE_D, BLUE_E],
            resolution=32,
        ),
        # the bus stop example
        # 'my_surface': ParametricSurface(
        #     lambda u, v: self.get_output_point(u, v),
        #     u_min=-4, u_max=4,
        #     v_min=-4, v_max=4,
        #     # checkerboard_colors=[BLUE_D, BLUE_E],
        #     resolution=32,
        # ),
    }

    # def surface_func(self, u, v):
    #     return total_distance(u, v) * self.scale_factor

    def get_input_dot(self):
        return Dot(self.input_point, color=GREEN).scale(2)

    def get_output_point(self, input_point=None):
        if input_point is None:
            input_point = self.input_point
        u, v, _ = input_point
        return np.array([u, v, self.my_surface.func(u, v)])

    def get_output_dot(self, input_point=None):
        point = self.get_output_point(input_point)
        point[2] -= EPSILON
        # print(point)
        return Dot(point)

    # def get_surface(self):
    #     # paraboloid
    #     a = 0.1
    #     b = 0.1
    #     return ParametricSurface(
    #         lambda u, v: np.array([u, v, 1 + a * (u ** 2) + b * (v ** 2)]),
    #         u_min=-4, u_max=4,
    #         v_min=-4, v_max=4,
    #         # checkerboard_colors=[BLUE_D, BLUE_E],
    #         resolution=32,
    #     )

    def get_tangent_plane(self):
        x, y, _ = self.input_point
        return self.my_surface.tangent_plane(x, y)
        # return ParametricSurface(
        #     lambda u, v: np.array([u, v, tangent_plane_value(x, y, u, v) * self.scale_factor - EPSILON]),
        #     u_min=x - 4, u_max=x + 4,
        #     v_min=y - 4, v_max=y + 4,
        #     resolution=1,
        # ).set_style(
        #     fill_color=GREEN,
        #     fill_opacity=1.0,
        #     stroke_color=WHITE,
        #     stroke_width=0.5,
        # )

    def get_vertical_line(self, input_point=None, color=YELLOW):
        if input_point is None:
            input_point = self.input_point
        return Line(
            input_point,
            self.get_output_point(input_point),
            color=color,
        )

    @staticmethod
    def get_input_point(start, end, t):
        return (1 - t) * start + t * end

    def move_input_point(self, end_point, rate_func, run_time=2):
        start_point = self.input_point
        vt = ValueTracker(0)

        def update_input_dot(dot):
            self.input_point = self.get_input_point(start_point, end_point, vt.get_value())
            dot.move_to(self.input_point)

        def update_output_dot(dot):
            dot.move_to(self.get_output_point())

        def update_vertical_line(line):
            line.become(self.get_vertical_line())

        def update_tangent_plane(plane):
            plane.become(self.get_tangent_plane())

        self.play(
            ApplyMethod(
                vt.set_value, 1,
                rate_func=rate_func,
            ),
            UpdateFromFunc(self.input_dot, update_input_dot),
            UpdateFromFunc(self.output_dot, update_output_dot),
            UpdateFromFunc(self.vertical_line, update_vertical_line),
            UpdateFromFunc(self.tangent_plane, update_tangent_plane),
            run_time=run_time,
        )
        # self.input_point = self.get_input_point(start_point, end_point, 1)

    def setup_axes(self):
        axes = self.get_axes()
        labels = VGroup(*[
            TexMobject(tex).set_color(color)
            for tex, color in zip(
                ["x", "y", "z"],
                [GREEN, RED, BLUE]
            )
        ])
        labels[0].next_to(axes.coords_to_point(self.three_d_axes_config['x_max'], 0, 0), DOWN + IN, SMALL_BUFF)
        labels[1].next_to(axes.coords_to_point(0, self.three_d_axes_config['y_max'], 0), RIGHT, SMALL_BUFF)
        labels[2].next_to(axes.coords_to_point(0, 0, self.three_d_axes_config['z_max'] - 0.5), RIGHT, SMALL_BUFF)
        self.add(axes)
        self.add(labels)
        for label in labels:
            self.add_fixed_orientation_mobjects(label)

    def init(self):
        self.setup_axes()
        self.scale_factor = 1 / 10
        self.input_point = ORIGIN
        self.input_dot = Dot(self.input_point)
        self.output_dot = self.get_output_dot()
        self.vertical_line = self.get_vertical_line()
        self.surface = self.my_surface.surface()
        self.tangent_plane = self.my_surface.tangent_plane(self.input_point[0], self.input_point[1])

    def construct(self):
        self.init()

        # input plane
        input_plane = ParametricSurface(
            lambda x, y: np.array([x, y, 0]),
            u_min=-4, u_max=4,
            v_min=-4, v_max=4,
            resolution=16,
        ).set_style(
            fill_opacity=0.2,
            fill_color=BLUE_B,
            stroke_width=0.5,
            stroke_color=WHITE,
        )

        # camera position uses spherical coordinates
        # default is phi = 70 degrees, theta = -110 degrees

        self.move_camera(**self.get_default_camera_position())
        # self.begin_ambient_camera_rotation(rate=0.05)  # comment out for faster rendering
        self.wait(2)

        self.play(Write(input_plane))
        self.wait(2)

        self.play(Write(self.surface))
        self.wait(2)

        # input stuff
        self.play(ShowCreation(self.input_dot))
        self.play(
            ShowCreation(self.output_dot),
            ShowCreation(self.vertical_line),
        )
        self.play(Write(self.tangent_plane))
        self.wait(1)

        # view from the side to see the curvature of the plane
        self.move_camera(phi=80 * DEGREES)
        self.wait(2)

        # start at a non-optimal bus stop location
        self.move_input_point(
            np.array([-2, -2, 0]),
            rate_func=my_smooth,
        )
        self.wait(2)

        # move to the optimal (?) location
        self.move_input_point(
            np.array([1.9, 1.1, 0]),
            rate_func=linear,
            run_time=8,
        )
        self.wait(5)


class NotationScene(Scene):
    def construct(self):
        function_notation = TexMobject(
            'f : S \\rightarrow \\mathbb{R}}'
        ).scale(1.5)
        min_notation = TexMobject('\\min_{x \\in S} f(x)').scale(1.5)
        VGroup(function_notation, min_notation).arrange(DOWN)
        self.play(
            Write(function_notation),
            FadeInFromDown(min_notation),
        )
        self.wait()

        # transform to the more concise version
        transform_min_notation = TexMobject('\\min f').scale(1.5)
        # transform_min_notation.to_corner(UL)
        self.play(
            Transform(min_notation, transform_min_notation),
            LaggedStart(*map(FadeOutAndShiftDown, function_notation)),
        )
        self.wait()


class MaxRelatedToMin(Scene):
    def construct(self):
        formula = TexMobject(
            '\\max_{x \\in S} f(x)'
            ' = -\\min_{x \\in S} (-f(x))'
        ).scale(2)
        # formula.arrange(DOWN)
        self.play(Write(formula))


# non-deleted scenes
class MovingPointScene(Scene):
    CONFIG = {
        'label_height': 0.2,
        'total_height': 0.35,
        'starting_stop_point': ORIGIN,
        'test_stop_points': [
        ],
        'tmp_wiggle_points': [
        ],
    }

    def get_points(self):
        points = self.points = [
            np.array([x, y, 0])
            for x, y in self.passenger_locations
        ]
        return points

    def get_stop_points(self):
        stop_points = self.stop_points = np.array([
            np.array([x, y, 0])
            for x, y in self.test_stop_points
        ])
        return stop_points

    def get_dots(self):
        dots = VGroup(*[
            Dot(p)
            for p in self.points
        ])
        return dots

    def get_stop_dot(self):
        stop_dot = Dot(self.stop_point)
        stop_dot.set_color(BLUE)
        stop_dot.scale(2)
        return stop_dot

    def get_lines(self, stop_point=None):
        if stop_point is None:
            stop_point = self.stop_point
        lines = VGroup(*[
            Line(stop_point, p)
            for p in self.get_points()
        ])
        lines.set_stroke(width=3)
        lines.set_color(ORANGE)
        return lines

    def get_labels(self, num_decimal_places=3):
        assert hasattr(self, 'lines')
        labels = VGroup()
        if not self.show_labels:
            return labels
        for line in self.lines:
            label = DecimalNumber(
                line.get_length(),
                num_decimal_places=num_decimal_places,
                show_ellipsis=True,
            )
            label.set_height(self.label_height)
            max_width = 0.5 * max(line.get_length(), 0.1)
            if label.get_width() > max_width:
                label.set_width(max_width)
            angle = (line.get_angle() % TAU) - TAU / 2
            if np.abs(angle) > TAU / 4:
                angle += np.sign(angle) * np.pi
            label.angle = angle
            label.next_to(line.get_center(), UP, SMALL_BUFF)
            label.rotate(angle, about_point=line.get_center())
            labels.add(label)
        return labels

    def get_distance_sum(self):
        return sum(
            line.get_length()
            for line in self.lines
        ) / len(self.passenger_locations)

    def get_stop_point(self, start_point, end_point, t):
        return (1 - t) * start_point + t * end_point

    def get_wiggle_points(self):
        wiggle_points = [
            np.array([x, y, 0])
            for x, y in self.tmp_wiggle_points
        ]
        return wiggle_points

    def move_stop_point(self, point, rate_func, run_time=2):
        # XXX: commented out the "recurse" code in the mobject.py file
        # because non-linear asymptotic growth for animation "compile time" is unacceptable
        vt = ValueTracker(0)

        start = self.stop_point

        def update_stop_dot(dot):
            self.stop_point = self.get_stop_point(start, point, vt.get_value())
            dot.move_to(self.stop_point)

        def update_lines(lines):
            lines.become(self.get_lines())
            self.bring_to_front(
                self.stop_dot,
                self.dots,
            )

        def update_labels(labels):
            labels.become(self.get_labels())

        self.play(
            ApplyMethod(
                vt.set_value, 1,
                rate_func=rate_func,
            ),
            UpdateFromFunc(self.stop_dot, update_stop_dot),
            UpdateFromFunc(self.lines, update_lines),
            UpdateFromFunc(self.labels, update_labels),
            run_time=run_time,
        )
        # update the stop_point attribute
        # self.stop_point = self.get_stop_point(start, point, rate_func(1))

    def get_move_stop_point_animations(self, point, rate_func):
        vt = ValueTracker(0)

        start = self.stop_point

        def update_stop_dot(dot):
            self.stop_point = self.get_stop_point(start, point, vt.get_value())
            dot.move_to(self.stop_point)

        def update_lines(lines):
            lines.become(self.get_lines())
            self.bring_to_front(
                self.stop_dot,
                self.dots,
            )

        def update_labels(labels):
            labels.become(self.get_labels())

        anims = [
            ApplyMethod(
                vt.set_value, 1,
                rate_func=rate_func,
            ),
            UpdateFromFunc(self.stop_dot, update_stop_dot),
            UpdateFromFunc(self.lines, update_lines),
            UpdateFromFunc(self.labels, update_labels),
        ]
        return anims

    def init(self):
        self.stop_point = self.starting_stop_point
        self.stop_dot = self.get_stop_dot()
        self.lines = self.get_lines()
        self.points = self.get_points()
        self.stop_points = self.get_stop_points()
        self.labels = self.get_labels()
        self.dots = self.get_dots()
        self.wiggle_points = self.get_wiggle_points()

    def construct(self):
        self.init()

        # grid and title
        grid = NumberPlane()
        grid_title = TextMobject('{\\large City of Cartesia}')
        # grid_title = TextMobject('{\\large Building a Store}')
        grid_title.to_edge(UP)

        # shop
        shop = self.stop_dot.copy()
        shop_label = TextMobject(': your shop').scale(0.8)

        # customer
        customer = self.dots[0].copy()
        customer_label = TextMobject(': customer').scale(0.8)

        # arrangement
        shop_label.to_corner(UR)
        shop.next_to(shop_label, direction=LEFT, buff=SMALL_BUFF)
        customer_label.next_to(shop_label, direction=DOWN, aligned_edge=LEFT)
        customer.move_to(np.array([shop.get_center()[0], customer_label.get_center()[1], 0]))

        # groups
        shop_group = VGroup(shop, shop_label)
        # shop_group.arrange(RIGHT)
        customer_group = VGroup(customer, customer_label)
        # customer_group.arrange(RIGHT)
        # legend_group = VGroup(shop_group, customer_group)
        # legend_group.arrange(DOWN, center=False)
        # legend_group.to_corner(UR)

        # for the DecimalNumber 'total_decimal'
        def update_total_decimal(d):
            # d.next_to(self.stop_dot, DOWN)
            d.set_value(self.get_distance_sum())

        total_decimal = DecimalNumber(
            0,
            num_decimal_places=3,
            show_ellipsis=True,
        )

        # header
        distance_title = TextMobject('\\underline{Average distance}')
        distance_title.to_corner(UL)
        distance_title.set_style(GREEN)

        # arrangement
        total_decimal.next_to(
            distance_title,
            direction=DOWN,
            buff=MED_SMALL_BUFF,
        )
        total_decimal.set_height(self.total_height)
        total_decimal.add_updater(update_total_decimal)

        # minimization text
        minimize_text = TextMobject('Formally, we want to minimize the function')
        # minimize_text.set_color(GREEN)
        expression = TexMobject('f(x, y) = \\dfrac{1}{8} \\sum_{i=1}^{8} \\sqrt{ (x - x_i)^2 + (y - y_i)^2.')
        min_jargon = VGroup(minimize_text, expression)
        min_jargon.arrange(RIGHT)
        min_jargon.scale(0.5)
        min_jargon.to_edge(DOWN, buff=SMALL_BUFF)

        min_text = TextMobject('Minimize this!')
        min_text.next_to(total_decimal, DOWN, buff=1.2)
        arrow = CurvedArrow(
            min_text.get_right() + np.array([0.4, 0.4, 0]),
            total_decimal.get_center() + np.array([1.1, 0, 0])
        )

        # explaining that optimization is just minimization
        equiv_group = VGroup(
            TextMobject('optimizing'),
            TexMobject('\\Updownarrow'),
            TextMobject('minimizing'),
        )
        equiv_group.arrange(DOWN)
        equiv_group.to_edge(RIGHT)

        # maximizing and minimizing are the same problem
        max_text = TextMobject('Maximization and minimization are equivalent problems, since')
        max_tex = TexMobject('\\max f = -\\min (-f).')
        max_min_jargon = VGroup(max_text, max_tex)
        max_min_jargon.arrange(direction=RIGHT)
        max_min_jargon.scale(0.5)
        max_min_jargon.to_edge(DOWN, buff=0.33)

        # all animations
        self.play(
            ShowCreation(grid),
            FadeInFromDown(grid_title),
        )
        self.wait()
        # add all dots and the legend
        self.play(
            ShowCreation(self.stop_dot),
            LaggedStartMap(ShowCreation, self.dots),
            Write(shop_group),
            Write(customer_group),
        )
        # self.wait(1)

        # lines and labels for distances
        self.play(
            LaggedStartMap(ShowCreation, self.lines),
            LaggedStartMap(ShowCreation, self.labels),
        )

        # workaround to bring the dots above the lines; doesn't move anywhere
        self.move_stop_point(self.stop_point, rate_func=my_smooth, run_time=0.1)

        # distance title and number
        self.play(
            FadeIn(distance_title),
            FadeInFrom(total_decimal, UP),
        )
        self.wait(0.5)
        self.play(
            FadeInFromDown(min_jargon),
            Write(min_text),
            ShowCreation(arrow),
            run_time=1.5,
        )

        # self.move_stop_point(np.array([]))

        # testing stop points far from origin, i.e. far from "centre of group"
        # for p in self.stop_points:
        #     self.move_stop_point(p, rate_func=my_smooth)
        #     self.wait(1)

        for p in self.stop_points:
            anims = self.get_move_stop_point_animations(p, my_smooth)
            wait_time = 1
            if p[0] == -1 and p[1] == 0:
                anims.append(Write(equiv_group))
            elif p[0] == 0 and p[1] == 0:
                self.play(
                    FadeOutAndShiftDown(min_jargon),
                    FadeInFromDown(max_min_jargon),
                    run_time=0.8,
                )
                wait_time -= 0.8
            self.wait(wait_time)
            self.play(
                *anims,
                run_time=2,
            )
            # below line: the original way, but no other animations could be played simultaneously
            # self.move_stop_point(p, rate_func=my_smooth)
        self.wait(1)

        # wiggle points
        for p in self.wiggle_points:
            self.move_stop_point(p, rate_func=wiggle, run_time=0.8)


class MovingPointScene1(MovingPointScene):
    CONFIG = {
        'passenger_locations': [
            (-2, 3),
            (-5, 1),
            (-3, -2),
            (4, 3),
            (2, -2),
            (4, -1),
        ],
        'show_labels': True,
        'test_stop_points': [
            (1, 1),
            (-1, 1),
            (0, -2),
            (0, 0),
        ],
        'tmp_wiggle_points': [
            (0.5, 0),
            (-0.5, 0),
            (0, 0.5),
            (0, -0.5),
        ],
    }


passenger_locations = [
    (-2, 3),
    (-5, 1),
    (-4, -2),
    (-4, 3),
    (-2, -3),
    # (3.1, -1.4),
    # (2.8, -0.9),
    # (3.4, -0.4),
    # (3.3, -0.6),
    # (3.5, -1),
    *[
        (2 + 0.3 * i, 1 + 0.3 * j)
        for j in range(-1, 2)
        for i in range(-1, 2)
    ],
    # (4.1, -1.3),
    # (3.8, -0.9),
    # (4.3, -1.4),
]


def total_distance(x0, y0):
    return sum(
        np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        for x, y in passenger_locations
    )


def tangent_plane_value(x0, y0, u, v):
    z0 = total_distance(x0, y0)
    partial_x = 0
    partial_y = 0
    for x, y in passenger_locations:
        tmp = 1 / np.sqrt((x0 - x) ** 2 + (y0 - y) ** 2)
        partial_x += (x0 - x) * tmp
        partial_y += (y0 - y) * tmp
    # print(u, v, z0 + partial_x * (u - x0) + partial_y * (v - y0))
    return z0 + partial_x * (u - x0) + partial_y * (v - y0)


class MovingPointSceneWithCluster(MovingPointScene):
    CONFIG = {
        'passenger_locations': passenger_locations,
        'show_labels': False,
        'test_stop_points': [
            (1, 1),
            (-1, 1),
            (0, -2),
            (2, 1),
        ],
        'tmp_wiggle_points': [
            (2 + 0.5 * i, 1 + 0.5 * j)
            for i, j in [
                (1, 0),
                (0, 1),
                (-1, 0),
                (0, -1),
            ]
        ],
    }


class MovingPointScene2(MovingPointScene):
    CONFIG = {
        'passenger_locations': [
            (2, 1),
            (-2, 1),
            (-2, -1),
            (2, -1),
            (1, 2),
            (-1, 2),
            (-1, -2),
            (1, -2),
        ],
        'show_labels': True,
        'starting_stop_point': ORIGIN,
        'test_stop_points': [
            (0.9999, 1),
            (-1, 0),
            (0, 0),
        ],
    }


class ParabolaExample(GraphScene):
    CONFIG = {
        'x_min': -1,
        'x_max': 13,
        'x_labeled_nums': list(range(1, 11)),
        'y_min': -10,
        'y_max': 100,
        'y_tick_frequency': 10,
        'y_labeled_nums': list(range(20, 120, 20)),
        'y_axis_label': '$y$',
        'sweep_start': 1,
        'sweep_end': 9,
        'parabola_vertex': 5,  # x-coordinate only
    }

    def get_vertical_line_to_function(self, x, func, color=YELLOW):
        return Line(
            self.coords_to_point(x, 0),
            self.coords_to_point(x, func(x)),
            color=color,
        )

    def get_horizontal_line_to_function(self, x, func, color=YELLOW):
        return Line(
            self.coords_to_point(x, func(x)),
            self.coords_to_point(0, func(x)),
            color=color,
        )

    def construct(self):
        # self.set_camera_background('image?')
        self.setup_axes()

        def f_func(x):
            return 90 - (x - 5) ** 2

        # def f_tangent(x0):
        #     def f(x):
        #         return -2 * (x - 5) * (x - x0) + f_func(x0)
        #     return f

        f = self.get_graph(
            f_func,
            color=ORANGE,
            x_min=0,
            x_max=12,
        )
        f_label = TexMobject(
            'y'
            ' = 90 - (x - 5)^2'
        )
        f_label.next_to(f.get_point_from_function(0.6), UR)

        def g_func(x):
            return 100 - f_func(x)

        # def g_tangent(x0):
        #     def f(x):
        #         return 2 * (x - 5) * (x - x0) + g_func(x)
        #     return f

        g = self.get_graph(
            g_func,
            color=BLUE,
            x_min=0,
            x_max=12,
        )
        g_label = TexMobject(
            'y'
            ' = 100 - (90 - (x - 5)^2)'
            # ' = 100 - f(x)'
        )
        g_label.next_to(
            g.get_point_from_function(1.0),
            direction=UP,
            buff=LARGE_BUFF,
        )

        vt = ValueTracker(self.sweep_start)  # x-coordinate
        vertical_line = self.get_vertical_line_to_function(
            vt.get_value(), f_func,
            # line_class=DashedLine,
            color=YELLOW,
        )
        horizontal_line = self.get_horizontal_line_to_function(
            vt.get_value(), f_func,
            color=YELLOW,
        )
        secant_group = self.get_secant_slope_group(vt.get_value(), f, dx=0.001)

        def get_vertical_line_updater(func):
            def updater(line):
                line.become(self.get_vertical_line_to_function(vt.get_value(), func))
            return updater

        def get_horizontal_line_updater(func):
            def updater(line):
                line.become(self.get_horizontal_line_to_function(vt.get_value(), func))
            return updater

        def get_secant_group_updater(graph):
            def updater(s):
                s.become(self.get_secant_slope_group(vt.get_value(), graph, dx=0.001))
            return updater

        # animations
        self.play(FadeIn(self.axes))
        self.wait()
        self.play(
            ShowCreation(f),
            Write(f_label),
        )
        self.wait()
        self.play(
            ShowCreation(vertical_line),
        )
        self.play(
            ShowCreation(horizontal_line),
            ShowCreation(secant_group),
        )
        self.wait()
        self.play(
            ApplyMethod(
                vt.set_value, self.sweep_end,
                rate_func=my_smooth,
            ),
            UpdateFromFunc(vertical_line, get_vertical_line_updater(f_func)),
            UpdateFromFunc(horizontal_line, get_horizontal_line_updater(f_func)),
            UpdateFromFunc(secant_group, get_secant_group_updater(f)),
            run_time=2,
        )
        self.wait()
        vt.set_value(self.sweep_start)
        self.play(
            ShowCreation(g),
            Write(g_label),
            FadeOut(f),
            FadeOutAndShiftDown(f_label),
            Transform(
                vertical_line,
                self.get_vertical_line_to_function(vt.get_value(), g_func)
            ),
            Transform(
                horizontal_line,
                self.get_horizontal_line_to_function(vt.get_value(), g_func)
            ),
            Transform(
                secant_group,
                self.get_secant_slope_group(vt.get_value(), g, dx=0.001)
            ),
        )
        vertical_line.become(self.get_vertical_line_to_function(vt.get_value(), g_func))
        horizontal_line.become(self.get_horizontal_line_to_function(vt.get_value(), g_func))
        secant_group.become(self.get_secant_slope_group(vt.get_value(), g, dx=0.001))
        self.wait(2)
        # self.remove(
        #     vertical_line,
        #     horizontal_line,
        #     secant_group,
        # )
        self.play(
            ApplyMethod(
                vt.set_value, self.sweep_end,
                rate_func=my_smooth,
            ),
            UpdateFromFunc(vertical_line, get_vertical_line_updater(g_func)),
            UpdateFromFunc(horizontal_line, get_horizontal_line_updater(g_func)),
            UpdateFromFunc(secant_group, get_secant_group_updater(g)),
            run_time=2,
        )

        self.wait(2)

        self.play(
            ApplyMethod(
                vt.set_value, self.parabola_vertex,
                rate_func=smooth,
            ),
            UpdateFromFunc(vertical_line, get_vertical_line_updater(g_func)),
            UpdateFromFunc(horizontal_line, get_horizontal_line_updater(g_func)),
            UpdateFromFunc(secant_group, get_secant_group_updater(g)),
            run_time=2,
        )
        self.wait(2)

        self.play(
            ApplyMethod(
                vt.set_value, self.parabola_vertex + 0.3,
                rate_func=my_wiggle,
            ),
            UpdateFromFunc(vertical_line, get_vertical_line_updater(g_func)),
            UpdateFromFunc(horizontal_line, get_horizontal_line_updater(g_func)),
            UpdateFromFunc(secant_group, get_secant_group_updater(g)),
            run_time=2,
        )
        self.play(
            ApplyMethod(
                vt.set_value, self.parabola_vertex - 0.3,
                rate_func=my_wiggle,
            ),
            UpdateFromFunc(vertical_line, get_vertical_line_updater(g_func)),
            UpdateFromFunc(horizontal_line, get_horizontal_line_updater(g_func)),
            UpdateFromFunc(secant_group, get_secant_group_updater(g)),
            run_time=2,
        )

        # f_prime = self.get_derivative_graph(f)
        # self.play(
        #     ShowCreation(f_prime),
        # )
        # self.wait()
        self.wait(2)


class GradientScene(Scene):
    def construct(self):
        # self.set_camera_background()

        SHOW_GRID = False
        X_CUTOFF = -0.8

        if SHOW_GRID:
            grid = NumberPlane()
            self.add(grid)

        line = Line(
            np.array([X_CUTOFF, -4, 0]),
            np.array([X_CUTOFF, 4, 0])
        )
        self.add(line)

        gradient = TextMobject('Gradient')
        descent = TextMobject('Descent')
        title = VGroup(gradient, descent)
        title.arrange()
        # title.scale(1.5)
        # title.to_edge(UP, buff=MED_LARGE_BUFF)
        title.move_to(np.array([-4, 3, 0]))

        description = TextMobject('optimizing by moving')
        downhill = TextMobject('downhill')
        downhill.rotate(-20 * DEGREES)
        downhill.next_to(description)
        downhill.move_to(downhill.get_center() + np.array([0, -0.2, 0]))
        definition = VGroup(description, downhill)
        definition.next_to(title, DOWN, buff=MED_SMALL_BUFF)
        definition.scale(0.6)

        # Calculus stuff
        calculus = TextMobject('\\underline{Calculus}').scale(0.7)
        derivative = TexMobject('\\dfrac{d}{dx} f(x)').scale(0.6)
        integral = TexMobject('\\int f(x) \, dx').scale(0.6)

        derivative.next_to(calculus, direction=DOWN, buff=0.4, aligned_edge=LEFT)
        integral.next_to(derivative, direction=RIGHT, buff=0.2)

        calculus_group = VGroup(calculus, derivative, integral)
        calculus_group.to_edge(LEFT)
        calculus_group.move_to(calculus_group.get_center() + np.array([0, -1, 0]))

        # Linear Algebra stuff
        linear_algebra = TextMobject('\\underline{Linear Algebra}').scale(0.7)
        vector = TexMobject('\\vec{u} = \\begin{bmatrix}1 \\\\ 2\\end{bmatrix}').scale(0.6)
        matrix = TexMobject('\\begin{bmatrix}3 & 1 & 4 \\\\ 1 & 5 & 9 \\\\ 2 & 6 & 5\\end{bmatrix}').scale(0.6)

        linear_algebra.next_to(calculus, buff=1.6)
        vector.next_to(linear_algebra, direction=DOWN, buff=0.3, aligned_edge=LEFT)
        matrix.next_to(vector, direction=RIGHT, buff=0.2)

        linalg_group = VGroup(linear_algebra, vector, matrix)
        # linalg_group.next_to(calculus_group, RIGHT, buff=2)

        # symbols
        nabla = TexMobject('\\nabla f')
        nabla.next_to(definition, DOWN, buff=0.6)
        # next_spot = np.array([-5.68595127, 1.37763566, 0])
        func_args = TexMobject('(x_1, x_2, \\cdots, x_n)')

        # with_args = TexMobject('\\nabla f(x_1, x_2, \\cdots, x_n)')
        # with_args.move_to(nabla)
        nabla2 = TexMobject('\\nabla f')
        nabla_group = VGroup(nabla2, func_args)
        nabla_group.arrange(RIGHT, buff=0.1)
        nabla_group.move_to(nabla)
        # print(nabla2.get_center())

        # left brace
        # brace = TexMobject('\\left\\{')
        # brace.scale(2.5)
        # brace.rotate(90 * DEGREES)
        # # brace.next_to(with_args, DOWN, buff=SMALL_BUFF)
        # brace.move_to(np.array([-3, 1, 0]))
        brace = Brace(func_args, DOWN)

        # caption
        any_point = TextMobject('any point')
        any_point.scale(0.6)
        any_point.next_to(brace, DOWN, buff=SMALL_BUFF)

        # footnote to clarify gradient
        footnote = TextMobject(
            'Technically, the gradient gives us a vector pointing us in the\n'
            'direction of steepest \\textit{ascent}. Therefore, the direction\n'
            'of steepest \\textit{descent} is given by the \\textit{negative} of the gradient.'
        )
        # footnote.to_edge(DOWN, buff=SMALL_BUFF)
        footnote.scale(0.35)
        footnote.arrange(RIGHT, center=False)
        footnote.to_corner(DL, buff=0.2)

        # animations start below
        self.play(Write(title))
        self.wait(0.5)
        self.play(Write(definition))
        self.wait(0.5)

        EM_SCALE = 1.1  # scale factor used to emphasize a word
        EM_SHIFT = np.array([0, 0.2, 0])  # shift to emphasize

        gradient.set_color(YELLOW)
        self.play(
            # ApplyMethod(gradient.scale, EM_SCALE),
            # ApplyMethod(gradient.set_color, YELLOW),
            ApplyMethod(gradient.move_to, gradient.get_center() + EM_SHIFT),
        )
        self.play(Write(nabla))
        self.wait(1)

        # "calculus and linear algebra"
        self.play(
            Write(calculus_group),
            Write(linalg_group),
        )

        # animate in the arguments of the function
        self.play(
            ApplyMethod(
                nabla.move_to, nabla2.get_center()
            ),
            FadeInFromDown(func_args),
        )
        self.wait(1)
        # show brace and label
        self.play(
            Write(brace),
            Write(any_point),
            FadeInFromDown(footnote),
        )
        self.wait(2)

        gradient.set_color(WHITE)
        descent.set_color(YELLOW)
        self.play(
            # ApplyMethod(gradient.scale, 1 / EM_SCALE),
            ApplyMethod(gradient.move_to, gradient.get_center() - EM_SHIFT),
            # ApplyMethod(gradient.set_color, WHITE),

            # ApplyMethod(descent.scale, EM_SCALE),
            ApplyMethod(descent.move_to, descent.get_center() + EM_SHIFT),
            # ApplyMethod(gradient.set_color, YELLOW),
        )
        self.wait(2)
        self.play(
            # ApplyMethod(descent.scale, 1 / EM_SCALE),
            ApplyMethod(descent.move_to, descent.get_center() - EM_SHIFT),
            # ApplyMethod(gradient.set_color, WHITE),
        )
        descent.set_color(WHITE)
        self.wait(1)


class AlphaGoScene(Scene):
    def construct(self):
        pass
