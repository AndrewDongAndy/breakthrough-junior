"""
TODO:
- check the Quaternion video source code to see how to add a short video to the background with transparency:
    https://www.youtube.com/watch?v=d4EgbgTm0Bg, at 27:44
"""

from manimlib.imports import *


def my_wiggle(t):
    return wiggle(t, wiggles=1)


def my_smooth(t):
    return smooth(t, inflection=10)


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


class MovingPointScene(Scene):
    CONFIG = {
        'label_height': 0.25,
        'total_height': 0.35,
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
        self.bring_to_front(self.stop_dot)
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
        )

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
            self.bring_to_front(dot)

        def update_lines(lines):
            lines.become(self.get_lines())

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
        self.stop_point = self.get_stop_point(start, point, rate_func(1))

    def init(self):
        self.stop_point = ORIGIN
        self.stop_dot = self.get_stop_dot()
        self.lines = self.get_lines()
        self.points = self.get_points()
        self.stop_points = self.get_stop_points()
        self.labels = self.get_labels()
        self.dots = self.get_dots()
        self.wiggle_points = self.get_wiggle_points()

    def construct(self):
        grid = NumberPlane()
        grid_title = TextMobject('{\\large Passengers of a Bus}')
        grid_title.to_edge(UP)
        self.add(grid, grid_title)
        self.play(
            FadeInFromDown(grid_title),
            ShowCreation(grid),
        )
        self.wait()

        # this can't be earlier, as it shows the stop_dot being added
        self.init()

        # dot denoting the bus stop
        self.add(self.stop_dot)
        self.play(ShowCreation(self.stop_dot))

        # dots denoting the passengers
        self.add(self.dots)
        self.play(LaggedStartMap(ShowCreation, self.dots))
        self.wait()

        # lines and labels for distances
        self.add(
            self.lines,
            self.labels,
        )
        self.play(
            LaggedStartMap(ShowCreation, self.lines),
            LaggedStartMap(ShowCreation, self.labels),
        )
        self.bring_to_front(
            self.stop_dot,
            self.dots,
        )

        # for the DecimalNumber 'total_distance'
        def update_total_distance(d):
            # d.next_to(self.stop_dot, DOWN)
            d.set_value(self.get_distance_sum())

        total_distance = DecimalNumber(
            0,
            num_decimal_places=3,
            show_ellipsis=True,
        )

        # header
        distance_title = TextMobject('\\underline{Total distance}')
        distance_title.to_corner(UL)
        distance_title.set_style(GREEN)

        # arrangement
        total_distance.next_to(
            distance_title,
            direction=DOWN,
            buff=MED_SMALL_BUFF,
        )
        total_distance.set_height(self.total_height)
        total_distance.add_updater(update_total_distance)

        # distance title and number
        self.add(distance_title, total_distance)
        self.play(
            FadeIn(distance_title),
            FadeInFrom(total_distance, UP),
        )
        self.wait()

        # testing stop points far from origin, i.e. far from "centre of group"
        for p in self.stop_points:
            self.move_stop_point(p, rate_func=my_smooth)
            self.wait()
        self.wait(2)  # pause for 2 seconds

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
            'z_min': -1, 'z_max': 3,
        },
    }

    def surface_func(self, u, v):
        return total_distance(u, v) / 30

    def get_input_dot(self):
        d = Dot(
            self.input_point,
            color=GREEN,
        )
        d.scale(2)

    def get_output_point(self, input_point=None):
        if input_point is None:
            input_point = self.input_point
        return np.array([
            input_point[0],
            input_point[1],
            self.surface_func(input_point[0], input_point[1])
        ])

    def get_output_dot(self, input_point=None):
        point = self.get_output_point(input_point)
        point[2] -= EPSILON
        return Dot(
            point,
            # color=GREEN,
        )

    def get_surface(self):
        return ParametricSurface(
            lambda u, v: self.get_output_point(np.array([u, v, 0])),
            u_min=-4, u_max=4,
            v_min=-4, v_max=4,
            checkerboard_colors=[BLUE_D, BLUE_E],
            resolution=16,
        )

    def get_vertical_line(self, input_point=None, color=YELLOW):
        if input_point is None:
            input_point = self.input_point
        return DashedLine(
            input_point,
            self.get_output_point(input_point),
            color=color,
        )

    def get_input_point(self, start, end, t):
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

        self.play(
            ApplyMethod(
                vt.set_value, 1,
                rate_func=rate_func,
            ),
            UpdateFromFunc(self.input_dot, update_input_dot),
            UpdateFromFunc(self.output_dot, update_output_dot),
            UpdateFromFunc(self.vertical_line, update_vertical_line),
            run_time=2,
        )
        self.input_point = self.get_input_point(start_point, end_point, 1)

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
        labels[2].next_to(axes.coords_to_point(0, 0, self.three_d_axes_config['z_max']), RIGHT, SMALL_BUFF)
        self.add(axes)
        self.add(labels)
        for label in labels:
            self.add_fixed_orientation_mobjects(label)

    def construct(self):
        self.setup_axes()

        # input plane
        input_plane = ParametricSurface(
            lambda x, y: np.array([x, y, 0]),
            u_min=-4, u_max=4,
            v_min=-4, v_max=4,
            resolution=16,
        )
        input_plane.set_style(
            fill_opacity=0.2,
            fill_color=BLUE_B,
            stroke_width=0.5,
            stroke_color=WHITE,
        )

        self.input_point = ORIGIN
        self.input_dot = Dot(self.input_point)
        self.output_dot = self.get_output_dot()
        self.vertical_line = self.get_vertical_line()
        surface = self.get_surface()

        # spherical coordinates
        # default is phi = 70 degrees, theta = -110 degrees

        self.move_camera(**self.get_default_camera_position())
        # self.begin_ambient_camera_rotation(rate=0.05)
        self.wait(2)

        self.play(Write(input_plane))
        self.wait(2)

        self.play(Write(surface))
        self.wait(2)

        self.play(ShowCreation(self.input_dot))
        self.play(
            ShowCreation(self.output_dot),
            ShowCreation(self.vertical_line),
        )

        self.move_camera(phi=80 * DEGREES)
        self.wait(2)

        self.move_input_point(
            np.array([-2, -2, 0]),
            rate_func=my_smooth,
        )
        self.wait(2)
        self.move_input_point(
            np.array([2, 1, 0]),
            rate_func=linear,
            run_time=8,
        )
        self.wait(5)
