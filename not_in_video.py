"""
The code that is not used in the video.
"""


from manimlib.imports import *


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


class NotationScene(Scene):  # not used
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
