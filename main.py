"""
HACK: to run any MovingPoint scenes, comment out lines 990 and 991 in mobject/mobject.py to speed up rendering
and avoid O(n^2) behaviour. However, using Transform requires uncommenting those back, e.g. in the explanation scenes
near the bottom of this file.

Another "hack", but less so. The original value of "open_angle" in the Laptop class of mobject/svg/drawings.py is
np.pi / 4, but I changed it to 70 * DEGREES as the default.

TODO:
- check the Quaternion video source code to see how to add a short video to the background with transparency:
    https://www.youtube.com/watch?v=d4EgbgTm0Bg, at 27:44

Ideas:
When going from 2d to 3d, add the 3rd axis as 3Blue1Brown did in the Fourier DE2 video.
Then, also add the tangent plane.
- draw connection between gravity and direction of steepest descent?

June 22, 2020:
Today, I learned (TIL) that changing the underlying library can be detrimental, because the "hack" at the top of this
comment block is the very reason Transform() animations weren't working.
"""

from manimlib.imports import *


def my_wiggle(t):
    return wiggle(t, wiggles=1)


def my_smooth(t):
    return smooth(t, inflection=10)


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

        # minimization text; TODO: use \\text{stuff} instead of VGroup?
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
        # TODO: use \text{stuff} instead of VGroup.arrange?
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


FOOTNOTE_SCALE = 0.35


class GradientScene(Scene):
    def construct(self):
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

        # divider
        divider = Line(np.array([-3.94, -0.5, 0]), np.array([-3.94, -2.7, 0]))
        divider.set_stroke(width=1)

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

        # annotations

        # "the gradient of f"
        # TODOne: use \text{stuff} instead of VGroup().arrange()?
        # gradient_text = TextMobject('``gradient of')
        # f = TexMobject('f')
        # right_quote = TextMobject("''")
        # term_group = VGroup(gradient_text, f, right_quote)
        # term_group.arrange(RIGHT, aligned_edge=UP, buff=SMALL_BUFF)
        term_group = TexMobject("\\text{``gradient of } f \\text{''}")
        term_group.scale(0.5)
        term_group.next_to(nabla, UP, buff=SMALL_BUFF)

        # left brace
        brace = Brace(func_args, DOWN)
        # any point
        any_point = TextMobject('any point')
        any_point.scale(0.5)
        any_point.next_to(brace, DOWN, buff=SMALL_BUFF)

        # footnote to clarify gradient
        footnote = TextMobject(
            'Technically, the gradient gives us a vector pointing us in the '
            'direction of steepest \\textit{ascent}. Therefore, the direction '
            'of steepest \\textit{descent} is given by the \\textit{negative} of the gradient.'
        )
        # footnote.to_edge(DOWN, buff=SMALL_BUFF)
        footnote.scale(FOOTNOTE_SCALE)
        footnote.arrange(RIGHT, center=False)
        footnote.to_corner(DL, buff=0.2)

        # animations start below
        self.wait(0.3)
        self.play(Write(definition))
        self.wait(1.4)
        self.play(FadeInFromLarge(title))
        self.wait(1)

        EM_SCALE = 1.1  # scale factor used to emphasize a word
        EM_SHIFT = 0.1  # shift to emphasize

        gradient.set_color(YELLOW)
        self.play(
            LaggedStart(
                AnimationGroup(
                    ApplyMethod(gradient.set_y, gradient.get_y() + EM_SHIFT),
                    Write(nabla),
                    Write(term_group),
                ),
                AnimationGroup(
                    FadeInFrom(calculus_group, LEFT),
                    FadeInFrom(linalg_group, RIGHT),
                    FadeIn(divider),
                ),
                lag_ratio=0.3,
            ),
        )
        self.wait(3)

        # animate in the arguments of the function
        self.play(
            ApplyMethod(
                nabla.move_to, nabla2.get_center()
            ),
            MaintainPositionRelativeTo(term_group, nabla),
            FadeInFromDown(func_args),
            Write(brace),
            Write(any_point),
        )
        # show brace and label
        self.play(
            FadeInFromDown(footnote),
        )

        # emphasize the word "descent"
        descent.set_color(YELLOW)
        self.play(
            # ApplyMethod(gradient.scale, 1 / EM_SCALE),
            ApplyMethod(gradient.set_y, gradient.get_y() - EM_SHIFT),
            # ApplyMethod(gradient.set_color, WHITE),

            # ApplyMethod(descent.scale, EM_SCALE),
            ApplyMethod(descent.set_y, descent.get_y() + EM_SHIFT),
            # ApplyMethod(gradient.set_color, YELLOW),
        )
        gradient.set_color(WHITE)
        self.wait(1.5)

        # unemphasize the word "descent"
        self.play(
            # ApplyMethod(descent.scale, 1 / EM_SCALE),
            ApplyMethod(descent.set_y, descent.get_y() - EM_SHIFT),
            # ApplyMethod(gradient.set_color, WHITE),
        )
        descent.set_color(WHITE)
        self.wait(1)


class AlphaGoScene(Scene):
    def construct(self):
        # self.set_camera_background()

        # for debugging purposes; not in final video
        SHOW_GRID = False
        X_CUTOFF = +0.8
        X_CENTRE = 4

        if SHOW_GRID:
            grid = NumberPlane()
            self.add(grid)

        line = Line(
            np.array([X_CUTOFF, -4, 0]),
            np.array([X_CUTOFF, 4, 0])
        )
        self.add(line)

        # AlphaGo logo .png file
        logo = ImageMobject('AlphaGo_Logo.png')
        logo.scale(0.6)
        logo.move_to(np.array([X_CENTRE, 3, 0]))
        self.add(logo)

        how = TextMobject('How!?')
        how.next_to(logo, DOWN, LARGE_BUFF)
        good_strategy = TextMobject('A good strategy.')
        good_strategy.next_to(how, DOWN)

        header = TextMobject('\\underline{Strategy for playing Go}')
        header.scale(0.75)
        header.next_to(logo, DOWN, buff=MED_LARGE_BUFF)

        # disclaimer
        # TODO: verify/edit this
        # talk about how AlphaGo uses this idea to direct its computational power
        # "reinforcement learning"
        disclaimer = TextMobject(
            'Of course, this is not the exact way AlphaGo learns; the state-of-the-art algorithms used by '
            'AlphaGo are much more intricate. Nonetheless, some degree of quantifiability is '
            'certainly required. AlphaGo uses a technique known as \\textit{reinforcement learning}.'
        )
        disclaimer.scale(FOOTNOTE_SCALE)
        disclaimer.to_edge(DOWN, buff=0.2)
        disclaimer.set_x(X_CENTRE)

        # strategy
        strategy_words = TextMobject('DEFEND TERRITORY AT ALL COSTS')
        # TODO: use \text{stuff} instead of VGroup().arrange()?
        arrow = TexMobject('\downarrow')
        convert = TextMobject('encoded as').scale(0.9)
        arrow_group = VGroup(arrow, convert).arrange(RIGHT, buff=MED_SMALL_BUFF)
        strategy_code = TexMobject('(0.3, -0.5, -0.8, 0.9, 0.7)')
        strategy_group = VGroup(
            strategy_words,
            arrow_group,
            strategy_code,
        )
        strategy_group.arrange(DOWN)
        strategy_group.scale(0.6)
        strategy_group.next_to(header, DOWN, MED_SMALL_BUFF)
        strategy_code_general = TexMobject('(x_1, x_2, x_3, \\cdots, x_n)').scale(0.6)
        strategy_code_general.move_to(strategy_code)

        # arrow = CurvedArrow()
        optimize = TextMobject('optimize these numbers')

        # spaces inside the braces are required here
        where_n_is_large = TexMobject('\\text{where } n \\text{ is {\LARGE large}}').scale(0.6)
        where_n_is_large.next_to(strategy_code_general, RIGHT)

        # for positioning purposes
        strategy_code_general2 = strategy_code_general.copy()
        # where_n_is_large2 = where_n_is_large
        VGroup(strategy_code_general2, where_n_is_large).arrange(RIGHT).next_to(arrow_group, DOWN)

        # show that this means the input is n-dimensional
        arg_brace = Brace(strategy_code_general2, direction=DOWN, buff=SMALL_BUFF)
        dim_text = TexMobject('n \\text{-dimensional space}')
        dim_text.scale(0.4)
        dim_text.next_to(arg_brace, DOWN, buff=SMALL_BUFF)
        strat = TextMobject('of strategies')
        strat.scale(0.4)
        strat.next_to(dim_text, DOWN, buff=SMALL_BUFF)
        brace_group = VGroup(arg_brace, dim_text, strat)

        footnote = TextMobject(
            "Here, ``strategy'' refers to the procedure, the set of instructions, that is followed to "
            "determine how a task is gone about being achieved."
        )
        footnote.scale(FOOTNOTE_SCALE)
        footnote.to_edge(DOWN, buff=0.2)
        footnote.set_x(X_CENTRE)

        battle = TextMobject('Which strategy is better?').scale(0.75)
        blue = Laptop(width=0.5, body_color=BLUE)
        vs = TextMobject('vs.').scale(0.5)
        orange = Laptop(width=0.5, body_color=ORANGE)
        comps = VGroup(blue, vs, orange).arrange(RIGHT)
        battle.next_to(comps, UP)
        comp_group = VGroup(battle, blue, vs, orange)
        comp_group.next_to(disclaimer, UP, buff=0.5)

        # after transform
        centred_comp_group = comp_group.copy().move_to(ORIGIN).scale(2)
        # blue2 = blue.copy().move_to(np.array([-3, 3, 0]))
        # orange2 = orange.copy().move_to(np.array([-3, 3, 0]))

        # table showing results
        table_lines = VGroup(
            Line(np.array([-4, -3.5, 0]), np.array([-4, 3.5, 0])),      # left vertical line
            Line(np.array([-2.5, -3.5, 0]), np.array([-2.5, 3.5, 0])),  # right vertical line
            Line(np.array([-6, 2.5, 0]), np.array([-1, 2.5, 0])),       # horizontal line below headers
            Line(np.array([-6, -3, 0]), np.array([-1, -3, 0])),         # horizontal line above total
        )

        # constants
        IMAGE_Y = 3
        # ROW1_Y = -0.5
        # ROW2_Y = -2.5
        COL1_X = (-4 + -2.5) / 2
        COL2_X = (-2.5 + -1) / 2
        BOTTOM_Y = -3.25

        opponent = TextMobject('Opposing\\\\Strategy')
        opponent.scale(0.55).move_to(np.array([-5, 3, 0]))
        blue2 = blue.copy().move_to(np.array([COL1_X, IMAGE_Y, 0]))
        orange2 = orange.copy().move_to(np.array([COL2_X, IMAGE_Y, 0]))
        score = TextMobject('\\textbf{Score}')
        score.scale(0.6).move_to(np.array([-5, BOTTOM_Y, 0]))

        blue3 = blue.copy().move_to(np.array([1.25, 0, 0])).scale(3)
        better = TextMobject('is better than')
        # better = TexMobject('>').move_to(np.array([3, 0, 0]))
        orange3 = orange.copy().move_to(np.array([4.75, 0, 0])).scale(3)
        VGroup(blue3, better, orange3).arrange(DOWN, buff=0.9).set_x(3.5)

        # start of animations
        self.play(
            FadeIn(logo),
            Write(how),
        )
        self.wait(1.8)
        self.play(
            Write(good_strategy),
            FadeInFromDown(footnote),
        )
        self.wait(1)

        # emphasize the word "how" again
        end_height = how.get_height() * 1.2
        end_y = how.get_y() + 0.2
        how.set_color(YELLOW)
        self.play(
            # ApplyMethod(how.set_height, end_height),  # this isn't working
            ApplyMethod(how.set_y, end_y),
        )
        self.play(
            how.set_height, end_height,
            rate_func=linear,
            run_time=2,
        )

        # start showing the strategy
        self.play(
            ReplacementTransform(good_strategy, header),
            FadeOutAndShiftDown(how),
        )
        self.play(ShowCreation(strategy_words))
        self.play(
            FadeInFrom(arrow_group, UP),
            ReplacementTransform(strategy_words.copy(), strategy_code),
            FadeOutAndShiftDown(footnote),
            FadeInFromDown(disclaimer),
        )
        self.wait(1)
        self.play(ReplacementTransform(strategy_code, strategy_code_general))
        self.wait(1)
        self.play(
            LaggedStart(
                ApplyMethod(strategy_code_general.move_to, strategy_code_general2.get_center()),
                FadeInFromDown(where_n_is_large),
            ),
        )
        self.play(
            FadeIn(brace_group),
        )
        self.wait(1)
        self.play(
            Write(battle),
            Write(comps),
        )
        self.wait(1)

        # reveal the full screen and move the computers to the middle
        self.remove(line)
        self.play(
            # line.move_to, np.array([-8, 0, 0]),
            FadeOut(logo),
            FadeOut(header),
            FadeOut(strategy_words),
            FadeOut(arrow_group),
            # FadeOut(strategy_group),
            FadeOut(strategy_code_general),
            FadeOut(where_n_is_large),
            FadeOut(brace_group),
            FadeOutAndShiftDown(disclaimer),
            Transform(comp_group, centred_comp_group),
            run_time=1.5,
        )

        # centre the two strategies we are following
        self.wait(0.5)

        # move them in place for the table and write the table
        self.play(
            Transform(blue, blue2),
            Transform(orange, orange2),
            FadeOut(battle),
            FadeOut(vs),
            Write(table_lines[:3]),
            Write(opponent),
            run_time=1,
        )
        self.wait(0.2)

        scores = [0, 0]
        rates = [80, 60]
        w = TextMobject('W').set_color(GREEN).scale(0.5)
        l = TextMobject('L').set_color(RED).scale(0.5)

        # opponent counter
        self.x = Integer(group_with_commas=True).move_to(np.array([-5, 2.5, 0])).scale(0.5)

        def play_games(start, end, wait_time):
            for i in range(start, end + 1):
                new_x = self.x.copy().set_value(i).next_to(self.x, DOWN, buff=0.09)
                self.x = new_x
                self.add(self.x)
                for j in range(2):
                    res = l
                    if np.random.randint(0, 100) < rates[j]:
                        scores[j] += 1
                        res = w
                    x_coord = (COL1_X if j == 0 else COL2_X)
                    self.add(res.copy().move_to(np.array([x_coord, self.x.get_y(), 0])))
                self.wait(wait_time)

        vdots1 = TexMobject('\\vdots').move_to(np.array([COL1_X, -1.5, 0]))
        vdots2 = TexMobject('\\vdots').move_to(np.array([COL2_X, -1.5, 0]))
        vdots0 = TexMobject('\\vdots').move_to(np.array([self.x.get_x(), -1.5, 0]))

        play_games(1, 4, 0.3)
        play_games(5, 20, 0.1)
        # play_games(5, 10, 0.1)
        # self.play(
        #     FadeIn(vdots0),
        #     FadeIn(vdots1),
        #     FadeIn(vdots2),
        # )
        # self.x.move_to(vdots0.get_bottom())
        # play_games(9999, 10000, 0.7)

        # compile scores
        blue_score = Integer(scores[0], group_with_commas=True).move_to([COL1_X, BOTTOM_Y, 0]).scale(0.6)
        orange_score = Integer(scores[1], group_with_commas=True).move_to([COL2_X, BOTTOM_Y, 0]).scale(0.6)

        # show results
        self.wait(0.5)
        self.play(
            Write(table_lines[3]),
            Write(score),
            FadeIn(blue_score),
            FadeIn(orange_score),
        )
        self.play(
            Transform(blue.copy(), blue3),
            Transform(orange.copy(), orange3),
            Write(better),
        )
        self.wait(1)


class WhyComputersScene(Scene):
    def construct(self):
        # for debugging purposes; not in final video
        SHOW_GRID = False
        X_CUTOFF = -0.8

        if SHOW_GRID:
            grid = NumberPlane()
            self.add(grid)

        # title
        # computer = ImageMobject('Computer.png')
        computer = Laptop(width=2)
        vs = TextMobject('{\\LARGE vs.}')
        screen = ScreenRectangle(height=computer.get_height())
        title_group = Group(computer, vs, screen)
        title_group.arrange(RIGHT, buff=MED_LARGE_BUFF)

        header = title_group.copy()
        header.scale(0.4)
        header.to_edge(UP)

        # table stuff
        table_lines = VGroup(
            Line(np.array([-6, 0.5, 0]), np.array([6, 0.5, 0])),            # upper horizontal line
            Line(np.array([-6, -1.5, 0]), np.array([6, -1.5, 0])),      # lower horizontal line
            Line(np.array([-1.5, -3.5, 0]), np.array([-1.5, 3.5, 0])),  # left vertical line
            Line(np.array([2.25, -3.5, 0]), np.array([2.25, 3.5, 0])),    # right vertical line
        )

        # constants
        LABEL_X = -3.75
        IMAGE_Y = 2
        ROW1_Y = -0.5
        ROW2_Y = -2.5
        COL1_X = (-1.5 + 2.25) / 2
        COL2_X = (2.25 + 6) / 2

        # images
        computer2 = computer.copy().move_to(np.array([COL1_X, IMAGE_Y, 0]))
        screen2 = screen.copy().move_to(np.array([COL2_X, IMAGE_Y, 0]))

        # row 1
        capacity = TextMobject('Capacity').move_to(np.array([LABEL_X, ROW1_Y, 0]))
        millions = TextMobject('millions of\\\\numbers').set_color(GREEN).move_to(np.array([COL1_X, ROW1_Y, 0])).scale(0.8)
        two = TextMobject('two.').set_color(RED).move_to(np.array([COL2_X, ROW1_Y, 0])).scale(0.8)

        # row 2
        feasibility = TextMobject('Feasibility on\\\\a large scale').move_to(np.array([LABEL_X, ROW2_Y, 0]))
        doable = TextMobject('doable').set_color(GREEN).move_to(np.array([COL1_X, ROW2_Y, 0])).scale(0.8)
        no_way = TextMobject('no way').set_color(RED).move_to(np.array([COL2_X, ROW2_Y, 0])).scale(0.8)

        # self.add(
        #     capacity,
        #     millions,
        #     two,
        #     feasibility,
        #     doable,
        #     no_way,
        # )

        # animations start
        self.add(title_group)
        self.wait(1)
        self.play(
            ReplacementTransform(computer, computer2),
            ReplacementTransform(screen, screen2),
            FadeOut(vs),
            FadeIn(table_lines),
        )
        self.wait(1)
        self.play(Write(capacity))
        self.wait(1)
        self.play(FadeInFromLarge(millions))
        self.wait(1)
        self.play(FadeIn(two))
        self.wait(2)
        self.play(Write(feasibility))
        self.wait(1)
        self.play(FadeIn(no_way))
        self.wait(1)
        self.play(FadeInFromLarge(doable))
        self.wait(2)
