import pygame
from enum import Enum
from sklearn import svm
import numpy as np

COLOR_WHITE = pygame.Color(255, 255, 255)
COLOR_BLACK = pygame.Color(0, 0, 0)
COLOR_GREY = pygame.Color(128, 128, 128)
COLOR_RED = pygame.Color(255, 0, 0)
COLOR_BLUE = pygame.Color(0, 0, 255)


class Flag(Enum):
    def __init__(self, id, color):
        self.id = id
        self.color = color

    RED = 1, COLOR_RED
    BLUE = 2, COLOR_BLUE


class Point:
    def __init__(self, pos, flag):
        self.pos = pos
        self.flag = flag

    def __repr__(self) -> str:
        return "{} - {}".format(self.flag, self.pos)


# define a main function
def main():
    # initialize the pygame module
    pygame.init()
    # load and set the logo
    pygame.display.set_caption("SVM")

    # create a surface on screen that has the size of 240 x 180
    screen = pygame.display.set_mode((1024, 768))
    screen.fill(COLOR_WHITE)
    pygame.display.update()

    # points drawn on the screen
    drawn_points = []
    # define a variable to control the main loop
    running = True
    # main loop
    while running:
        # event handling, gets all event from the event queue
        for event in pygame.event.get():
            # only do something if the event is of type QUIT
            if event.type == pygame.QUIT:
                # change the value to False, to exit the main loop
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1 or event.key == pygame.K_2:
                    point_coords = draw_point(screen, event.key)
                    drawn_points.append(point_coords)
                elif event.key == pygame.K_RETURN:
                    classify_svm(screen, drawn_points)
                elif event.key == pygame.K_c:
                    drawn_points = []
                    clear_screen(screen)


def draw_point(screen, key):
    cursor_pos = pygame.mouse.get_pos()
    flag = None
    if key == pygame.K_1:
        flag = Flag.RED
    elif key == pygame.K_2:
        flag = Flag.BLUE
    pygame.draw.circle(screen, flag.color, cursor_pos, 10)
    pygame.display.update()
    return Point(cursor_pos, flag)


def classify_svm(screen, points):
    clf_svm = create_svm_model(points)
    draw_svm(screen, clf_svm)


def create_svm_model(points):
    clf = svm.SVC(kernel='linear')
    positions = list(map(lambda point: list(point.pos), points))
    flags = list(map(lambda point: point.flag.id, points))
    clf.fit(positions, flags)
    return clf


def draw_svm(screen, clf_svm):
    # get the separating hyperplane
    w = clf_svm.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(0, 1024)
    yy = a * xx - (clf_svm.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors (margin away from hyperplane in direction
    # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
    # 2-d.
    margin = 1 / np.sqrt(np.sum(clf_svm.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin

    line = list(zip(xx, yy))
    line_down = list(zip(xx, yy_down))
    line_up = list(zip(xx, yy_up))
    for i in range(1, len(line)):
        pygame.draw.line(screen, COLOR_BLACK, line[i - 1], line[i], 5)
        pygame.draw.line(screen, COLOR_GREY, line_down[i - 1], line_down[i])
        pygame.draw.line(screen, COLOR_GREY, line_up[i - 1], line_up[i])
    pygame.display.update()


def clear_screen(screen):
    screen.fill(COLOR_WHITE)
    pygame.display.flip()


# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__ == "__main__":
    # call the main function
    main()
