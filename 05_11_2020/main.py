import pygame
from enum import Enum

COLOR_WHITE = pygame.Color(255, 255, 255)
COLOR_RED = pygame.Color(255, 0, 0)
COLOR_BLUE = pygame.Color(0, 0, 255)


class Flag(Enum):
    def __init__(self, color):
        self.color = color

    RED = COLOR_RED
    BLUE = COLOR_BLUE


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
                if event.key == pygame.K_RETURN:
                    start_svm(drawn_points)


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


def start_svm(points):
    print("Drawn points: {}".format(points))


# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__ == "__main__":
    # call the main function
    main()
