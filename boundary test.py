import pygame
import numpy as np
from scipy.interpolate import splprep, splev

pygame.init()

# Screen dimensions and colors
WIDTH, HEIGHT = 1080, 1080
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (50, 50, 50)

# Screen and clock
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Track Boundary Fix")
clock = pygame.time.Clock()

TRACK_WIDTH = 80


class Track:
    def __init__(self, width):
        self.centerline = []
        self.inner_boundary = []
        self.outer_boundary = []
        self.width = width
        self.finalized = False

    def add_point(self, point):
        if not self.centerline or point != self.centerline[-1]:
            self.centerline.append(point)

    def _spline_smooth(self, points, smooth_factor=2, num_points=200):
        x, y = zip(*points)
        tck, _ = splprep([x, y], s=smooth_factor)
        u = np.linspace(0, 1, num_points)
        x_smooth, y_smooth = splev(u, tck)
        return list(zip(x_smooth, y_smooth))

    def smooth_centerline(self):
        if len(self.centerline) > 2:
            self.centerline = self._spline_smooth(self.centerline, smooth_factor=2, num_points=200)

    def finalize(self):
        if len(self.centerline) > 2:
            self.smooth_centerline()
            self.centerline.append(self.centerline[0])  # Close the centerline
            self.inner_boundary, self.outer_boundary = self.calculate_boundaries()
            self.finalized = True

    def calculate_boundaries(self):
        """Calculate inner and outer boundaries based on the centerline."""
        inner, outer = [], []

        for i in range(len(self.centerline)):
            p1 = np.array(self.centerline[i])
            p2 = np.array(self.centerline[(i + 1) % len(self.centerline)])
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length == 0:
                continue
            unit_direction = direction / length
            offset = np.array([-unit_direction[1], unit_direction[0]]) * (self.width / 2)
            inner.append(tuple(p1 + offset))
            outer.append(tuple(p1 - offset))

        inner.append(inner[0])
        outer.append(outer[0])

        adjusted_inner, adjusted_outer = [], []
        for i in range(len(inner)):
            p_inner = np.array(inner[i])
            p_outer = np.array(outer[i])
            midpoint = (p_inner + p_outer) / 2
            distance = np.linalg.norm(p_inner - p_outer)

            if distance < self.width:
                scale = (self.width / 2) / distance if distance != 0 else 0.5
                adjusted_inner.append(tuple(midpoint + (p_inner - midpoint) * scale))
                adjusted_outer.append(tuple(midpoint + (p_outer - midpoint) * scale))
            else:
                adjusted_inner.append(inner[i])
                adjusted_outer.append(outer[i])

        adjusted_inner = self._spline_smooth(adjusted_inner, smooth_factor=2, num_points=200)
        adjusted_outer = self._spline_smooth(adjusted_outer, smooth_factor=2, num_points=200)

        return adjusted_inner, adjusted_outer

    def draw(self):
        if self.finalized:
            pygame.draw.polygon(screen, GREY, self.inner_boundary + self.outer_boundary[::-1])
            pygame.draw.lines(screen, WHITE, True, self.inner_boundary, 2)
            pygame.draw.lines(screen, WHITE, True, self.outer_boundary, 2)
        elif len(self.centerline) > 1:
            pygame.draw.lines(screen, WHITE, False, self.centerline, 2)


def handle_events(track):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if not track.finalized:
            if event.type == pygame.MOUSEBUTTONDOWN:
                track.add_point(pygame.mouse.get_pos())
            if event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[0]:
                track.add_point(pygame.mouse.get_pos())
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                track.finalize()
    return True


def main():
    track = Track(TRACK_WIDTH)
    running = True

    while running:
        screen.fill(BLACK)
        running = handle_events(track)
        track.draw()
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
