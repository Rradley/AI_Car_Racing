import pygame
import numpy as np
import random
from scipy.interpolate import splprep, splev

# Initialize pygame
pygame.init()

# Screen dimensions and colors
WIDTH, HEIGHT = 1080, 1080
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (50, 50, 50)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Screen and clock
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AIRacing")
clock = pygame.time.Clock()

# Track width
TRACK_WIDTH = 80

# Font for labels
font = pygame.font.Font(None, 30)


class Track:
    """Handles the track's centerline, boundaries, and rendering."""

    def __init__(self, width):
        self.centerline = []  # Track centerline points
        self.inner_boundary = []  # Inner boundary points
        self.outer_boundary = []  # Outer boundary points
        self.width = width
        self.finalized = False
        self.track_polygon = None  # Combined polygon for collision detection
        self.track_mask = None  # Mask for collision detection

    def add_point(self, point):
        """Add a point to the centerline."""
        if not self.centerline or point != self.centerline[-1]:
            self.centerline.append(point)

    def _spline_smooth(self, points, smooth_factor=2, num_points=200):
        """Smooth the track points using cubic splines."""
        x, y = zip(*points)  # Separate x and y coordinates
        tck, _ = splprep([x, y], s=smooth_factor)  # Fit spline with smoothing factor
        u = np.linspace(0, 1, num_points)  # Generate parameter values for interpolation
        x_smooth, y_smooth = splev(u, tck)  # Evaluate spline at generated points
        return list(zip(x_smooth, y_smooth))

    def smooth_centerline(self):
        """Smooth the track centerline using splines."""
        if len(self.centerline) > 2:
            self.centerline = self._spline_smooth(self.centerline, smooth_factor=2, num_points=200)

    def finalize(self):
        """Finalize the track by calculating boundaries."""
        if len(self.centerline) > 2:          
            self.smooth_centerline()  # Apply smoothing to the centerline
            self.centerline.append(self.centerline[0])  # Close the centerline loop
            self.inner_boundary, self.outer_boundary = self.calculate_boundaries()
            
            # Calculate inner and outer boundaries
            self.inner_boundary, self.outer_boundary = self.calculate_boundaries()

            # Create the track polygon by combining boundaries
            self.track_polygon = self.outer_boundary + self.inner_boundary[::-1]

            # Create a mask for collision detection
            self.track_mask = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            self.track_mask.fill((0, 0, 0, 0))  # Transparent background
            pygame.draw.polygon(self.track_mask, WHITE, self.track_polygon)
            self.track_mask = pygame.mask.from_surface(self.track_mask)
            self.finalized = True

    def calculate_boundaries(self):
        """Calculate inner and outer boundaries based on the centerline."""
        inner, outer = [], []
        for i in range(len(self.centerline)):
            # Current and next points
            p1 = np.array(self.centerline[i])
            p2 = np.array(self.centerline[(i + 1) % len(self.centerline)])  # Loop back to start

            # Direction vector and perpendicular offset
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length == 0:
                continue  # Skip zero-length segments
            unit_direction = direction / length
            offset = np.array([-unit_direction[1], unit_direction[0]]) * (self.width / 2)

            # Add inner and outer points
            inner.append(tuple(p1 + offset))
            outer.append(tuple(p1 - offset))

        # Ensure boundaries are closed
        inner.append(inner[0])
        outer.append(outer[0])
        return inner, outer

    def draw(self):
        """Render the track."""
        if self.finalized:
            # Draw the filled track polygon
            pygame.draw.polygon(screen, GREY, self.track_polygon)
            # Draw boundaries
            pygame.draw.lines(screen, WHITE, True, self.inner_boundary, 3)
            pygame.draw.lines(screen, WHITE, True, self.outer_boundary, 3)
        else:
            # Draw the centerline if not finalized
            if len(self.centerline) > 1:
                pygame.draw.lines(screen, WHITE, False, self.centerline, 3)

class Car:
    """Represents a car with AI navigation."""

    def __init__(self, x, y, color, ai_model, track):
        self.start_x = x
        self.start_y = y
        self.x = x
        self.y = y
        self.angle = 0  # Angle of movement
        self.speed = 2
        self.color = color
        self.ai_model = ai_model
        self.track = track
        self.sensors = [200] * 5  # Five sensors for detecting track boundaries
        self.collided = False

    def draw(self):
        """Render the car."""
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), 5)

    def move(self):
        """Move the car based on its current speed and angle."""
        self.x += np.cos(np.radians(self.angle)) * self.speed
        self.y -= np.sin(np.radians(self.angle)) * self.speed

    def sense(self):
        """Simulate sensors to detect track boundaries."""
        sensor_angles = [-45, -20, 0, 20, 45]  # Sensor angles relative to the car's angle
        for i, sensor_angle in enumerate(sensor_angles):
            ray_angle = self.angle + sensor_angle
            for dist in range(1, 200):  # Sensor range
                ray_x = self.x + np.cos(np.radians(ray_angle)) * dist
                ray_y = self.y - np.sin(np.radians(ray_angle)) * dist
                if not self.is_inside_track(ray_x, ray_y):
                    self.sensors[i] = dist
                    pygame.draw.line(screen, RED, (self.x, self.y), (ray_x, ray_y), 1)  # Debug line
                    break

    def is_inside_track(self, x, y):
        """Check if a point is inside the track polygon."""
        if self.track.finalized and self.track.track_mask:
            # Ensure x and y are within bounds
            if 0 <= int(x) < WIDTH and 0 <= int(y) < HEIGHT:
                return self.track.track_mask.get_at((int(x), int(y))) != 0
            return False  # Treat out-of-bounds as outside the track

    def reset(self):
        """Reset the car to its starting position."""
        self.x = self.start_x
        self.y = self.start_y
        self.angle = 0
        self.speed = 2
        self.collided = False

    def update(self):
        """Update the car's position based on AI logic."""
        if not self.collided:
            self.sense()
            turn_angle = self.ai_model(self.sensors)  # AI decides turn
            self.angle += turn_angle
            self.move()
            
            # Check if the car is outside the track or out of bounds
            if not self.is_inside_track(self.x, self.y) or not (0 <= self.x < WIDTH and 0 <= self.y < HEIGHT):
                self.collided = True
                self.reset()  # Reset the car
                        
def simple_logic(sensors):
    """Enhanced simple AI model for better responsiveness."""
    if sensors[2] < 50:  # Obstacle ahead
        return random.choice([-10, 10])  # Turn left or right
    elif sensors[0] < 50:  # Obstacle on the left
        return 5  # Turn right
    elif sensors[4] < 50:  # Obstacle on the right
        return -5  # Turn left
    return 0  # Go straight

def advanced_logic(sensors):
    """Advanced logic with sharper wall avoidance."""
    left_score = sensors[0] + sensors[1]
    right_score = sensors[3] + sensors[4]
    center_steering = (right_score - left_score) * 0.02

    if sensors[2] < 30:  # Obstacle ahead
        return -20 if left_score > right_score else 20

    if sensors[0] < 20 or sensors[4] < 20:  # Very close to a wall
        return -15 if sensors[0] < sensors[4] else 15

    return center_steering

def handle_events(track):
    """Handle user input events."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False

        # Track drawing
        if not track.finalized:
            if event.type == pygame.MOUSEBUTTONDOWN:
                track.add_point(pygame.mouse.get_pos())

            if event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[0]:  # Check if the left mouse button is pressed
                    track.add_point(pygame.mouse.get_pos())

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    track.finalize()

    return True


def draw_labels():
    """Draw labels for car behaviors."""
    labels = [
        (RED, "Red: Simple"),
        (BLUE, "Blue: Advanced"),
        (GREEN, "Green: Advanced"),
    ]
    for i, (color, text) in enumerate(labels):
        label = font.render(text, True, color)
        screen.blit(label, (10, 10 + i * 30))  # Stack labels vertically


# Main loop
def main():
    track = Track(TRACK_WIDTH)  # Create a track instance
    cars = []  # List of cars
    running = True

    while running:
        screen.fill(BLACK)

        # Handle events
        running = handle_events(track)

        # Draw the track
        track.draw()

        # Add cars after the track is finalized
        if track.finalized and not cars:
            start_pos = track.centerline[0]  # Start position is the first point of the track
            cars.append(Car(start_pos[0], start_pos[1], RED, simple_logic, track))
            cars.append(Car(start_pos[0] + 10, start_pos[1] + 10, BLUE, advanced_logic, track))
            cars.append(Car(start_pos[0] - 10, start_pos[1] - 10, GREEN, advanced_logic, track))

        # Update and draw cars
        for car in cars:
            car.update()
            car.draw()

        # Draw labels
        draw_labels()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
