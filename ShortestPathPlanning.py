import pygame
import heapq
import threading  # Na podporu reálnom čase vizualizácie

# Constants
WIDTH, HEIGHT = 900, 800  # Pridaný priestor na bočné GUI
ROWS, COLS = 50, 50
TILE_SIZE = min((WIDTH - 200) / COLS, HEIGHT / ROWS)  # Priestor na bočné tlačidlá
WHITE, BLACK, RED, GREEN, BLUE, GREY, ORANGE, TURQUOISE = (255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128), (255, 165, 0), (64, 224, 208)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pathfinding Visualization")


class Node:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.x = col * TILE_SIZE
        self.y = row * TILE_SIZE
        self.color = WHITE
        self.neighbors = []

    def draw(self):
        pygame.draw.rect(screen, self.color, (self.x, self.y, TILE_SIZE, TILE_SIZE))

    def set_start(self):
        self.color = ORANGE

    def set_end(self):
        self.color = TURQUOISE

    def set_obstacle(self):
        self.color = BLACK

    def set_open(self):
        self.color = GREEN

    def set_closed(self):
        self.color = RED

    def set_path(self):
        self.color = BLUE

    def reset(self):
        self.color = WHITE

    def is_obstacle(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE

    def update_neighbors(self, grid):
        self.neighbors = []
        if self.row > 0 and not grid[self.row - 1][self.col].is_obstacle():  # Up
            self.neighbors.append(grid[self.row - 1][self.col])
        if self.row < ROWS - 1 and not grid[self.row + 1][self.col].is_obstacle():  # Down
            self.neighbors.append(grid[self.row + 1][self.col])
        if self.col > 0 and not grid[self.row][self.col - 1].is_obstacle():  # Left
            self.neighbors.append(grid[self.row][self.col - 1])
        if self.col < COLS - 1 and not grid[self.row][self.col + 1].is_obstacle():  # Right
            self.neighbors.append(grid[self.row][self.col + 1])


def make_grid():
    return [[Node(row, col) for col in range(COLS)] for row in range(ROWS)]


def draw_grid():
    for row in range(ROWS):
        for col in range(COLS):
            pygame.draw.rect(screen, GREY, (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE), 1)


def reconstruct_path(came_from, current):
    while current in came_from:
        current = came_from[current]
        current.set_path()
        pygame.display.update()


def heuristic(node1, node2):
    return abs(node1.row - node2.row) + abs(node1.col - node2.col)


def draw_buttons():
    font = pygame.font.SysFont("arial", 20)
    buttons = [
        {"rect": pygame.Rect(WIDTH - 190, 50, 180, 50), "text": "A* Algorithm", "color": GREEN, "action": "A*"},
        {"rect": pygame.Rect(WIDTH - 190, 120, 180, 50), "text": "Dijkstra", "color": BLUE, "action": "Dijkstra"},
        {"rect": pygame.Rect(WIDTH - 190, 190, 180, 50), "text": "Greedy BFS", "color": TURQUOISE, "action": "Greedy BFS"},
        {"rect": pygame.Rect(WIDTH - 190, 260, 180, 50), "text": "Reset Map", "color": RED, "action": "Reset"},
    ]

    for button in buttons:
        pygame.draw.rect(screen, button["color"], button["rect"])
        text = font.render(button["text"], True, WHITE)
        screen.blit(text, (button["rect"].x + 10, button["rect"].y + 15))

    return buttons


def handle_button_click(buttons, pos):
    for button in buttons:
        if button["rect"].collidepoint(pos):
            return button["action"]
    return None


def algorithm_execution(grid, algorithm, start, end):
    for row in grid:
        for node in row:
            node.update_neighbors(grid)

    if algorithm == "A*":
        astar(grid, start, end)
    elif algorithm == "Dijkstra":
        dijkstra(grid, start, end)
    elif algorithm == "Greedy BFS":
        greedy_bfs(grid, start, end)


def dijkstra(grid, start, end):
    open_set = []
    heapq.heappush(open_set, (0, id(start), start))
    came_from = {}
    g_score = {node: float("inf") for row in grid for node in row}
    visited = set()

    g_score[start] = 0

    while open_set:
        current = heapq.heappop(open_set)[2]

        if current in visited:
            continue
        visited.add(current)

        if current == end:
            reconstruct_path(came_from, end)
            return True

        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_obstacle():
                temp_g_score = g_score[current] + 1

                if temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    heapq.heappush(open_set, (g_score[neighbor], id(neighbor), neighbor))
                    neighbor.set_open()

        draw_grid()
        pygame.display.update()
        if current != start:
            current.set_closed()

        pygame.time.delay(30)

    return False


def astar(grid, start, end):
    open_set = []
    heapq.heappush(open_set, (0, id(start), start))
    came_from = {}
    g_score = {node: float("inf") for row in grid for node in row}
    f_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    f_score[start] = heuristic(start, end)

    while open_set:
        current = heapq.heappop(open_set)[2]

        if current == end:
            reconstruct_path(came_from, end)
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
                heapq.heappush(open_set, (f_score[neighbor], id(neighbor), neighbor))
                neighbor.set_open()

        draw_grid()
        pygame.display.update()
        if current != start:
            current.set_closed()

        pygame.time.delay(30)

    return False


def greedy_bfs(grid, start, end):
    open_set = []
    heapq.heappush(open_set, (heuristic(start, end), id(start), start))
    came_from = {}
    visited = set()

    while open_set:
        current = heapq.heappop(open_set)[2]

        if current == end:
            reconstruct_path(came_from, end)
            return True

        if current in visited:
            continue
        visited.add(current)

        for neighbor in current.neighbors:
            if neighbor not in visited:
                came_from[neighbor] = current
                heapq.heappush(open_set, (heuristic(neighbor, end), id(neighbor), neighbor))
                neighbor.set_open()

        draw_grid()
        pygame.display.update()
        if current != start:
            current.set_closed()

        pygame.time.delay(30)

    return False


def main():
    grid = make_grid()
    start, end = None, None
    algorithm = None
    running = True

    while running:
        screen.fill(WHITE)

        for row in grid:
            for node in row:
                node.draw()

        draw_grid()
        buttons = draw_buttons()
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return

            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                if pos[0] < WIDTH - 200:  # Kliknutie v mriežku
                    row, col = int(pos[1] // TILE_SIZE), int(pos[0] // TILE_SIZE)
                    node = grid[row][col]
                    if not start:
                        start = node
                        start.set_start()
                    elif not end and node != start:
                        end = node
                        end.set_end()
                    elif node != start and node != end:
                        node.set_obstacle()
                else:  # Kliknutie na tlačidlá
                    action = handle_button_click(buttons, pos)
                    if action == "Reset":
                        grid = make_grid()
                        start, end = None, None
                    else:
                        algorithm = action

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    thread = threading.Thread(target=algorithm_execution, args=(grid, algorithm, start, end))
                    thread.start()


if __name__ == "__main__":
    main()
