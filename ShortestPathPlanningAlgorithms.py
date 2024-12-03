import pygame # for user input and visualization
import heapq # priority queue library
import threading # for real-time visualizaiton

# Constants
WIDTH = 900
HEIGHT = 800
ROWS = 50
COLS = 50
TILE_SIZE = min((WIDTH - 200) / COLS, HEIGHT / ROWS) # Calculation of tile (cell) size
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GREY = (128, 128, 128)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("Pathfinding Algorithms Visualization")

# Node class and his methods
class Node:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.x = col * TILE_SIZE
        self.y = row * TILE_SIZE
        self.color = WHITE
        self.neighbors = []

    def draw(self): # to represent node (grid cell)
        pygame.draw.rect(screen, self.color, (self.x, self.y, TILE_SIZE, TILE_SIZE))

    def set_start(self):
        self.color = ORANGE
    
    def set_end(self):
        self.color = YELLOW

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
        return self.color == YELLOW
    
    def update_neighbors(self, grid):# Identifies and updates neighbors that are not obstacles
        self.neighbors = []
        if self.row > 0 and not grid[self.row - 1][self.col].is_obstacle(): # Upside
            self.neighbors.append(grid[self.row - 1][self.col])
        if self.row < ROWS - 1 and not grid[self.row + 1][self.col].is_obstacle(): # Downside
            self.neighbors.append(grid[self.row + 1][self.col])
        if self.col > 0 and not grid[self.row][self.col - 1].is_obstacle(): # Leftside
            self.neighbors.append(grid[self.row][self.col - 1])
        if self.col < COLS - 1 and not grid [self.row][self.col + 1].is_obstacle(): # Rightside
            self.neighbors.append(grid[self.row][self.col + 1])


def make_grid(): # Fill grids with nodes
    return[[Node(row, col) for col in range(COLS)] for row in range(ROWS)]


def draw_grid():
    for row in range(ROWS):
        for col in range(COLS):
            pygame.draw.rect(screen, GREY, (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE), 1)


def reconstruct_path(came_from, current): # reconstruct path once algorithm findes the destination
    while current in came_from:
        current = came_from[current]
        current.set_path()
        pygame.display.update()


def heuristic(node1, node2): # For A* and GreedyBFS, calculates an estimate of cost (distance) from one node to another
    return abs(node1.row - node2.row) + abs(node1.col - node2.col)


def draw_buttons():
    font = pygame.font.SysFont("consolas", 20)
    buttons = [
        {"rect": pygame.Rect(WIDTH - 190, 50, 180, 50), "text": "A* Algorithm", "color": GREY, "action": "A*"},
        {"rect": pygame.Rect(WIDTH - 190, 120, 180, 50), "text": "Dijkstra", "color": GREY, "action": "Dijkstra"},
        {"rect": pygame.Rect(WIDTH - 190, 190, 180, 50), "text": "Greedy BFS", "color": GREY, "action": "Greedy BFS"},
        {"rect": pygame.Rect(WIDTH - 190, 260, 180, 50), "text": "Reset Map", "color": RED, "action": "Reset"},
        ]

    for button in buttons:
        pygame.draw.rect(screen, button["color"],button["rect"])
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


def dijkstra(grid, start, end): # Dijkstras's algorithm to find shortest path in graph
    open_set = [] # priority queue that stores nodes to explore, prioritized by their cost from start node
    heapq.heappush(open_set, (0, id(start), start))
    came_from = {} # dicitonary used to track previous node at each new node, than used to reconstruct the path
    g_score = {node: float("inf") for row in grid for node in row} # a dictionary mapping each node to the cost to reach it form the start
    visited = set() # a set to keep track of nodes that have already been processed

    g_score[start] = 0

    while open_set: # checking until open_set is empty or detination node found
        current = heapq.heappop(open_set)[2] # the node with lowest cost is popped from open_set

        if current in visited: # final node condition
            continue
        visited.add(current)

        if current == end:
            reconstruct_path(came_from, end)
            return True

        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_obstacle():
                temp_g_score = g_score[current] + 1 # temporary cost to reach neighbor

                if temp_g_score < g_score[neighbor]: # if the cost is lower than previously recorded cost
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    heapq.heappush(open_set, (g_score[neighbor], id(neighbor), neighbor)) # add the neighbor to the open_set with its updated cost and priority
                    neighbor.set_open()

        draw_grid()
        pygame.display.update()
        if current != start:
            current.set_closed() # set red those explored nodes

        pygame.time.delay(30)

    raise ValueError("No path found!") # if the open_set is empty and the end node was not reached, path doesnt exist


def astar(grid, start, end):
    open_set = [] # priority queue that stores nodes to explore, prioritized by their cost from start node
    heapq.heappush(open_set, (0, id(start), start))
    came_from = {} # dicitonary used to track previous node at each new node, than used to reconstruct the path
    g_score = {node: float('inf') for row in grid for node in row} # dictionary holding actual cost from start node to each node, initialized all nodes to infinity except start node = 0
    f_score = {node: float('inf') for row in grid for node in row} # dicitonary holding estimated total cost for each node, initialized all nodes to infinity except start node = 0
    g_score[start] = 0
    f_score [start] = heuristic(start, end)

    while open_set: # checking until open_set is empty or detination node found
        current = heapq.heappop(open_set)[2] # the node with lowest cost is popped from open_set

        if current == end: # final node condition
            reconstruct_path(came_from, end)
            return True
        
        for neighbor in current.neighbors: # for each neighbor of current node
            temp_g_score = g_score[current] + 1 # temp score for neighbor
            if temp_g_score < g_score[neighbor]: # if the score is better than currently known
                came_from[neighbor] = current # current pointer leads to less g_score neighbor
                g_score[neighbor] = temp_g_score  # update g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end) # update heuristics = actual cost from start + estimated cost to end
                heapq.heappush(open_set, (f_score[neighbor], id(neighbor), neighbor)) # adds neighbor node to the priority queue, prioritizing nodes eith lowest f_score
                neighbor.set_open() # add neighbor to the open_set

        draw_grid()
        pygame.display.update()
        if current != start:
            current.set_closed() # set red explored nodes

        pygame.time.delay(30)

    raise ValueError("No path found!") # if the open_set is empty and the end node was not reached, path doesnt exist


def greedy_bfs(grid, start, end):
    open_set = [] # priority queue that stores nodes to explore, prioritized by their cost from start node
    heapq.heappush(open_set, (heuristic(start, end), id(start), start))
    came_from = {} # dicitonary used to track previous node at each new node, than used to reconstruct the path
    visited = set() # to keep track of previously explored nodes

    while open_set: # checking until open_set is empty or detination node found
        current = heapq.heappop(open_set)[2] # the node with lowest cost is popped from open_set

        if current == end: # final node condition
            reconstruct_path(came_from, end)
            return True

        if current in visited:
            continue
        visited.add(current)

        for neighbor in current.neighbors: # for each neighbor of current node
            if neighbor not in visited: # if neighbor has not been visited
                came_from[neighbor] = current
                heapq.heappush(open_set, (heuristic(neighbor, end), id(neighbor), neighbor)) # pushed to open_set priority queue, prioritized by heuristic value = distance between actual node to destiantion node
                neighbor.set_open()

        draw_grid()
        pygame.display.update()
        if current != start:
            current.set_closed() # set red explored nodes

        pygame.time.delay(30)

    raise ValueError("No path found!") # if the open_set is empty and the end node was not reached, path doesnt exist


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
                if pos[0] < WIDTH - 200:
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
                else:
                    action = handle_button_click(buttons, pos)
                    if action == "Reset":
                        grid = make_grid()
                        start, end = None, None
                    else:
                        algorithm = action

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and start and end: # Enter to run algorithm
                    thread = threading.Thread(target=algorithm_execution, args=(grid, algorithm, start, end))
                    thread.start()


if __name__ == "__main__":
    main()
