import os
import time
import heapq
import cv2
import numpy as np


# PROJECT SETTINGS

WIDTH = 180
HEIGHT = 50
CLEARANCE = 2
TEXT_OBSTACLE = "JS5690"
DISPLAY_SCALE = 6
FPS = 30
VIDEO_NAME = "BW-dijkstra_Jigar_Shah.mp4"

# To keep the video size reasonable
EXPLORATION_SKIP = 8
PATH_FRAME_REPEAT = 4
FINAL_HOLD_FRAMES = 45



# COORDINATE HELPERS
def map_to_image_coords(x: int, y: int) -> tuple[int, int]:
    return x, HEIGHT - 1 - y


def is_within_bounds(x: int, y: int) -> bool:
    return 0 <= x < WIDTH and 0 <= y < HEIGHT



# MAP BUILDING
def build_wall_mask() -> np.ndarray:
    wall_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    wall_mask[:CLEARANCE, :] = 255
    wall_mask[HEIGHT - CLEARANCE:, :] = 255
    wall_mask[:, :CLEARANCE] = 255
    wall_mask[:, WIDTH - CLEARANCE:] = 255

    return wall_mask


def build_text_obstacle_mask(text: str) -> np.ndarray:
    temp = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2

    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    x = max(3, (WIDTH - text_w) // 2)
    y = min(HEIGHT - 4, max(text_h + 2, (HEIGHT + text_h) // 2 - 2))

    cv2.putText(
        temp,
        text,
        (x, y),
        font,
        font_scale,
        255,
        thickness,
        lineType=cv2.LINE_8,
    )

    contours, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(temp)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    return filled


def inflate_mask(mask: np.ndarray, clearance: int) -> np.ndarray:
    if clearance <= 0:
        return mask.copy()

    kernel_size = 2 * clearance + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)


def build_map(text: str):
    wall_mask = build_wall_mask()
    obstacle_mask = build_text_obstacle_mask(text)

    inflated_text_mask = inflate_mask(obstacle_mask, CLEARANCE)
    clearance_mask = cv2.subtract(inflated_text_mask, obstacle_mask)
    blocked_mask = cv2.bitwise_or(wall_mask, inflated_text_mask)

    color_map = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    color_map[blocked_mask > 0] = (220, 220, 220)
    color_map[obstacle_mask > 0] = (0, 0, 0)

    return obstacle_mask, clearance_mask, wall_mask, blocked_mask, color_map


def is_blocked(x: int, y: int, blocked_mask: np.ndarray) -> bool:
    if not is_within_bounds(x, y):
        return True

    img_x, img_y = map_to_image_coords(x, y)
    return blocked_mask[img_y, img_x] != 0



# USER INPUT
def parse_point_input(user_text: str) -> tuple[int, int]:
    cleaned = user_text.replace(",", " ").split()
    if len(cleaned) != 2:
        raise ValueError("Please enter exactly 2 integers like: 10 20")

    x = int(cleaned[0])
    y = int(cleaned[1])
    return x, y


def get_valid_point(prompt_name: str, blocked_mask: np.ndarray) -> tuple[int, int]:
    while True:
        try:
            user_text = input(f"Enter {prompt_name} coordinates as x y: ")
            x, y = parse_point_input(user_text)

            if not is_within_bounds(x, y):
                print(
                    f"{prompt_name} is outside map bounds. "
                    f"Valid range: x in [0, {WIDTH - 1}], y in [0, {HEIGHT - 1}]"
                )
                continue

            if is_blocked(x, y, blocked_mask):
                print(
                    f"{prompt_name} is in obstacle / clearance / wall space. "
                    f"Please enter again."
                )
                continue

            print(f"{prompt_name} accepted: ({x}, {y})")
            return x, y

        except ValueError as exc:
            print(f"Invalid input. {exc}")


# 8 ACTION FUNCTIONS (Dijkstra uses 1.0 for straight and 1.4 for diagonal)
def move_right(node: tuple[int, int]) -> tuple[int, int, float]:
    x, y = node
    return x + 1, y, 1.0


def move_left(node: tuple[int, int]) -> tuple[int, int, float]:
    x, y = node
    return x - 1, y, 1.0


def move_up(node: tuple[int, int]) -> tuple[int, int, float]:
    x, y = node
    return x, y + 1, 1.0


def move_down(node: tuple[int, int]) -> tuple[int, int, float]:
    x, y = node
    return x, y - 1, 1.0


def move_up_right(node: tuple[int, int]) -> tuple[int, int, float]:
    x, y = node
    return x + 1, y + 1, 1.4


def move_up_left(node: tuple[int, int]) -> tuple[int, int, float]:
    x, y = node
    return x - 1, y + 1, 1.4


def move_down_right(node: tuple[int, int]) -> tuple[int, int, float]:
    x, y = node
    return x + 1, y - 1, 1.4


def move_down_left(node: tuple[int, int]) -> tuple[int, int, float]:
    x, y = node
    return x - 1, y - 1, 1.4


def get_all_actions():
    return [
        ("RIGHT", move_right),
        ("LEFT", move_left),
        ("UP", move_up),
        ("DOWN", move_down),
        ("UP_RIGHT", move_up_right),
        ("UP_LEFT", move_up_left),
        ("DOWN_RIGHT", move_down_right),
        ("DOWN_LEFT", move_down_left),
    ]


def get_valid_neighbors(node: tuple[int, int], blocked_mask: np.ndarray):
    valid_neighbors = []

    for action_name, action_fn in get_all_actions():
        nx, ny, step_cost = action_fn(node)

        if not is_within_bounds(nx, ny):
            continue

        if is_blocked(nx, ny, blocked_mask):
            continue

        valid_neighbors.append((action_name, (nx, ny), step_cost))

    return valid_neighbors



# BACKWARD DIJKSTRA (Search starts from GOAL and stops when START is reached)
def reached_start(current_node: tuple[int, int], start_node: tuple[int, int]) -> bool:
    return current_node == start_node


def backtrack_path(
    parent: dict[tuple[int, int], tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [start]
    current = start

    while current != goal:
        if current not in parent:
            raise ValueError("Backtracking failed: missing parent for a node.")
        current = parent[current]
        path.append(current)

    return path


def run_backward_dijkstra(
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked_mask: np.ndarray,
):
    open_heap = []
    heapq.heappush(open_heap, (0.0, goal))

    best_cost = {goal: 0.0}
    parent = {}
    visited = set()
    explored_order = []

    while open_heap:
        current_cost, current_node = heapq.heappop(open_heap)

        if current_node in visited:
            continue

        visited.add(current_node)
        explored_order.append(current_node)

        if reached_start(current_node, start):
            path = backtrack_path(parent, start, goal)
            return {
                "success": True,
                "path": path,
                "explored_order": explored_order,
                "total_cost": best_cost[start],
            }

        for _, neighbor, step_cost in get_valid_neighbors(current_node, blocked_mask):
            if neighbor in visited:
                continue

            new_cost = current_cost + step_cost

            if (neighbor not in best_cost) or (new_cost < best_cost[neighbor]):
                best_cost[neighbor] = new_cost
                parent[neighbor] = current_node
                heapq.heappush(open_heap, (new_cost, neighbor))

    return {
        "success": False,
        "path": [],
        "explored_order": explored_order,
        "total_cost": None,
    }


# DRAWING HELPERS
def draw_point_on_map(
    image: np.ndarray,
    point: tuple[int, int],
    bgr_color: tuple[int, int, int],
    label: str,
):
    x, y = point
    img_x, img_y = map_to_image_coords(x, y)

    cv2.circle(image, (img_x, img_y), 1, bgr_color, thickness=-1)

    tx = min(img_x + 2, WIDTH - 18)
    ty = max(img_y - 2, 8)

    cv2.putText(
        image,
        label,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        bgr_color,
        1,
        lineType=cv2.LINE_AA,
    )


def draw_explored_node(image: np.ndarray, node: tuple[int, int]):
    x, y = node
    img_x, img_y = map_to_image_coords(x, y)

    if image[img_y, img_x].tolist() == [255, 255, 255]:
        image[img_y, img_x] = (255, 200, 0)


def draw_path_segment(
    image: np.ndarray,
    p1_map: tuple[int, int],
    p2_map: tuple[int, int],
):
    p1 = map_to_image_coords(*p1_map)
    p2 = map_to_image_coords(*p2_map)
    cv2.line(image, p1, p2, (255, 0, 255), 1)


def make_display_frame(image: np.ndarray) -> np.ndarray:
    return cv2.resize(
        image,
        (WIDTH * DISPLAY_SCALE, HEIGHT * DISPLAY_SCALE),
        interpolation=cv2.INTER_NEAREST,
    )



# VIDEO CREATION (Visualization starts only after search is completed)

def create_animation_video(
    base_map: np.ndarray,
    explored_order: list[tuple[int, int]],
    path: list[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    video_path: str,
):
    frame_width = WIDTH * DISPLAY_SCALE
    frame_height = HEIGHT * DISPLAY_SCALE

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, FPS, (frame_width, frame_height))

    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter for mp4 output.")

    canvas = base_map.copy()
    draw_point_on_map(canvas, start, (0, 255, 0), "S")
    draw_point_on_map(canvas, goal, (0, 0, 255), "G")

    first_frame = make_display_frame(canvas)
    for _ in range(15):
        writer.write(first_frame)

    for idx, node in enumerate(explored_order):
        draw_explored_node(canvas, node)
        draw_point_on_map(canvas, start, (0, 255, 0), "S")
        draw_point_on_map(canvas, goal, (0, 0, 255), "G")

        if idx % EXPLORATION_SKIP == 0 or idx == len(explored_order) - 1:
            writer.write(make_display_frame(canvas))

    for i in range(len(path) - 1):
        draw_path_segment(canvas, path[i], path[i + 1])
        draw_point_on_map(canvas, start, (0, 255, 0), "S")
        draw_point_on_map(canvas, goal, (0, 0, 255), "G")

        frame = make_display_frame(canvas)
        for _ in range(PATH_FRAME_REPEAT):
            writer.write(frame)

    final_frame = make_display_frame(canvas)
    for _ in range(FINAL_HOLD_FRAMES):
        writer.write(final_frame)

    writer.release()

    final_png_path = os.path.splitext(video_path)[0] + "_final.png"
    cv2.imwrite(final_png_path, canvas)
    return final_png_path



def main():
    print("\n===== BUILDING MAP =====")
    _, _, _, blocked_mask, color_map = build_map(TEXT_OBSTACLE)

    print("\n===== ENTER VALID START AND GOAL =====")
    start = get_valid_point("START", blocked_mask)
    goal = get_valid_point("GOAL", blocked_mask)

    while goal == start:
        print("Goal cannot be the same as start. Please enter goal again.")
        goal = get_valid_point("GOAL", blocked_mask)

    print("\n===== RUNNING BACKWARD DIJKSTRA =====")
    print("Search starts from GOAL and stops when START is popped.")

    t0 = time.time()
    result = run_backward_dijkstra(start, goal, blocked_mask)
    t1 = time.time()

    if not result["success"]:
        print("\nNo path found.")
        print(f"Explored nodes: {len(result['explored_order'])}")
        print(f"Runtime: {t1 - t0:.6f} seconds")
        return

    path = result["path"]
    explored_order = result["explored_order"]
    total_cost = result["total_cost"]

    print("\n===== RESULT =====")
    print(f"Start              : {start}")
    print(f"Goal               : {goal}")
    print("Path found         : YES")
    print(f"Total path cost    : {total_cost:.2f}")
    print(f"Explored nodes     : {len(explored_order)}")
    print(f"Path length (nodes): {len(path)}")
    print(f"Runtime            : {t1 - t0:.6f} seconds")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, VIDEO_NAME)

    print("\n===== CREATING ANIMATION VIDEO =====")
    final_png_path = create_animation_video(
        base_map=color_map,
        explored_order=explored_order,
        path=path,
        start=start,
        goal=goal,
        video_path=video_path,
    )

    print("\n===== FILE OUTPUT =====")
    print(f"Video saved      : {video_path}")
    print(f"Final image saved: {final_png_path}")


if __name__ == "__main__":
    main()