# ENPM661 Project 02 - Backward BFS and Backward Dijkstra

This project implements **Backward Dijkstra** and **Backward BFS** for a **point robot** on a custom 2D map.

The map size is **180 x 50**. The obstacle space is created using the custom text **JS5690**. The robot has **2 mm clearance**, so both the text obstacle and the walls are bloated by 2 pixels. The search starts from the **goal** and stops when the **start** is reached.

## Files

- `BW-dijkstra_Jigar_Shah.py` - Final backward Dijkstra implementation with animation video output
- `BW-BFS_Jigar_Shah.py` - Final backward BFS implementation with animation video output

## Requirements

Install these libraries before running:

```bash
pip install numpy opencv-python
```

Standard library modules used:

- `os`
- `time`
- `heapq` (Dijkstra)
- `collections.deque` (BFS)

## How to Run

Run either file from the terminal.

### Backward Dijkstra

```bash
python BW-dijkstra_Jigar_Shah.py
```

### Backward BFS

```bash
python BW-BFS_Jigar_Shah.py
```

## Input Format

The program asks for:

- `START` coordinates as `x y`
- `GOAL` coordinates as `x y`

Example:

```text
10 10
160 40
```

You may also enter values like `10,10`.

## Coordinate Convention

The start and goal coordinates must be given with respect to the **map origin at the bottom-left corner**.

Valid ranges:

- `x` from `0` to `179`
- `y` from `0` to `49`

If a point is:

- outside the map,
- inside obstacle space,
- inside clearance space, or
- inside wall clearance,

then the program will reject it and ask again.

## Actions Used

The workspace is **8-connected**.

Actions:

- Right `(1, 0)`
- Left `(-1, 0)`
- Up `(0, 1)`
- Down `(0, -1)`
- Up-Right `(1, 1)`
- Up-Left `(-1, 1)`
- Down-Right `(1, -1)`
- Down-Left `(-1, -1)`

### Cost used in Dijkstra

- Straight moves = `1.0`
- Diagonal moves = `1.4`

### Cost used in BFS

- All moves = `1.0`

## Output Files

### Dijkstra output

- `BW-dijkstra_Jigar_Shah.mp4`
- `BW-dijkstra_Jigar_Shah_final.png`

### BFS output

- `BW-BFS_Jigar_Shah.mp4`
- `BW-BFS_Jigar_Shah_final.png`
