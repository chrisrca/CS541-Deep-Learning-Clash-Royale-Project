import io
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

# Grid parameters
Y_OFFSET_TOP = 62
Y_OFFSET_BOTTOM = 7
X_OFFSET_LEFT = 0
X_OFFSET_RIGHT = 0
NUM_ROWS = 32
NUM_COLS = 18

# Load the parquet file
table = pq.read_table("./new_arena_placement.parquet")

# Get the first row
row_table = table.slice(0, 1)
data = row_table.to_pydict()

# Get the image bytes
raw_image_bytes = data["png_bytes"][0]

# Handle case where it might be a list of frames
if isinstance(raw_image_bytes, list):
    raw_image_bytes = raw_image_bytes[0]

# Open the image
img = Image.open(io.BytesIO(raw_image_bytes))
width, height = img.size

# Calculate tile dimensions from offsets
grid_width = width - X_OFFSET_LEFT - X_OFFSET_RIGHT
grid_height = height - Y_OFFSET_TOP - Y_OFFSET_BOTTOM
TILE_WIDTH = grid_width / NUM_COLS
TILE_HEIGHT = grid_height / NUM_ROWS

print(f"=== IMAGE DIMENSIONS ===")
print(f"Image size: {width} x {height} pixels")

print(f"\n=== CALCULATED TILE SIZE ===")
print(f"Grid area: {grid_width} x {grid_height} pixels")
print(f"Tile width: {TILE_WIDTH:.2f} pixels ({grid_width} / {NUM_COLS})")
print(f"Tile height: {TILE_HEIGHT:.2f} pixels ({grid_height} / {NUM_ROWS})")

# Create a drawing context
draw = ImageDraw.Draw(img)

# Draw vertical lines (columns)
for col in range(NUM_COLS + 1):
    x = X_OFFSET_LEFT + col * TILE_WIDTH
    y_start = Y_OFFSET_TOP
    y_end = Y_OFFSET_TOP + NUM_ROWS * TILE_HEIGHT
    draw.line([(x, y_start), (x, y_end)], fill="red", width=1)

# Draw horizontal lines (rows)
for row in range(NUM_ROWS + 1):
    x_start = X_OFFSET_LEFT
    x_end = X_OFFSET_LEFT + NUM_COLS * TILE_WIDTH
    y = Y_OFFSET_TOP + row * TILE_HEIGHT
    draw.line([(x_start, y), (x_end, y)], fill="red", width=1)

# Draw middle line
middle_y = Y_OFFSET_TOP + (grid_height / 2)
print(f"Middle Y Position: {middle_y}")
draw.line([(0, middle_y), (width, middle_y)], fill="blue", width=2)

print(f"\n=== GRID PARAMETERS ===")
print(f"Offsets: left={X_OFFSET_LEFT}, right={X_OFFSET_RIGHT}, top={Y_OFFSET_TOP}, bottom={Y_OFFSET_BOTTOM}")
print(f"Grid: {NUM_COLS} columns x {NUM_ROWS} rows")
print(f"Grid covers: x=[{X_OFFSET_LEFT}, {width - X_OFFSET_RIGHT}], y=[{Y_OFFSET_TOP}, {height - Y_OFFSET_BOTTOM}]")

# Show the image
img.show()
