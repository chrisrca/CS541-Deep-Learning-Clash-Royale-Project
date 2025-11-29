import pyarrow.parquet as pq
import random
from dataset import ALL_CARDS, CARD_TO_ID

table = pq.read_table("./initial_training_hand_elixir.parquet")
total_rows = table.num_rows

print(f"Analyzing {total_rows} samples...")

# Initialize counters
hand_size_counts = {}  # {hand_size: count}
hand_size_examples = {}  # {hand_size: [(replay, frame), ...]}
none_counts = {}  # {none_count: count}
none_examples = {}  # {none_count: [(replay, frame), ...]}
ground_truth_in_hand = 0
ground_truth_not_in_hand = 0
empty_hands = 0
valid_hands = 0

for i in range(total_rows):
    row_table = table.slice(i, 1)
    data = row_table.to_pydict()

    # Get hand data
    raw_hand = data["hand"][0]

    # Count None values
    none_count = sum(1 for c in raw_hand if c is None)
    none_counts[none_count] = none_counts.get(none_count, 0) + 1

    # Store ALL examples for each none count
    if none_count not in none_examples:
        none_examples[none_count] = []
    replay = data["replay"][0] if "replay" in data else "unknown"
    frame = data["frame"][0] if "frame" in data else "unknown"
    none_examples[none_count].append((replay, frame))

    hand_ids = []
    for c in raw_hand:
        if c in CARD_TO_ID:
            hand_ids.append(CARD_TO_ID[c])

    hand_size = len(hand_ids)
    hand_size_counts[hand_size] = hand_size_counts.get(hand_size, 0) + 1

    # Store ALL examples for each hand size
    if hand_size not in hand_size_examples:
        hand_size_examples[hand_size] = []
    replay = data["replay"][0] if "replay" in data else "unknown"
    frame = data["frame"][0] if "frame" in data else "unknown"
    hand_size_examples[hand_size].append((replay, frame))

    if hand_size == 0:
        empty_hands += 1

    # Get ground truth card
    card_name = data["card"][0]

    # Check if ground truth is in hand
    if card_name in CARD_TO_ID:
        card_id = CARD_TO_ID[card_name]
        if card_id in hand_ids:
            ground_truth_in_hand += 1
        else:
            ground_truth_not_in_hand += 1
    else:
        # Card name not recognized (e.g., "none")
        ground_truth_not_in_hand += 1

# Print statistics
print("\n=== HAND SIZE DISTRIBUTION ===")
total_samples = sum(hand_size_counts.values())
for size in sorted(hand_size_counts.keys()):
    count = hand_size_counts[size]
    percentage = (count / total_samples) * 100
    all_examples = hand_size_examples.get(size, [])
    # Randomly select up to 10 examples
    selected_examples = random.sample(all_examples, min(10, len(all_examples)))
    examples_str = ", ".join([f"<{replay}, {frame}>" for replay, frame in selected_examples])
    print(f"Hands with {size} cards: {count} ({percentage:.1f}%) [{examples_str}]")

print("\n=== NONE VALUE ANALYSIS ===")
for none_count in sorted(none_counts.keys()):
    count = none_counts[none_count]
    percentage = (count / total_samples) * 100
    all_examples = none_examples.get(none_count, [])
    # Randomly select up to 10 examples
    selected_examples = random.sample(all_examples, min(10, len(all_examples)))
    examples_str = ", ".join([f"<{replay}, {frame}>" for replay, frame in selected_examples])
    print(f"Hands with {none_count} None values: {count} ({percentage:.1f}%) [{examples_str}]")

print("\n=== GROUND TRUTH ANALYSIS ===")
total_ground_truth_checks = ground_truth_in_hand + ground_truth_not_in_hand
if total_ground_truth_checks > 0:
    in_hand_percentage = (ground_truth_in_hand / total_ground_truth_checks) * 100
    not_in_hand_percentage = (ground_truth_not_in_hand / total_ground_truth_checks) * 100
    print(f"Ground truth in hand: {ground_truth_in_hand} ({in_hand_percentage:.1f}%)")
    print(f"Ground truth NOT in hand: {ground_truth_not_in_hand} ({not_in_hand_percentage:.1f}%)")
else:
    print("No ground truth checks performed")