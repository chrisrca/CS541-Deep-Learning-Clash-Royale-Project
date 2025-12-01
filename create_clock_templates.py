import cv2
import numpy as np
from pathlib import Path
import json


def find_frames_with_clocks(frames_dir: str, num_samples: int = 20) -> list:
    frames_path = Path(frames_dir)
    all_frames = sorted(frames_path.glob("*.png"))
    
    candidates = []
    
    print(f"Scanning {len(all_frames)} frames for clocks...")
    
    for i, frame_path in enumerate(all_frames):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(all_frames)}")
        
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blue_lower = np.array([100, 100, 100])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        # Check for small circular blue regions
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Clock-sized regions (200-2000 pixels)
            if 200 < area < 2000:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    # Fairly circular
                    if circularity > 0.5:
                        candidates.append({
                            'path': frame_path,
                            'area': area,
                            'circularity': circularity,
                            'contour': contour
                        })
                        break  # One clock per frame is enough
    
    print(f"\nFound {len(candidates)} frames with potential clocks")
    
    # Sample evenly from candidates
    if len(candidates) > num_samples:
        step = len(candidates) // num_samples
        sampled = [candidates[i * step] for i in range(num_samples)]
    else:
        sampled = candidates
    
    return sampled


def extract_templates(candidates: list, output_dir: str, target_count: int = 30):
    """
    Interactively extract clock templates from candidate frames.
    
    Args:
        candidates: List of dicts with 'path' and 'contour' keys
        output_dir: Directory to save templates
        target_count: Target number of templates to create
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check for existing templates
    existing_templates = sorted(output_path.glob("clock_template_*.png"))
    template_count = len(existing_templates)
    
    if template_count > 0:
        print(f"\nFound {template_count} existing templates in {output_dir}")
        response = input(f"Continue adding more? (y/n): ").strip().lower()
        if response != 'y':
            return template_count
    
    print("\n" + "="*70)
    print("INTERACTIVE CLOCK TEMPLATE EXTRACTION")
    print("="*70)
    print(f"\nTarget: {target_count} templates (current: {template_count})")
    print("\nInstructions:")
    print("  - A frame with a potential clock will be shown")
    print("  - Click and drag to select the clock region")
    print("  - Press 's' to save this template")
    print("  - Press 'n' to skip to next frame")
    print("  - Press 'q' to quit (you can resume later)")
    print("\nTip: Select clocks at different animation stages")
    print("="*70 + "\n")
    
    for idx, candidate in enumerate(candidates):
        frame_path = candidate['path']
        img = cv2.imread(str(frame_path))
        
        if img is None:
            continue
        
        # Draw a hint circle around detected region
        contour = candidate.get('contour')
        if contour is not None:
            hint_img = img.copy()
            cv2.drawContours(hint_img, [contour], -1, (0, 255, 0), 2)
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(hint_img, "Clock here?", (cx-40, cy-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            hint_img = img
        
        clone = img.copy()
        roi_selected = False
        start_point = None
        end_point = None
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal start_point, end_point, roi_selected, hint_img
            
            if event == cv2.EVENT_LBUTTONDOWN:
                start_point = (x, y)
                roi_selected = False
            
            elif event == cv2.EVENT_MOUSEMOVE and start_point:
                hint_img = clone.copy()
                cv2.rectangle(hint_img, start_point, (x, y), (0, 255, 255), 2)
            
            elif event == cv2.EVENT_LBUTTONUP:
                end_point = (x, y)
                roi_selected = True
                cv2.rectangle(hint_img, start_point, end_point, (0, 255, 0), 2)
        
        window_name = f"Frame {idx+1}/{len(candidates)} - {frame_path.name}"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        while True:
            cv2.imshow(window_name, hint_img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') and roi_selected and start_point and end_point:
                # Save template
                x1 = min(start_point[0], end_point[0])
                y1 = min(start_point[1], end_point[1])
                x2 = max(start_point[0], end_point[0])
                y2 = max(start_point[1], end_point[1])
                
                # Add padding for better matching
                padding = 5
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(img.shape[1], x2 + padding)
                y2 = min(img.shape[0], y2 + padding)
                
                roi = clone[y1:y2, x1:x2]
                
                if roi.size > 0:
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    template_count += 1
                    template_path = output_path / f"clock_template_{template_count:02d}.png"
                    cv2.imwrite(str(template_path), gray_roi)
                    
                    print(f"✓ Saved template {template_count}: {template_path.name} ({x2-x1}x{y2-y1})")
                    
                    # Save metadata
                    metadata = {
                        'template_id': template_count,
                        'source_frame': str(frame_path),
                        'region': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                        'size': {'width': x2-x1, 'height': y2-y1}
                    }
                    
                    meta_path = output_path / f"clock_template_{template_count:02d}.json"
                    with open(meta_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                
                break
            
            elif key == ord('n'):
                print(f"  Skipped frame {idx+1}")
                break
            
            elif key == ord('q'):
                print(f"\nQuitting. Total templates: {template_count}")
                cv2.destroyAllWindows()
                return template_count
        
        cv2.destroyAllWindows()
        
        # Check if we've reached target
        if template_count >= target_count:
            print(f"\n✓ Target reached: {template_count} templates created!")
            break
    
    print(f"\n{'='*70}")
    print(f"COMPLETED: Total {template_count} clock templates")
    print(f"Templates saved to: {output_path}")
    if template_count < target_count:
        print(f"Note: Created {template_count}/{target_count} templates")
        print(f"      You can run again to add more")
    print(f"{'='*70}")
    
    return template_count

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create clock templates for placement detection (10fps optimized)"
    )
    parser.add_argument("frames_dir", help="Directory containing game frames")
    parser.add_argument("--output", "-o", default="clock_templates",
                       help="output directory for templates (default: clock_templates)")
    parser.add_argument("--samples", "-s", type=int, default=50,
                       help="number of frames to check")
    parser.add_argument("--target", "-t", type=int, default=30,
                       help="number of templates to create")
    
    args = parser.parse_args()
    print(f"Frames directory: {args.frames_dir}")
    print(f"Output directory: {args.output}")
    print(f"Target templates: {args.target}")
    print("="*70 + "\n")
    # Find frames
    candidates = find_frames_with_clocks(args.frames_dir, args.samples)    
    if not candidates:
        print("No potential clocks found. Try adjusting detection parameters.")
        return
    count = extract_templates(candidates, args.output, target_count=args.target)
        
    print(f"{count} templates created")

if __name__ == "__main__":
    main()
