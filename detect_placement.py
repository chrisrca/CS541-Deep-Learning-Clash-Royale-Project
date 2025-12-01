import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


class PlacementDetector:    
    def __init__(self, template_path: Optional[str] = None, threshold: float = 0.7):
        self.threshold = threshold
        self.template = None
        self.template_w = None
        self.template_h = None
        
        if template_path and Path(template_path).exists():
            self.load_template(template_path)
    
    def load_template(self, template_path: str):
        """Load the clock template image."""
        self.template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        self.template_h, self.template_w = self.template.shape
        print(f"Loaded template: {self.template_w}x{self.template_h}")
    
    def create_template_from_roi(self, image_path: str, x: int, y: int, w: int, h: int, 
                                 save_path: str = "clock_template.png"):
        """
        extract a clock region from an image to use as template
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Extract ROI
        roi = img[y:y+h, x:x+w]
        
        # Convert to grayscale
        self.template = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        self.template_h, self.template_w = self.template.shape
        cv2.imwrite(save_path, self.template)
        print(f"Created and saved template to {save_path}")
        print(f"Template size: {self.template_w}x{self.template_h}")
        
        return self.template
    
    def load_multiple_templates(self, template_dir: str):
        template_path = Path(template_dir)
        self.templates = []
        
        for img_path in sorted(template_path.glob("clock_*.png")):
            template = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if template is not None:
                self.templates.append({
                    'image': template,
                    'width': template.shape[1],
                    'height': template.shape[0],
                    'name': img_path.name
                })
        print(f"Loaded {len(self.templates)} clock templates:")
        for t in self.templates:
            print(f"  - {t['name']} ({t['width']}x{t['height']})")
        self.template = self.templates[0]['image']
        self.template_w = self.templates[0]['width']
        self.template_h = self.templates[0]['height']
    
    def detect_placements(self, img, 
                         method: int = cv2.TM_CCOEFF_NORMED,
                         multi_scale: bool = False,
                         scales: List[float] = None,
                         use_all_templates: bool = True,
                         return_template_id: bool = False) -> List[Tuple]:
        """
        Detect placement locations in an image.
        
        Args:
            image_path: Path to the game frame image
            method: OpenCV template matching method
            multi_scale: Whether to try multiple scales (usually not needed at fixed FPS)
            scales: List of scales to try (default: [0.9, 1.0, 1.1])
            use_all_templates: If True and multiple templates loaded, tries all of them
            return_template_id: If True, returns template ID that matched
        
        Returns:
            List of (x, y, confidence) or (x, y, confidence, template_id) tuples
        """
        # Check if we have templates
        has_multiple = hasattr(self, 'templates') and len(self.templates) > 0
        
        if not has_multiple and self.template is None:
            raise ValueError("No template loaded. Call load_template() or load_multiple_templates() first.")
        
        # Load image
        if img is None:
            raise ValueError(f"Could not load image")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # For 10fps with 1-second clock, multi-scale usually not needed
        if scales is None:
            scales = [0.9, 1.0, 1.1] if multi_scale else [1.0]
        
        all_detections = []
        
        # Determine which templates to use
        if has_multiple and use_all_templates:
            templates_to_try = self.templates
        else:
            # Single template mode
            templates_to_try = [{
                'image': self.template,
                'width': self.template_w,
                'height': self.template_h,
                'name': 'single'
            }]
        
        # Try each template
        for template_info in templates_to_try:
            template_img = template_info['image']
            template_w = template_info['width']
            template_h = template_info['height']
            
            # Try multiple scales for this template
            for scale in scales:
                # Resize template
                if scale != 1.0:
                    w = int(template_w * scale)
                    h = int(template_h * scale)
                    template_scaled = cv2.resize(template_img, (w, h))
                else:
                    template_scaled = template_img
                    w, h = template_w, template_h
                
                # Skip if template is larger than image
                if w > gray.shape[1] or h > gray.shape[0]:
                    continue
                
                # Perform template matching
                result = cv2.matchTemplate(gray, template_scaled, method)
                
                # Find locations above threshold
                locations = np.where(result >= self.threshold)
                
                # Store detections with their confidence scores
                for pt in zip(*locations[::-1]):  # Switch x and y
                    confidence = result[pt[1], pt[0]]
                    # Store center point instead of top-left
                    center_x = pt[0] + w // 2
                    center_y = pt[1] + h // 2
                    template_id = template_info['name'] if return_template_id else None
                    all_detections.append((center_x, center_y, confidence, scale, template_id))
        
        # Apply Non-Maximum Suppression to remove duplicate detections
        detections = self._non_max_suppression(all_detections, overlap_threshold=30, return_template_id=return_template_id)
        
        return detections
    
    def _non_max_suppression(self, detections: List[Tuple], 
                            overlap_threshold: int = 30,
                            return_template_id: bool = False) -> List[Tuple]:
        """
        Remove overlapping detections, keeping only the one with highest confidence.
        
        Args:
            detections: List of (x, y, confidence, scale, template_id) or (x, y, confidence, scale)
            overlap_threshold: Maximum distance (pixels) between detections to be considered duplicates
            return_template_id: Whether to include template_id in output
        
        Returns:
            Filtered list of (x, y, confidence) or (x, y, confidence, template_id)
        """
        if not detections:
            return []
        
        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda x: x[2], reverse=True)
        
        kept = []
        
        for det in detections:
            x, y, conf, scale = det[:4]
            template_id = det[4] if len(det) > 4 else 'unknown'
            
            # Check if this detection overlaps with any kept detection
            is_duplicate = False
            for kept_det in kept:
                kx, ky = kept_det[0], kept_det[1]
                distance = np.sqrt((x - kx)**2 + (y - ky)**2)
                
                if distance < overlap_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                if return_template_id:
                    kept.append((x, y, conf, template_id))
                else:
                    kept.append((x, y, conf))
        
        return kept
    
    def visualize_detections(self, image_path: str, detections: List[Tuple[int, int, float]], 
                            output_path: str = None, show: bool = False):
        """
        Visualize detected placements on the image.
        """
        img = cv2.imread(image_path)
        
        for x, y, conf in detections:
            # Draw circle at placement location
            cv2.circle(img, (x, y), 10, (0, 255, 0), 2)
            # Draw crosshair
            cv2.line(img, (x-15, y), (x+15, y), (0, 255, 0), 2)
            cv2.line(img, (x, y-15), (x, y+15), (0, 255, 0), 2)
            # Add confidence text
            cv2.putText(img, f"{conf:.2f}", (x+15, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if output_path:
            cv2.imwrite(output_path, img)
            print(f"Saved visualization to {output_path}")
        
        if show:
            cv2.imshow("Placement Detections", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return img

