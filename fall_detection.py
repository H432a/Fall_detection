import numpy as np

# Keep track of last known positions of humans to calculate movement
previous_positions = {}

def detect_fall(humans):
    global previous_positions

    is_fall = False
    for bbox in humans:
        x1, y1, x2, y2 = map(int, bbox[:4])
        height = y2 - y1
        width = x2 - x1
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # Get previous center position if it exists
        previous_center = previous_positions.get(tuple(center), None)

        if previous_center:
            speed = np.linalg.norm(np.array(center) - np.array(previous_center))
            
            # Heuristic: if speed is high and width > height (person is horizontal), assume fall
            if width > height and speed > 15:  # Tweak this threshold based on testing
                is_fall = True
        
        # Update the position of the person
        previous_positions[tuple(center)] = center

    return is_fall
