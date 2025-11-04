#!/usr/bin/env python3
"""
Create synthetic rice grain images for testing
"""

import cv2
import numpy as np
import random

def draw_rice_grain(img, x, y, grain_type='white', angle=0):
    """Draw a rice grain shape at given position"""
    # Rice grain dimensions (approximate)
    length = random.randint(25, 35)
    width = random.randint(8, 12)
    
    # Colors for different rice types
    colors = {
        'white': (240, 240, 240),
        'brown': (139, 90, 43),
        'black': (40, 40, 40),
        'thai': (255, 235, 180)
    }
    
    color = colors.get(grain_type, (200, 200, 200))
    
    # Create ellipse for rice grain
    axes = (length, width)
    cv2.ellipse(img, (x, y), axes, angle, 0, 360, color, -1)
    
    # Add slight border for realism
    border_color = tuple(int(c * 0.8) for c in color)
    cv2.ellipse(img, (x, y), axes, angle, 0, 360, border_color, 1)
    
    return img

def create_rice_test_image():
    """Create a test image with various rice grains"""
    # Create wooden background texture
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 220
    
    # Add wood grain texture
    for i in range(0, 720, 3):
        color_variation = random.randint(-10, 10)
        cv2.line(img, (0, i), (1280, i), 
                (210 + color_variation, 200 + color_variation, 190 + color_variation), 1)
    
    # Add rice grains
    rice_types = ['white', 'brown', 'black', 'thai']
    
    # Place rice grains in a grid pattern with some randomness
    for row in range(3, 10):
        for col in range(5, 20):
            if random.random() > 0.3:  # 70% chance to place a grain
                x = col * 60 + random.randint(-20, 20)
                y = row * 60 + random.randint(-20, 20)
                
                # Ensure within bounds
                if 50 < x < 1230 and 50 < y < 670:
                    grain_type = random.choice(rice_types)
                    angle = random.randint(0, 180)
                    draw_rice_grain(img, x, y, grain_type, angle)
    
    # Save the test image
    cv2.imwrite('test_rice_image.jpg', img)
    print("Created test_rice_image.jpg")
    
    # Also create a simpler test with fewer grains for easier debugging
    img_simple = np.ones((640, 640, 3), dtype=np.uint8) * 220
    
    # Add wood texture
    for i in range(0, 640, 3):
        color_variation = random.randint(-10, 10)
        cv2.line(img_simple, (0, i), (640, i), 
                (210 + color_variation, 200 + color_variation, 190 + color_variation), 1)
    
    # Place specific grains at known positions
    test_positions = [
        (100, 100, 'white', 45),
        (200, 100, 'brown', 90),
        (300, 100, 'black', 135),
        (400, 100, 'thai', 0),
        (100, 300, 'white', 30),
        (200, 300, 'brown', 60),
        (300, 300, 'black', 120),
        (400, 300, 'thai', 150),
        (100, 500, 'white', 0),
        (200, 500, 'brown', 45),
        (300, 500, 'black', 90),
        (400, 500, 'thai', 135),
    ]
    
    for x, y, grain_type, angle in test_positions:
        draw_rice_grain(img_simple, x, y, grain_type, angle)
    
    cv2.imwrite('test_rice_simple.jpg', img_simple)
    print("Created test_rice_simple.jpg")
    
    return 'test_rice_image.jpg', 'test_rice_simple.jpg'

if __name__ == "__main__":
    create_rice_test_image()
    print("\nTest images created successfully!")
    print("- test_rice_image.jpg: Complex scene with many rice grains")
    print("- test_rice_simple.jpg: Simple grid pattern for debugging")