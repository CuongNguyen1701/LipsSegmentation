import numpy as np
def hex_to_rgb(hex_color):
    """Convert hex color to RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

def create_colormap(color1, color2):
    """Create a colormap for cv2.applyColorMap from two hex RGB values(BGR value)."""
    # Convert hex to RGB
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)
    
    # Generate gradient
    gradient = np.linspace(rgb1, rgb2, 256)
    
    # Convert to uint8
    colormap = gradient.astype(np.uint8)
    
    # Reshape to 256x1x3
    colormap = colormap.reshape(256, 1, 3)
    
    return colormap