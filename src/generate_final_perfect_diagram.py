import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, RegularPolygon
import numpy as np

# Ultra-clean setup
plt.style.use('default')
fig, ax = plt.subplots(figsize=(18, 12), dpi=150)
ax.set_facecolor('#FFFFFF')
fig.patch.set_facecolor('#FFFFFF')

# Professional LinkedIn colors
colors = {
    'primary': '#0A66C2',
    'secondary': '#378FE9', 
    'success': '#057642',
    'warning': '#F5B800',
    'accent': '#7B68EE'
}

def create_perfect_box(ax, center, width, height, title, subtitle, color):
    """Create perfectly spaced professional box with BIGGER TEXT"""
    x, y = center
    
    # Subtle shadow
    shadow = FancyBboxPatch(
        (x - width/2 + 0.03, y - height/2 - 0.03), width, height,
        boxstyle="round,pad=0.1", 
        facecolor='#00000015',
        edgecolor='none'
    )
    ax.add_patch(shadow)
    
    # Main box
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.1", 
        facecolor=color, 
        edgecolor='white',
        linewidth=2
    )
    ax.add_patch(box)
    
    # BIGGER Title text - increased from 11 to 14
    ax.text(x, y + 0.15, title, ha='center', va='center', 
           fontsize=14, weight='bold', color='white')
    
    # BIGGER Subtitle text - increased from 9 to 12
    if subtitle:
        ax.text(x, y - 0.15, subtitle, ha='center', va='center', 
               fontsize=12, color='white', alpha=0.9)
    
    # Return box boundaries for precise connection
    return {
        'left': x - width/2,
        'right': x + width/2, 
        'top': y + height/2,
        'bottom': y - height/2,
        'center_x': x,
        'center_y': y
    }

def _edge_point(box, edge):
    if edge == 'left':
        return box['left'], box['center_y']
    if edge == 'right':
        return box['right'], box['center_y']
    if edge == 'top':
        return box['center_x'], box['top']
    if edge == 'bottom':
        return box['center_x'], box['bottom']
    # default center
    return box['center_x'], box['center_y']


def create_consistent_arrow(ax, start_box, end_box, color, start_edge=None, end_edge=None, offset=0):
    """Create CONSISTENT arrows with proper color coding.
    Allows explicit control of start/end edges to avoid wrong-side connections.
    Offset parameter allows for parallel arrows that don't overlap."""
    
    # Determine connection direction
    dx = end_box['center_x'] - start_box['center_x']
    dy = end_box['center_y'] - start_box['center_y']
    
    # Calculate precise connection points (with optional edge overrides)
    if start_edge is not None and end_edge is not None:
        start_x, start_y = _edge_point(start_box, start_edge)
        end_x, end_y = _edge_point(end_box, end_edge)
    else:
        if abs(dx) > abs(dy):  # Horizontal connection
            if dx > 0:  # Left to right
                start_x = start_box['right']
                start_y = start_box['center_y']
                end_x = end_box['left']
                end_y = end_box['center_y']
            else:  # Right to left
                start_x = start_box['left']
                start_y = start_box['center_y']
                end_x = end_box['right']
                end_y = end_box['center_y']
        else:  # Vertical connection
            if dy > 0:  # Bottom to top
                start_x = start_box['center_x']
                start_y = start_box['top']
                end_x = end_box['center_x']
                end_y = end_box['bottom']
            else:  # Top to bottom
                start_x = start_box['center_x']
                start_y = start_box['bottom']
                end_x = end_box['center_x']
                end_y = end_box['top']
    
    # Apply offset for parallel arrows
    if offset != 0:
        # Calculate perpendicular direction for offset
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            perp_x = -dy / length * offset
            perp_y = dx / length * offset
            start_x += perp_x
            start_y += perp_y
            end_x += perp_x
            end_y += perp_y
    
    # CONSISTENT main arrow line with PROPER COLOR AND VISIBLE ARROWHEADS
    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(
                    arrowstyle='->', 
                    lw=4,  # Slightly thicker for better visibility
                    color=color,
                    shrinkA=10, shrinkB=10,  # Shrink arrows so they just touch the boxes
                    mutation_scale=20,  # Slightly smaller arrowhead
                    connectionstyle="arc3,rad=0"  # Force STRAIGHT arrows, no curves
                ))
    
    # CONSISTENT vector flow diamonds with SAME COLOR as arrow
    num_diamonds = 3
    diamond_size = 0.08  # Slightly bigger diamonds
    
    for i in range(num_diamonds):
        t = (i + 1) / (num_diamonds + 1)
        diamond_x = start_x + t * (end_x - start_x)
        diamond_y = start_y + t * (end_y - start_y)
        
        diamond = RegularPolygon(
            (diamond_x, diamond_y), 4, radius=diamond_size,
            orientation=np.pi/4, facecolor=color, 
            edgecolor='white', linewidth=1, alpha=0.8
        )
        ax.add_patch(diamond)

# Define components with perfect spacing
box_width = 3.2  # Slightly wider boxes for bigger text
box_height = 1.2  # Slightly taller boxes for bigger text

components = [
    # Top row - Sequential document processing
    ((3, 9), "Hebrew Medical PDF", "Raw scanned document", colors['warning']),
    ((7.5, 9), "OCR Processing", "Text extraction", colors['primary']),
    ((12, 9), "Text Processing", "Unicode + BiDi cleanup", colors['secondary']),
    
    # Middle row - Text intelligence
    ((3, 6.5), "Smart Chunking", "Document segmentation", colors['accent']),
    ((7.5, 6.5), "Embedding Model", "Vector generation", colors['success']),
    ((12, 6.5), "Vector Database", "Similarity indexing", colors['primary']),
    
    # Bottom row - Query processing
    ((3, 4), "Hebrew Query", "User question", colors['warning']),
    ((7.5, 4), "Similarity Search", "Context retrieval", colors['accent']),
    ((12, 4), "LLM Generation", "Final answer", colors['success'])
]

# Create all boxes
boxes = []
for center, title, subtitle, color in components:
    box = create_perfect_box(ax, center, box_width, box_height, title, subtitle, color)
    boxes.append(box)

# Define ALL connections with PROPER COLOR CODING
connections = [
    # Document processing pipeline
    (0, 1, colors['warning'], 'right', 'left'),     # PDF → OCR (YELLOW)
    (1, 2, colors['primary'], 'right', 'left'),     # OCR → Text Processing (BLUE)
    
    # Text intelligence pipeline - FIXED COLORS AS REQUESTED
    (2, 3, colors['secondary'], 'bottom', 'top'),   # Text → Chunking (SAME BLUE AS TEXT PROCESSING BOX!)
    (3, 4, colors['accent'], 'right', 'left'),      # Chunking → Embedding (PURPLE as requested!)
    (4, 5, colors['success'], 'right', 'left'),     # Embedding → Vector DB (GREEN as requested!)
    
    # Query processing pipeline
    (6, 7, colors['warning'], 'right', 'left'),     # Query → Search (YELLOW)
    
    # Context retrieval - STRAIGHT BIDIRECTIONAL ARROWS
    (5, 7, colors['primary'], 'bottom', 'top'),           # Vector DB → Search (BLUE, straight vertical)
    
    # BIDIRECTIONAL: Purple arrow going back - straight and separate
    (7, 5, colors['accent'], 'top', 'bottom'),            # Search → Vector DB (PURPLE, straight vertical back)
    
    # Final generation
    (7, 8, colors['success'], 'right', 'left'),     # Search → Generation (GREEN)
]

# Draw ALL connections with PERFECT COLOR CODING
for conn in connections:
    if len(conn) == 6:  # With offset
        start_idx, end_idx, color, start_edge, end_edge, offset = conn
        create_consistent_arrow(ax, boxes[start_idx], boxes[end_idx], color, start_edge, end_edge, offset)
    elif len(conn) == 5:  # Without offset
        start_idx, end_idx, color, start_edge, end_edge = conn
        create_consistent_arrow(ax, boxes[start_idx], boxes[end_idx], color, start_edge, end_edge)
    else:  # Basic connection
        start_idx, end_idx, color = conn
        create_consistent_arrow(ax, boxes[start_idx], boxes[end_idx], color)

# BIGGER Clean title - increased font sizes
ax.text(7.5, 10.7, "Hebrew Medical RAG Pipeline", 
        ha='center', va='center', fontsize=26, weight='bold',  # Increased from 22 to 26
        color=colors['primary'])

ax.text(7.5, 10.2, "Architecture Flow", 
        ha='center', va='center', fontsize=18, style='italic',  
        color='#666666')

# BIGGER text for pipeline order - increased font size
flow_text = """RAG Pipeline Order:
1. PDF → OCR (extract text from images)
2. OCR → Text Processing (clean Hebrew text)  
3. Text Processing → Chunking (segment clean text)
4. Chunking → Embedding (vectorize segments)
5. Embedding → Vector DB (store for search)"""

ax.text(0.5, 2, flow_text, fontsize=12,  # Increased from 10 to 12
        bbox=dict(boxstyle="round,pad=0.5", 
                 facecolor='#F8F9FA', 
                 edgecolor=colors['primary'],
                 linewidth=1),
        verticalalignment='top')

# Perfect boundaries
ax.set_xlim(0, 15)
ax.set_ylim(1, 11.5)
ax.axis('off')

plt.tight_layout()

# Save PERFECT final diagram
output_path = "hebrew_rag_pipeline_perfect.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
print(f"PERFECT final diagram with BIGGER text and ALL connections saved to {output_path}")
