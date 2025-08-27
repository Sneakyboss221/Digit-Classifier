"""
Interactive drawing canvas for digit input
Based on the implementation from xEC40/MNIST-ResNet-digit-recognizer
"""

import pygame
import numpy as np
import torch
from data.data_loader import Preprocessor

class DrawingCanvas:
    """Interactive drawing canvas for digit input"""
    
    def __init__(self, width=280, height=280, brush_size=15, bg_color=(255, 255, 255), 
                 brush_color=(0, 0, 0)):
        """
        Initialize drawing canvas
        
        Args:
            width: Canvas width in pixels
            height: Canvas height in pixels
            brush_size: Size of the drawing brush
            bg_color: Background color (RGB)
            brush_color: Brush color (RGB)
        """
        self.width = width
        self.height = height
        self.brush_size = brush_size
        self.bg_color = bg_color
        self.brush_color = brush_color
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("MNIST Digit Drawing Canvas")
        
        # Create drawing surface
        self.surface = pygame.Surface((width, height))
        self.surface.fill(bg_color)
        
        # Drawing state
        self.drawing = False
        self.last_pos = None
        
        # Store drawing as numpy array
        self.drawing_array = np.zeros((height, width), dtype=np.uint8)
    
    def clear(self):
        """Clear the canvas"""
        self.surface.fill(self.bg_color)
        self.drawing_array.fill(0)
        pygame.display.flip()
    
    def draw_circle(self, pos, size):
        """Draw a circle at the given position"""
        pygame.draw.circle(self.surface, self.brush_color, pos, size)
        
        # Update numpy array
        x, y = pos
        y_start = max(0, y - size)
        y_end = min(self.height, y + size + 1)
        x_start = max(0, x - size)
        x_end = min(self.width, x + size + 1)
        
        for dy in range(y_start, y_end):
            for dx in range(x_start, x_end):
                if (dx - x) ** 2 + (dy - y) ** 2 <= size ** 2:
                    self.drawing_array[dy, dx] = 255
    
    def draw_line(self, start_pos, end_pos):
        """Draw a line between two points"""
        pygame.draw.line(self.surface, self.brush_color, start_pos, end_pos, self.brush_size)
        
        # Update numpy array by drawing circles along the line
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        # Bresenham's line algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        n = 1 + dx + dy
        x_inc = 1 if x2 > x1 else -1
        y_inc = 1 if y2 > y1 else -1
        error = dx - dy
        dx *= 2
        dy *= 2
        
        for _ in range(n):
            self.draw_circle((x, y), self.brush_size // 2)
            
            if x == x2 and y == y2:
                break
            
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    self.drawing = True
                    self.last_pos = event.pos
                    self.draw_circle(event.pos, self.brush_size // 2)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    self.drawing = False
                    self.last_pos = None
            
            elif event.type == pygame.MOUSEMOTION:
                if self.drawing and self.last_pos:
                    self.draw_line(self.last_pos, event.pos)
                    self.last_pos = event.pos
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:  # Clear canvas
                    self.clear()
                elif event.key == pygame.K_ESCAPE:  # Quit
                    return False
        
        pygame.display.flip()
        return True
    
    def get_drawing_array(self):
        """Get the current drawing as a numpy array"""
        return self.drawing_array.copy()
    
    def preprocess_for_prediction(self):
        """Preprocess the drawing for model prediction"""
        return Preprocessor.preprocess_drawn_image(self.drawing_array)
    
    def save_drawing(self, filename):
        """Save the drawing as an image"""
        pygame.image.save(self.surface, filename)
    
    def run(self):
        """Run the drawing canvas"""
        running = True
        while running:
            running = self.handle_events()
        
        pygame.quit()

class StreamlitDrawingCanvas:
    """Streamlit-compatible drawing canvas"""
    
    def __init__(self, width=280, height=280):
        self.width = width
        self.height = height
        self.drawing_array = np.zeros((height, width), dtype=np.uint8)
    
    def create_canvas(self):
        """Create a Streamlit canvas for drawing"""
        import streamlit as st
        
        # Create canvas using streamlit-canvas
        try:
            from streamlit_canvas import st_canvas
            
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=15,
                stroke_color="black",
                background_color="white",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
            
            if canvas_result.image_data is not None:
                # Convert to grayscale
                image_data = canvas_result.image_data
                if len(image_data.shape) == 3:
                    # Convert RGB to grayscale
                    gray = np.dot(image_data[..., :3], [0.299, 0.587, 0.114])
                    self.drawing_array = gray.astype(np.uint8)
                else:
                    self.drawing_array = image_data.astype(np.uint8)
            
            return canvas_result
            
        except ImportError:
            st.error("streamlit-canvas not installed. Please install it with: pip install streamlit-canvas")
            return None
    
    def get_drawing_array(self):
        """Get the current drawing as a numpy array"""
        return self.drawing_array.copy()
    
    def preprocess_for_prediction(self):
        """Preprocess the drawing for model prediction"""
        return Preprocessor.preprocess_drawn_image(self.drawing_array)
    
    def clear(self):
        """Clear the drawing"""
        self.drawing_array.fill(0)

class GradioDrawingCanvas:
    """Gradio-compatible drawing canvas"""
    
    def __init__(self, width=280, height=280):
        self.width = width
        self.height = height
    
    def create_interface(self, predict_function):
        """Create a Gradio interface for drawing and prediction"""
        import gradio as gr
        
        def predict_digit(image):
            if image is None:
                return "Please draw a digit"
            
            # Convert image to numpy array
            if hasattr(image, 'shape'):
                # Already numpy array
                drawing_array = image
            else:
                # PIL Image or other format
                import cv2
                drawing_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # Preprocess for prediction
            try:
                preprocessed = Preprocessor.preprocess_drawn_image(drawing_array)
                prediction = predict_function(preprocessed)
                return f"Predicted digit: {prediction}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Create Gradio interface
        interface = gr.Interface(
            fn=predict_digit,
            inputs=gr.Sketchpad(shape=(28, 28), brush_radius=2),
            outputs=gr.Textbox(label="Prediction"),
            title="MNIST Digit Recognition",
            description="Draw a digit and get the prediction",
            examples=[],
            cache_examples=False
        )
        
        return interface

def test_drawing_canvas():
    """Test the drawing canvas"""
    canvas = DrawingCanvas()
    print("Drawing canvas opened. Draw a digit and press ESC to quit.")
    print("Press 'C' to clear the canvas.")
    canvas.run()

if __name__ == "__main__":
    test_drawing_canvas()
