import random
import time
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, Arrow
import matplotlib.animation as animation

class Bead:
    def __init__(self, color: str):
        self.color = color
        
    def get_color_rgb(self):
        color_map = {
            'red': '#FF0000',
            'green': '#00FF00',
            'blue': '#0000FF',
            None: '#EEEEEE'
        }
        return color_map.get(self.color, '#EEEEEE')

class ConveyorBelt:
    def __init__(self, rows: int = 10, columns: int = 5):
        self.rows = rows
        self.columns = columns
        self.belt = [[Bead(None) for _ in range(columns)] for _ in range(rows)]
        self.ejector_states = [[0 for _ in range(columns)] for _ in range(3)]
        self.buckets = {'red': 0, 'green': 0, 'blue': 0}
        self.ejector_commands = []
        
        # Setup matplotlib figure
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title('Conveyor Belt Simulation')
        
    def generate_new_row(self) -> List[Bead]:
        colors = ['red', 'green', 'blue', None]
        return [Bead(random.choice(colors)) for _ in range(self.columns)]
    
    def add_ejector_commands(self, new_row: List[Bead]):
        ejector_rows = {'red': 7, 'green': 8, 'blue': 9}
        for pos, bead in enumerate(new_row):
            if bead and bead.color:
                steps = ejector_rows[bead.color]
                self.ejector_commands.append((bead.color, pos, steps))
    
    def update_ejector_states(self):
        self.ejector_states = [[0 for _ in range(self.columns)] for _ in range(3)]
        remaining_commands = []
        for color, pos, steps in self.ejector_commands:
            if steps == 1:
                ejector_row = 0 if color == 'red' else (1 if color == 'green' else 2)
                self.ejector_states[ejector_row][pos] = 1
            else:
                remaining_commands.append((color, pos, steps - 1))
        self.ejector_commands = remaining_commands
    
    def move_belt(self):
        self.update_ejector_states()
        new_row = self.generate_new_row()
        self.add_ejector_commands(new_row)
        
        for i in reversed(range(1, self.rows)):
            for j in range(self.columns):
                bead = self.belt[i-1][j]
                
                if bead and (
                    (i == 7 and bead.color == 'red' and self.ejector_states[0][j]) or
                    (i == 8 and bead.color == 'green' and self.ejector_states[1][j]) or
                    (i == 9 and bead.color == 'blue' and self.ejector_states[2][j])
                ):
                    self.buckets[bead.color] += 1
                    self.belt[i][j] = Bead(None)
                else:
                    self.belt[i][j] = bead
        
        self.belt[0] = new_row
        self.display()
    
    def display(self):
        self.ax.clear()
        
        # Set plot limits and title
        self.ax.set_xlim(-2, self.columns + 4)
        self.ax.set_ylim(-3, self.rows + 1)
        self.ax.set_title('Conveyor Belt Simulation', pad=20)
        
        # Draw conveyor belt background
        belt_bg = Rectangle((-0.5, -0.5), self.columns + 1, self.rows + 1,
                          facecolor='#DDDDDD', edgecolor='#888888')
        self.ax.add_patch(belt_bg)
        
        # Draw beads
        for i, row in enumerate(self.belt):
            for j, bead in enumerate(row):
                circle = Circle((j, self.rows - 1 - i), 0.4, 
                              color=bead.get_color_rgb(),
                              edgecolor='black')
                self.ax.add_patch(circle)
        
        # Draw ejectors
        ejector_colors = ['red', 'green', 'blue']
        for i, (states, color) in enumerate(zip(self.ejector_states, ejector_colors)):
            row = 7 + i
            for j, state in enumerate(states):
                # Draw ejector base
                ejector = Rectangle((j - 0.2, self.rows - 1 - row - 0.2),
                                  0.4, 0.4, color=color, alpha=0.3)
                self.ax.add_patch(ejector)
                
                # Draw activation arrow if ejector is active
                if state:
                    arrow = Arrow(j, self.rows - 1 - row,
                                0.5, 0, width=0.3, color=color)
                    self.ax.add_patch(arrow)
        
        # Draw buckets
        bucket_positions = {'red': 0, 'green': 1, 'blue': 2}
        for color, count in self.buckets.items():
            pos = bucket_positions[color]
            # Draw bucket
            bucket = Rectangle((self.columns + 1, pos * 2 - 2),
                             2, 1.5, color=color, alpha=0.3)
            self.ax.add_patch(bucket)
            # Add count text
            self.ax.text(self.columns + 2, pos * 2 - 1.25,
                        str(count), ha='center', va='center')
        
        # Remove axes
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        plt.draw()
        plt.pause(0.1)

def main():
    belt = ConveyorBelt()
    belt.display()
    
    # Mode selection
    while True:
        mode = input("Select mode (1 for manual, 2 for automatic): ").strip()
        if mode in ['1', '2']:
            break
        print("Invalid input. Please enter 1 or 2.")

    if mode == '1':
        # Manual mode
        while True:
            user_input = input("Press Enter to advance the belt, or 'q' to quit: ")
            if user_input.lower() == 'q':
                break
            belt.move_belt()
    else:
        # Automatic mode
        while True:
            try:
                speed = float(input("Enter speed (rows per minute): "))
                if speed > 0:
                    break
                print("Speed must be positive.")
            except ValueError:
                print("Please enter a valid number.")
        
        delay = 60.0 / speed  # Convert speed to delay in seconds
        print("Belt running automatically. Press Ctrl+C to stop...")
        
        try:
            while True:
                belt.move_belt()
                time.sleep(delay)
        except KeyboardInterrupt:
            print("\nStopping simulation...")
    
    plt.ioff()
    plt.close()

if __name__ == "__main__":
    main()