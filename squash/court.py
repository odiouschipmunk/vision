class court:
    def __init__(self, width, depth, height):
        self.width = width
        self.depth = depth
        self.height = height
    
    def is_in_court(self, position):
        x, y, z = position
        return 0 <= x <= self.width and 0 <= y <= self.depth and 0 <= z <= self.height
    
    def get_zone(self, position):
        x, y, z = position
        if y < self.depth / 3:
            return 'Front'
        elif y < 2 * self.depth / 3:
            return 'Middle'
        else:
            return 'Back'
