from PyQt5.QtGui import QColor, QPen, QBrush
from pyqtgraph import GraphicsObject

class CircleItem(GraphicsObject):
    def __init__(self, pos, radius, pen=None, brush=None):
        super().__init__()

        self.pos = pos
        self.radius = radius
        self.pen = pen or QPen()
        self.brush = brush or QBrush()

    def boundingRect(self):
        br = self.radius + self.pen.widthF() / 2
        return self.pos[0] - br, self.pos[1] - br, 2 * br, 2 * br

    def paint(self, painter, option, widget):
        painter.setPen(self.pen)
        painter.setBrush(self.brush)
        painter.drawEllipse(self.pos[0] - self.radius, self.pos[1] - self.radius, 2 * self.radius, 2 * self.radius)