class FrameImage:
    def __init__(self, image, img_name, cx, cy, dim_height, dim_width):
        self.image = image
        self.img_name = img_name
        self.cx = cx
        self.cy = cy
        self.dim_height = dim_height
        self.dim_width = dim_width

    def get_img_name(self):
        return self.img_name
