# This script can display the png image of the saved Plotly 3D Figure.
# Just for demonstration purpose on GitHub
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_image(image_path):
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    display_image('fig1.png')