import random

from keras.datasets import mnist, fashion_mnist
import pygame

(trainX, trainY), (testX, testY) = mnist.load_data()

pygame.init()
screen = pygame.display.set_mode((200, 200))

running = True
i = 0
pygame.display.set_caption(f"{i}")
while running:
    surface = pygame.Surface(screen.get_size())
    pixelArray = pygame.PixelArray(surface)
    for x in range(28):
        for y in range(28):
            pixelArray[x, y] = (trainX[i][y][x], trainX[i][y][x], trainX[i][y][x])

    pixelArray.close()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                #i = random.randint(0, len(trainX) - 1)
                i += 1
                pygame.display.set_caption(f"{i}")
                print(trainY[i])
                for row in trainX[i]:
                    str = ""
                    for pixel in row:
                        if pixel > 0:
                            str += "#"
                        else:
                            str += " "
                    print(str)


    # Blit the pixel_surface onto the screen
    screen.blit(surface, (86, 86))

    # Update the display
    pygame.display.flip()

pygame.quit()