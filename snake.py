from cv2 import imread
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import random

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # width
cap.set(4, 720)  # height

# Create HandDetector object
detector = HandDetector(detectionCon=0.8, maxHands=1)


class snakeGameClass:
    """Class for the actual game"""

    def __init__(self, food_path):
        self.points = []     # list of all points of the snake body
        self.lengths = []    # distance between each point
        self.current_len = 0  # total snake length
        self.allowed_len = 150  # start with 150, increases when eats food
        self.previous_head = 0, 0  # previous head point
        self.score = 0  # initialize score
        self.game_over = False

        # Food
        # IMREAD_UNCHANGED: remove the pixels from the sides of the png image
        self.img_food = imread(food_path, cv2.IMREAD_UNCHANGED)
        self.food_height, self.food_width, _ = self.img_food.shape
        # food points
        self.food_points = 0, 0  # initial points
        self.random_food_location()

    def random_food_location(self):
        # we have width from 0 to 1280 and length from 0 to 720
        self.food_points = random.randint(100, 1000), random.randint(100, 600)

    def update(self, imgMain, currentHead):

        if self.game_over:
            cvzone.putTextRect(imgMain, "GAME OVER", [
                               300, 400], scale=7, thickness=5, offset=10)
            cvzone.putTextRect(imgMain, f"Score: {self.score}", [
                               300, 600], scale=7, thickness=5, offset=10)
        else:
            # break down previous head in x and y
            px, py = self.previous_head
            # break down current head in x and y
            cx, cy = currentHead

            # distance between previous and current points
            self.points.append([cx, cy])
            distance = np.hypot(cx-px, cy-py)
            self.lengths.append(distance)
            self.current_len += distance  # add distance to current length
            self.previous_head = cx, cy  # update previous head

            # Reduce length (so it doesn't keep adding points until inf.)
            if self.current_len > self.allowed_len:
                for i, len in enumerate(self.lengths):
                    self.current_len -= len
                    self.lengths.pop(i)  # remove len from list
                    self.points.pop(i)  # remove point from list
                    if self.current_len < self.allowed_len:
                        break

            # Check if snake ate food
            rand_x, rand_y = self.food_points
            # Check if our index finger is in this region
            cond1 = rand_x - self.food_width // 2
            cond2 = rand_x + self.food_width // 2
            cond3 = rand_y - self.food_height // 2
            cond4 = rand_y + self.food_height // 2
            if cond1 < cx < cond2 and cond3 < cy < cond4:
                # generate a random location for the food
                # every time it's eaten
                self.random_food_location()
                # increase the allowed length, so the snake can grow
                self.allowed_len += 50
                self.score += 1
                #self.game_over = False

            # Draw the snake
            if self.points:  # only draw if we have something
                for i, p in enumerate(self.points):
                    if i != 0:
                        cv2.line(imgMain, self.points[i-1],
                                 self.points[i], (50, 205, 50), 20)
                        cv2.circle(img, self.points[-1],
                                   10, (255, 0, 255), cv2.FILLED)

            # Draw the food
            # overlayPNG (background image, front image)
            rand_x, rand_y = self.food_points
            food_x = rand_x - self.food_width // 2
            food_y = rand_y - self.food_height // 2
            imgMain = cvzone.overlayPNG(
                imgMain, self.img_food, (food_x, food_y))

            cvzone.putTextRect(imgMain, f"Score: {self.score}",
                               [50, 60], scale=3, thickness=3, offset=5)
            # Check for collision
            # Collision : one point lands on one of the polygon points
            pts = np.array(self.points[:-2], np.int32)  # don't use the last 2
            # so it's compatible with the cv2 function
            pts = pts.reshape((-1, 1, 2))
            # returns the minimum distance between all of the points (in the polygon)
            cv2.polylines(imgMain, [pts], False, (0, 200, 0), 3)
            # check the polygon against the head point, if it hits we know it's a collision
            min_dist = cv2.pointPolygonTest(pts, (cx, cy), True)

            if abs(min_dist) < 1:  # between -1 and 1
                print("Collision!")
                self.game_over = True

                # Reset everything
                self.points = []     # list of all points of the snake body
                self.lengths = []    # distance between each point
                self.current_len = 0  # total snake length
                self.allowed_len = 150  # start with 150, increases when eats food
                self.previous_head = 0, 0  # previous head point
                self.random_food_location()

        return imgMain


# Initialize the game (create snake object)
snake = snakeGameClass('ih.png')

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # flip image in axis 1
    # detect the hand in each frame
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        # get the landmark point of the index finger
        lmList = hands[0]['lmList']
        pointIndex = lmList[8][0:2]  # we only want x and y for index finger
        img = snake.update(img, pointIndex)

    cv2.imshow("Snake Game", img)
    key = cv2.waitKey(1)  # 1 ms delay
    # Reset game with 'r' key
    if key == ord('r'):
        snake.game_over = False
        snake.score = 0
