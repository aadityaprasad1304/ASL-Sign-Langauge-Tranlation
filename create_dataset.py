import cv2
import numpy as np
import os
from cvzone.HandTrackingModule import HandDetector
import math
import time

# Initialize video capture
cap = cv2.VideoCapture(0)

# Hand detector with max 2 hands
detector = HandDetector(maxHands=2)

# Set image size and folder base path
imgSize = 300
offset = 20

# Alphabet list to create folders for each letter
alphabets = [chr(i) for i in range(65, 91)]  # 'A' to 'Z'

# Loop through the alphabets and create folders if they don't exist
for letter in alphabets:
    folder = f"Data/{letter}"
    if not os.path.exists(folder):
        os.makedirs(folder)

for selected_letter in alphabets:
    # Ask the user for confirmation
    response = input(f"Are you ready to start capturing images for '{selected_letter}'? (yes/no): ").lower()

    if response == "yes":
        print(f"Starting to capture images for '{selected_letter}' in 2 seconds...")
        time.sleep(2)  # Wait for 2 seconds before starting
        folder = f"Data/{selected_letter}"
        counter = 0
        start_time = time.time()

        while time.time() - start_time < 10:  # Capture images for 10 seconds
            success, img = cap.read()
            hands, img = detector.findHands(img)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                # Create white background for hand crop
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                # Resizing the cropped image based on aspect ratio
                imgCropShape = imgCrop.shape
                aspectRatio = h / w

                if aspectRatio > 1:  # Height is greater than width
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:  # Width is greater than height
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                # Save the processed image
                counter += 1
                cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                print(f"Saved Image {counter} in {folder}")

                # Display the cropped and resized image
                cv2.imshow("ImageWhite", imgWhite)

            # Show the original webcam image
            cv2.imshow("Image", img)

            # Exit if 'q' is pressed
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

        print(f"Finished capturing images for '{selected_letter}'.")

    # Ask if the user wants to move to the next alphabet
    next_letter = input(f"Do you want to move to the next alphabet '{selected_letter}'? (yes/no): ").lower()
    if next_letter != "yes":
        print("Exiting...")
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
