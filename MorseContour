import cv2
import numpy as np
import time

# Define the morse code dictionary.
morse_dict = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..', '0': '-----', '1': '.----', '2': '..---',
    '3': '...--', '4': '....-', '5': '.....', '6': '-....', '7': '--...',
    '8': '---..', '9': '----.', ' ': '/'
}

# Reverse the morse code dictionary to map morse code to characters.
morse_dict = {v: k for k, v in morse_dict.items()}

# Initialize variables to keep track of the state and timing of finger movements.
move_state = False
prev_time = 0
dots_dashes = ''

# Start the video capture.
cap = cv2.VideoCapture(0)
while True:
    # Read a frame from the video.
    ret, frame = cap.read()
    # Convert the frame to grayscale.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale frame to create a binary image.
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image.
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours.
    for contour in contours:
        # Check if the contour is large enough to be a hand.
        if cv2.contourArea(contour) > 3000:
            # Draw a rectangle around the hand.
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Check if the hand is moving.
            if move_state == False:
                move_state = True
                prev_time = time.time()
            else:
                # Check the elapsed time since the previous movement.
                elapsed_time = time.time() - prev_time
                # If the elapsed time is less than 1 second, it's a dot.
                if elapsed_time < 1:
                    dots_dashes += '.'
                # If the elapsed time is between 1 and 2 seconds, it's a dash.
                elif elapsed_time >= 1 and elapsed_time < 2:
                    dots_dashes += '-'
                # If the elapsed time is more than 2 seconds, it's a new character.
                else:
                    # Check if the current sequence of dots and dashes is a valid morse code.
                    if dots_dashes in morse_dict:
                        # Print the corresponding character.
                        print(morse_dict[dots_dashes], end='')
                        dots_dashes = ''
                    else:
                        # If it's not a valid morse code, reset the sequence.
                        dots_dashes = ''
                prev_time = time.time()
                move_state = False

    # Show the frame with the hand contour and rectangle.
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Break the loop if the 'q' key is pressed.
    if key == ord("q"):
        break

# Release the video capture and close the window.
cap.release()
cv2.destroyAllWindows()

