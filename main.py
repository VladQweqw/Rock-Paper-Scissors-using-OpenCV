import cv2
import mediapipe as mp

import time
import threading

mp_hands = mp.solutions.hands

# dimensions
WIDTH, HEIGHT = 800, 500

newGame = True
hasFinished = True
countdownTimer = 3
winner = 0

hand_choice = {
    "Left": "",
    "Right": ""
}

# hands offsets
hands = mp_hands.Hands(
    max_num_hands=2,     
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

def displayText(text, coords=(150, 150)):
    cv2.putText(
        frame,
        text,
        coords,                      
        cv2.FONT_HERSHEY_SIMPLEX,       
        1,                             
        (0, 225, 0),                  
        2,                              
        cv2.LINE_AA
    )

def countdown():
    global hasFinished, countdownTimer

    for idx in range(3, -1, -1): 
        countdownTimer = idx
        time.sleep(1)

    hasFinished = True

def compareHandsChoice():
    global winner
    print(hand_choice)

    if hand_choice['Left'] == 'Paper' and hand_choice['Right'] == 'Rock':
        winner = 1
    elif hand_choice['Left'] == 'Rock' and hand_choice['Right'] == 'Scissors':
        winner = 1
    elif hand_choice['Left'] == 'Scissors' and hand_choice['Right'] == 'Paper':
        winner = 1
    else:
        winner = 2
    

def startGame():
    global hasFinished, newGame
    displayText(f"Start in {countdownTimer}")

    if hasFinished and newGame:
        hasFinished = False
        countdownThread = threading.Thread(target=countdown)
        countdownThread.start()

    for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
        # display scores
        displayText(f"Player 1", coords=(50, 40))
        displayText(f"Player 2", coords=(600, 40))

        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        hand_side = handedness.classification[0].label
        hl = hand_landmarks.landmark

        fingers = {
            "Left": {
                "THUMB": hl[4].x > hl[3].x,
                "INDEX": hl[8].y < hl[6].y,
                "MIDDLE": hl[12].y < hl[10].y,
                "RING": hl[16].y < hl[14].y,
                "PINKY": hl[20].y < hl[18].y
            },
            "Right": {
                "THUMB": hl[4].x < hl[3].x,
                "INDEX": hl[8].y < hl[6].y,
                "MIDDLE": hl[12].y < hl[10].y,
                "RING": hl[16].y < hl[14].y,
                "PINKY": hl[20].y < hl[18].y
            }
        }

        all_fingers = [finger for finger, is_up in fingers[hand_side].items() if is_up]
        if len(all_fingers) == 5:
            hand_choice[hand_side] = "Paper"
        elif fingers[hand_side]['INDEX'] and fingers[hand_side]['MIDDLE']:
            hand_choice[hand_side] = "Scissors"
        elif not all_fingers:
            hand_choice[hand_side] = "Rock"

        if countdownTimer == 0:
            compareHandsChoice()
            newGame = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        # start the game only if there are 2 hands
        if len(result.multi_hand_landmarks) == 2:
            if newGame:
                startGame()
            elif winner:
                displayText(f"Player {winner} wins!!!")
        else:
            displayText("Show 2 hands")
            winner = 0
    else:
        displayText("Show hands to start")
        newGame = True

    cv2.imshow("Rock, Paper, Scissors with ML", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
