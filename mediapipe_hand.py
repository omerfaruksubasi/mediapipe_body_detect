import cv2
import mediapipe as mp
import os

# MediaPipe Hands modelini yükleyin
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# Çizim yardımcılarını yükleyin
mp_drawing = mp.solutions.drawing_utils

# Girdiyi yükleyin
input_img = '/Users/omerfaruksubasi/Desktop/sunum/mediapipe/hand.jpg'
if not os.path.isfile(input_img):
    print(f"Error: The file {input_img} does not exist.")
else:
    image = cv2.imread(input_img)

    if image is None:
        print(f"Error: The file {input_img} could not be read.")
    else:
        # BGR görüntüsünü RGB'ye çevir
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # El tespiti yapın
        results = hands.process(image_rgb)

        # Sonuçları görselleştirin
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Sonucu kaydedin
        output_img = 'hand_detected.jpg'
        cv2.imwrite(output_img, image)

        print(f"Sonuç kaydedildi: {output_img}")

# Kaynakları serbest bırakın
hands.close()
