import cv2
import mediapipe as mp

# MediaPipe Pose modelini yükleyin
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Çizim yardımcılarını yükleyin
mp_drawing = mp.solutions.drawing_utils

# Girdiyi yükleyin
input_img = '/Users/omerfaruksubasi/Desktop/sunum/mediapipe/deneme.jpg'
image = cv2.imread(input_img)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Pose detection yapın
results = pose.process(image_rgb)

# Sonuçları görselleştirin
if results.pose_landmarks:
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# Sonucu kaydedin
output_img = 'pose_detected.jpg'
cv2.imwrite(output_img, image)

print(f"Sonuç kaydedildi: {output_img}")