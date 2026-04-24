import cv2
import mediapipe as mp

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Captura de webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Falha ao capturar imagem.")
        break

    # CONSERTO: Este bloco agora está FORA do 'if not success'
    # Converter cor de BGR para RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Verificar a detecção de mãos
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Desenha os pontos e as conexões na mão
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Pegar a coordenada de um ponto específico
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 4: # Ponta do polegar
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

    # Exibe a imagem (também deve estar dentro do while)
    cv2.imshow("Detector de Maos", img)

    # 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()