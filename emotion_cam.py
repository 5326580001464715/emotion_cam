import cv2
import numpy as np
from deepface import DeepFace
import threading
import time
import os
import sys
from collections import deque
from datetime import datetime
import io
import sounddevice as sd
import queue
import math
import warnings
warnings.filterwarnings('ignore')

# ========== FIX ENCODING ==========
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ========== SETTINGS ==========
EMOTION_IMAGES = {
    "angry": "angry.png", "disgust": "disgust.png", "fear": "fear.png",
    "happy": "happy.png", "sad": "sad.png", "surprise": "surprise.png",
    "neutral": "neutral.png"
}
TALK_IMAGE = "talk.png"

EMOTION_EMOJI = {
    "angry": "😠", "disgust": "🤢", "fear": "😨",
    "happy": "😊", "sad": "😢", "surprise": "😲",
    "neutral": "😐"
}
TALK_EMOJI = "🔊"

EMOTION_NAMES_EN = {
    "angry": "ANGRY", "disgust": "DISGUST", "fear": "FEAR",
    "happy": "HAPPY", "sad": "SAD", "surprise": "SURPRISE",
    "neutral": "NEUTRAL"
}
TALK_NAME_EN = "SPEAKING"

EMOTION_COLORS = {
    "angry": (0, 0, 255), "disgust": (0, 255, 0), "fear": (255, 0, 255),
    "happy": (0, 255, 255), "sad": (255, 0, 0), "surprise": (255, 255, 0),
    "neutral": (128, 128, 255)
}
TALK_COLOR = (0, 200, 200)

# ========== OPTIMIZED SETTINGS ==========
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 30

FRAME_SKIP = 4
DETECTION_COOLDOWN = 0.3

EMOTION_WINDOW_SIZE = 500

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_DURATION = 0.05
AUDIO_CHUNK_SIZE = int(AUDIO_SAMPLE_RATE * AUDIO_CHUNK_DURATION)

# ========== PSYCHOLOGICAL SETTINGS ==========
NEUTRAL_IS_DEFAULT = True
NEUTRAL_THRESHOLD = 0.3

EMOTION_CONFIRMATION_FRAMES = 3
EMOTION_REQUIRED_BOOST = 1.5

EMOTION_PRIORITY = {
    "happy": 1.3,
    "sad": 1.2,
    "angry": 1.2,
    "surprise": 1.1,
    "fear": 1.0,
    "disgust": 1.0,
    "neutral": 0.8
}

EMOTION_MIN_CONFIDENCE = {
    "happy": 0.35,
    "sad": 0.4,
    "angry": 0.4,
    "surprise": 0.45,
    "fear": 0.45,
    "disgust": 0.45,
    "neutral": 0.2
}

# ========== ANIMATION SETTINGS ==========
ENABLE_PULSE_ANIMATION = True
ENABLE_GLOW_EFFECT = True
ENABLE_SMOOTH_TRANSITIONS = True
PULSE_SPEED = 0.1
GLOW_INTENSITY = 0.3

current_emotion = "neutral"
emotion_history = deque(maxlen=10)
emotion_confidence = 0.0
emotion_lock = threading.Lock()
image_cache = {}
running = True
talking = False
talk_lock = threading.Lock()
current_volume = 0.0

emotion_threshold = 0.3
talk_threshold = 0.02
threshold_lock = threading.Lock()

menu_visible = False
settings_button_rect = (CAMERA_WIDTH-150, 8, 140, 28)
slider_dragging = False
active_slider = None

pulse_phase = 0.0
glow_phase = 0.0
transition_alpha = 1.0
last_emotion_display = "neutral"

emotion_confirmation = {emotion: 0 for emotion in EMOTION_NAMES_EN.keys()}
emotion_pending = None
pending_counter = 0

face_cascade = None
eye_cascade = None
smile_cascade = None

audio_queue = queue.Queue(maxsize=20)
audio_history = deque(maxlen=20)


class PsychologicalEmotionDetector:
    def __init__(self):
        self.history = deque(maxlen=15)
        self.current_emotion = "neutral"
        self.confirmation_counter = 0
        self.pending_emotion = None
        self.last_change_time = time.time()
        self.emotion_start_time = {}
        
    def detect_emotion(self, face_img):
        try:
            result = DeepFace.analyze(
                face_img,
                actions=['emotion'],
                detector_backend='opencv',
                enforce_detection=False,
                silent=True
            )
            if result and len(result) > 0:
                emotion = result[0]['dominant_emotion']
                confidence = result[0]['emotion'][emotion] / 100
                return emotion, confidence
        except:
            pass
        return None, 0
    
    def process_emotion(self, new_emotion, confidence):
        global emotion_pending, pending_counter
        
        if not new_emotion:
            return self.current_emotion, False, 0
        
        priority = EMOTION_PRIORITY.get(new_emotion, 1.0)
        min_conf = EMOTION_MIN_CONFIDENCE.get(new_emotion, 0.3)
        
        if confidence < min_conf:
            return self.current_emotion, False, 0
        
        weighted_confidence = confidence * priority
        self.history.append((new_emotion, weighted_confidence, confidence))
        
        if self.current_emotion == "neutral":
            if new_emotion != "neutral":
                if emotion_pending == new_emotion:
                    pending_counter += 1
                else:
                    emotion_pending = new_emotion
                    pending_counter = 1
                
                if pending_counter >= EMOTION_CONFIRMATION_FRAMES:
                    if weighted_confidence > EMOTION_REQUIRED_BOOST * EMOTION_MIN_CONFIDENCE.get(new_emotion, 0.3):
                        emotion_pending = None
                        pending_counter = 0
                        self.current_emotion = new_emotion
                        self.last_change_time = time.time()
                        print(f"🎭 EMOTION DETECTED: {new_emotion.upper()} ({int(confidence*100)}%)")
                        return new_emotion, True, weighted_confidence
            return self.current_emotion, False, weighted_confidence
        else:
            if new_emotion == "neutral" and confidence > EMOTION_MIN_CONFIDENCE["neutral"]:
                if emotion_pending == "neutral":
                    pending_counter += 1
                    if pending_counter >= 2:
                        emotion_pending = None
                        pending_counter = 0
                        self.current_emotion = "neutral"
                        return "neutral", True, confidence
                else:
                    emotion_pending = "neutral"
                    pending_counter = 1
                return self.current_emotion, False, weighted_confidence
            elif new_emotion != self.current_emotion and new_emotion != "neutral":
                if emotion_pending == new_emotion:
                    pending_counter += 1
                    if pending_counter >= EMOTION_CONFIRMATION_FRAMES + 1:
                        emotion_pending = None
                        pending_counter = 0
                        self.current_emotion = new_emotion
                        return new_emotion, True, weighted_confidence
                else:
                    emotion_pending = new_emotion
                    pending_counter = 1
            elif new_emotion == self.current_emotion:
                emotion_pending = None
                pending_counter = 0
            return self.current_emotion, False, weighted_confidence


def audio_callback(indata, frames, time_info, status):
    global current_volume
    if status:
        print(f"Audio status: {status}")
    volume_norm = np.linalg.norm(indata) * 10
    current_volume = volume_norm
    audio_history.append(volume_norm)
    
    if audio_queue.qsize() > 15:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            pass
    try:
        audio_queue.put_nowait(volume_norm)
    except queue.Full:
        pass


def audio_worker():
    global talking
    try:
        stream = sd.InputStream(samplerate=AUDIO_SAMPLE_RATE, channels=1,
                                blocksize=AUDIO_CHUNK_SIZE, callback=audio_callback)
        stream.start()
        while running:
            try:
                vol = audio_queue.get(timeout=0.1)
                with threshold_lock:
                    thr = talk_threshold
                is_talking = vol > thr
                with talk_lock:
                    talking = is_talking
            except queue.Empty:
                pass
            time.sleep(0.01)
        stream.stop()
    except Exception as e:
        print(f"⚠️ Microphone error: {e}")


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


def load_detectors():
    global face_cascade, eye_cascade, smile_cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    print("✅ Detectors loaded")


def optimize_camera_settings(cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))


def draw_face_features(frame, faces, is_active_emotion=False):
    for (x, y, w, h) in faces:
        if is_active_emotion:
            glow = int(15 * (1 + math.sin(glow_phase)))
            for i in range(3):
                alpha = 1 - i * 0.3
                color = (0, int(255 * alpha), int(100 * alpha))
                cv2.rectangle(frame, (x-i-glow, y-i-glow), (x+w+i+glow, y+h+i+glow), color, 1)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 150, 0), 1)
        
        roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
        for (ex, ey, ew, eh) in eyes:
            size = 2 + int(math.sin(pulse_phase + ex) * 1)
            cv2.circle(frame, (x+ex+ew//2, y+ey+eh//2), size, (255, 100, 100), -1)
        
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 20)
        for (sx, sy, sw, sh) in smiles:
            thickness = 1 + int(math.sin(pulse_phase + sx) * 1)
            cv2.ellipse(frame, (x+sx+sw//2, y+sy+sh//2), 
                       (sw//2, sh//4), 0, 0, 180, (100, 100, 255), thickness)
    return frame


def draw_mic_visualizer(frame, x, y, width, height, volume, is_talking):
    num_bars = 12
    bar_width = width // num_bars
    spacing = 2
    
    for i in range(num_bars):
        bar_height_factor = (math.sin(i * 0.5 + pulse_phase) * 0.3 + 0.7)
        bar_height = int(height * (volume / talk_threshold) * bar_height_factor)
        bar_height = max(2, min(height, bar_height))
        
        if is_talking:
            if volume > talk_threshold * 1.5:
                color = (0, 255, 0)
            elif volume > talk_threshold:
                color = (0, 200, 100)
            else:
                color = (0, 150, 0)
        else:
            color = (80, 80, 80)
        
        bar_x = x + i * (bar_width + spacing)
        cv2.rectangle(frame, 
                     (bar_x, y + height - bar_height),
                     (bar_x + bar_width - spacing, y + height),
                     color, -1)
    return frame


def draw_ui(frame, fps, emotion, confidence, faces_count, volume, is_talking, weighted_score):
    global transition_alpha, last_emotion_display
    
    h, w = frame.shape[:2]
    
    if emotion != last_emotion_display:
        transition_alpha = 0.3
        last_emotion_display = emotion
    else:
        transition_alpha = min(1.0, transition_alpha + 0.05)
    
    overlay = frame.copy()
    for i in range(30):
        alpha = i / 30
        color = (int(15 * alpha), int(15 * alpha), int(20 * alpha))
        cv2.line(overlay, (0, i), (w, i), color, 1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    sx, sy, sw, sh = settings_button_rect
    if slider_dragging:
        cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (50, 50, 70), -1)
    else:
        cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (30, 30, 40), -1)
    cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (80, 80, 100), 1)
    cv2.putText(frame, "SETTINGS", (sx+15, sy+18), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 220), 1)
    
    cv2.putText(frame, f"FPS:{fps}", (w-50, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    emoji = EMOTION_EMOJI.get(emotion.lower(), "😐")
    emotion_color = EMOTION_COLORS.get(emotion.lower(), (200, 200, 200))
    
    font_scale = 0.5 if emotion.lower() == "neutral" else 0.6
    if emotion.lower() != "neutral":
        emotion_color = tuple(min(255, int(c * 1.3)) for c in emotion_color)
    
    if transition_alpha < 1.0:
        alpha = transition_alpha
        cv2.putText(frame, f"{emoji} {emotion}", (5, h-25), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                    (int(emotion_color[0]*alpha), int(emotion_color[1]*alpha), int(emotion_color[2]*alpha)), 1)
    else:
        cv2.putText(frame, f"{emoji} {emotion}", (5, h-25), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, emotion_color, 1)
    
    bar_x, bar_y, bar_w, bar_h = 5, h-15, 80, 5
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (40, 40, 40), -1)
    
    display_confidence = weighted_score if weighted_score > 0 else confidence
    fill_w = int(bar_w * display_confidence)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill_w, bar_y+bar_h), emotion_color, -1)
    
    mic_x = w - 150
    mic_y = h - 30
    mic_w = 100
    mic_h = 20
    frame = draw_mic_visualizer(frame, mic_x, mic_y, mic_w, mic_h, volume, is_talking)
    
    if is_talking:
        cv2.putText(frame, "SPEAKING", (w-120, h-35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    else:
        cv2.putText(frame, "SILENT", (w-120, h-35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
    
    return frame


def draw_settings_menu(frame, e_thr, t_thr):
    h, w = frame.shape[:2]
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (w-180, 40), (w-20, 140), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
    cv2.rectangle(frame, (w-180, 40), (w-20, 140), (60, 60, 80), 1)
    
    cv2.putText(frame, "SETTINGS", (w-160, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 220), 1)
    
    cv2.putText(frame, "EMOTION", (w-170, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)
    cv2.rectangle(frame, (w-170, 83), (w-40, 97), (40, 40, 40), -1)
    fill_e = int(130 * (e_thr-0.2) / 0.6)
    cv2.rectangle(frame, (w-170, 83), (w-170+fill_e, 97), (100, 100, 200), -1)
    
    if active_slider == 'emotion':
        cv2.putText(frame, f"{e_thr:.2f}", (w-45, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    else:
        cv2.putText(frame, f"{e_thr:.2f}", (w-45, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (150, 150, 150), 1)
    
    cv2.putText(frame, "MIC", (w-170, 115), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)
    cv2.rectangle(frame, (w-170, 118), (w-40, 132), (40, 40, 40), -1)
    fill_t = int(130 * t_thr / 0.1)
    cv2.rectangle(frame, (w-170, 118), (w-170+fill_t, 132), (0, 200, 200), -1)
    
    if active_slider == 'talk':
        cv2.putText(frame, f"{t_thr:.3f}", (w-45, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    else:
        cv2.putText(frame, f"{t_thr:.3f}", (w-45, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (150, 150, 150), 1)
    
    cv2.rectangle(frame, (w-40, 43), (w-25, 58), (40, 40, 50), -1)
    cv2.putText(frame, "X", (w-35, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
    
    return frame


def mouse_callback(event, x, y, flags, param):
    global emotion_threshold, talk_threshold, slider_dragging, active_slider, menu_visible

    if event == cv2.EVENT_LBUTTONDOWN:
        sx, sy, sw, sh = settings_button_rect
        if sx <= x <= sx+sw and sy <= y <= sy+sh:
            menu_visible = not menu_visible
            return
        
        if menu_visible:
            w = CAMERA_WIDTH
            if w-170 <= x <= w-40 and 83 <= y <= 97:
                slider_dragging = True
                active_slider = 'emotion'
                val = 0.2 + ((x - (w-170)) / 130) * 0.6
                val = max(0.2, min(0.8, val))
                with threshold_lock:
                    emotion_threshold = val
                return
            if w-170 <= x <= w-40 and 118 <= y <= 132:
                slider_dragging = True
                active_slider = 'talk'
                val = (x - (w-170)) / 130 * 0.1
                val = max(0.005, min(0.1, val))
                with threshold_lock:
                    talk_threshold = val
                return
            if w-40 <= x <= w-25 and 43 <= y <= 58:
                menu_visible = False
                return

    elif event == cv2.EVENT_MOUSEMOVE and slider_dragging:
        w = CAMERA_WIDTH
        if active_slider == 'emotion' and w-170 <= x <= w-40:
            val = 0.2 + ((x - (w-170)) / 130) * 0.6
            val = max(0.2, min(0.8, val))
            with threshold_lock:
                emotion_threshold = val
        elif active_slider == 'talk' and w-170 <= x <= w-40:
            val = (x - (w-170)) / 130 * 0.1
            val = max(0.005, min(0.1, val))
            with threshold_lock:
                talk_threshold = val

    elif event == cv2.EVENT_LBUTTONUP:
        slider_dragging = False
        active_slider = None


def camera_worker():
    global current_emotion, running, emotion_threshold, talk_threshold, menu_visible, emotion_confidence
    global pulse_phase, glow_phase

    cap = cv2.VideoCapture(0)
    optimize_camera_settings(cap)
    if not cap.isOpened():
        print("❌ NO CAMERA!")
        running = False
        return

    print("✅ Camera working")
    cv2.namedWindow("Face Tracking")
    cv2.setMouseCallback("Face Tracking", mouse_callback)

    detector = PsychologicalEmotionDetector()
    
    frame_count = 0
    last_detection = 0
    fps = 0
    fps_time = time.time()
    no_face_counter = 0
    current_weighted_score = 0.0

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        current_time = time.time()
        
        pulse_phase += PULSE_SPEED
        glow_phase += 0.05

        if current_time - fps_time >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = current_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        if len(faces) > 0:
            no_face_counter = 0
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            
            if (frame_count % FRAME_SKIP == 0 and 
                current_time - last_detection > DETECTION_COOLDOWN):
                
                try:
                    x, y, w, h = largest_face
                    face_img = frame[y:y+h, x:x+w]
                    
                    emotion, confidence = detector.detect_emotion(face_img)
                    
                    if emotion:
                        stable_emotion, changed, weighted_score = detector.process_emotion(emotion, confidence)
                        current_weighted_score = weighted_score
                        
                        if changed:
                            with emotion_lock:
                                current_emotion = stable_emotion
                                emotion_confidence = confidence
                    
                    last_detection = current_time
                    
                except Exception as e:
                    pass
            
            is_active = current_emotion != "neutral"
            frame = draw_face_features(frame, [largest_face], is_active)
        else:
            no_face_counter += 1
            if no_face_counter > 45:
                with emotion_lock:
                    current_emotion = "neutral"
                    emotion_confidence = 0.5

        with talk_lock:
            is_talking = talking
        
        en_cur = EMOTION_NAMES_EN.get(current_emotion, current_emotion)
        frame = draw_ui(frame, fps, en_cur, emotion_confidence, len(faces), 
                        current_volume, is_talking, current_weighted_score)

        with threshold_lock:
            e_val = emotion_threshold
            t_val = talk_threshold
        if menu_visible:
            frame = draw_settings_menu(frame, e_val, t_val)

        cv2.imshow("Face Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()


def emotion_display_worker():
    global running
    cv2.namedWindow("Emotion Display", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Emotion Display", EMOTION_WINDOW_SIZE, EMOTION_WINDOW_SIZE)

    while running:
        with talk_lock:
            is_talking = talking
        with emotion_lock:
            cur_em = current_emotion
            conf = emotion_confidence

        if is_talking:
            if "talk" in image_cache:
                display = image_cache["talk"].copy()
                title = TALK_NAME_EN
                emoji = TALK_EMOJI
            else:
                display = create_placeholder("talk", EMOTION_WINDOW_SIZE, EMOTION_WINDOW_SIZE)
                title = TALK_NAME_EN
                emoji = TALK_EMOJI
        else:
            if cur_em in image_cache:
                display = image_cache[cur_em].copy()
                title = EMOTION_NAMES_EN.get(cur_em, cur_em)
                emoji = EMOTION_EMOJI.get(cur_em, "😐")
            else:
                display = create_placeholder(cur_em, EMOTION_WINDOW_SIZE, EMOTION_WINDOW_SIZE)
                title = EMOTION_NAMES_EN.get(cur_em, cur_em)
                emoji = EMOTION_EMOJI.get(cur_em, "😐")

        h, w = display.shape[:2]
        
        font_scale = 1.0 if cur_em == "neutral" else 1.3
        cv2.putText(display, f"{emoji} {title}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        
        tm = datetime.now().strftime("%H:%M")
        cv2.putText(display, tm, (w-70, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        cv2.imshow("Emotion Display", display)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            running = False
            break
        time.sleep(0.03)


def create_placeholder(name, width, height):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if name == "talk":
        color = TALK_COLOR
        text = TALK_NAME_EN
    else:
        color = EMOTION_COLORS.get(name, (128,128,128))
        text = EMOTION_NAMES_EN.get(name, name)
    img[:] = color
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img


def load_all_images():
    print("🖼️ Loading images...")
    loaded = 0
    total = len(EMOTION_IMAGES) + 1
    
    for emotion, fname in EMOTION_IMAGES.items():
        path = resource_path(fname)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, (EMOTION_WINDOW_SIZE, EMOTION_WINDOW_SIZE))
                image_cache[emotion] = img
                print(f"✅ {emotion.upper():10} -> {fname}")
                loaded += 1
                continue
        print(f"⚠️ {emotion.upper():10} -> placeholder")
        image_cache[emotion] = create_placeholder(emotion, EMOTION_WINDOW_SIZE, EMOTION_WINDOW_SIZE)
        loaded += 1

    talk_path = resource_path(TALK_IMAGE)
    if os.path.exists(talk_path):
        img = cv2.imread(talk_path)
        if img is not None:
            img = cv2.resize(img, (EMOTION_WINDOW_SIZE, EMOTION_WINDOW_SIZE))
            image_cache["talk"] = img
            print(f"✅ SPEAKING   -> {TALK_IMAGE}")
            loaded += 1
        else:
            print(f"⚠️ SPEAKING   -> placeholder")
            image_cache["talk"] = create_placeholder("talk", EMOTION_WINDOW_SIZE, EMOTION_WINDOW_SIZE)
            loaded += 1
    else:
        print(f"⚠️ SPEAKING   -> placeholder")
        image_cache["talk"] = create_placeholder("talk", EMOTION_WINDOW_SIZE, EMOTION_WINDOW_SIZE)
        loaded += 1

    print(f"📊 Loaded: {loaded}/{total} images")


def main():
    global running
    print("\n" + "="*70)
    print("🎭 EMOTION CAMERA - PSYCHOLOGICAL EDITION")
    print("="*70)
    print("🧠 PSYCHOLOGICAL SETTINGS:")
    print(f"   • Neutral is default: {NEUTRAL_IS_DEFAULT}")
    print(f"   • Confirmation frames: {EMOTION_CONFIRMATION_FRAMES}")
    print(f"   • Required boost: {EMOTION_REQUIRED_BOOST}x")
    print("\n🎯 EMOTION PRIORITIES:")
    for emotion, priority in EMOTION_PRIORITY.items():
        print(f"   • {emotion.upper()}: {priority}x")
    print("="*70)

    load_detectors()
    load_all_images()

    t1 = threading.Thread(target=camera_worker, daemon=True)
    t2 = threading.Thread(target=emotion_display_worker, daemon=True)
    t3 = threading.Thread(target=audio_worker, daemon=True)
    t1.start()
    t2.start()
    t3.start()

    print("\n✅ Program started!")
    print("❌ Press Q in any window to exit")
    print("="*70)

    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        running = False
        print("\n👋 Exiting...")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
