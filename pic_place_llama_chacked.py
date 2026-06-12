import cv2
import cv2.aruco as aruco
import numpy as np
import serial
import time
import threading
import requests
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

# ==========================================
# 1. HARDWARE & ROBOT CONFIGURATION
# ==========================================
ARDUINO_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200

# Arm Measurements (in cm)
BASE_HEIGHT = 7.6    
BICEP_LEN = 10.5     
FOREARM_LEN = 14.8   
WRIST_LEN = 17.9

HOVER_HEIGHT = 5.0
GRIPPER_OPEN  = 60
GRIPPER_CLOSE = 125

HOME_X, HOME_Y, HOME_Z = 0.0, 0.0, 50.8

# Build the IK Chain
my_arm = Chain(name='arduino_arm', links=[
    OriginLink(),
    URDFLink(name="base_pan",       origin_translation=[0, 0, BASE_HEIGHT], origin_orientation=[0, 0, 0], rotation=[0, 0, 1], bounds=(-1.57, 1.57)),
    URDFLink(name="shoulder_pitch", origin_translation=[0, 0, 0],           origin_orientation=[0, 0, 0], rotation=[0, 1, 0], bounds=(-1.57, 1.57)),
    URDFLink(name="elbow_pitch",    origin_translation=[0, 0, BICEP_LEN],   origin_orientation=[0, 0, 0], rotation=[0, 1, 0], bounds=(-0.1, 2.269)),
    URDFLink(name="wrist_pitch",    origin_translation=[0, 0, FOREARM_LEN], origin_orientation=[0, 0, 0], rotation=[0, 1, 0], bounds=(-0.1, 2.007)),
    URDFLink(name="gripper_tip",    origin_translation=[0, 0, WRIST_LEN],   origin_orientation=[0, 0, 0], rotation=[0, 0, 0])
])

# ==========================================
# 2. VISION & WORKSPACE CONFIGURATION
# ==========================================
worldx = 595
worldy = 545

ROBOT_BASE_X_MM = 303  
ROBOT_BASE_Y_MM = 545 - 210       

DEFAULT_CLICK_Z_CM = 0

# ArUco Setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
parameters = aruco.DetectorParameters()
h_matrix = None

# --- COLOR CALIBRATION DATA ---
lower_green = np.array([30, 80, 50], dtype=np.uint8)
upper_green = np.array([60, 255, 255], dtype=np.uint8)

lower_blue = np.array([90, 80, 50], dtype=np.uint8)
upper_blue = np.array([130, 255, 255], dtype=np.uint8)

lower_pink = np.array([160, 80, 50], dtype=np.uint8)
upper_pink = np.array([180, 255, 255], dtype=np.uint8)

targets_config = {
    'g': (lower_green, upper_green, (0, 255, 0), "Green"),
    'b': (lower_blue, upper_blue, (255, 0, 0), "Blue"),
    'p': (lower_pink, upper_pink, (203, 192, 255), "Pink")
}

# ==========================================
# 3. BASKET POSITIONS
# ==========================================
BASKETS = {
    1: {'name': 'Basket 1', 'pos': (17.40, 20.60, 0.0)},
    2: {'name': 'Basket 2', 'pos': (6.80,  14.30, 0.0)},
    3: {'name': 'Basket 3', 'pos': (7.30,  -14.70, 0.0)},
}

# ==========================================
# 4. OBJECT ZONE
# ==========================================
OBJECT_ZONE = (147, 64, 455, 208)  # (x1, y1, x2, y2) in warped image pixels

# ==========================================
# 5. SHARED STATE (thread-safe)
# ==========================================
pending_command = None
command_lock = threading.Lock()
quit_flag = threading.Event()
robot_busy = threading.Event()
serial_lock = threading.Lock()
cap = None

# ==========================================
# 6. HELPER FUNCTIONS
# ==========================================
def homography(ids, corners):
    top_left, top_right, bottom_right, bottom_left = 1, 2, 3, 4
    zero = corners[np.where(ids == top_left)[0][0]][0][0]
    x    = corners[np.where(ids == top_right)[0][0]][0][1]
    y    = corners[np.where(ids == bottom_left)[0][0]][0][3]
    xy   = corners[np.where(ids == bottom_right)[0][0]][0][2]
    workspace_world_corners = np.array([[0.0, 0.0], [worldx, 0.0], [0.0, worldy], [worldx, worldy]], np.float32)
    workspace_pixel_corners = np.array([zero, x, y, xy], np.float32)
    h_mat, _ = cv2.findHomography(workspace_pixel_corners, workspace_world_corners)
    return h_mat

def clean_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

def send_to_arduino(x, y, z, gripper=GRIPPER_OPEN):
    print(f"\n[IK] Target: X={x:.2f}cm, Y={y:.2f}cm, Z={z:.2f}cm | Gripper={gripper}")
    y = y - 1.5
    ik_radians = my_arm.inverse_kinematics(target_position=[x, y, z])
    ik_degrees = np.degrees(ik_radians)
    motor_angles = [
        int(ik_degrees[1] + 90),
        int(ik_degrees[2] + 90),
        int(ik_degrees[3] + 0),
        int(ik_degrees[4] + 0),
        90,
        gripper
    ]
    motor_angles = [max(0, min(180, angle)) for angle in motor_angles]
    command = f"<{','.join(map(str, motor_angles))}>\n"
    print(f"[SERIAL] Sending: {command.strip()}")

    with serial_lock:
        try:
            # Clear both buffers before sending
            arduino.reset_input_buffer()
            arduino.reset_output_buffer()

            arduino.write(command.encode('utf-8'))
            arduino.flush()  # ensure all bytes are sent

            timeout = time.time() + 20  # 20 second timeout
            while True:
                if arduino.in_waiting > 0:
                    try:
                        line = arduino.readline().decode('utf-8').strip()
                    except UnicodeDecodeError:
                        print("  [SERIAL] Decode error — skipping corrupted line")
                        continue

                    if line:
                        print(f"  Arduino: {line}")
                    if line == "DONE":
                        print("✅ Move complete.")
                        return True  # success

                if time.time() > timeout:
                    print("⚠️  WARNING: Timeout waiting for DONE!")
                    print("   Flushing serial buffers and waiting for Arduino to recover...")
                    arduino.reset_input_buffer()
                    arduino.reset_output_buffer()
                    time.sleep(2)  # give Arduino time to recover
                    return False  # signal move failed

        except serial.SerialException as e:
            print(f"[SERIAL ERROR] {e}")
            return False

def go_home():
    print("\n🏠 Going home...")
    for attempt in range(3):
        success = send_to_arduino(HOME_X, HOME_Y, HOME_Z, gripper=GRIPPER_OPEN)
        if success:
            return
        print(f"  [HOME] Retry {attempt + 1}/3...")
        time.sleep(1)
    print("  [HOME] Could not confirm home position — continuing anyway.")

def is_object_in_zone(color_key):
    """
    Check if the colored object is still detected inside the single object zone.
    Returns True if object still in zone, False if zone is clear.
    """
    ret, frame = cap.read()
    if not ret or h_matrix is None:
        return False

    warped = cv2.warpPerspective(frame, h_matrix, (worldx, worldy))
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    lower, upper, color, name = targets_config[color_key]
    mask = clean_mask(cv2.inRange(hsv, lower, upper))

    x1, y1, x2, y2 = OBJECT_ZONE
    zone_mask = np.zeros_like(mask)
    zone_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

    cnts, _ = cv2.findContours(zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_cnts = [c for c in cnts if cv2.contourArea(c) > 200]

    return len(valid_cnts) > 0

def get_current_centroid(color_key):
    """
    Get fresh centroid of object detected only within the object zone.
    Returns (cx, cy) or None if not found.
    """
    ret, frame = cap.read()
    if not ret or h_matrix is None:
        return None

    warped = cv2.warpPerspective(frame, h_matrix, (worldx, worldy))
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    lower, upper, color, name = targets_config[color_key]
    mask = clean_mask(cv2.inRange(hsv, lower, upper))

    x1, y1, x2, y2 = OBJECT_ZONE
    zone_mask = np.zeros_like(mask)
    zone_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

    cnts, _ = cv2.findContours(zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_cnts = [c for c in cnts if cv2.contourArea(c) > 200]

    if valid_cnts:
        largest = max(valid_cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        (cx, cy), _, _ = rect
        return (int(cx), int(cy))

    return None

def pick_and_place(target_data, color_key, basket_num):
    color_name = targets_config[color_key][3]
    basket = BASKETS[basket_num]
    bx, by, bz = basket['pos']

    base_x_cm = ROBOT_BASE_X_MM / 10.0
    base_y_cm = ROBOT_BASE_Y_MM / 10.0

    MAX_RETRIES = 3

    for attempt in range(MAX_RETRIES):
        print(f"\n{'='*40}")
        print(f"[PICK & PLACE] Attempt {attempt + 1} of {MAX_RETRIES}")

        # Check if object is in zone before attempting
        if not is_object_in_zone(color_key):
            print(f"[ZONE] {color_name} not in zone. Already picked or missing.")
            return

        # Get fresh centroid from zone
        centroid = get_current_centroid(color_key)
        if centroid is None:
            print(f"[VISION] {color_name} not detected in zone. Aborting.")
            go_home()
            return

        pixel_x, pixel_y = centroid
        robot_x = base_y_cm - (pixel_y / 10.0)
        robot_y = base_x_cm - (pixel_x / 10.0)
        robot_z = DEFAULT_CLICK_Z_CM

        print(f"[VISION] {color_name} at pixel=({pixel_x},{pixel_y}) → X={robot_x:.2f}, Y={robot_y:.2f}")

        # --- PICK SEQUENCE ---
        go_home()

        print("\n📍 Moving above object...")
        if not send_to_arduino(robot_x, robot_y, robot_z + HOVER_HEIGHT, gripper=GRIPPER_OPEN):
            print("[ABORT] Move failed. Retrying from scratch...")
            go_home()
            continue

        print("\n📍 Descending to object...")
        if not send_to_arduino(robot_x, robot_y, robot_z, gripper=GRIPPER_OPEN):
            print("[ABORT] Move failed. Retrying from scratch...")
            go_home()
            continue

        print("\n📍 Closing gripper...")
        if not send_to_arduino(robot_x, robot_y, robot_z, gripper=GRIPPER_CLOSE):
            print("[ABORT] Move failed. Retrying from scratch...")
            go_home()
            continue
        time.sleep(0.5)

        print("\n📍 Rising with object...")
        if not send_to_arduino(robot_x, robot_y, robot_z + HOVER_HEIGHT, gripper=GRIPPER_CLOSE):
            print("[ABORT] Move failed. Retrying from scratch...")
            go_home()
            continue

        # --- PLACE SEQUENCE ---
        print(f"\n📦 Moving to {basket['name']}...")
        if not send_to_arduino(bx, by, bz + HOVER_HEIGHT, gripper=GRIPPER_CLOSE):
            print("[ABORT] Move failed. Retrying from scratch...")
            go_home()
            continue

        print(f"\n📦 Dropping into {basket['name']}...")
        send_to_arduino(bx, by, bz + HOVER_HEIGHT, gripper=GRIPPER_OPEN)
        time.sleep(0.3)

        go_home()

        # --- VERIFY BY CHECKING ZONE ---
        print(f"\n🔍 Checking zone for {color_name}...")
        time.sleep(0.5)

        if not is_object_in_zone(color_key):
            print(f"✅ Zone clear! {color_name} successfully placed in {basket['name']}.")
            return  # success
        else:
            print(f"❌ {color_name} still in zone! Operation failed.")
            if attempt < MAX_RETRIES - 1:
                print(f"   Retrying...")
            else:
                print(f"\n[FAILED] All {MAX_RETRIES} attempts failed. Giving up.")
                go_home()

def run_pick_and_place(target_data, color_key, basket_num):
    """Runs pick_and_place in a separate thread so camera stays responsive."""
    robot_busy.set()
    try:
        pick_and_place(target_data, color_key, basket_num)
    finally:
        robot_busy.clear()
        print("\n[ROBOT] Ready for next command.")
        print("Command: ", end='', flush=True)

# ==========================================
# 7. LLAMA 3.2 NATURAL LANGUAGE PARSER
# ==========================================
def parse_with_llama(user_input):
    prompt = f"""
You are a robot arm controller. Extract the color and basket number from the user command.

Rules:
- Color must be one of: green (g), blue (b), pink (p)
- Basket number must be one of: 1, 2, 3
- Reply with ONLY two characters separated by a space, like: g 1
- If you cannot extract both pieces of information, reply with: UNKNOWN
- If the sentence contains negation (like "not basket 2"), ignore the negated basket and use the intended one.

User command: "{user_input}"

Reply:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0}
            },
            timeout=10
        )
        response.raise_for_status()
        reply = response.json()["response"].strip().lower()
        print(f"[LLAMA] Interpreted as: '{reply}'")

        if reply == "unknown":
            print("[ERROR] Llama could not understand the command. Please try again.")
            print("        Example: 'pick the green object and place it in basket 2'")
            return None, None

        parts = reply.split()
        if len(parts) != 2:
            print(f"[ERROR] Unexpected Llama reply: '{reply}'. Please try again.")
            return None, None

        color_key, basket_str = parts

        if color_key not in targets_config:
            print(f"[ERROR] Llama returned unknown color '{color_key}'. Please try again.")
            return None, None

        if not basket_str.isdigit() or int(basket_str) not in BASKETS:
            print(f"[ERROR] Llama returned unknown basket '{basket_str}'. Please try again.")
            return None, None

        return color_key, int(basket_str)

    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to Ollama. Run: ollama serve")
        return None, None
    except requests.exceptions.Timeout:
        print("[ERROR] Llama took too long to respond. Please try again.")
        return None, None
    except Exception as e:
        print(f"[ERROR] Llama request failed: {e}")
        return None, None

# ==========================================
# 8. INPUT THREAD
# ==========================================
def input_thread_fn():
    global pending_command
    print("\nType your command in natural language and press Enter:")
    print("  Example: 'pick the green object and place it in basket 2'")
    print("  Example: 'take blue to basket 1'")
    print("  Type 'q' to quit.\n")

    while not quit_flag.is_set():
        try:
            raw = input("Command: ")

            if raw.strip().lower() == 'q':
                quit_flag.set()
                break

            if not raw.strip():
                continue

            if robot_busy.is_set():
                print("[WARNING] Robot is still busy! Please wait for it to finish.")
                continue

            print(f"[LLAMA] Processing: '{raw}'")
            color_key, basket_num = parse_with_llama(raw)

            if color_key is not None:
                color_name = targets_config[color_key][3]
                basket_name = BASKETS[basket_num]['name']

                confirm = input(f"[CONFIRM] Pick {color_name} → Place in {basket_name}? (y/n): ").strip().lower()

                if confirm == 'y':
                    print("[CONFIRMED] Executing command...")
                    with command_lock:
                        pending_command = (color_key, basket_num)
                else:
                    print("[CANCELLED] Command cancelled. Please try again.")

        except EOFError:
            break

# ==========================================
# 9. MAIN EXECUTION
# ==========================================
try:
    print("Connecting to Arduino...")
    arduino = serial.Serial(port=ARDUINO_PORT, baudrate=BAUD_RATE, timeout=1)
    time.sleep(2)
    arduino.reset_input_buffer()
    print("Arduino Connected!")

    go_home()

    cap = cv2.VideoCapture(2)
    cv2.namedWindow("Live Workspace")

    # Start input thread
    t = threading.Thread(target=input_thread_fn, daemon=True)
    t.start()

    print("\n--- SYSTEM READY ---")
    print("Waiting for all 4 ArUco markers to lock workspace...")
    print("Press 'q' in terminal to quit.\n")

    while not quit_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            ids_flat = ids.flatten()
            if all(m in ids_flat for m in [1, 2, 3, 4]) and h_matrix is None:
                h_matrix = homography(ids, corners)
                print("\n[VISION] Workspace Locked! Ready for commands.\n")

        if h_matrix is not None:
            warped_image = cv2.warpPerspective(frame, h_matrix, (worldx, worldy))
            hsv_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2HSV)
            current_targets = {}

            for key_char, (lower, upper, color, name) in targets_config.items():
                mask = clean_mask(cv2.inRange(hsv_image, lower, upper))
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid_cnts = [c for c in cnts if 200 < cv2.contourArea(c) < 5000]

                if valid_cnts:
                    largest_cnt = max(valid_cnts, key=cv2.contourArea)
                    rect = cv2.minAreaRect(largest_cnt)
                    (cx_float, cy_float), (w, h), angle = rect
                    cx, cy = int(cx_float), int(cy_float)
                    current_targets[key_char] = {"centroid": (cx, cy), "angle": angle, "name": name}
                    box = np.int32(cv2.boxPoints(rect))
                    cv2.drawContours(warped_image, [box], 0, color, 2)
                    cv2.circle(warped_image, (cx, cy), 4, color, -1)
                    cv2.putText(warped_image, f"{name} A:{angle:.0f}", (cx - 40, cy - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw object zone
            x1, y1, x2, y2 = OBJECT_ZONE
            cv2.rectangle(warped_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(warped_image, "Object Zone", (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Draw basket markers
            for bnum, bdata in BASKETS.items():
                bx, by, bz = bdata['pos']
                base_x_cm = ROBOT_BASE_X_MM / 10.0
                base_y_cm = ROBOT_BASE_Y_MM / 10.0
                vis_px = int((base_x_cm - by) * 10)
                vis_py = int((base_y_cm - bx) * 10)
                if 0 <= vis_px < worldx and 0 <= vis_py < worldy:
                    cv2.rectangle(warped_image,
                                  (vis_px - 20, vis_py - 20),
                                  (vis_px + 20, vis_py + 20),
                                  (255, 255, 255), 2)
                    cv2.putText(warped_image, f"B{bnum}",
                                (vis_px - 10, vis_py + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show robot status
            status = "BUSY" if robot_busy.is_set() else "READY"
            status_color = (0, 0, 255) if robot_busy.is_set() else (0, 255, 0)
            cv2.putText(warped_image, f"Robot: {status}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # Execute pending command
            with command_lock:
                cmd = pending_command
                pending_command = None

            if cmd is not None:
                if robot_busy.is_set():
                    print("[WARNING] Robot is still busy! Wait for current operation to finish.")
                else:
                    color_key, basket_num = cmd
                    color_name = targets_config[color_key][3]
                    if color_key in current_targets:
                        robot_thread = threading.Thread(
                            target=run_pick_and_place,
                            args=(current_targets[color_key], color_key, basket_num),
                            daemon=True
                        )
                        robot_thread.start()
                    else:
                        print(f"[WARNING] {color_name} not visible! Place it in the workspace and try again.")

            cv2.imshow("Live Workspace", warped_image)

        else:
            cv2.imshow("Live Workspace", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            quit_flag.set()
            break

except serial.SerialException:
    print("Error: Could not open port. Is the Arduino IDE Serial Monitor closed?")
finally:
    quit_flag.set()
    if 'arduino' in locals() and arduino.is_open:
        arduino.close()
    if cap is not None and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    print("\nSystem safely shut down.")