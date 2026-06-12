#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver board = Adafruit_PWMServoDriver();

#define SERVOMIN  150
#define SERVOMAX  600
#define SERVO_FREQ 50

int offsets[]       = {-15, -15, -10, -12, 0, 0};
int currentAngles[] = {90, 90, 90, 90, 90, 90};

// Each motor has its own speed (ms per step, lower = faster)
int motorStepDelay[6] = {40, 40, 60, 40, 40, 40};
//                        ^   ^   ^   ^   ^   ^
//                       M0  M1  M2  M3  M4  M5
//                      fast                 medium

// Movement order: base → shoulder → elbow → wrist → roll → gripper
int moveOrder[6] = {0, 1, 2, 3, 4, 5};

const byte numChars = 32;
char receivedChars[numChars];
boolean newData = false;
int targetAngles[6];

void setup() {
  Serial.begin(115200);
  board.begin();
  board.setPWMFreq(SERVO_FREQ);

  Serial.println("<Arduino is ready and waiting...>");

  // Move all servos to starting position one by one
  for (int i = 0; i < 6; i++) {
    setServoAngle(moveOrder[i], currentAngles[moveOrder[i]]);
    delay(300);
  }
}

void loop() {
  recvWithStartEndMarkers();
  if (newData == true) {
    parseData();
    Serial.println("Moving motors one by one...");
    moveOneByOne(
      targetAngles[0], targetAngles[1], targetAngles[2],
      targetAngles[3], targetAngles[4], targetAngles[5]
    );
    Serial.println("Movement complete.");
    Serial.println("DONE");   // ← Python listens for this
    newData = false;
  }
}

// --- MOVE ONE MOTOR AT A TIME WITH PER-MOTOR SPEED ---
void moveOneByOne(int t0, int t1, int t2, int t3, int t4, int t5) {
  int targets[6] = {t0, t1, t2, t3, t4, t5};
  int steps = 30;

  // ✅ If Motor 3's current angle is > 80, move it first
  if (currentAngles[3] > 80) {
    Serial.println("Motor 3 priority move (angle > 80)...");

    if (targets[3] != currentAngles[3]) {
      Serial.print("Moving motor 3 first from ");
      Serial.print(currentAngles[3]);
      Serial.print(" to ");
      Serial.println(targets[3]);

      float current   = currentAngles[3];
      float target    = targets[3];
      float increment = (target - current) / steps;

      for (int s = 1; s <= steps; s++) {
        current += increment;
        setServoAngle(3, round(current));
        delay(motorStepDelay[3]);
      }

      setServoAngle(3, targets[3]);
      currentAngles[3] = targets[3];
      delay(200);
    }
  }

  // Then move remaining motors in normal order
  for (int idx = 0; idx < 6; idx++) {
    int motor = moveOrder[idx];

    // ✅ Skip Motor 3 if it already moved above
    if (motor == 3 && currentAngles[3] == targets[3]) continue;

    if (targets[motor] == currentAngles[motor]) continue;

    Serial.print("Moving motor ");
    Serial.print(motor);
    Serial.print(" from ");
    Serial.print(currentAngles[motor]);
    Serial.print(" to ");
    Serial.println(targets[motor]);

    float current   = currentAngles[motor];
    float target    = targets[motor];
    float increment = (target - current) / steps;

    for (int s = 1; s <= steps; s++) {
      current += increment;
      setServoAngle(motor, round(current));
      delay(motorStepDelay[motor]);
    }

    setServoAngle(motor, targets[motor]);
    currentAngles[motor] = targets[motor];
    delay(200);
  }
}

// --- SET SERVO ANGLE ---
void setServoAngle(int channel, int angle) {
  int finalAngle = angle;

  // Invert Elbow (Channel 2)
  if (channel == 2) {
    finalAngle = 180 - angle;
    if (finalAngle >= 170) finalAngle = 170;
  }

  // Apply offset
  finalAngle = finalAngle + offsets[channel];

  // Safety limits
  if (finalAngle < 0)   finalAngle = 0;
  if (finalAngle > 180) finalAngle = 180;

  Serial.print("  M"); Serial.print(channel);
  Serial.print(" -> sent: "); Serial.print(finalAngle);
  Serial.print(" (requested: "); Serial.print(angle);
  Serial.print(", offset: "); Serial.print(offsets[channel]);
  Serial.println(")");

  int pulse = map(finalAngle, 0, 180, SERVOMIN, SERVOMAX);
  board.setPWM(channel, 0, pulse);
}

// --- RECEIVE SERIAL DATA ---
void recvWithStartEndMarkers() {
  static boolean recvInProgress = false;
  static byte ndx = 0;
  char startMarker = '<';
  char endMarker   = '>';
  char rc;

  while (Serial.available() > 0 && newData == false) {
    rc = Serial.read();

    if (recvInProgress == true) {
      if (rc != endMarker) {
        receivedChars[ndx] = rc;
        ndx++;
        if (ndx >= numChars) ndx = numChars - 1;
      } else {
        receivedChars[ndx] = '\0';
        recvInProgress = false;
        ndx = 0;
        newData = true;
      }
    } else if (rc == startMarker) {
      recvInProgress = true;
    }
  }
}


// --- PARSE INCOMING DATA ---
void parseData() {
  char* strtokIndx;
  strtokIndx = strtok(receivedChars, ",");
  targetAngles[0] = atoi(strtokIndx);

  for (int i = 1; i < 6; i++) {
    strtokIndx = strtok(NULL, ",");
    if (strtokIndx != NULL) {
      targetAngles[i] = atoi(strtokIndx);
    }
  }

  // ✅ Minimum angle limits for Motor 3 and Motor 4
  if (targetAngles[2] < 15) targetAngles[2] = 15;
  if (targetAngles[3] < 13) targetAngles[3] = 13;

  Serial.print("Received targets: ");
  for (int i = 0; i < 6; i++) {
    Serial.print(targetAngles[i]);
    if (i < 5) Serial.print(", ");
  }
  Serial.println();
}