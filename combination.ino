#include <Servo.h>

Servo doorServo;

// =========================
// PINS
// =========================

const int piezoPin = A0;
const int redLED = 8;
const int greenLED = 7;
const int servoPin = 6;
const int buzzer = 9;

// =========================
// KNOCK SETTINGS
// =========================

const int knockThreshold = 180;

const unsigned long DEBOUNCE = 350;

// P1: 5 seconds ± 1 second
const unsigned long P1_MIN = 4000;
const unsigned long P1_MAX = 6000;

// P2: within 1 second
const unsigned long P2_MAX = 1000;


// =========================
// STATES
// =========================

bool waitingForFace = true;
bool waitingForPassword = false;

int passwordNumber = 1;

int knockCount = 0;

unsigned long firstKnockTime = 0;
unsigned long lastKnockTime = 0;
unsigned long lastKnockDetected = 0;

bool previousKnock = false;


// =========================
// LED BLINKING
// =========================

unsigned long lastBlinkTime = 0;
bool greenBlinkState = false;

const unsigned long BLINK_INTERVAL = 300;


// =========================
// SETUP
// =========================

void setup() {

  Serial.begin(9600);

  pinMode(piezoPin, INPUT);
  pinMode(redLED, OUTPUT);
  pinMode(greenLED, OUTPUT);
  pinMode(buzzer, OUTPUT);

  doorServo.attach(servoPin);
  doorServo.write(0);

  digitalWrite(redLED, LOW);
  digitalWrite(greenLED, LOW);
  noTone(buzzer);

  Serial.println("READY_FACE");
}


// =========================
// GREEN SUCCESS
// =========================

void greenSuccess() {

  digitalWrite(greenLED, HIGH);

  delay(1000);

  digitalWrite(greenLED, LOW);
}


// =========================
// BLINK GREEN UNTIL KNOCK
// =========================

void blinkGreen() {

  unsigned long now = millis();

  if (now - lastBlinkTime >= BLINK_INTERVAL) {

    lastBlinkTime = now;

    greenBlinkState = !greenBlinkState;

    digitalWrite(
      greenLED,
      greenBlinkState ? HIGH : LOW
    );
  }
}


// =========================
// STOP GREEN BLINK
// =========================

void stopGreenBlink() {

  greenBlinkState = false;

  digitalWrite(greenLED, LOW);
}


// =========================
// ERROR
// =========================

void errorSignal() {

  digitalWrite(redLED, HIGH);

  // BUZZER ON
  tone(buzzer, 2000);
  delay(300);

  // BUZZER OFF
  noTone(buzzer);
  delay(150);

  // BUZZER ON AGAIN
  tone(buzzer, 2000);
  delay(300);

  // BUZZER OFF
  noTone(buzzer);

  delay(500);

  digitalWrite(redLED, LOW);
}


// =========================
// RESET TO FACE
// =========================

void resetToFace() {

  waitingForFace = true;
  waitingForPassword = false;

  knockCount = 0;

  previousKnock = false;

  stopGreenBlink();

  digitalWrite(redLED, LOW);
  noTone(buzzer);

  doorServo.write(0);

  Serial.println("READY_FACE");
}


// =========================
// WRONG PASSWORD
// =========================

void wrongPassword() {

  Serial.println("WRONG_PASSWORD");

  stopGreenBlink();

  errorSignal();

  resetToFace();
}


// =========================
// UNLOCK
// =========================

void unlockDoor() {

  Serial.println("PASSWORD_CORRECT");

  stopGreenBlink();

  // Green ON for 1 second
  greenSuccess();

  Serial.println("UNLOCK");

  doorServo.write(90);

  // Door stays unlocked for 2 seconds
  delay(2000);

  Serial.println("LOCK");

  doorServo.write(0);

  // RESET TIME = 5 SECONDS
  delay(5000);

  // Alternate password
  if (passwordNumber == 1) {
    passwordNumber = 2;
  }
  else {
    passwordNumber = 1;
  }

  resetToFace();
}


// =========================
// CHECK P1
// =========================

void checkP1() {

  unsigned long gap =
    lastKnockTime - firstKnockTime;

  Serial.print("P1 GAP = ");
  Serial.print(gap);
  Serial.println(" ms");

  if (gap >= P1_MIN && gap <= P1_MAX) {

    Serial.println("P1_CORRECT");

    unlockDoor();

  }
  else {

    wrongPassword();
  }
}


// =========================
// CHECK P2
// =========================

void checkP2() {

  unsigned long gap =
    lastKnockTime - firstKnockTime;

  Serial.print("P2 GAP = ");
  Serial.print(gap);
  Serial.println(" ms");

  if (gap <= P2_MAX) {

    Serial.println("P2_CORRECT");

    unlockDoor();

  }
  else {

    wrongPassword();
  }
}


// =========================
// READ KNOCK
// =========================

void readKnocks() {

  int value = analogRead(piezoPin);

  unsigned long now = millis();

  bool knockDetected =
    value > knockThreshold;


  // Detect LOW → HIGH
  if (knockDetected && !previousKnock) {

    if (now - lastKnockDetected >= DEBOUNCE) {

      lastKnockDetected = now;


      // FIRST KNOCK
      if (knockCount == 0) {

        // STOP GREEN BLINK
        stopGreenBlink();

        knockCount = 1;

        firstKnockTime = now;
        lastKnockTime = now;

        Serial.println("KNOCK_1");
      }


      // SECOND KNOCK
      else if (knockCount == 1) {

        knockCount = 2;

        lastKnockTime = now;

        Serial.println("KNOCK_2");


        if (passwordNumber == 2) {

          checkP2();

          return;
        }
      }
    }
  }

  previousKnock = knockDetected;


  // P1 TIMEOUT AFTER FIRST KNOCK
  if (passwordNumber == 1 &&
      knockCount == 1) {

    if (now - firstKnockTime > P1_MAX) {

      Serial.println("P1_TIMEOUT");

      wrongPassword();

      return;
    }
  }


  // P1 SECOND KNOCK
  if (passwordNumber == 1 &&
      knockCount == 2) {

    checkP1();

    return;
  }


  // P2 TIMEOUT AFTER FIRST KNOCK
  if (passwordNumber == 2 &&
      knockCount == 1) {

    if (now - firstKnockTime > P2_MAX) {

      Serial.println("P2_TIMEOUT");

      wrongPassword();

      return;
    }
  }
}


// =========================
// MAIN LOOP
// =========================

void loop() {

  // =========================
  // RECEIVE PYTHON MESSAGE
  // =========================

  if (Serial.available()) {

    String command =
      Serial.readStringUntil('\n');

    command.trim();


    // =========================
    // CORRECT FACE
    // =========================

    if (command == "FACE_OK" &&
        waitingForFace) {

      Serial.println("FACE_VERIFIED");

      // Green ON for 1 second
      greenSuccess();

      // Start password mode
      waitingForFace = false;
      waitingForPassword = true;

      knockCount = 0;

      // Start blinking
      lastBlinkTime = millis();
      greenBlinkState = true;

      digitalWrite(greenLED, HIGH);

      Serial.print("ENTER_P");
      Serial.println(passwordNumber);
    }


    // =========================
    // WRONG FACE
    // =========================

    else if (command == "FACE_BAD" &&
             waitingForFace) {

      Serial.println("WRONG_FACE");

      errorSignal();

      Serial.println("READY_FACE");
    }
  }


  // =========================
  // PASSWORD MODE
  // =========================

  if (waitingForPassword) {

    // Blink until first knock
    if (knockCount == 0) {

      blinkGreen();
    }

    readKnocks();
  }
}