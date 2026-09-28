"""
robotic_hand_manual.py - Manual keyboard control of the robotic hand.

Flexion keys:
  a -> thumb,  s -> index,  d -> middle,  f -> ring,  g -> pinky

Extension keys:
  z -> thumb,  x -> index,  c -> middle,  v -> ring,  b -> pinky

  h -> extend all fingers
  q -> quit

Requires: pip install pyserial

Message format (26 bytes, matching MotorsMessage C++ struct with default alignment):
  1 byte  : motor_mask   (bitmask, bit N = motor N is active)
  8 bytes : motor_pwms   (uint8 x8, power per motor)
  1 byte  : padding      (compiler alignment for uint16_t)
  16 bytes: encoder_positions (uint16 x8 LE, target position 0-255)

Motor index mapping:
  0 = index (F2)
  1 = middle (F3)
  2 = ring (F4)
  3 = pinky (F5)
  4 = thumb (F1)

Encoder position convention:
  Fingers 0-3: 0 = open, 255 = closed/flexed
  Thumb (4):   255 = open, 0 = closed/flexed  (hardware-inverted)

Reader's guide: ``build_message`` is the protocol boundary and can be tested
without hardware. ``main`` maps keyboard commands to motor targets, while
``send`` performs the serial write used by every command.
"""

import argparse
import struct
import time
import serial

NUM_MOTORS = 8
BAUD_RATE  = 57600

# Motor indices
MOTOR_INDEX  = 0
MOTOR_MIDDLE = 1
MOTOR_RING   = 2
MOTOR_PINKY  = 3
MOTOR_THUMB  = 4

# Flexion keys (close finger)
FLEX_KEYS = {
    'a': MOTOR_THUMB,
    's': MOTOR_INDEX,
    'd': MOTOR_MIDDLE,
    'f': MOTOR_RING,
    'g': MOTOR_PINKY,
}

# Extension keys (open finger)
EXTEND_KEYS = {
    'z': MOTOR_THUMB,
    'x': MOTOR_INDEX,
    'c': MOTOR_MIDDLE,
    'v': MOTOR_RING,
    'b': MOTOR_PINKY,
}

FINGER_NAMES = {
    MOTOR_THUMB:  'thumb',
    MOTOR_INDEX:  'index',
    MOTOR_MIDDLE: 'middle',
    MOTOR_RING:   'ring',
    MOTOR_PINKY:  'pinky',
}

# Encoder positions per state
CLOSED = {MOTOR_INDEX:   0, MOTOR_MIDDLE:   0, MOTOR_RING:   0, MOTOR_PINKY:   0, MOTOR_THUMB:   0}
OPEN   = {MOTOR_INDEX: 255, MOTOR_MIDDLE: 255, MOTOR_RING: 255, MOTOR_PINKY: 255, MOTOR_THUMB: 255}

# Struct format: 1B motor_mask + 8B pwms + 1B padding + 8H positions = 26 bytes
# Matches the C++ MotorsMessage struct with default compiler alignment.
# If the hand is not responding, check that sizeof(struct_message) on the Arduino is 26.
MSG_FORMAT = '<B8Bx8H'
MSG_SIZE   = struct.calcsize(MSG_FORMAT)  # should be 26


def build_message(motor_commands: dict, pwm: int) -> bytes:
    """
    Build a MotorsMessage bytestring.

    motor_commands: {motor_index: encoder_position (0-255)}
    pwm: motor speed for all active motors (0-255)
    """
    motor_mask = 0
    pwms      = [0] * NUM_MOTORS
    positions = [0] * NUM_MOTORS

    for idx, pos in motor_commands.items():
        motor_mask      |= (1 << idx)
        pwms[idx]        = pwm
        positions[idx]   = pos

    return struct.pack(MSG_FORMAT, motor_mask, *pwms, *positions)


def send(ser: serial.Serial, motor_commands: dict, pwm: int) -> None:
    msg = build_message(motor_commands, pwm)
    assert len(msg) == MSG_SIZE, f"Message size mismatch: {len(msg)} != {MSG_SIZE}"
    ser.write(msg)


def main():
    parser = argparse.ArgumentParser(description='Manual robotic hand control via keyboard')
    parser.add_argument('--port', '-p', required=True,
                        help='Serial COM port (e.g. COM3 or /dev/ttyUSB0)')
    parser.add_argument('--pwm', type=int, default=120,
                        help='Motor PWM speed 0-255 (default: 120)')
    args = parser.parse_args()

    print(f"Message size: {MSG_SIZE} bytes")
    print(f"Connecting to {args.port} at {BAUD_RATE} baud...")
    ser = serial.Serial(args.port, BAUD_RATE, timeout=1)
    time.sleep(2)  # allow ESP32 to finish booting after serial open
    print("Connected.\n")

    # finger state: True = closed/flexed, False = open/extended
    states = {m: False for m in FINGER_NAMES}

    print("Controls:")
    print("  Flex:   a=thumb  s=index  d=middle  f=ring  g=pinky")
    print("  Extend: z=thumb  x=index  c=middle  v=ring  b=pinky")
    print("  h = extend all fingers")
    print("  q = quit")
    print()

    print("Hand ready.")

    while True:
        try:
            key = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if key == 'q':
            break

        elif key == 'h':
            for m in states:
                states[m] = False
            send(ser, {m: OPEN[m] for m in states}, args.pwm)
            print("All fingers: open")

        elif key in FLEX_KEYS:
            motor = FLEX_KEYS[key]
            states[motor] = True
            send(ser, {motor: CLOSED[motor]}, args.pwm)
            print(f"{FINGER_NAMES[motor]}: closed")

        elif key in EXTEND_KEYS:
            motor = EXTEND_KEYS[key]
            states[motor] = False
            send(ser, {motor: OPEN[motor]}, args.pwm)
            print(f"{FINGER_NAMES[motor]}: open")

        else:
            print(f"Unknown key '{key}'. Valid: a-g (flex), z-b (extend), h, q")

    # Open all fingers before exit
    send(ser, {m: OPEN[m] for m in states}, args.pwm)
    ser.close()
    print("Disconnected.")


if __name__ == '__main__':
    main()
