/*
 * TGAM: one CSV row per valid summary packet, at 57600 baud.
 * Columns: Delta,Theta,LowAlpha,HighAlpha,LowBeta,HighBeta,
 *          LowGamma,MiddleGamma,Attention.
 * Only signal quality 0 is printed. Raw samples are consumed but not printed.
 */
#define BAUDRATE 57600

byte payloadData[32];
long feature[9];

void setup() {
  Serial.begin(BAUDRATE);
}

byte ReadOneByte() {
  while (!Serial.available()) {}
  return (byte)Serial.read();
}

void passFeature(long values[], int len) {
  for (int i = 0; i < len; ++i) {
    if (i > 0) Serial.print(',');
    Serial.print(values[i]);
  }
  Serial.println();
}

void read_serial_data() {
  if (ReadOneByte() != 0xAA) return;
  if (ReadOneByte() != 0xAA) return;

  byte payloadLength = ReadOneByte();
  // Extra sync bytes may precede the length. Valid TGAM payloads are <=169.
  while (payloadLength == 0xAA) payloadLength = ReadOneByte();
  if (payloadLength > 169) return;

  byte sum = 0;
  for (int i = 0; i < payloadLength; ++i) {
    byte value = ReadOneByte();
    sum += value;
    if (i < 32) payloadData[i] = value;
  }
  byte checksum = ReadOneByte();
  if (checksum != (byte)(~sum)) return;
  // Consume every complete packet before ignoring non-summary packets.
  if (payloadLength != 32) return;
  if (payloadData[0] != 0x02 || payloadData[2] != 0x83 ||
      payloadData[3] != 24 || payloadData[28] != 0x04 ||
      payloadData[30] != 0x05) return;
  if (payloadData[1] != 0) return;

  for (int i = 0; i < 8; ++i) {
    int j = 4 + 3 * i;
    // Promote BEFORE shifting: AVR int is only 16 bits.
    feature[i] = ((long)payloadData[j] << 16) |
                 ((long)payloadData[j + 1] << 8) |
                 (long)payloadData[j + 2];
  }
  feature[8] = payloadData[29];
  passFeature(feature, 9);
}

void loop() {
  read_serial_data();
}
