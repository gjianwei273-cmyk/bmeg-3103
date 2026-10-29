#include <cstdio>
typedef unsigned char byte;
struct MockSerial {
  byte input[1024]; int count = 0, pos = 0;
  char output[2048]; int used = 0;
  void begin(int) {} void setTimeout(int) {} void flush() {}
  int available() { return count - pos; }
  int read() { return input[pos++]; }
  void print(char c) { output[used++] = c; }
  void print(long n) {
    char digits[12]; int len = 0;
    if (n < 0) { print('-'); n = -n; }
    do { digits[len++] = '0' + n % 10; n /= 10; } while(n);
    while(len) print(digits[--len]);
  }
  void println(long n) { print(n); println(); }
  void println() { print('\n'); }
} Serial;
#include "../demo_code/BMEG3103/Arduino/Arduino.ino"
extern "C" {
void reset() { Serial.count = Serial.pos = Serial.used = 0; }
void push(int b) { Serial.input[Serial.count++] = b; }
void run() { while(Serial.available()) read_serial_data(); }
int output_size() { return Serial.used; }
int output_at(int i) { return Serial.output[i]; }
}
int main() {
  int b;
  while (std::scanf("%d", &b) == 1) push(b);
  run();
  for (int i = 0; i < output_size(); ++i) std::putchar(output_at(i));
}
