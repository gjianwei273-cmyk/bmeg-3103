Arduino regression tests compile the real sketch with an in-memory serial adapter.
Requires a C++11 compiler and Node.js. On Windows add the compiler bin to PATH.

```powershell
g++ -std=c++11 -Wall -Wextra -Werror -static tests/arduino_harness.cpp -o tests/arduino-test.exe
node tests/arduino.test.mjs tests/arduino-test.exe
arduino-cli compile --fqbn arduino:avr:uno demo_code/BMEG3103/Arduino
```

Checks big-endian 24-bit powers (including 0, 0x800000 and 0xFFFFFF),
nine-column CSV rows, consecutive packets, checksum rejection, quality filtering,
small packets containing sync bytes, and invalid summary field markers.
The host tests do not replace an AVR build or a physical hardware test.
