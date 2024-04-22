#!/bin/bash
#Turn off emulator
port=$1
adb -s emulator-$port emu kill;
echo "Emulator turn off complete"
sleep 10