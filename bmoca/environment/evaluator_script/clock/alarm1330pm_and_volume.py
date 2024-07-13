import os
import time
import sqlite3
import subprocess
import regex as re
import pandas as pd


_WORK_PATH = os.environ["BMOCA_HOME"]


def check_alarm1330pm_and_volume(driver):
    command = "adb shell"
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    )
    process.stdin.write("su\n")
    process.stdin.flush()
    process.stdin.write(
        "cp /data/user_de/0/com.google.android.deskclock/databases/alarms.db /sdcard/alarms.db\n"
    )
    process.stdin.flush()
    time.sleep(0.5)
    command = f"adb pull /sdcard/alarms.db {_WORK_PATH}/bmoca/environment/evaluator_script/clock\n"
    _ = subprocess.run(command, text=True, shell=True)

    # check 9am alarm

    conn = sqlite3.connect(
        f"{_WORK_PATH}/bmoca/environment/evaluator_script/clock/alarms.db"
    )
    table_name = "alarm_templates"
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    result_1330 = df[(df["hour"] == 13) & (df["minutes"] == 30)]

    # check volume
    command = "adb shell media volume --show --stream 4 --get"
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    )
    stdout, stderr = process.communicate()
    volume_match = re.search(r"volume is (\d+)", stdout)
    volume = int(volume_match.group(1))

    return not result_1330.empty and volume > 1
