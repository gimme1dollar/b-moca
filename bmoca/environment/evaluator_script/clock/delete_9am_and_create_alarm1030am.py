import os
import time
import sqlite3
import subprocess
import pandas as pd


_WORK_PATH = os.environ['BMOCA_HOME']
    
def check_delete_9am_and_create_alarm1030am(driver):
    command = 'adb shell'
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    process.stdin.write('su\n')
    process.stdin.flush()
    process.stdin.write('cp /data/user_de/0/com.google.android.deskclock/databases/alarms.db /sdcard/alarms.db\n')
    process.stdin.flush()
    time.sleep(0.5)
    command = f'adb pull /sdcard/alarms.db {_WORK_PATH}/bmoca/environment/evaluator_script/clock\n'
    _ = subprocess.run(command, text=True, shell=True)    
    
    conn = sqlite3.connect(f'{_WORK_PATH}/bmoca/environment/evaluator_script/clock/alarms.db')
    table_name = 'alarm_templates'
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    
    result_0900 = df[(df['hour'] == 9) & (df['minutes'] == 00)]
    result_1030 = df[(df['hour'] == 10) & (df['minutes'] == 30)]
    if result_0900.empty and not result_1030.empty:
        return True
    return False