import sqlite3
import json

conn = sqlite3.connect('pipeline/checkpoints.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print('Tables:', cursor.fetchall())

cursor.execute("SELECT thread_id, checkpoint FROM checkpoints ORDER BY thread_id DESC LIMIT 1;")
row = cursor.fetchone()
if row:
    print('Thread ID:', row[0])
    cp = json.loads(row[1])
    state = cp.get('channel_values', {})
    errors = state.get('errors', [])
    print('Errors:')
    for err in errors:
        print(err)
    print('\nLatest Logs (last 10):')
    logs = state.get('logs', [])
    for log in logs[-10:]:
        print(log)
else:
    print("No checkpoints found.")
