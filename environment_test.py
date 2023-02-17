import json
import psycopg2
from datetime import datetime

key = json.load(open('key.json'))

conn = psycopg2.connect(dbname=key['database'], user=key['user'], host=key['host'],
                        password=key['password'], port=key['port'])
cursor = conn.cursor()

timestamp = datetime.utcnow().timestamp() * 1000000
temperature = 23
pressure = 1000
humidity = 40
light = 1500
proximity = 0
reducing = 1
oxidizing = 1
ammonia = 1

query = f'INSERT INTO environment (timestamp, temperature, pressure, humidity, ' \
        f'light, proximity, reducing, oxidizing, ammonia) VALUES({timestamp}, {temperature}, {pressure}, {humidity}, ' \
        f'{light}, {proximity}, {reducing}, {oxidizing}, {ammonia})'
print(query)

cursor.execute(query)
conn.commit()

query = f'SELECT * FROM environment'
print(query)

cursor.execute(query)
data = cursor.fetchall()
print(data)

query = f'DELETE FROM environment'
print(query)
cursor.execute(query)
conn.commit()

query = f'SELECT * FROM environment'
print(query)

cursor.execute(query)
data = cursor.fetchall()
print(data)

cursor.close()
conn.close()
