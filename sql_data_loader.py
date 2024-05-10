import _sqlite3

conn = _sqlite3.connect("data/historic_matches.db")

cursor = conn.cursor()

cursor.execute("SELECT * FROM match_data")

rows = cursor.fetchall()


conn.close()