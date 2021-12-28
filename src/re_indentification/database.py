import sqlite3

con = None


class Tracker:
    def __init__(self, tracker_id, hash, predict_name, location, start, end):
        self.tracker_id = tracker_id
        self.hash = hash
        self.predict_name = predict_name
        self.location = location
        self.start = start
        self.end = end


def getConnection():
    databaseFile = "./tracker_db.db"
    global con
    if con == None:
        con = sqlite3.connect(databaseFile)
    return con


def createTable(con):
    try:
        c = con.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS Tracker (tracker_id INTEGER PRIMARY KEY AUTOINCREMENT,hash,predict_name,location, start, end)""")
    except Exception as e:
        pass


def insert(con, hash, predict_name, location, start, end):
    c = con.cursor()
    c.execute("""INSERT INTO Tracker (hash,predict_name,location, start, end) values(?, ?, ?,?,?)""", (
        hash, predict_name, location, start, end))
    con.commit()


def update(con, tracker_id, end):
    c = con.cursor()
    c.execute("UPDATE Tracker SET end = ? WHERE tracker_id = ? ",
              (end, tracker_id))
    con.commit()


def select(con, hash, predict_name, location):
    cur = con.execute(
        "SELECT * FROM Tracker WHERE hash = ? and predict_name = ? and location = ? ORDER BY tracker_id DESC LIMIT 1", (hash, predict_name, location))
    return cur.fetchall()


def select_all(con, hash):
    cur = con.execute(
        "SELECT * FROM Tracker WHERE hash = ? ORDER BY predict_name ", (hash,))
    return cur.fetchall()
