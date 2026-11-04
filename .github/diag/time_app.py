import sys, time
sys.path.insert(0, ".")
t = time.time()
from src.database import read_table
print("import + engine", round(time.time() - t, 2))
for name in ["players_master", "club_profiles", "player_value_history"]:
    t = time.time(); d = read_table(name); print(name, d.shape, round(time.time() - t, 2), "s")
t = time.time()
read_table("player_timeline", columns=lambda c: c.lower() in {"pkey","player","season","squad","min"} or c.lower().endswith("_per90"))
print("player_timeline (filtered)", round(time.time() - t, 2), "s")
from streamlit.testing.v1 import AppTest
for i in range(2):
    t = time.time()
    at = AppTest.from_file("/home/runner/work/StatTrick/StatTrick/frontend/appsujal.py", default_timeout=300).run()
    print(f"app run {i+1}: {round(time.time() - t, 1)} s, exceptions: {[e.value for e in at.exception][:2]}")
