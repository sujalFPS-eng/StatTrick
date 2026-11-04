import os, time
from sqlalchemy import create_engine, text
url = os.environ["DATABASE_URL"]
if url.startswith("postgres://"): url = "postgresql://" + url[len("postgres://"):]
if url.startswith("postgresql://"): url = "postgresql+psycopg://" + url[len("postgresql://"):]
t = time.time()
try:
    e = create_engine(url, connect_args={"connect_timeout": 30})
    with e.connect() as c:
        print("connected in", round(time.time()-t, 1), "s")
        for tbl in ["players_master", "club_profiles", "player_timeline", "player_value_history"]:
            try:
                print(tbl, c.execute(text(f'select count(*) from "{tbl}"')).scalar())
            except Exception as ex:
                print(tbl, "ERROR", type(ex).__name__, str(ex)[:300]); c.rollback()
except Exception as ex:
    print("DB CONNECT FAILED after", round(time.time()-t, 1), "s:", type(ex).__name__, str(ex)[:500])
