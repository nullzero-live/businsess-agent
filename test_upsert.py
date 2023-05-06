import storage as stg

with open("chat_logs.txt", "r") as f:
    text = f.read()
    stg.embed_upsert(text)