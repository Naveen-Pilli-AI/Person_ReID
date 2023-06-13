from datetime import datetime
prev= datetime.today()  # Get timezone naive now
seconds = prev.timestamp()
for i in range(0, int(seconds)):
    cur=datetime.today() 
    cur_seconds = cur.timestamp()
    total_seconds=cur_seconds-seconds
    print(total_seconds)