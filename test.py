import time

print("start loop...")
for _ in range(100):
    try:
        for _ in range(100):
            try:
                print("loop")
                time.sleep(5)
            except KeyboardInterrupt:
                break
    except KeyboardInterrupt:
        break

print("continue after loop...")

