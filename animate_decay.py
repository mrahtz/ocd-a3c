import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque

fig, ax = plt.subplots()
maxlen = 100
ax.set_xlim([0, maxlen])
ax.set_ylim([0, 11])
line, = ax.plot([], [])

fig.show()
fig.canvas.draw()
bg = fig.canvas.copy_from_bbox(ax.bbox)

d = deque(maxlen=maxlen)
d.append(0)


def run():
    ydata = list(d)
    xdata = list(range(len(d)))
    d.append(np.random.randint(low=1, high=10))
    line.set_data(xdata, ydata)
    return line


start = time.time()
frames = 0
while True:
    line = run()

    fig.canvas.restore_region(bg)
    ax.draw_artist(line)
    fig.canvas.update()

    fig.canvas.flush_events()
    time.sleep(1/60.0)

    frames += 1
    if (time.time() - start) > 1.0:
        print("%d fps" % frames)
        start = time.time()
        frames = 0
