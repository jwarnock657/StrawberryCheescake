import time
import matplotlib.pyplot as plt
import numpy as np
import csv
import os


def get_stats():
    dict_ = {}
    with open(os.path.join("soundAnalysis", "voiceAnalysis.csv"), "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for d in reader:
            dict_=d

    return dict_


fig, ax = plt.subplots()
ind = np.arange(1, 11)

# show the figure, but do not block
plt.show(block=False)



txt = ax.text(0,0,"")
fap, fcp, ffp, fhp, fsp, map, mcp, mfp, mhp, msp = plt.bar(ind, 0)

ax.set_xticks(ind)
ax.set_xticklabels(get_stats().keys())
ax.set_ylim([0, 1])
ax.set_ylabel('Percent usage')
ax.set_title('System Monitor')
for tick in ax.get_xticklabels():
    tick.set_rotation(45)


start = time.time()
for i in range(200):  # run for a little while
    d = get_stats()
    print(type(d["female_angry"]))
    # update the animated artists
    fap.set_height(float(d["female_angry"]))
    fcp.set_height(float(d["female_calm"]))
    ffp.set_height(float(d["female_fearful"]))
    fhp.set_height(float(d["female_happy"]))
    fsp.set_height(float(d["female_sad"]))
    map.set_height(float(d["male_angry"]))
    mcp.set_height(float(d["male_calm"]))
    mfp.set_height(float(d["male_fearful"]))
    mhp.set_height(float(d["male_happy"]))
    msp.set_height(float(d["male_sad"]))

    with open(os.path.join("soundAnalysis", "textAnalysisOutput.txt"), 'r') as file:
        val = int(file.readline())
        if val == 0:
            txt.set_text(str(val))
        elif val==1:
            txt.set_text(str(val))

    # ask the canvas to re-draw itself the next time it
    # has a chance.
    # For most of the GUI backends this adds an event to the queue
    # of the GUI frameworks event loop.
    fig.canvas.draw_idle()
    try:
        # make sure that the GUI framework has a chance to run its event loop
        # and clear any GUI events.  This needs to be in a try/except block
        # because the default implementation of this method is to raise
        # NotImplementedError
        fig.canvas.flush_events()
    except NotImplementedError:
        pass

    time.sleep(1)
stop = time.time()
print("{fps:.1f} frames per second".format(fps=200 / (stop - start)))