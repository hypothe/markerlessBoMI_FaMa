import scripts.JointMapper as JointMapper
import scripts.tk_utilities as tk_utils
import tkinter as tk

# MAIN
if __name__ == "__main__":
    # initialize tkinter window
    win = tk_utils.win_init("BoMi Settings")

    JointMapper.JointMapper(win, 2)

    # initiate Tkinter mainloop
    win.mainloop()
