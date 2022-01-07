import tkinter as tk
import math
import pyautogui

# Define some colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
CURSOR = (0.19 * 255, 0.65 * 255, 0.4 * 255)

# WORKAROUND in order to get the correct screen resolution in a
# multi-screen setup
def get_curr_screen_geometry():
	"""
	Workaround to get the size of the current screen in a multi-screen setup.

	Returns:
		geometry (str): The standard Tk geometry string.
			[width]x[height]+[left]+[top]
	"""
	root = tk.Tk()
	root.update_idletasks()
	root.attributes('-fullscreen', True)
	#root.attributes('-alpha', 0.1)
	root.state('iconic')
	geometry = root.winfo_geometry()
	root.destroy()
	return geometry

def get_window_res_from_geometry(geomStr):
	geo = [s.split('+') for s in geomStr.split('x')]
	geo = [it for its in geo for it in its]
	return (int(geo[0]), int(geo[1]))

def win_init(win_title):

	win = tk.Tk()
	win.title(win_title)
	#user32 = ctypes.windll.user32
	#screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

	#screen_width = win.winfo_screenwidth()
	#screen_height = win.winfo_screenheight()

	#screensize = get_window_res_from_geometry(get_curr_screen_geometry())
	
	#screen_width = screensize[0]
	#screen_height = screensize[1]
	
	screen_width, screen_height = pyautogui.size()

	window_width = math.ceil(screen_width / 1.2)
	window_height = math.ceil(screen_height / 1.2)


	x_cordinate = int((screen_width / 2) - (window_width / 2))
	y_cordinate = int((screen_height / 2) - (window_height / 2))

	win.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

	return win


def raise_above_all(window):
	window.lift()
	window.attributes('-topmost', True)
	window.attributes('-topmost', False)


class popupWindow(object):
	"""
	class that defines the popup tkinter window
	"""

	def __init__(self, master, msg):
		top = self.top = tk.Toplevel(master)
		self.lbl = tk.Label(top, text=msg)
		self.lbl.pack()
		self.btn = tk.Button(top, text='Ok', command=self.cleanup)
		self.btn.pack()
		self.status = False

	def cleanup(self):
		self.status = True
		self.top.destroy()
