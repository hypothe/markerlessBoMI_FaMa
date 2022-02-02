import tkinter as tk
import math
import pyautogui

# GLobal styling (CSS-like)
GUIStyle = {
		'font_main' : ("Times", 18),
		'font_help' : ("Times", 8, 'normal', 'italic'),
		'activeforeground' : 'blue',
		'activebackground' : '#4682b4',
		'bg' : '#abcdef',
		'btn_padx' : 20,
		'btn_pady' : (20, 30),
		'check_padx' : (0, 20),
		'check_pady' : (20, 30),
		'lbl_pady' : (20, 30),
		'ent_pady' : (20, 30),
		'txt_custom_pady' : (40, 20),
		'lbl_custom_padx' : (40, 20),
		'lbl_custom_pady' : (40, 20),
		'btn_custom_padx' : (80, 0),
		'btn_custom_pady' : (40, 20),
		'close_bg' : '#dd1122',
		'close_activebackground' : 'red',
		'fg_help' : '#111111'
}

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
	win['bg'] = GUIStyle['bg']

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

class HelpBoxCollection(object):
	"""
	Class managing 'Help' boxes.
	Boxes can be added 
	"""
	def __init__(self, win):
		self.labels = {}
		self.win = win
		self.font_size = 8
		self.visible = False
		self.max_txt_len = 30

	def add_info(self, tk_obj, tk_name, tk_text):
		# Overwrite if existent
		#if tk_name in self.labels.keys():
		#	return
		#tk_text = '-\n'.join([tk_text[i:min(len(tk_text), i+self.max_txt_len)] for i in range(0, len(tk_text), self.max_txt_len)])
		try:
			info = tk_obj.grid_info()
		except AttributeError:
			pass
		else:
			wd = tk.Label(self.win, text=tk_text, bg=GUIStyle['bg'], wraplength=300, justify=tk.LEFT, fg=GUIStyle['fg_help'])
			wd.config(font=GUIStyle['font_help'])
			wd.grid(row=info["row"], column=self._rightmost_free_column(), columnspan=2, pady=GUIStyle['lbl_pady'], sticky='w')
			wd.grid_remove() # do not display it on startup
			wd.update()
			self.labels[tk_name] = wd
	
	def toggle_help(self):
		self.visible = not self.visible

		for lbl in self.labels.values():
			if (self.visible):
				lbl.grid()
			else:
				lbl.grid_remove()
	
	def _rightmost_free_column(self, row=None, col=None):
		col = col and col is not None or 0
		for wd in self.win.grid_slaves(row=row):
			try:
				_col = wd.grid_info()["column"]
				if _col >= col:
					col = _col + wd.grid_info()["columnspan"]
			except KeyError:
				pass
		
		return col

