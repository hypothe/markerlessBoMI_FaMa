import math
import tkinter as tk

import pyautogui
import scripts.tk_utils as tk_utils

uppercase = False  # use uppercase chars.
shift_on = False



class KeyBoard_Top(object):

	def __init__(self, master):
		top = self.top = tk.Toplevel(master)

		self.top.title('keyboard 12')
		self.top['bg'] = 'powder blue'
	
		#screensize = tk_utils.get_window_res_from_geometry(tk_utils.get_curr_screen_geometry())

		#screen_width = screensize[0]
		#screen_height = screensize[1]
		screen_width, screen_height = pyautogui.size()

		window_width = math.ceil(screen_width / 2)
		window_height = math.ceil(screen_height / 2)


		x_cordinate = int((screen_width / 2) - (window_width / 2))
		y_cordinate = int((screen_height / 2) - (window_height / 2))

		self.top.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

		#self.top.resizable(0, 0)

		self.label1 = tk.Label(self.top, text="Group 12 keyboard", font=('arial', 30, 'bold'),
					   bg='powder blue', fg="#000000")
		self.label1.grid(row=0, column=0, columnspan=14)
		self.entry = tk.Text(self.top, height=4, font=('arial', 10, 'bold'))
		self.entry.grid(row=1, column=0, columnspan=14)

		buttons = ['1','2','3','4','5','6','7','8','9','0','-','=','<--',
				   'Tab','q','w','e','r','t','y','u','i','o','p','[',']',"\\",
				'Caps','a','s','d','f','g','h','j','k','l',';',"'",'Enter',
				'Shift','z','x','c','v','b','n','m',',','.','/','Shift',
				 'Space']

		var_row = 2
		var_col = 0
		# List of buttons
		self.btns = []

		for button in buttons:
			command = lambda x = button: self.select(self.entry, x)
			if button != 'Space':
				btn = tk.Button(self.top, text=button, width=4, bd=12, font=('arial', 10, 'bold'),
							   activebackground="#ffffff", activeforeground="#000990", relief='raised',
							   command=command)
				btn.grid(row=var_row, column=var_col)
			else:
				btn = tk.Button(self.top, text=button, width=100, padx=3, pady=3, bd=12, font=('arial', 10, 'bold'),
							   bg="#cccccc", activebackground="#ffffff", activeforeground="#000990", relief='raised',
							   command=command)
				btn.grid(row=var_row + 1, column=var_col, columnspan=14)

			self.btns.append(btn)
			var_col += 1

			if var_col > 12:
				var_col = 0
				var_row += 1

		# Add save button
		self.save_btn = tk.Button(self.top, text='Save', width=4, padx=3, pady=3, bd=12, font=('arial', 10, 'bold'),
						bg="#22bb44", activebackground="#11aa33", activeforeground="#000990", relief='raised',
						command=command).grid(row=var_row + 2, column=1)
		self.top.update_idletasks()
	
	def select(self, entry, value):
		global uppercase, shift_on

		if value == 'Space':
			entry.insert(tk.END, ' ')
		elif value == 'Tab':
			entry.insert(tk.END, '\t')
		elif value == 'Enter':
			entry.insert(tk.END, '\n')
		elif value == '<--':
			if isinstance(entry, tk.Entry):
				entry.delete(len(entry.get())-1, 'end')
			#elif isinstance(entry, tk.Text):
			else: # tk.Text
				entry.delete('end - 2c', 'end')
		elif value in ('Caps', 'Shift'):
			uppercase = not uppercase
			if value == 'Shift':
				shift_on = True# change True to False, or False to True
		else:
			if uppercase:
				value = value.upper()
				if shift_on == True:
					uppercase = not uppercase
					shift_on == False
			entry.insert('end', value)

	def cleanup(self):
		self.top.destroy()