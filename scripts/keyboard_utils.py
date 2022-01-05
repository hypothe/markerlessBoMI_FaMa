"""
This scripts contains all the function needed for interacting with a digital keyboard.
"""
import time
import tkinter
import tkinter as tk
from tkinter import *
from tkinter import messagebox
import scripts.tk_utils as tk_utils

uppercase = False  # use uppercase chars.
shift_on = False

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        keyboard_app.destroy()

def select(entry, value):
    global uppercase, shift_on

    if value == 'Space':
        entry.insert(tkinter.END, ' ')
    elif value == 'Tab':
        entry.insert(tkinter.END, '\t')
    elif value == 'Enter':
        entry.insert(tkinter.END, '\n')
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

def keyboard_interface(root = None, start = True):
    """
    function to define the digital keyboard
    :return:
    """
    #global keyboard_app

    if start:

        if root is None:
            keyboard_app = tk.Tk()
            print('Main')
        else:
            keyboard_app = root #tk.Toplevel(root)
            print('Top')

        keyboard_app.title("My keyboard")
        keyboard_app['bg'] = 'powder blue'
        keyboard_app.resizable(0, 0)

        label1 = Label(keyboard_app, text="Group 12 keyboard", font=('arial', 30, 'bold'),
                       bg='powder blue', fg="#000000").grid(row=0, column=0, columnspan=14)
        entry = Text(keyboard_app, width=140, font=('arial', 10, 'bold'))
        entry.grid(row=1, column=0, columnspan=14)

        buttons = ['1','2','3','4','5','6','7','8','9','0','-','=','<--',
                   'Tab','q','w','e','r','t','y','u','i','o','p','[',']',"\\",
                'Caps','a','s','d','f','g','h','j','k','l',';',"'",'Enter',
                'Shift','z','x','c','v','b','n','m',',','.','/','Shift',
                 'Space']

        var_row = 2
        var_col = 0

        for button in buttons:
            command = lambda x = button: select(entry, x)
            if button != 'Space':
                btn = tkinter.Button(keyboard_app, text=button, width=4, bd=12, font=('arial', 10, 'bold'),
                               activebackground="#ffffff", activeforeground="#000990", relief='raised',
                               command=command)
                btn.grid(row=var_row, column=var_col)
            if button == 'Space':
                spc_btn = tkinter.Button(keyboard_app, text=button, width=100, padx=3, pady=3, bd=12, font=('arial', 10, 'bold'),
                               activebackground="#ffffff", activeforeground="#000990", relief='raised',
                               command=command)
                spc_btn.grid(row=var_row + 1, column=var_col, columnspan=14)
            var_col += 1

            if var_col > 12:
                var_col = 0
                var_row += 1

        # Add save button
        tkinter.Button(keyboard_app, text='Save', width=4, padx=3, pady=3, bd=12, font=('arial', 10, 'bold'),
                        activebackground="#ffffff", activeforeground="#000990", relief='raised',
                        command=command).grid(row=var_row + 2, column=1)

        # Visualize the keyboard
        # keyboard_app.protocol("WM_DELETE_WINDOW", on_closing())
        if root is None:
            keyboard_app.mainloop()
        else:
            keyboard_app.grab_set()

        try:
            tk_utils.raise_above(keyboard_app)
        except:
            pass

        time.sleep(0.001)

    else:
        pass
        #try:
        #    keyboard_app.destroy()
        #except:
        #    print('Nothing to destroy')
    return keyboard_app



if __name__ == "__main__":

    keyboard_interface()
