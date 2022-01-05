import tkinter as tk
from tkinter import *
import tkinter

uppercase = False  # use uppercase chars.
shift_on = False



def raise_above(window):
    window.lift()
    window.attributes('-topmost', True)
    window.attributes('-topmost', False)

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


class KeyBoard_Top(tk.Toplevel):

    def __init__(self, master):
        super().__init__(master)
        self.title('My keyboard')
        self['bg'] = 'powder blue'
        self.resizable(0, 0)


        label1 = Label(self, text="Group 12 keyboard", font=('arial', 30, 'bold'),
                       bg='powder blue', fg="#000000").grid(row=0, column=0, columnspan=14)
        entry = Text(self, width=140, font=('arial', 10, 'bold'))
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
                btn = tkinter.Button(self, text=button, width=4, bd=12, font=('arial', 10, 'bold'),
                               activebackground="#ffffff", activeforeground="#000990", relief='raised',
                               command=command)
                btn.grid(row=var_row, column=var_col)
            if button == 'Space':
                spc_btn = tkinter.Button(self, text=button, width=100, padx=3, pady=3, bd=12, font=('arial', 10, 'bold'),
                               activebackground="#ffffff", activeforeground="#000990", relief='raised',
                               command=command)
                spc_btn.grid(row=var_row + 1, column=var_col, columnspan=14)
            var_col += 1

            if var_col > 12:
                var_col = 0
                var_row += 1

        # Add save button
        tkinter.Button(self, text='Save', width=4, padx=3, pady=3, bd=12, font=('arial', 10, 'bold'),
                        activebackground="#ffffff", activeforeground="#000990", relief='raised',
                        command=command).grid(row=var_row + 2, column=1)


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.geometry('300x200')
        self.title('Main Window')

        # place a button on the root window
        tk.Button(self,
                text='Open a window',
                command=self.open_window).pack(expand=True)

    def open_window(self):
        kb = KeyBoard_Top(self)
        kb.grab_set()

if __name__ == "__main__":
    app = App()
    app.mainloop()