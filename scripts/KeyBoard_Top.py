import tkinter as tk
import scripts.tk_utils as tk_utils

uppercase = False  # use uppercase chars.
shift_on = False



class KeyBoard_Top(tk_utils.popupWindow):

    def __init__(self, master):
        super().__init__(master, "keyboard_12")
        self.top.title('My keyboard')
        self.top['bg'] = 'powder blue'
        self.top.geometry('300x200')
        self.top.resizable(0, 0)

        self.label1 = tk.Label(self.top, text="Group 12 keyboard", font=('arial', 30, 'bold'),
                       bg='powder blue', fg="#000000").grid(row=0, column=0, columnspan=14)
        self.entry = tk.Text(self.top, width=140, font=('arial', 10, 'bold'))
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
                               activebackground="#ffffff", activeforeground="#000990", relief='raised',
                               command=command)
                btn.grid(row=var_row + 1, column=var_col, columnspan=14)

            self.btns.append(btn)
            var_col += 1

            if var_col > 12:
                var_col = 0
                var_row += 1

        # Add save button
        self.save_btn = tk.Button(self.top, text='Save', width=4, padx=3, pady=3, bd=12, font=('arial', 10, 'bold'),
                        activebackground="#ffffff", activeforeground="#000990", relief='raised',
                        command=command).grid(row=var_row + 2, column=1)
    
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