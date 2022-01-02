import numpy as np


class FilterButter3:
    """
        Class to perform online filter of the cursor
    """

    def __init__(self, pass_type):
        self._pass_type = pass_type

        # Array of input values, latest are in front
        # self._input_history = np.zeros([3, 8], 'float')
        self._in_cursor_history = np.zeros([3, 2], 'float')

        # Array of output values, latest are in front
        # self._output_history = np.zeros([3, 8], 'float')
        self._out_cursor_history = np.zeros([3, 2], 'float')

        # Coefficients for cursor (50 Hz) 4Hz - 3Hz - 2Hz - 1Hz
        if self._pass_type == "lowpass_4":
            self._a1 = -2.498608344691178  # -2.003797477370017; -2.250085081726394; -2.498608344691178
            self._a2 = 2.115254127003159  # 1.447054019489380; 1.756401381785953; 2.115254127003159
            self._a3 = -0.604109699507275  # -0.361795928227867; -0.468312111171712; -0.604109699507275
            self._b0 = 0.001567010350588  # 0.010182576736437; 0.004750523610981; 0.001567010350588
            self._b1 = 0.004701031051765  # 0.030547730209311; 0.014251570832943; 0.004701031051765
            self._b2 = 0.004701031051765  # 0.030547730209311; 0.014251570832943; 0.004701031051765
            self._b3 = 0.001567010350588  # 0.010182576736437; 0.004750523610981; 0.001567010350588

    def update_cursor(self, new_input, coord):
        new_output = self._b0 * new_input + self._b1 * self._in_cursor_history[0, coord] + \
                     self._b2 * self._in_cursor_history[1, coord] + self._b3 * self._in_cursor_history[2, coord] - \
                     self._a1 * self._out_cursor_history[0, coord] - self._a2 * self._out_cursor_history[1, coord] - \
                     self._a3 * self._out_cursor_history[2, coord]

        self._in_cursor_history[2, coord] = self._in_cursor_history[1, coord]
        self._in_cursor_history[1, coord] = self._in_cursor_history[0, coord]
        self._in_cursor_history[0, coord] = new_input

        self._out_cursor_history[2, coord] = self._out_cursor_history[1, coord]
        self._out_cursor_history[1, coord] = self._out_cursor_history[0, coord]
        self._out_cursor_history[0, coord] = new_output

    @property
    def filtered_value(self):
        return self._out_cursor_history[0, :]
