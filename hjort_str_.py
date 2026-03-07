import shutil
from math import floor, ceil
from hjortMatrixHelper import CFunc
from hjortMatrix import SETTINGS

def round_to_sig_figs(x, n):
    if x == 0 and not SETTINGS.suppress_zeroes:
        return "0." + "0" * (n - 1)
    elif x == 0 and SETTINGS.suppress_zeroes:
        return " "*(floor(n/2))+SETTINGS.suppress_zeroes[0]+" "*(ceil(n/2))
    
    formatted = f"{x:#.{n}g}"
    
    return formatted.replace("e+0", "e").replace("e-0", "e-")

def colorize(value: str, float_value: float, min_val: float, max_val: float, use_color: bool) -> str:
    if not use_color:
        return value
    range_val = max_val - min_val
    if range_val == 0:
        return value
    normalized = max(0.0, min(1.0, (float_value - min_val) / range_val))
    colors = [82, 118, 226, 214, 196]
    color_idx = int(normalized * (len(colors) - 1))
    return f"\033[38;5;{colors[color_idx]}m{value}\033[0m"

def hjort_str_(self) -> str:
    if self.m == 0 or self.n == 0:
        return "[]"

    entries = CFunc.matrix_to_list(self._ptr)
    digits = SETTINGS.sig_digits
    use_color = SETTINGS.use_color
    limit_prints = SETTINGS.limit_prints
    
    min_val = CFunc.matrix_get_min(self._ptr)
    max_val = CFunc.matrix_get_max(self._ptr)
    term_width = shutil.get_terminal_size().columns

    max_str_len = 0
    check_rows = entries[:limit_prints] if limit_prints else entries
    for r in check_rows:
        for val in r[:10]:
            s_len = len(round_to_sig_figs(val, digits))
            if s_len > max_str_len:
                max_str_len = s_len
    
    col_width = max_str_len + 1
    
    max_cols_terminal = (term_width - 8) // col_width
    limit_cols = min(limit_prints, max_cols_terminal) if limit_prints else max_cols_terminal

    output = ""
    num_rows = len(entries)
    num_cols = len(entries[0]) if entries else 0
    
    for a, row_data in enumerate(entries):
        if limit_prints and a == limit_prints - 1 and a != num_rows - 1:
            dot_row = ""

            if limit_cols and num_cols > limit_cols:
                visible_count = limit_cols + 1
            else:
                visible_count = num_cols
            
            for _ in range(visible_count):
                dot_row += ".".center(max_str_len) + " "
            output += f"⎢ {dot_row.rstrip()}{" ".center(int(max_str_len / 2))} ⎥\n"
            continue
        
        if limit_prints and a >= limit_prints and a != num_rows - 1:
            continue

        row_content = ""
        for b, val in enumerate(row_data):
            if limit_cols and b >= limit_cols - 1 and b != num_cols - 1:
                if b == limit_cols - 1:
                    row_content += ".".center(max_str_len) + " "
                continue

            raw_str = round_to_sig_figs(val, digits)
            padded_str = raw_str.rjust(max_str_len)
            row_content += colorize(padded_str, val, min_val, max_val, use_color) + " "

        if num_rows == 1:
            left, right = "[", "]"
        elif a == 0:
            left, right = "⎡", "⎤"
        elif a == num_rows - 1:
            left, right = "⎣", "⎦"
        else:
            left, right = "⎢", "⎥"

        output += f"{left} {row_content.rstrip()} {right}{"\n" if a != num_rows - 1 else ""}"

    return output