from IPython.core.magic import register_cell_magic


def row(*args):
    from IPython import display
    from IPython.utils.capture import capture_output
    
    with capture_output(stdout=False, stderr=False, display=True) as c:
        for arg in args:
            display.display_html(arg)

    tds = [f"<td>{item.data['text/html']}</td>" for item in c.outputs]
    html = f"<table><tr>{''.join(tds)}</tr></table>"
    
    display.display_html(html, raw=True)


try:
    @register_cell_magic
    def capture_except(line, cell):
        from IPython import get_ipython
        ip = get_ipython()
        try:
            result = ip.ex(cell)
        except Exception as ex:
            print(f'{type(ex).__name__}: {ex}')
            result = None
        return result
except NameError:
    pass    