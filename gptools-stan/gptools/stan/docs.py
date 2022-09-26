from IPython.core.magic import register_line_magic
from IPython.display import display_html
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name


@register_line_magic
def stan_file(line):
    lexer = get_lexer_by_name("stan")
    formatter = HtmlFormatter(noclasses=True)
    with open(line) as fp:
        code = highlight(fp.read(), lexer, formatter)

    display_html(code, raw=True)
