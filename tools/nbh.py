"""notebook 生成ヘルパ（完成版 full / 穴埋め版 blank を切替）."""
import nbformat as nbf

MODE = "full"  # "full"=ans用完成版 / "blank"=main用穴埋め


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(s):
    return nbf.v4.new_code_cell(s)


def sol(full, blank):
    """穴埋め対象のコードセル．full=解答, blank=TODO版."""
    return code(full if MODE == "full" else blank)


def build(cells, path):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nbf.write(nb, path)
    print("wrote", path, f"({len(cells)} cells, mode={MODE})")
