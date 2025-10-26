import datetime
import multiprocessing as mp
from pathlib import Path

from aoc.utils.template_tools import make_ascii_header


_here = Path(__file__).parent

files = list((_here / "aoc").glob("aoc_*/*.py"))


def file_name_good(year, day):
    try:
        _ = int(year)
        _ = int(day)
        return True
    except:
        return False


def year_day_from_file(f: Path) -> tuple[int, int]:
    year = int(f.parent.name.split("_")[-1])
    day = int(f.name.split(".py")[0].split("solution")[-1])
    return year, day


def fix(f: Path) -> str:
    year = int(f.parent.name.split("_")[-1])
    day = int(f.name.split(".py")[0].split("solution")[-1])
    s = f"year {year}, day {day}"
    print(f"Processing year {year}, day {day}")
    
    lines = f.read_text().splitlines()
    
    inside_main = False
    
    for i, line in reversed(list(enumerate(lines))):
        
        if line == "if __name__ == '__main__':":
            inside_main = True
        
        if line.startswith("def main():"):
            inside_main = False
        
        if line.startswith("def main():"):
            lines[i] = "def main() -> None:"
            print("Type hinted main")
        if line.startswith("def solve("):
            n_args = line.count(",") + 1
            if n_args != 1:
                raise RuntimeError(f"{s} has multiple arguments")
            assert line.count(":") == 2
            lines[i] = line.strip()[:-1]+" -> tuple[int|str, ...]:"
            print("Type hinted solve method")
        elif "import check_examples" in line:
            del lines[i]
            print("Removed example import")
        elif line.strip().startswith("check_examples("):
            del lines[i]
            print("Removed check examples")
        elif inside_main and line.strip().startswith("#"):
            del lines[i]
            print("Removed comment in main method")
        else:
            continue
        print()
        
    
    res = "\n".join(lines)
    return res


def replace_header(content: str, new_header: str) -> str:
    lines = content.splitlines()
    cut = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    header_lines = new_header.splitlines()
    
    reslines = header_lines + lines[cut:]
    assert len(reslines) == len(lines), new_header
    
    return "\n".join(reslines)


def main():
    now = datetime.datetime.now()
    solution_files = sorted([fn for fn in files if fn.name != "__init__.py"], key=year_day_from_file)
    
    new_content = dict()

    with mp.Pool() as pool:
        # Just start creating the ascii art headers in the background
        
        jobs = {
            fn: pool.apply_async(make_ascii_header, args=year_day_from_file(fn))
            for fn in solution_files
        }
        
        for fn in solution_files:
            new_header = jobs[fn].get()
            print(new_header)
            print()
            
            fixed = fix(fn)
            updated = replace_header(content=fixed, new_header=new_header)
            new_content[fn] = updated
            print()
        #
    
    delta = datetime.datetime.now() - now
    print(f"Finished in {delta}.")


if __name__ == '__main__':
    main()
