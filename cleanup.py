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
    except Exception:
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
        if i < 10 and inside_main:
            raise RuntimeError(f"Concluded we're in main at line {i} - shouldn't happen")
        
        if line == "if __name__ == '__main__':":
            inside_main = True
        
        if line.startswith("def main()"):
            inside_main = False
        
        if line.startswith("def main():"):
            lines[i] = "def main() -> None:"
            print("Type hinted main")
        elif line.startswith("def solve("):
            n_args = line.split(')')[0].count(",") + 1
            if n_args != 1:
                raise RuntimeError(f"{s} has multiple arguments")
            assert line.count(":") == 2
            lines[i] = line.strip()[:-1].split("->")[0].strip()+" -> tuple[int|str, int|str]:"
            print("Type hinted solve method")
        elif line.strip().startswith("def parse("):
            with_hint = line.replace('def parse(s)', 'def parse(s: str)')
            lines[i] = with_hint
            print(f"{line} -> {with_hint}")
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
    
    if len(reslines) != len(lines):
        msg = f"Attempted replacing header of {cut} lines with {len(header_lines)}\n\n{new_header}"
        raise RuntimeError(msg)
    
    return "\n".join(reslines)


def main(save=False) -> None:
    now = datetime.datetime.now()
    solution_files = sorted([fn for fn in files if fn.name != "__init__.py"], key=year_day_from_file)
    
    new_content: dict[Path, str] = dict()
    
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
    
    if save:
        for p, s in new_content.items():
            p.write_text(s)
        
    delta = datetime.datetime.now() - now
    print(f"Finished in {delta}.")


if __name__ == '__main__':
    main(save=True)
