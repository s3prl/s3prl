import sys
import traceback
from s3prl import problem


def main():
    try:
        cls = getattr(problem, sys.argv[1])
    except:
        available_problems = [
            name
            for name in dir(problem)
            if not name.startswith("_") and isinstance(getattr(problem, name), type)
        ]
        print(traceback.format_exc())
        print(
            "Usage:\n"
            "1. s3prl-main [PROBLEM] -h\n"
            "2. python3 -m s3prl.main [PROBLEM] -h\n"
            "3. python3 s3prl/main.py [PROBLEM] -h\n"
            "\nPROBLEM should be an available class name in the s3prl.problem package.\n"
            f"Available options: {', '.join(available_problems)}"
        )
        exit(0)

    cls().main(sys.argv[2:])


if __name__ == "__main__":
    main()

