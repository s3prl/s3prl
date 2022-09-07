import sys
from s3prl import problem


if __name__ == "__main__":
    try:
        cls = getattr(problem, sys.argv[1])
    except:
        available_problems = [
            name
            for name in dir(problem)
            if not name.startswith("_") and isinstance(getattr(problem, name), type)
        ]
        print(
            "Usage:\n1. python3 -m s3prl.main [PROBLEM] -h\n2. python3 s3prl/main.py [PROBLEM] -h\n\n"
            f"PROBLEM should be an available class name in the s3prl.problem package: {', '.join(available_problems)}"
        )
        exit(0)

    cls().main(sys.argv[2:])
