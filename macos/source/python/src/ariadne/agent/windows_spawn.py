"""Internal start gate for a child that must be assigned to a Windows Job."""
import subprocess
import sys


if __name__ == "__main__":
    if sys.stdin.buffer.read(1) != b"1":
        raise SystemExit(125)
    raise SystemExit(subprocess.call(sys.argv[1:], stdin=subprocess.DEVNULL))
