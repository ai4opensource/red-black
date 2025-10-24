# red-black
Demonstration of red black trees with bottom up and top down insertion 

## Operating System Python Setup
You need to have Python-tk installed, which is not the default. 

### Ubuntu
```bash
sudo apt update
sudo apt install python3-tk # Use reinstall if this causes an error
python3 -m tkinter # to test
```

### OSX
```bash
brew install python@3.12-tk
brew reinstall python python@3.12-tk --with-tcl-tk # Use reinstall 
python3.12 -m tkinter
```

### Windows
#✅ If you installed Python from python.org, Tkinter is included by default.
#To verify, open a terminal (Command Prompt or PowerShell) and run:

```bash
python -m tkinter
python -m tkinter
```

If a small blank window appears, Tkinter is installed and working.

If you get an error like ModuleNotFoundError: No module named 'tkinter', reinstall Python and make sure to check:

✅ “tcl/tk and IDLE” during installation.

You can also repair the installation by re-running the Python installer and selecting Modify → tcl/tk and IDLE.

## Python virtual environment setup
1. `python3 -m venv .venv`
2. `source .venv/bin/activate`

## Running
1. `python redblack.py`