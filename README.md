# gSTED-sFCS

Python-based software for our optical measurement system.

## How this project works:

#### <u>GUI</u>

1. `.ui` files are created graphically using Qt Designer, and are dynamically compiled into Python from within the `gui.py` module.

2. Signals/slots (similar to callbacks) are coded as methods for the classes that are named after the windows (`MainWindow`, `SettingsWindow`, etc.).

3. Finally, when running the project (`gSTEDsFCS.py`), a `MainWindow()` object is instantiated and the window is shown on screen.

#### <u>Physical Device Control</u>

1. the `drivers` package contains an `__init__.py` file which imports the necessary drivers from existing packages (pyVISA, nidaqmx, instrumental etc.), which provide the interface to the physical devices.

2. `drivers` is imported into the implementation modules which communicate with the devices.

#### <u>Implemetation</u>

1. ## What's here/Noteworthy files:
- `./gSTEDsFCS.e4p` - 'eric IDE' project file, which nicely ties everything together.

- `./gui/gui.py` - GUI module.

- `./implementation/implementation.py` - main implementation module.

- `./implementation/constants.py` - constants used across the project.

- `./drivers` - package for physical devices interfaces (used only in implementation modules).

- `./settings/default_settings.csv` - default setting file, initial values for all GUI forms.

## Dependencies/Imports:

- Python 3.x

- PyQt5

- #pyqt5ac (placeholder)

- pandas

- matplotlib

- pyVISA

- instrumental-lib

- nicelib

- Thorlabs [DCx Camera interfaces](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam)

## Installing Eric IDE on Windows:

Eric IDE is notoriously complicated to install properly. I've tries to make it easier with the following steps (assuming Python is installed and added to PATH):

1. [Download](https://sourceforge.net/projects/eric-ide/) Eric IDE distribution and extract to temporary folder.

2. In Command Prompt:
   
   * Go to a directory where you want to install Eric (along with a virtual environment), e.g. `cd c:\Python`
   
   * Create a virtual environment, e.g. `python -m venv eric_env`
   
   * Activate virtual environment, e.g. `eric_env\Scripts\activate.bat` (`(eric_env)` should appear on left of next).
   
   * Install eric via pip e.g. `python -m pip install eric-ide`, say `y` to all.
   
   * Run install.py from the distribution (still in the virtual environment) e.g. `python C:\temporary folder\eric6-20.10\install.py`
   
   * To have access to Qt Designer, install pyqt5-tools e.g. `pip install pyqt5-tools`, then copy the contents of `\eric_env\Lib\site-packages\pyqt5_tools\Qt` into `\eric_env\Lib\site-packages\PyQt5\Qt`

3. That's it, icons should appear on desktop.
