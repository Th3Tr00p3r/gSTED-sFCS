# gSTED-sFCS

Python-based control GUI and analysis software for our optical measurement system.

## How this project works:

#### <u>GUI</u>

1. `.ui` files are created graphically using Qt Designer, and are dynamically compiled into Python from within the `gui.py` module.

2. Signals/slots (similar to callbacks) are coded as methods for the classes that are named after the windows (`MainWindow`, `SettingsWindow`, etc.).

3. Finally, when running the project (`main.py`), a `MainWindow()` object is instantiated and the window is shown on screen.

#### <u>Implemetation</u>

1. The`./logic` package contains everything other than the most basic GUI - any button press triggers a slot in `gui.py` which in turn calls a function from the `slot_implementations.py` module, which in turn calls upon any of the other modules. In this way the GUI is completely seperated from the implementation and the physical devices.

#### <u>Physical Device Control</u>

1. The `devices.py` module contains a high-level API for communicating with the instruments, subclassed from the more general `drivers.py`, where existing packages (pyVISA, nidaqmx, instrumental etc.) are imported.

## What's here/Noteworthy files:

- `./sfcs_sted.epj` - 'Eric IDE' project file, which nicely ties everything together.

- `./gui/gui.py` - GUI module, containing strictly user interaction signals and slots, the slots implemented in `logic.py`.

- `./logic/slot_implementations.py` - general implementations, slowly taking shape when things are moved to seperate modules. Currently contains mostly implementations of the main application stuff (`App()` class) and the GUI windows (i.e. `MainWin()` class).

- `./logic/devices.py` - module implementing front-end device interaction with `logic.py` and back-end driver communication with physical instruments through subclassing driver implementations in `drivers.py`

- `./logic/helper.py` - general helper functions used throughout the project.

- `./settings/default_settings.csv` - default setting file, initial values for all editable GUI widgets.

- `.pre-commit-config.yaml`, `.flake8`, `pyproject.toml` - these are configuration files for git pre-commit hooks (stuff performed right before commiting). They enforce better code quality (and let us focus on the content).

## Notable Dependencies:

- Python 3.10.10

- Eric IDE (optional, but was invalueable during development) - very useful for Python GUI projects.

- [PyQt5](https://pypi.org/project/PyQt5/) - Qt-Python API, used for GUI and timers (**installed with Eric IDE**).

- [qasync](https://github.com/CabbageDevelopment/qasync) - to make PyQt5 work with Python's `asyncio` library.

- NumPy.

- matplotlib - used for plotting (currently only the camera, later the graphs).

- [pyVISA](https://pypi.org/project/PyVISA/) - communication with VISA-supported devices.

- [nidaqmx](https://pypi.org/project/nidaqmx/) - communication with NI-DAQmx supported devices.

- [DCx Camera interfaces](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam), [Instrumental-lib](https://pypi.org/project/Instrumental-lib/), [nicelib](https://pypi.org/project/NiceLib/),  - communication with DC480 cameras.

- [ftd2xx](https://github.com/snmishra/ftd2xx) - communication with UM232, which is a usb port to the FPGA (TDC data).


## Installing Eric IDE on Windows:

Eric IDE is notoriously complicated to install properly. I've tried to make it easier with the following steps (assuming Python is installed and added to PATH):

1. [Download](https://sourceforge.net/projects/eric-ide/) Eric IDE distribution and extract to temporary folder.

2. In Command Prompt:
   
   * Go to a directory where you want to install Eric (along with a virtual environment), e.g. `cd c:\Python`
   
   * Create a virtual environment, e.g. `python -m venv eric_env`
   
   * Activate virtual environment, e.g. `eric_env\Scripts\activate.bat` (or `& .\activate` in powershell). `(eric_env)` should appear on left of text.
   
   * Install eric via pip e.g. `pip install eric-ide`.
   
   * Run install.py from the distribution (still in the virtual environment) e.g. `python C:\temporary folder\eric6-20.10\install.py`.
   
   * To have access to Qt Designer from within Eric IDE, go to 'settings' --> 'Qt' --> 'Qt Tools' and set 'Tools Directory' to `\eric_env\Lib\site-packages\qt5_applications\Qt\bin`.

3. That's it, icons should appear on desktop.

## Compiling the Software Correlator Dynamic Library

Windows:

1. Install [MinGW](https://sourceforge.net/projects/mingw/)

2. add /bin to Path

3. in cmd or PowerShell, cd to `/SoftCorrelatorDynamicLib`

4. Compile: `g++ -o SoftCorrelatorDynamicLib_win32.so -shared -fPIC -O2 SoftCorrelatorDynamicLib.cpp Correlator.cpp CountCorrelator.cpp CPhDelayCrossCorrelator.cpp CPhDelayLifeTimeCrossCorrelator.cpp`
